"""
evaluate.py
-----------
Production-grade DeepEval benchmark runner for Papeer Agentic RAG.
Evaluates 10 golden test cases against 'documents/Openclaw_Research_Report.pdf'.

Architecture:
- RAG Pipeline Generator : 'openai/gpt-oss-120b' (production RAG synthesis & tool planning)
- RAG Pipeline Classifier: 'openai/gpt-oss-20b'  (contextualizer, router, relevancy check)
- DeepEval Judge Model   : 'openai/gpt-oss-20b'  (consistent 20B judge across all 10 cases)

Features:
- Separate per-case logging: saves individual JSON + human-readable .log files under 'eval_logs/'
- Incremental disk checkpointing after EVERY case (saved directly to 'eval_results.json')
- Automatic resume capability: inspects completed per-case logs/results and starts from the next unfinished case
- Smart Groq 429 backoff: parses exact 'try again in XmYs' retry durations from Groq rate-limit messages
"""

import argparse
import asyncio
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Force unbuffered UTF-8 output for real-time tracking
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

import backend.rag_graph as rag_graph_module
from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.vector_store import add_paper

load_dotenv(override=True)

PDF_PATH         = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE     = Path("goldens.json")
RESULTS_FILE     = Path("eval_results.json")
BACKUP_FILE      = Path("eval_results.backup.json")
LOGS_DIR         = Path("eval_logs")
JUDGE_MODEL      = "openai/gpt-oss-20b"
METRIC_THRESHOLD = 0.70

EXPECTED_METRICS = {
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
    "ContextualRelevancyMetric",
    "AnswerRelevancyMetric",
    "FaithfulnessMetric",
}

# Concurrency limiter: prevent DeepEval from firing parallel async bursts at Groq
_semaphore = asyncio.Semaphore(1)


def _extract_wait_seconds(error_msg: str, default_wait: float) -> float:
    """Extracts exact wait time from Groq error messages like 'Please try again in 4m48.144s' or '12.5s'."""
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d\.]+)s", error_msg, re.IGNORECASE)
    if m:
        hrs = float(m.group(1)) if m.group(1) else 0.0
        mins = float(m.group(2)) if m.group(2) else 0.0
        secs = float(m.group(3))
        return hrs * 3600.0 + mins * 60.0 + secs + 3.0
    return default_wait


class DailyQuotaExhaustedError(RuntimeError):
    """Raised when Groq requires waiting > 8 minutes (TPD exhaustion), allowing clean checkpoint exit."""
    pass


# ── Groq Judge Wrapper for DeepEval (gpt-oss-20b) ────────────────────────────

class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = JUDGE_MODEL):
        self.model_name = model_name
        super().__init__(model_name)

    def load_model(self):
        return ChatGroq(
            model=self.model_name,
            model_kwargs={"response_format": {"type": "json_object"}},
            max_tokens=4096,
            request_timeout=60,
            max_retries=2,
        )

    def _safe_prompt(self, prompt: str) -> str:
        if "json" not in prompt.lower():
            prompt += "\n\nCRITICAL INSTRUCTION: You must respond entirely with a valid JSON object matching the requested schema."
        return prompt

    def generate(self, prompt: str) -> str:
        p = self._safe_prompt(prompt)
        for attempt in range(8):
            try:
                time.sleep(1.5)
                model = self.load_model()
                return model.invoke(p).content
            except Exception as e:
                msg = str(e)
                low = msg.lower()
                if "429" in low or "rate limit" in low or "too many requests" in low:
                    wait = _extract_wait_seconds(msg, 15.0 * (attempt + 1))
                    if wait > 480.0:
                        raise DailyQuotaExhaustedError(f"Groq TPD quota requires waiting {wait/60:.1f}m.")
                    print(f"\n      ⚠️ Judge Sync 429 hit. Sleeping {wait:.1f}s... (attempt {attempt+1}/8)", flush=True)
                    time.sleep(wait)
                elif "400" in low or "json_validate_failed" in low or "failed to generate json" in low:
                    print(f"\n      ⚠️ Judge Sync JSON validation hiccup. Retrying in 3s... (attempt {attempt+1}/8)", flush=True)
                    time.sleep(3.0)
                else:
                    raise
        raise RuntimeError("Groq judge sync generation exhausted all retries.")

    async def a_generate(self, prompt: str) -> str:
        async with _semaphore:
            await asyncio.sleep(2.0)
            p = self._safe_prompt(prompt)
            for attempt in range(8):
                try:
                    model = self.load_model()
                    res = await model.ainvoke(p)
                    return res.content
                except Exception as e:
                    msg = str(e)
                    low = msg.lower()
                    if "429" in low or "rate limit" in low or "too many requests" in low:
                        wait = _extract_wait_seconds(msg, 15.0 * (attempt + 1))
                        if wait > 480.0:
                            raise DailyQuotaExhaustedError(f"Groq TPD quota requires waiting {wait/60:.1f}m.")
                        print(f"\n      ⚠️ Judge Async 429 hit. Sleeping {wait:.1f}s... (attempt {attempt+1}/8)", flush=True)
                        await asyncio.sleep(wait)
                    elif "400" in low or "json_validate_failed" in low or "failed to generate json" in low:
                        print(f"\n      ⚠️ Judge Async JSON validation hiccup. Retrying in 3s... (attempt {attempt+1}/8)", flush=True)
                        await asyncio.sleep(3.0)
                    else:
                        raise
        raise RuntimeError("Groq judge async generation exhausted all retries.")

    def get_model_name(self) -> str:
        return self.model_name


# ── Per-Case Logging & Incremental Checkpoint Store ──────────────────────────

def is_case_complete(case: dict) -> bool:
    """Returns True if this test case record has valid output and all 5 metrics evaluated without error."""
    if not isinstance(case, dict):
        return False
    if case.get("success") is not True:
        return False
    actual_output = case.get("actual_output") or ""
    if not actual_output or "RAG Error:" in actual_output or "RateLimitError" in actual_output:
        return False
    metrics = case.get("metrics") or []
    if len(metrics) < 5:
        return False
    names = {m.get("name") for m in metrics}
    return EXPECTED_METRICS.issubset(names)


def load_completed_cases() -> dict[int, dict]:
    """Loads all completed cases from both per-case logs and RESULTS_FILE."""
    completed: dict[int, dict] = {}

    if RESULTS_FILE.exists():
        try:
            data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    cnum = item.get("case_number")
                    if cnum is not None and is_case_complete(item):
                        completed[int(cnum)] = item
        except Exception as e:
            print(f"⚠️ Error reading {RESULTS_FILE}: {e}")

    if LOGS_DIR.exists():
        for log_file in sorted(LOGS_DIR.glob("case_[0-9][0-9].json")):
            try:
                item = json.loads(log_file.read_text(encoding="utf-8"))
                cnum = item.get("case_number")
                if cnum is not None and is_case_complete(item):
                    completed[int(cnum)] = item
            except Exception as e:
                print(f"⚠️ Error reading {log_file}: {e}")

    return completed


def load_partial_case(case_num: int) -> dict | None:
    """Loads intra-case partial progress (RAG answer + already evaluated metrics) if present."""
    partial_path = LOGS_DIR / f"case_{case_num:02d}_partial.json"
    if partial_path.exists():
        try:
            data = json.loads(partial_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("case_number") == case_num:
                return data
        except Exception:
            pass
    return None


def save_partial_case(partial_record: dict, case_num: int):
    """Persists partial RAG answer and individual completed metrics mid-case to avoid re-running on 429."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    partial_path = LOGS_DIR / f"case_{case_num:02d}_partial.json"
    partial_path.write_text(json.dumps(partial_record, indent=2, ensure_ascii=False), encoding="utf-8")


def save_case_checkpoint(case_record: dict, case_num: int):
    """Saves an individual case log file (JSON + human-readable .log), removes partial cache, and updates eval_results.json."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Remove partial checkpoint now that all 5 metrics are complete
    partial_path = LOGS_DIR / f"case_{case_num:02d}_partial.json"
    if partial_path.exists():
        try:
            partial_path.unlink()
        except Exception:
            pass

    # 1. Save dedicated per-case JSON log
    case_json_path = LOGS_DIR / f"case_{case_num:02d}.json"
    case_json_path.write_text(json.dumps(case_record, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. Save dedicated human-readable per-case .log report
    case_txt_path = LOGS_DIR / f"case_{case_num:02d}.log"
    lines = [
        "=" * 72,
        f"CASE [{case_num}/10] EVALUATION LOG",
        f"RAG Generator  : {case_record.get('rag_model')}",
        f"DeepEval Judge : {case_record.get('judge_model')}",
        f"Timestamp      : {case_record.get('timestamp')}",
        f"RAG Latency    : {case_record.get('rag_latency_sec', 0):.2f}s",
        f"Judge Latency  : {case_record.get('eval_latency_sec', 0):.2f}s",
        f"Retrieved Docs : {case_record.get('retrieved_chunks_count', 0)} chunks",
        f"All Metrics OK : {case_record.get('success')}",
        "=" * 72,
        "",
        "--- INPUT QUERY ---",
        case_record.get("input", ""),
        "",
        "--- EXPECTED OUTPUT ---",
        case_record.get("expected_output", ""),
        "",
        "--- ACTUAL RAG OUTPUT (120B) ---",
        case_record.get("actual_output", ""),
        "",
        "--- RETRIEVED CONTEXT CHUNKS ---",
    ]
    for idx, chunk in enumerate(case_record.get("retrieval_context", []), start=1):
        lines.append(f"[Chunk {idx}] {chunk}\n")

    lines.append("--- DEEPEVAL METRICS SCORECARD (20B JUDGE) ---")
    for m in case_record.get("metrics", []):
        status = "PASSED" if m.get("passed") else "FAILED"
        lines.append(
            f"• {m['name']:<28}: {m['score']:.4f} [{status}] ({m.get('duration_sec', 0):.1f}s)\n"
            f"  Reason: {m.get('reason', 'n/a')}"
        )
    lines.append("=" * 72 + "\n")
    case_txt_path.write_text("\n".join(lines), encoding="utf-8")

    # 3. Merge into consolidated RESULTS_FILE and BACKUP_FILE
    completed_map = load_completed_cases()
    completed_map[case_num] = case_record

    ordered_list = [completed_map[cnum] for cnum in sorted(completed_map.keys())]
    payload = json.dumps(ordered_list, indent=2, ensure_ascii=False)
    RESULTS_FILE.write_text(payload, encoding="utf-8")
    try:
        BACKUP_FILE.write_text(payload, encoding="utf-8")
    except Exception:
        pass

    print(f"  💾 Saved separate case logs: {case_json_path} & {case_txt_path}", flush=True)
    print(f"  💾 Updated master checkpoint: {RESULTS_FILE} ({len(ordered_list)}/10 cases stored)", flush=True)


# ── RAG Pipeline Invocation with Backoff ─────────────────────────────────────

def run_rag_query(graph, query: str, session_id: str, thread_id: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": thread_id}}
    for attempt in range(8):
        try:
            state = graph.invoke(
                {
                    "messages": [HumanMessage(content=query)],
                    "session_id": session_id,
                    "query": query,
                    "retrieved_docs": [],
                    "retrieval_attempts": 0,
                    "rewrite_count": 0,
                },
                config=config,
            )
            answer = state.get("answer") or ""
            if "RateLimitError" in answer or "429" in answer:
                raise RuntimeError(f"RAG returned rate-limit response: {answer[:160]}")
            ctx = [d.page_content for d in (state.get("retrieved_docs") or [])]
            return answer, ctx
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            if "429" in low or "rate limit" in low or "ratelimiterror" in low:
                wait = _extract_wait_seconds(msg, 20.0 * (attempt + 1))
                if wait > 480.0:
                    raise DailyQuotaExhaustedError(f"Groq TPD quota requires waiting {wait/60:.1f}m.")
                print(f"\n    ⚠️ RAG Rate Limit hit. Backing off {wait:.1f}s... (attempt {attempt+1}/8)", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("RAG query failed after multiple rate limit retries.")


# ── Benchmark Summary Printer ────────────────────────────────────────────────

def print_summary_report(results: list[dict]):
    print("\n" + "=" * 72, flush=True)
    print(f"      FINAL DEEPEVAL BENCHMARK REPORT (RAG: 120B | Judge: 20B)", flush=True)
    print("=" * 72, flush=True)

    metric_scores: dict[str, list[float]] = {}
    metric_passed: dict[str, list[int]] = {}

    for item in results:
        for m in item.get("metrics", []):
            name = m["name"]
            metric_scores.setdefault(name, []).append(float(m.get("score", 0.0)))
            metric_passed.setdefault(name, []).append(1 if m.get("passed") else 0)

    for name, scores in metric_scores.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        passed_cnt = sum(metric_passed[name]) if name in metric_passed else 0
        total_cnt = len(scores)
        status = "PASSED ✅" if avg >= METRIC_THRESHOLD else "FAILED ❌"
        print(f"  {name:<30}: {avg:.4f}  [{passed_cnt}/{total_cnt} passed]  → {status}", flush=True)

    print("=" * 72, flush=True)
    print(f"Total evaluated cases recorded : {len(results)}/10")
    print(f"Per-case logs directory        : {LOGS_DIR.resolve()}")
    print(f"Consolidated results file      : {RESULTS_FILE.resolve()}\n", flush=True)


# ── Main Controller ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Papeer Resumable DeepEval Benchmark Runner")
    parser.add_argument("--reset", action="store_true", help="Archive existing results/logs and start fresh from Case 1.")
    parser.add_argument("--case", type=int, default=None, help="Force re-run of a single case number (1-10).")
    parser.add_argument("--start", type=int, default=1, help="Start case number (default: 1).")
    parser.add_argument("--end", type=int, default=10, help="End case number (default: 10).")
    args = parser.parse_args()

    if not GOLDENS_FILE.exists():
        print(f"❌ Error: {GOLDENS_FILE} not found. Cannot run evals.")
        sys.exit(1)

    goldens = json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))
    total_cases = len(goldens)

    print("=" * 72, flush=True)
    print("      PAPEER AGENTIC RAG — DEEPEVAL BENCHMARK (120B RAG + 20B JUDGE)", flush=True)
    print(f"      RAG Router & Gen  : {rag_graph_module.LLM_GEN}", flush=True)
    print(f"      RAG Fast Aux LLM  : {rag_graph_module.LLM_FAST}", flush=True)
    print(f"      DeepEval Judge    : {JUDGE_MODEL}", flush=True)
    print(f"      Per-Case Logs     : {LOGS_DIR}/case_XX.json & .log", flush=True)
    print(f"      Total Test Cases  : {total_cases}", flush=True)
    print("=" * 72, flush=True)

    if args.reset:
        ts = int(time.time())
        if RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 10:
            archive_name = f"eval_results_archive_{ts}.json"
            RESULTS_FILE.rename(archive_name)
            print(f"📦 Archived existing {RESULTS_FILE} to {archive_name}.", flush=True)
        if LOGS_DIR.exists():
            archive_dir = Path(f"eval_logs_archive_{ts}")
            LOGS_DIR.rename(archive_dir)
            print(f"📦 Archived existing {LOGS_DIR}/ to {archive_dir}/.", flush=True)
        RESULTS_FILE.write_text("[]", encoding="utf-8")
        print("🔄 Results and per-case logs reset. Starting fresh from Case 1.\n", flush=True)

    # Inspect existing completed cases on disk
    completed_map = load_completed_cases()
    completed_case_nums = sorted(completed_map.keys())

    if args.case is not None:
        target_case_nums = [args.case]
    else:
        target_case_nums = [i for i in range(args.start, min(args.end, total_cases) + 1)]

    pending_case_nums = [
        c for c in target_case_nums
        if c not in completed_map or args.case is not None
    ]

    print(f"📊 Checkpoint Inspection:")
    print(f"   • Already completed on disk : {completed_case_nums if completed_case_nums else 'None'}")
    print(f"   • Pending cases to run      : {pending_case_nums if pending_case_nums else 'None (All 10 Done!)'}\n", flush=True)

    if not pending_case_nums:
        print("🎉 All requested test cases have already been evaluated completely!")
        print_summary_report([completed_map[k] for k in sorted(completed_map.keys())])
        return

    # Initialize RAG Pipeline and Document Index
    print("-> [1/3] Loading PDF document...", flush=True)
    docs = load_document(PDF_PATH)
    print(f"   Loaded {len(docs)} document chunks from {PDF_PATH}.", flush=True)

    print("-> [2/3] Compiling LangGraph RAG workflow...", flush=True)
    graph = build_graph()

    print("-> [3/3] Initializing vector store collection...", flush=True)
    eval_session_id = f"eval_session_{uuid4()}"
    add_paper(docs, eval_session_id)
    print(f"   Indexed into Qdrant & BM25 with session ID: {eval_session_id}\n", flush=True)

    # Initialize 20B Judge and Metrics
    judge = GroqJudge(model_name=JUDGE_MODEL)
    metrics = [
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=judge),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=judge),
        ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge),
    ]

    # Process each pending case sequentially
    for loop_idx, case_num in enumerate(pending_case_nums):
        pair = goldens[case_num - 1]
        print("\n" + "-" * 72, flush=True)
        print(f"▶ EXECUTING CASE [{case_num}/{total_cases}]: '{pair['input'][:65]}...'", flush=True)
        print("-" * 72, flush=True)

        partial = load_partial_case(case_num)
        partial_metrics_map: dict[str, dict] = {}
        if partial and partial.get("actual_output") and "RAG Error:" not in partial.get("actual_output", ""):
            answer = partial["actual_output"]
            ctx = partial.get("retrieval_context") or []
            rag_time = float(partial.get("rag_latency_sec", 0.0))
            for pm in partial.get("metrics", []):
                if pm.get("name"):
                    partial_metrics_map[pm["name"]] = pm
            print(
                f"  [1] Reusing cached RAG answer from partial checkpoint "
                f"({len(ctx)} chunks, {len(partial_metrics_map)}/5 metrics already judged).",
                flush=True,
            )
        else:
            # 1. RAG Query via gpt-oss-120b
            query = pair["input"] + " as per the report in knowledge base"
            thread_id = f"eval_thread_{case_num}_{uuid4()}"
            print(f"  [1] Generating RAG Answer via LangGraph ({rag_graph_module.LLM_GEN})...", end=" ", flush=True)
            t0_rag = time.time()
            try:
                answer, ctx = run_rag_query(graph, query, eval_session_id, thread_id)
                rag_time = time.time() - t0_rag
                print(f"DONE in {rag_time:.1f}s ({len(ctx)} chunks retrieved)", flush=True)
                partial = {
                    "case_number": case_num,
                    "rag_model": rag_graph_module.LLM_GEN,
                    "judge_model": JUDGE_MODEL,
                    "rag_latency_sec": round(rag_time, 2),
                    "input": pair["input"],
                    "expected_output": pair["expected_output"],
                    "actual_output": answer,
                    "retrieval_context": ctx,
                    "metrics": [],
                }
                save_partial_case(partial, case_num)
            except DailyQuotaExhaustedError as dq:
                print(f"\n🛑 {dq}")
                print(f"💾 All previously completed cases are safely preserved in {LOGS_DIR}/ and {RESULTS_FILE}.")
                print(f"▶ Run `python evaluate.py` again later to automatically resume from Case [{case_num}/10].\n", flush=True)
                all_completed = load_completed_cases()
                if all_completed:
                    print_summary_report([all_completed[k] for k in sorted(all_completed.keys())])
                return
            except Exception as e:
                rag_time = time.time() - t0_rag
                print(f"FAILED: {e}", flush=True)
                answer = f"RAG Error: {e}"
                ctx = []
                LOGS_DIR.mkdir(parents=True, exist_ok=True)
                fail_log = LOGS_DIR / f"case_{case_num:02d}_incomplete.json"
                fail_log.write_text(json.dumps({
                    "case_number": case_num,
                    "rag_model": rag_graph_module.LLM_GEN,
                    "judge_model": JUDGE_MODEL,
                    "actual_output": answer,
                    "retrieval_context": ctx,
                    "success": False,
                }, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"▶ Stopping loop cleanly so next run resumes from Case [{case_num}/{total_cases}].", flush=True)
                break

        # Trim text to keep judge prompt well within Groq 8k TPM window
        ctx_trimmed = [c[:550] for c in ctx[:5]]
        answer_trimmed = answer[:1200]

        test_case = LLMTestCase(
            input=pair["input"],
            actual_output=answer_trimmed,
            expected_output=pair["expected_output"],
            retrieval_context=ctx_trimmed,
        )

        # 2. Evaluate 5 DeepEval Metrics via gpt-oss-20b Judge
        print(f"  [2] Evaluating 5 DeepEval Metrics via GroqJudge ({JUDGE_MODEL})...", flush=True)
        t0_eval = time.time()
        case_metrics = []
        partial_ok_metrics = []
        overall_ok = True

        for m_idx, metric in enumerate(metrics, start=1):
            m_name = metric.__class__.__name__
            if m_name in partial_metrics_map:
                cached_m = partial_metrics_map[m_name]
                status = "✅ PASSED" if cached_m.get("passed") else "❌ FAILED"
                print(
                    f"      ({m_idx}/5) {m_name:<28} ... ⏩ CACHED ({status}, score: {cached_m['score']:.2f})",
                    flush=True,
                )
                case_metrics.append(cached_m)
                partial_ok_metrics.append(cached_m)
                continue

            print(f"      ({m_idx}/5) {m_name:<28} ...", end=" ", flush=True)
            t0_m = time.time()
            ok = False

            for attempt in range(8):
                try:
                    metric.measure(test_case)
                    ok = True
                    break
                except DailyQuotaExhaustedError as dq:
                    print(f"\n\n🛑 {dq}")
                    print(f"💾 Partial progress ({len(partial_ok_metrics)}/5 metrics) cached in {LOGS_DIR}/case_{case_num:02d}_partial.json; all completed cases safe in {RESULTS_FILE}.")
                    print(f"▶ Run `python evaluate.py` again later to resume right at {m_name} of Case [{case_num}/10].\n", flush=True)
                    all_completed = load_completed_cases()
                    if all_completed:
                        print_summary_report([all_completed[k] for k in sorted(all_completed.keys())])
                    return
                except Exception as e:
                    msg = str(e)
                    low = msg.lower()
                    if "429" in low or "rate limit" in low or "too many requests" in low:
                        wait = _extract_wait_seconds(msg, 15.0 * (attempt + 1))
                        if wait > 480.0:
                            print(f"\n\n🛑 Groq TPD quota requires waiting {wait/60:.1f}m.")
                            print(f"💾 Partial progress ({len(partial_ok_metrics)}/5 metrics) cached in {LOGS_DIR}/case_{case_num:02d}_partial.json; all completed cases safe in {RESULTS_FILE}.")
                            print(f"▶ Run `python evaluate.py` again later to resume right at {m_name} of Case [{case_num}/10].\n", flush=True)
                            all_completed = load_completed_cases()
                            if all_completed:
                                print_summary_report([all_completed[k] for k in sorted(all_completed.keys())])
                            return
                        print(f"\n          ⚠️ Rate limit. Sleeping {wait:.1f}s... (attempt {attempt+1}/8)", end=" ", flush=True)
                        time.sleep(wait)
                    elif "400" in low or "json_validate_failed" in low or "failed to generate json" in low:
                        print(f"\n          ⚠️ JSON validation hiccup. Retrying in 3s... (attempt {attempt+1}/8)", end=" ", flush=True)
                        time.sleep(3.0)
                    else:
                        print(f"FAILED ({e})", end=" ", flush=True)
                        break

            m_time = time.time() - t0_m
            score = getattr(metric, "score", None) or 0.0
            passed = getattr(metric, "success", False) or False
            reason = getattr(metric, "reason", "n/a") or "n/a"
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status} (score: {score:.2f}) [{m_time:.1f}s]", flush=True)

            m_record = {
                "name": m_name,
                "score": float(score),
                "passed": bool(passed),
                "reason": str(reason),
                "duration_sec": round(m_time, 2),
            }
            case_metrics.append(m_record)

            if ok:
                partial_ok_metrics.append(m_record)
                if partial is not None:
                    partial["metrics"] = list(partial_ok_metrics)
                    save_partial_case(partial, case_num)
            else:
                overall_ok = False

            time.sleep(4)  # Gentle pacing between metrics to protect 8k TPM window

        eval_time = time.time() - t0_eval

        # 3. Construct complete per-case log record and save immediately ONLY if all 5 metrics ran cleanly
        case_record = {
            "case_number": case_num,
            "rag_model": rag_graph_module.LLM_GEN,
            "judge_model": JUDGE_MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rag_latency_sec": round(rag_time, 2),
            "eval_latency_sec": round(eval_time, 2),
            "retrieved_chunks_count": len(ctx),
            "input": pair["input"],
            "expected_output": pair["expected_output"],
            "actual_output": answer,
            "retrieval_context": ctx,
            "success": overall_ok,
            "metrics": case_metrics,
        }

        if overall_ok:
            save_case_checkpoint(case_record, case_num)
        else:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            fail_log = LOGS_DIR / f"case_{case_num:02d}_incomplete.json"
            fail_log.write_text(json.dumps(case_record, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\n⚠️ Case [{case_num}/{total_cases}] had metric failures (not marked complete). Saved debug log to {fail_log}.", flush=True)
            print(f"▶ Stopping loop cleanly so next run resumes from Case [{case_num}/{total_cases}].", flush=True)
            break

        # Print case scorecard
        print(f"\n  🎯 CASE [{case_num}/{total_cases}] SCORECARD (120B RAG | 20B Judge):", flush=True)
        for m in case_metrics:
            icon = "✅" if m["passed"] else "❌"
            print(f"     {icon} {m['name']:<28}: {m['score']:.2f}", flush=True)

        # 4. Inter-case cooldown to replenish Groq TPM bucket
        if loop_idx < len(pending_case_nums) - 1:
            print(f"\n  ⏳ Inter-case cooldown: 20s to replenish Groq TPM bucket...", flush=True)
            time.sleep(20)

    # Final summary over all completed cases
    all_completed = load_completed_cases()
    print_summary_report([all_completed[k] for k in sorted(all_completed.keys())])


if __name__ == "__main__":
    main()
