"""
eval_6_to_10.py
---------------
Dedicated benchmark runner for Cases 6 through 10 of the DeepEval golden test set.
Features:
- Live unbuffered progress output (flush=True everywhere) for real-time tracking
- Asynchronous semaphore in GroqJudge to serialize DeepEval requests (prevents 429 concurrency bursts)
- max_tokens=2048 to prevent Groq JSON truncation
- Immediate incremental disk saving to both 'eval_results_6_to_10.json' and 'eval_results.json'
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

# Force unbuffered UTF-8 output so every line is streamed live
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

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

from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.vector_store import add_paper

load_dotenv(override=True)

PDF_PATH             = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE         = Path("goldens.json")
RESULTS_6_TO_10_FILE = Path("eval_results_6_to_10.json")
RESULTS_MAIN_FILE    = Path("eval_results.json")
METRIC_THRESHOLD     = 0.7

# ── Concurrency limiter: prevent DeepEval from firing 20 async requests at once ──
_semaphore = asyncio.Semaphore(1)


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="openai/gpt-oss-20b"):
        self.model_name = model_name
        super().__init__(model_name)

    def load_model(self):
        return ChatGroq(
            model=self.model_name,
            model_kwargs={"response_format": {"type": "json_object"}},
            max_tokens=2048,
            max_retries=3,
        )

    def _safe_prompt(self, prompt: str) -> str:
        if "json" not in prompt.lower():
            prompt += "\n\nCRITICAL INSTRUCTION: You must respond entirely with a valid JSON object only."
        return prompt

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        p = self._safe_prompt(prompt)
        for attempt in range(8):
            try:
                return model.invoke(p).content
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "rate limit" in msg:
                    wait = 15 * (attempt + 1)
                    print(f"\n      ⚠️ Judge Rate Limit hit. Backing off {wait}s... (attempt {attempt+1}/8)", flush=True)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Groq judge sync generation exhausted all retries.")

    async def a_generate(self, prompt: str) -> str:
        async with _semaphore:
            # Pacing sleep inside semaphore: ensure 1.5s gap between sequential async requests
            await asyncio.sleep(1.5)
            model = self.load_model()
            p = self._safe_prompt(prompt)
            for attempt in range(8):
                try:
                    res = await model.ainvoke(p)
                    return res.content
                except Exception as e:
                    msg = str(e).lower()
                    if "429" in msg or "rate limit" in msg:
                        wait = 15 * (attempt + 1)
                        print(f"\n      ⚠️ Judge Async Rate Limit. Backing off {wait}s... (attempt {attempt+1}/8)", flush=True)
                        await asyncio.sleep(wait)
                    else:
                        raise
        raise RuntimeError("Groq judge async generation exhausted all retries.")

    def get_model_name(self) -> str:
        return self.model_name


# ── RAG execution with backoff ──────────────────────────────────────────────

def run_rag_query(graph, query: str, session_id: str, thread_id: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": thread_id}}
    for attempt in range(6):
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
                raise RuntimeError(f"RAG returned rate-limit text: {answer[:120]}")
            ctx = [d.page_content for d in (state.get("retrieved_docs") or [])]
            return answer, ctx
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg or "ratelimiterror" in msg:
                wait = 20 * (attempt + 1)
                print(f"\n    ⚠️ RAG 429 hit. Sleeping {wait}s... (attempt {attempt+1}/6)", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("RAG query failed after multiple rate limit retries.")


# ── Incremental save helper ─────────────────────────────────────────────────

def save_incremental(new_case: dict, case_num: int):
    # 1. Save to dedicated 6-to-10 file
    dedicated_data = []
    if RESULTS_6_TO_10_FILE.exists():
        try:
            dedicated_data = json.loads(RESULTS_6_TO_10_FILE.read_text(encoding="utf-8"))
        except Exception:
            dedicated_data = []
    # Replace if exists or append
    dedicated_data = [d for d in dedicated_data if d["input"] != new_case["input"]]
    dedicated_data.append(new_case)
    RESULTS_6_TO_10_FILE.write_text(json.dumps(dedicated_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. Merge into main eval_results.json
    main_data = []
    if RESULTS_MAIN_FILE.exists():
        try:
            main_data = json.loads(RESULTS_MAIN_FILE.read_text(encoding="utf-8"))
        except Exception:
            main_data = []
    main_data = [d for d in main_data if d["input"] != new_case["input"]]
    main_data.append(new_case)
    RESULTS_MAIN_FILE.write_text(json.dumps(main_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  💾 Incremental save successful! (Case {case_num} recorded to disk)", flush=True)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70, flush=True)
    print("      DEEPEVAL BENCHMARK RUNNER — CASES 6 TO 10 (LIVE TRACKING)", flush=True)
    print("=" * 70, flush=True)

    goldens = json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))
    
    # Target cases: index 5 to 9 (display numbers 6 to 10)
    target_pairs = goldens[5:10]
    print(f"-> Target cases: 5 cases ({len(target_pairs)} pairs: Cases 6 through 10)", flush=True)
    for i, p in enumerate(target_pairs, start=6):
        print(f"   [{i}/10] {p['input'][:65]}...", flush=True)

    print("\n-> [1/3] Loading PDF document...", flush=True)
    docs = load_document(PDF_PATH)
    print(f"   Loaded {len(docs)} document chunks.", flush=True)

    print("-> [2/3] Compiling LangGraph RAG pipeline...", flush=True)
    graph = build_graph(db_path="eval_6_to_10_checkpoints.db")

    print("-> [3/3] Initializing Qdrant vector store collection...", flush=True)
    global_session_id = f"eval_6_10_{uuid4()}"
    add_paper(docs, global_session_id)
    print(f"   Vector store indexed. Session ID: {global_session_id}\n", flush=True)

    judge = GroqJudge(model_name="openai/gpt-oss-20b")
    metrics = [
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=judge),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=judge),
        ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge),
    ]

    results_batch = []

    for rel_idx, pair in enumerate(target_pairs):
        case_num = 6 + rel_idx
        print("\n" + "-" * 70, flush=True)
        print(f"▶ STARTING CASE [{case_num}/10]: '{pair['input'][:60]}...'", flush=True)
        print("-" * 70, flush=True)

        query = pair["input"] + " as per the report in knowledge base"
        thread_id = f"eval_thread_{case_num}_{uuid4()}"

        print(f"  [1] Generating RAG Answer via GPT-OSS-120B...", end=" ", flush=True)
        rag_t0 = time.time()
        try:
            answer, ctx = run_rag_query(graph, query, global_session_id, thread_id)
            rag_duration = time.time() - rag_t0
            print(f"DONE in {rag_duration:.1f}s ({len(ctx)} chunks retrieved)", flush=True)
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            answer = f"RAG Error: {e}"
            ctx = []

        ctx_trimmed = [c[:600] for c in ctx[:5]]
        answer_trimmed = answer[:1200]

        tc = LLMTestCase(
            input=pair["input"],
            actual_output=answer_trimmed,
            expected_output=pair["expected_output"],
            retrieval_context=ctx_trimmed,
        )

        print(f"  [2] Evaluating 5 DeepEval Metrics via Judge...", flush=True)
        case_metrics = []
        overall_ok = True

        for m_idx, metric in enumerate(metrics, start=1):
            m_name = metric.__class__.__name__
            print(f"      ({m_idx}/5) {m_name:<28} ...", end=" ", flush=True)
            m_t0 = time.time()
            ok = False
            for attempt in range(8):
                try:
                    metric.measure(tc)
                    ok = True
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if "429" in msg or "rate limit" in msg:
                        wait = 15 * (attempt + 1)
                        print(f"\n          ⚠️ Rate limit. Sleeping {wait}s... (attempt {attempt+1}/8)", end=" ", flush=True)
                        time.sleep(wait)
                    else:
                        print(f"FAILED ({e})", end=" ", flush=True)
                        break

            m_duration = time.time() - m_t0
            score  = getattr(metric, "score", None) or 0.0
            passed = getattr(metric, "success", False) or False
            reason = getattr(metric, "reason", "n/a") or "n/a"
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status} (score: {score:.2f}) [{m_duration:.1f}s]", flush=True)

            if not ok:
                overall_ok = False

            case_metrics.append({
                "name": m_name,
                "score": float(score),
                "passed": bool(passed),
                "reason": str(reason),
            })
            time.sleep(4)  # short pause between metric measurements

        case_record = {
            "case_number": case_num,
            "input": pair["input"],
            "actual_output": answer,
            "success": overall_ok,
            "metrics": case_metrics,
        }
        results_batch.append(case_record)

        # Immediate incremental save!
        save_incremental(case_record, case_num)

        # Print summary for this case
        print(f"\n  🎯 CASE [{case_num}/10] COMPLETE:", flush=True)
        for m in case_metrics:
            p_icon = "✅" if m["passed"] else "❌"
            print(f"     {p_icon} {m['name']:<28}: {m['score']:.2f}", flush=True)

        if rel_idx < len(target_pairs) - 1:
            print(f"\n  ⏳ Inter-case cooldown: 20s to refresh TPM window...", flush=True)
            time.sleep(20)

    # ── Final Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("               FINAL SUMMARY REPORT — CASES 6 TO 10", flush=True)
    print("=" * 70, flush=True)
    metric_scores = {}
    metric_passed = {}
    for item in results_batch:
        for m in item["metrics"]:
            n = m["name"]
            metric_scores.setdefault(n, []).append(m["score"])
            metric_passed.setdefault(n, []).append(1 if m["passed"] else 0)

    for name, scores in metric_scores.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        passed_cnt = sum(metric_passed[name]) if name in metric_passed else 0
        status = "PASSED ✅" if avg >= METRIC_THRESHOLD else "FAILED ❌"
        print(f"  {name:<30}: {avg:.4f}  [{passed_cnt}/{len(scores)} passed]  → {status}", flush=True)
    print("=" * 70, flush=True)
    print(f"\n🚀 ALL TARGET CASES PROCESSED! Results safely stored in:", flush=True)
    print(f"   - {RESULTS_6_TO_10_FILE}", flush=True)
    print(f"   - {RESULTS_MAIN_FILE}\n", flush=True)


if __name__ == "__main__":
    main()

