import argparse
import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

# Force unbuffered UTF-8 output so every line is streamed live
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

from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.vector_store import add_paper

load_dotenv(override=True)

PDF_PATH = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE = Path("goldens.json")
RESULTS_FILE = Path("eval_results.json")
METRIC_THRESHOLD = 0.70

_semaphore = asyncio.Semaphore(1)


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="openai/gpt-oss-20b"):
        self.model_name = model_name
        super().__init__(model_name)

    def load_model(self):
        return ChatGroq(
            model=self.model_name,
            max_tokens=2048,
            model_kwargs={"response_format": {"type": "json_object"}},
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
                time.sleep(2.5)
                model = self.load_model()
                return model.invoke(p).content
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "rate limit" in msg:
                    wait = 15 * (attempt + 1)
                    print(f"\n      ⚠️ Judge Sync 429 hit. Backing off {wait}s... (attempt {attempt+1}/8)", flush=True)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Groq judge sync generation exhausted all retries.")

    async def a_generate(self, prompt: str) -> str:
        async with _semaphore:
            await asyncio.sleep(2.5)
            p = self._safe_prompt(prompt)
            for attempt in range(8):
                try:
                    model = self.load_model()
                    res = await model.ainvoke(p)
                    return res.content
                except Exception as e:
                    msg = str(e).lower()
                    if "429" in msg or "rate limit" in msg:
                        wait = 15 * (attempt + 1)
                        print(f"\n      ⚠️ Judge Async 429 hit. Backing off {wait}s... (attempt {attempt+1}/8)", flush=True)
                        await asyncio.sleep(wait)
                    else:
                        raise
        raise RuntimeError("Groq judge async generation exhausted all retries.")

    def get_model_name(self) -> str:
        return self.model_name


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


def save_incremental(new_case: dict, case_num: int):
    data = []
    if RESULTS_FILE.exists():
        try:
            data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = []
    data = [d for d in data if d.get("input") != new_case["input"]]
    data.append(new_case)
    # Sort by case_number if available
    data.sort(key=lambda x: x.get("case_number", 999))
    RESULTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  💾 Incremental save successful! (Case {case_num} recorded to {RESULTS_FILE})", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run DeepEval benchmark case-by-case.")
    parser.add_argument("--case", type=int, default=None, help="Run a single case (1-10).")
    parser.add_argument("--start", type=int, default=1, help="Start case number (default 1).")
    parser.add_argument("--end", type=int, default=10, help="End case number (default 10).")
    args = parser.parse_args()

    goldens = json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))

    if args.case is not None:
        target_cases = [(args.case, goldens[args.case - 1])]
    else:
        target_cases = [(i, goldens[i - 1]) for i in range(args.start, args.end + 1)]

    print("=" * 70, flush=True)
    print(f"      DEEPEVAL CASE-BY-CASE BENCHMARK — FLASHRANK PIPELINE")
    print(f"      Evaluating {len(target_cases)} case(s): {[c[0] for c in target_cases]}")
    print("=" * 70, flush=True)

    print("\n-> [1/3] Loading PDF with structure-aware Markdown parser...", flush=True)
    docs = load_document(PDF_PATH)
    print(f"   Loaded {len(docs)} structured chunks.", flush=True)

    print("-> [2/3] Compiling LangGraph RAG pipeline...", flush=True)
    graph = build_graph(db_path="eval_case_checkpoints.db")

    print("-> [3/3] Initializing Qdrant vector store & BM25 sparse index...", flush=True)
    global_session_id = f"eval_case_{uuid4()}"
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

    for idx, (case_num, pair) in enumerate(target_cases):
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

        ctx_trimmed = [c[:600] for c in ctx[:4]]
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
            score = getattr(metric, "score", None) or 0.0
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
            time.sleep(4)

        case_record = {
            "case_number": case_num,
            "input": pair["input"],
            "actual_output": answer,
            "success": overall_ok,
            "metrics": case_metrics,
        }

        save_incremental(case_record, case_num)

        print(f"\n  🎯 CASE [{case_num}/10] COMPLETE:", flush=True)
        for m in case_metrics:
            p_icon = "✅" if m["passed"] else "❌"
            print(f"     {p_icon} {m['name']:<28}: {m['score']:.2f}", flush=True)

        if idx < len(target_cases) - 1:
            print(f"\n  ⏳ Inter-case cooldown: 15s to refresh TPM window...", flush=True)
            time.sleep(15)

    print("\n" + "=" * 70, flush=True)
    print("  ALL TARGET CASES COMPLETED SUCCESSFULLY!")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()

