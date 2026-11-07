"""
eval_resume.py
--------------
Resumes evaluation from case 4 onwards (cases 1-3 preserved from eval_results.json).
- Uses llama-3.1-8b-instant as judge (20k TPM, no JSON truncation issues)
- Adds proper retry + sleep in RAG call to handle 429 gracefully
- Re-runs RAG for cases 4-6 since those had rate-limit errors in actual_output
- Merges final results back into eval_results.json
"""

import json
import sys
import time
import asyncio
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

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

PDF_PATH        = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE    = Path("goldens.json")
RESULTS_FILE    = Path("eval_results.json")
METRIC_THRESHOLD = 0.7

# ── Judge model: llama-3.1-8b-instant has 20k TPM — no JSON truncation ──────

class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="openai/gpt-oss-20b"):
        self.model_name = model_name
        super().__init__(model_name)

    def load_model(self):
        return ChatGroq(
            model=self.model_name,
            model_kwargs={"response_format": {"type": "json_object"}},
            max_retries=5,
        )

    def _safe_prompt(self, prompt: str) -> str:
        if "json" not in prompt.lower():
            prompt += "\n\nCRITICAL: You must respond with a valid JSON object only."
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
                    print(f"      ⚠️  Judge 429. Sleeping {wait}s... (attempt {attempt+1}/8)")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Judge exhausted retries.")

    async def a_generate(self, prompt: str) -> str:
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
                    print(f"      ⚠️  Judge 429. Sleeping {wait}s... (attempt {attempt+1}/8)")
                    await asyncio.sleep(wait)
                else:
                    raise
        raise RuntimeError("Judge exhausted async retries.")

    def get_model_name(self) -> str:
        return self.model_name


# ── RAG call with proper rate-limit backoff ──────────────────────────────────

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
            # Treat a rate-limit error string as a failure, not a real answer
            if "RateLimitError" in answer or "429" in answer:
                raise RuntimeError(f"RAG returned rate-limit error: {answer[:120]}")
            ctx = [d.page_content for d in (state.get("retrieved_docs") or [])]
            return answer, ctx
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg or "ratelimiterror" in msg.lower():
                wait = 20 * (attempt + 1)
                print(f"    ⚠️  RAG 429. Sleeping {wait}s... (attempt {attempt+1}/6)")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("RAG exhausted all retries due to rate limits.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    goldens = json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))
    existing = json.loads(RESULTS_FILE.read_text(encoding="utf-8")) if RESULTS_FILE.exists() else []

    # Keep only the first 3 (cases 1-3) which had real answers — discard 4-6 (had error strings)
    good_results = existing[:3]
    start_idx = len(good_results)  # 3 → will process from golden[3] onwards (cases 4-10)

    print(f"\n✅ Keeping {len(good_results)} already-good results (cases 1-{len(good_results)}).")
    print(f"🔄 Will re-run cases {start_idx+1} → {len(goldens)} (indices {start_idx}-{len(goldens)-1}).\n")

    print("→ Loading PDF...")
    docs = load_document(PDF_PATH)
    print("→ Compiling LangGraph graph...")
    graph = build_graph(db_path="eval_resume_checkpoints.db")
    print("→ Indexing paper into vector store...")
    global_session_id = f"eval_resume_{uuid4()}"
    add_paper(docs, global_session_id)
    print(f"→ Vector store ready. Session: {global_session_id}\n")

    judge = GroqJudge("openai/gpt-oss-20b")
    metrics = [
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=judge),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=judge),
        ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge),
    ]

    summary = list(good_results)  # start with the 3 preserved cases

    pairs_to_run = goldens[start_idx:]  # cases 4-10

    print("=" * 60)
    print(f"  STARTING RAG RUNS (cases {start_idx+1}–{len(goldens)})")
    print("=" * 60)

    for rel_idx, pair in enumerate(pairs_to_run):
        abs_idx = start_idx + rel_idx          # 3,4,...,9
        display_num = abs_idx + 1              # 4,5,...,10

        print(f"\n[{display_num}/{len(goldens)}] RAG → '{pair['input'][:60]}...'")

        query = pair["input"] + " as per the report in knowledge base"
        thread_id = f"eval_resume_thread_{uuid4()}"

        try:
            answer, ctx = run_rag_query(graph, query, global_session_id, thread_id)
        except Exception as e:
            print(f"  ❌ RAG failed permanently: {e}")
            answer = f"RAG FAILED: {e}"
            ctx = []

        ctx = ctx[:5]
        # Truncate each context chunk and the answer to keep judge prompt small
        # gpt-oss-20b has 8k TPM — keep each judge call under ~1500 tokens total
        ctx_trimmed = [c[:600] for c in ctx]
        answer_trimmed = answer[:1200]
        print(f"  → RAG done. Got {len(ctx)} context chunks.")

        # Sleep a bit between RAG and judge to stay under TPM
        time.sleep(3)

        tc = LLMTestCase(
            input=pair["input"],
            actual_output=answer_trimmed,
            expected_output=pair["expected_output"],
            retrieval_context=ctx_trimmed,
        )

        print(f"  → Evaluating metrics...")
        case_metrics = []
        overall_ok = True

        for metric in metrics:
            metric_name = metric.__class__.__name__
            print(f"     • {metric_name}...", end=" ", flush=True)
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
                        print(f"\n       ⚠️  Rate limit. Sleeping {wait}s... (attempt {attempt+1}/8)", end=" ", flush=True)
                        time.sleep(wait)
                    else:
                        print(f"FAILED: {e}")
                        break

            score  = getattr(metric, "score", None) or 0.0
            passed = getattr(metric, "success", False) or False
            reason = getattr(metric, "reason", "n/a") or "n/a"
            status = "✅" if passed else "❌"
            print(f"{status} {score:.2f}")

            if not ok:
                overall_ok = False

            case_metrics.append({
                "name": metric_name,
                "score": score,
                "passed": bool(passed),
                "reason": str(reason),
            })
            time.sleep(12)  # 12s gap keeps judge calls under 8k TPM

        summary.append({
            "input": pair["input"],
            "actual_output": answer,
            "success": overall_ok,
            "metrics": case_metrics,
        })

        RESULTS_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  💾 Saved. Case {display_num}/{len(goldens)} complete.")
        # Cool down between cases to let the 1-minute TPM window reset
        if rel_idx < len(pairs_to_run) - 1:
            print(f"  ⏳ Cooling down 30s before next case...")
            time.sleep(30)

    # ── Final aggregate report ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("         FINAL DEEPEVAL BENCHMARK — ALL 10 CASES")
    print("=" * 65)
    metric_scores: dict[str, list] = {}
    metric_passed: dict[str, list] = {}
    for item in summary:
        for m in item["metrics"]:
            n = m["name"]
            metric_scores.setdefault(n, []).append(m["score"])
            metric_passed.setdefault(n, []).append(1 if m["passed"] else 0)

    for name, scores in metric_scores.items():
        avg = sum(scores) / len(scores)
        passed_cnt = sum(metric_passed[name])
        status = "PASSED ✅" if avg >= METRIC_THRESHOLD else "FAILED ❌"
        print(f"  {name:<30}: {avg:.4f}  [{passed_cnt}/{len(scores)} cases]  → {status}")
    print("=" * 65)
    print(f"\n🚀 Done! Full results saved to {RESULTS_FILE}\n")


if __name__ == "__main__":
    main()

