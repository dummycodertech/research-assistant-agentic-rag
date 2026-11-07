import argparse
import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path

# Force unbuffered UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

from dotenv import load_dotenv
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

load_dotenv(override=True)

CACHE_FILE = Path("eval_rag_cache.json")
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
            max_tokens=4096,
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
                time.sleep(2.0)
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
                    print(f"\n      [Judge Error: {e}]", flush=True)
                    if "429" in msg or "rate limit" in msg:
                        wait = 15 * (attempt + 1)
                        print(f"\n      ⚠️ Judge Async 429 hit. Backing off {wait}s... (attempt {attempt+1}/8)", flush=True)
                        await asyncio.sleep(wait)
                    else:
                        raise
        raise RuntimeError("Groq judge async generation exhausted all retries.")

    def get_model_name(self) -> str:
        return self.model_name


def save_incremental(new_case: dict, case_num: int):
    data = []
    if RESULTS_FILE.exists():
        try:
            data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = []
    # Replace existing entry for this input or case_number
    data = [d for d in data if d.get("input") != new_case["input"] and d.get("case_number") != case_num]
    data.append(new_case)
    data.sort(key=lambda x: x.get("case_number", 999))
    RESULTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  💾 Incremental save successful! (Case {case_num} recorded to {RESULTS_FILE})", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate pregenerated RAG cache case-by-case.")
    parser.add_argument("--case", type=int, default=None, help="Run a single case (1-10).")
    parser.add_argument("--start", type=int, default=1, help="Start case number (default 1).")
    parser.add_argument("--end", type=int, default=10, help="End case number (default 10).")
    args = parser.parse_args()

    if not CACHE_FILE.exists():
        print(f"❌ Error: {CACHE_FILE} not found. Run pregenerate_rag_cases.py first.")
        sys.exit(1)

    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    if args.case is not None:
        target_keys = [str(args.case)]
    else:
        target_keys = [str(i) for i in range(args.start, args.end + 1) if str(i) in cache]

    print("=" * 70, flush=True)
    print(f"      DEEPEVAL JUDGE EVALUATION — CACHED RAG RESPONSES")
    print(f"      Target Cases: {target_keys}")
    print("=" * 70, flush=True)

    judge = GroqJudge(model_name="openai/gpt-oss-20b")
    metrics = [
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=judge),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=judge),
        ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge),
    ]

    for rel_idx, k in enumerate(target_keys):
        item = cache[k]
        case_num = item["case_number"]

        print("\n" + "-" * 70, flush=True)
        print(f"▶ STARTING EVALUATION ON CASE [{case_num}/10]: '{item['input'][:55]}...'", flush=True)
        print("-" * 70, flush=True)

        ctx_trimmed = [c[:600] for c in item["retrieval_context"][:4]]
        answer_trimmed = item["actual_output"][:1200]

        tc = LLMTestCase(
            input=item["input"],
            actual_output=answer_trimmed,
            expected_output=item["expected_output"],
            retrieval_context=ctx_trimmed,
        )

        print(f"  Evaluating 5 DeepEval Metrics via GroqJudge (gpt-oss-20b)...", flush=True)
        case_metrics = []
        overall_ok = True

        for m_idx, metric in enumerate(metrics, start=1):
            m_name = metric.__class__.__name__
            print(f"    ({m_idx}/5) {m_name:<28} ...", end=" ", flush=True)
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
                        print(f"\n        ⚠️ Rate limit. Sleeping {wait}s... (attempt {attempt+1}/8)", end=" ", flush=True)
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
            time.sleep(3)

        case_record = {
            "case_number": case_num,
            "input": item["input"],
            "actual_output": item["actual_output"],
            "success": overall_ok,
            "metrics": case_metrics,
        }

        save_incremental(case_record, case_num)

        print(f"\n  🎯 CASE [{case_num}/10] EVALUATION COMPLETE:", flush=True)
        for m in case_metrics:
            p_icon = "✅" if m["passed"] else "❌"
            print(f"     {p_icon} {m['name']:<28}: {m['score']:.2f}", flush=True)

        if rel_idx < len(target_keys) - 1:
            print(f"\n  ⏳ Inter-case cooldown: 10s to refresh TPM window...", flush=True)
            time.sleep(10)

    print("\n" + "=" * 70, flush=True)
    print("  EVALUATION BATCH COMPLETED!")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()

