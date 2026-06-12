import json
import sys
import time
import asyncio
from pathlib import Path
from uuid import uuid4

# Ensure terminal outputs handle complex text encoding cleanly
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.test_case import LLMTestCase

from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.vector_store import add_paper

# Robust DeepEval evaluator wrapper for Groq
class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model_name = model_name
        super().__init__(model_name)

    def load_model(self):
        return ChatGroq(
            model=self.model_name,
            # Guarantees valid JSON output
            model_kwargs={"response_format": {"type": "json_object"}},
            max_retries=3
        )

    def _prepare_prompt(self, prompt: str) -> str:
        if "json" not in prompt.lower():
            prompt += "\n\nCRITICAL INSTRUCTION: You must format your final response entirely as a valid JSON object."
        return prompt

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        safe_prompt = self._prepare_prompt(prompt)
        for attempt in range(6):
            try:
                return chat_model.invoke(safe_prompt).content
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                    time.sleep(10) # Increased sleep to give API more breathing room
                else:
                    raise e
        raise RuntimeError("Groq judge failed to generate output after multiple retries due to Rate Limits.")

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        safe_prompt = self._prepare_prompt(prompt)
        for attempt in range(8): # Increased retries
            try:
                res = await chat_model.ainvoke(safe_prompt)
                return res.content
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                    print(f"    ⚠️ Judge Rate Limit hit. Sleeping 10s... (Attempt {attempt + 1}/8)")
                    await asyncio.sleep(10)
                else:
                    raise e
        raise RuntimeError("Groq judge failed to generate async output after multiple retries due to Rate Limits.")

    def get_model_name(self) -> str:
        return self.model_name

load_dotenv(override=True)

PDF_PATH            = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE        = Path("goldens.json")
MAX_CONTEXTS        = 5
GOLDENS_PER_CONTEXT = 2
METRIC_THRESHOLD    = 0.7


def generate_goldens() -> list[dict]:
    print("-> goldens.json not found. Generating synthetic test cases using Synthesizer...")
    synthesizer = Synthesizer()
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[PDF_PATH],
        include_expected_output=True,
        max_goldens_per_context=GOLDENS_PER_CONTEXT,
        context_construction_config=ContextConstructionConfig(
            max_contexts_per_document=MAX_CONTEXTS,
        ),
    )
    pairs = [
        {"input": g.input, "expected_output": g.expected_output}
        for g in goldens
        if g.input and g.expected_output
    ]
    GOLDENS_FILE.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    return pairs

def load_goldens() -> list[dict]:
    return json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))

def run_rag_query(graph, query: str, doc_session_id: str, thread_id: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": thread_id}} 
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "session_id": doc_session_id, 
            "query": query,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "rewrite_count": 0,
        },
        config=config,
    )
    answer = final_state.get("answer") or ""
    retrieval_context = [doc.page_content for doc in (final_state.get("retrieved_docs") or [])]
    return answer, retrieval_context

def main() -> None:
    pairs = load_goldens() if GOLDENS_FILE.exists() else generate_goldens()
    print(f"-> Successfully loaded {len(pairs)} evaluation pairs.")

    print("-> Loading PDF documents...")
    docs = load_document(PDF_PATH)
    
    print("-> Compiling LangGraph RAG workflow...")
    graph = build_graph(db_path="eval_checkpoints.db")

    judge_model = GroqJudge(model_name="llama-3.1-8b-instant")

    metrics = [
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=judge_model),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=judge_model),
        ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge_model),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge_model),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge_model),
    ]

    print("-> Indexing paper context into vector store...")
    global_session_id = f"eval_global_docs_{uuid4()}"
    add_paper(docs, global_session_id)

    test_cases = []
    print("\n=== STARTING RAG PIPELINE RUNS ===")
    for idx, pair in enumerate(pairs, start=1):
        print(f"\n[{idx}/{len(pairs)}] Processing Input: '{pair['input'][:50]}...'")

        query = pair["input"] + " as per the report in knowledge base"
        current_thread_id = f"eval_thread_{uuid4()}"
        
        answer, retrieval_context = None, []
        for rag_attempt in range(4):
            try:
                answer, retrieval_context = run_rag_query(graph, query, global_session_id, current_thread_id)
                break
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                    print(f"    ⚠️ RAG Rate Limit hit. Sleeping 60 seconds... (Attempt {rag_attempt + 1}/4)")
                    time.sleep(60)
                else:
                    raise e
                    
        if answer is None:
            print("❌ Critical Error: RAG pipeline failed repeatedly.")
            return

        retrieval_context = retrieval_context[:4]
        print(f"    -> Response processed successfully (Context constrained to top {len(retrieval_context)} chunks).")
        
        test_cases.append(
            LLMTestCase(
                input=pair["input"],
                actual_output=answer,
                expected_output=pair["expected_output"],
                retrieval_context=retrieval_context,
            )
        )

    print("\n=== STARTING SLEDGEHAMMER METRIC ASSESSMENT ===")
    print("-> Bypassing DeepEval's brittle async execution. Processing manually...")

    summary = []
    results_path = Path("eval_results.json")

    # THE FIX: We iterate manually. No async timeouts can kill this script now.
    for i, test_case in enumerate(test_cases):
        print(f"\n-> Evaluating test case {i+1}/{len(test_cases)}...")
        
        case_metrics_results = []
        overall_success = True
        
        for metric in metrics:
            print(f"   -> Measuring {metric.__class__.__name__}...")
            metric_success_flag = False
            
            # Retry loop for each individual metric
            for attempt in range(6):
                try:
                    metric.measure(test_case)
                    metric_success_flag = True
                    break
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                        print(f"      ⚠️ Metric Rate limit hit. Sleeping 15s... (Attempt {attempt + 1}/6)")
                        time.sleep(15)
                    else:
                        print(f"      ❌ Metric {metric.__class__.__name__} failed: {e}")
                        break
            
            if not metric_success_flag:
                overall_success = False

            case_metrics_results.append({
                "name": metric.__class__.__name__,
                "score": getattr(metric, 'score', 0.0),
                "passed": getattr(metric, 'success', False),
                "reason": getattr(metric, 'reason', "Failed to compute."),
            })
            
            # Rest 2 seconds between metrics to keep Groq happy
            time.sleep(2)
            
        summary.append({
            "input": test_case.input,
            "actual_output": test_case.actual_output,
            "success": overall_success,
            "metrics": case_metrics_results,
        })
        
        # Incremental save! If it crashes on Question 5, you still have the results for 1-4.
        results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"   ✅ Saved intermediate results for test case {i+1}.")

    print(f"\n🚀 COMPLETE! Final evaluation metrics saved safely to {results_path}.")

if __name__ == "__main__":
    main()