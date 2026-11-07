import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.vector_store import add_paper

load_dotenv(override=True)

PDF_PATH = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE = Path("goldens.json")
CACHE_FILE = Path("eval_rag_cache.json")


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


def main():
    print("=" * 70, flush=True)
    print("    PRE-GENERATING RAG RESPONSES FOR BENCHMARK CASES 1 TO 10")
    print("=" * 70, flush=True)

    goldens = json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))

    # Load existing cache if any
    cache = {}
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    print("\n-> [1/3] Loading PDF with pymupdf4llm...", flush=True)
    docs = load_document(PDF_PATH)
    print(f"   Loaded {len(docs)} structured chunks.", flush=True)

    print("-> [2/3] Compiling LangGraph workflow...", flush=True)
    graph = build_graph(db_path="eval_pregen_checkpoints.db")

    print("-> [3/3] Indexing Qdrant & BM25...", flush=True)
    session_id = f"eval_pregen_{uuid4()}"
    add_paper(docs, session_id)
    print(f"   Indexed session: {session_id}\n", flush=True)

    for i, pair in enumerate(goldens, start=1):
        case_key = str(i)
        if case_key in cache and cache[case_key].get("actual_output"):
            print(f"▶ CASE [{i}/10]: Already cached, skipping.", flush=True)
            continue

        print(f"▶ CASE [{i}/10]: Generating answer for '{pair['input'][:55]}...' ...", end=" ", flush=True)
        t0 = time.time()
        query = pair["input"] + " as per the report in knowledge base"
        thread_id = f"pregen_thread_{i}_{uuid4()}"

        try:
            answer, ctx = run_rag_query(graph, query, session_id, thread_id)
            dur = time.time() - t0
            print(f"DONE in {dur:.1f}s ({len(ctx)} chunks)", flush=True)

            cache[case_key] = {
                "case_number": i,
                "input": pair["input"],
                "expected_output": pair["expected_output"],
                "actual_output": answer,
                "retrieval_context": [c[:600] for c in ctx[:4]],
                "generation_time_s": round(dur, 2),
            }

            CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print(f"FAILED: {e}", flush=True)

        time.sleep(4)

    print("\n" + "=" * 70, flush=True)
    print(f"  ALL {len(cache)} CASES PRE-GENERATED AND SAVED TO {CACHE_FILE}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()

