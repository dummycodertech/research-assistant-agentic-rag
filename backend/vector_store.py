import os
import pickle
import re
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage.file_system import LocalFileStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rank_bm25 import BM25Okapi

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 384          # sentence-transformers/all-MiniLM-L6-v2
RRF_K = 60                   # RRF constant — standard value
MAX_IN_MEMORY_SESSIONS = 10  # LRU threshold to prevent Streamlit RAM OOM

BM25_CACHE_DIR = Path("./embedding_cache/bm25")
BM25_CACHE_DIR.mkdir(parents=True, exist_ok=True)

FLASHRANK_CACHE_DIR = Path("./embedding_cache/flashrank")
FLASHRANK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Singletons ────────────────────────────────────────────────────────────────
base_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_file_store = LocalFileStore("./embedding_cache/")
embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace="all-MiniLM-L6-v2",
    query_embedding_cache=True,
)

# Connect to Qdrant Cloud if credentials are provided, otherwise fall back to
# an in-memory instance (useful for local dev without an env file).
_QDRANT_URL = os.getenv("QDRANT_URL")
_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if _QDRANT_URL and _QDRANT_API_KEY:
    qdrant_client = QdrantClient(url=_QDRANT_URL, api_key=_QDRANT_API_KEY, timeout=30)
else:
    # Fallback: in-memory — avoids external port restrictions on Streamlit Cloud.
    qdrant_client = QdrantClient(":memory:")


# ── FlashRank Cross-Encoder Singleton ─────────────────────────────────────────
_ranker: Ranker | None = None
_ranker_lock = Lock()


def get_ranker() -> Ranker:
    global _ranker
    with _ranker_lock:
        if _ranker is None:
            _ranker = Ranker(
                model_name="ms-marco-MiniLM-L-12-v2",
                cache_dir=str(FLASHRANK_CACHE_DIR.resolve()),
            )
        return _ranker


# ── BM25 Index Store with LRU Eviction & Disk Persistence ────────────────────
# Maps session_id -> (BM25Okapi index, ordered list of Documents)
_bm25_store: OrderedDict[str, Tuple["BM25Okapi", List[Document]]] = OrderedDict()
_bm25_lock = Lock()


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def _build_bm25(docs: List[Document]) -> "BM25Okapi":
    tokenized = [_tokenize(doc.page_content) for doc in docs]
    return BM25Okapi(tokenized)


def _save_bm25_to_disk(session_id: str, index: "BM25Okapi", docs: List[Document]) -> None:
    """Persist BM25 index & documents to disk so indices survive server reboots."""
    safe_sid = session_id.replace("-", "_")
    file_path = BM25_CACHE_DIR / f"{safe_sid}.pkl"
    try:
        with open(file_path, "wb") as f:
            pickle.dump((index, docs), f)
    except Exception as e:
        print(f"⚠️ Failed to persist BM25 to disk for session {session_id}: {e}")


def _load_bm25_from_disk(session_id: str) -> Tuple["BM25Okapi", List[Document]] | None:
    """Load persisted BM25 index & documents from disk if available."""
    safe_sid = session_id.replace("-", "_")
    file_path = BM25_CACHE_DIR / f"{safe_sid}.pkl"
    if file_path.exists():
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Failed to read BM25 disk cache for session {session_id}: {e}")
    return None


def _get_bm25_entry(session_id: str) -> Tuple["BM25Okapi", List[Document]] | None:
    """Retrieve BM25 entry from RAM LRU, or load from disk on miss."""
    if session_id in _bm25_store:
        _bm25_store.move_to_end(session_id)
        return _bm25_store[session_id]

    disk_entry = _load_bm25_from_disk(session_id)
    if disk_entry is not None:
        _bm25_store[session_id] = disk_entry
        if len(_bm25_store) > MAX_IN_MEMORY_SESSIONS:
            _bm25_store.popitem(last=False)
        return disk_entry

    return None


# ── Extractive Sentence Pruning ──────────────────────────────────────────────

def prune_chunk_sentences(query: str, text: str, min_sentences: int = 2) -> str:
    """
    Extracts the highest-signal sentences from a chunk matching the query,
    stripping boilerplate headers/footers to maximize Contextual Relevancy.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= min_sentences:
        return text

    q_tokens = set(re.findall(r"\w+", query.lower()))
    if not q_tokens:
        return text

    scored_sents = []
    for i, sent in enumerate(sentences):
        s_tokens = set(re.findall(r"\w+", sent.lower()))
        overlap = len(q_tokens.intersection(s_tokens))
        scored_sents.append((overlap, i, sent))

    # Keep top 75% most relevant sentences (or at least min_sentences)
    top_count = max(min_sentences, int(len(sentences) * 0.75))
    top_selected = sorted(
        sorted(scored_sents, key=lambda x: x[0], reverse=True)[:top_count],
        key=lambda x: x[1],
    )
    return " ".join(s[2] for s in top_selected)


# ── Collection ───────────────────────────────────────────────────────────────

def get_collection_name(session_id: str) -> str:
    return f"papeer_{session_id.replace('-', '_')}"


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> None:
    """Index documents into Qdrant (dense) and BM25 (sparse) with disk persistence."""
    get_vectorstore(session_id).add_documents(docs)
    with _bm25_lock:
        existing_entry = _get_bm25_entry(session_id)
        existing_docs = existing_entry[1] if existing_entry else []
        all_docs = existing_docs + list(docs)
        bm25_idx = _build_bm25(all_docs)

        # LRU in-memory management
        if session_id in _bm25_store:
            _bm25_store.move_to_end(session_id)
        _bm25_store[session_id] = (bm25_idx, all_docs)
        if len(_bm25_store) > MAX_IN_MEMORY_SESSIONS:
            _bm25_store.popitem(last=False)

        # Persist to disk
        _save_bm25_to_disk(session_id, bm25_idx, all_docs)


def cleanup_session(session_id: str) -> None:
    """Free memory and disk resources when a session is closed/deleted."""
    with _bm25_lock:
        _bm25_store.pop(session_id, None)
    safe_sid = session_id.replace("-", "_")
    disk_file = BM25_CACHE_DIR / f"{safe_sid}.pkl"
    if disk_file.exists():
        disk_file.unlink(missing_ok=True)

    collection_name = get_collection_name(session_id)
    try:
        if qdrant_client.collection_exists(collection_name):
            qdrant_client.delete_collection(collection_name)
    except Exception as e:
        print(f"⚠️ Failed to delete Qdrant collection {collection_name}: {e}")


def list_papers(session_id: str) -> list[str]:
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        return []
    seen: set[str] = set()
    titles: list[str] = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break
    return titles


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = RRF_K,
) -> List[Document]:
    """
    Merge multiple ranked document lists into one via Reciprocal Rank Fusion.
    score(d) = Σ  1 / (k + rank_i(d))
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

    return [doc_map[key] for key in sorted(scores, key=scores.__getitem__, reverse=True)]


def rerank_documents(
    query: str,
    docs: List[Document],
    top_k: int = 4,
    min_score: float = 0.20,
) -> List[Document]:
    """
    Re-scores candidate chunks with FlashRank local cross-encoder
    (ms-marco-MiniLM-L-12-v2), filtering out borderline/irrelevant noise.
    """
    if not docs:
        return []

    try:
        ranker = get_ranker()
        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
        rerank_req = RerankRequest(query=query, passages=passages)
        scored_results = ranker.rerank(rerank_req)

        filtered_docs = []
        for r in scored_results:
            idx = r["id"]
            score = r.get("score", 0.0)
            if score >= min_score or len(filtered_docs) < 2:  # Always retain top 2 candidates
                doc = docs[idx]
                doc.metadata["rerank_score"] = float(score)
                filtered_docs.append(doc)
            if len(filtered_docs) >= top_k:
                break

        return filtered_docs if filtered_docs else docs[:top_k]
    except Exception as e:
        print(f"⚠️ FlashRank reranking fallback to RRF order: {e}")
        return docs[:top_k]


def search(query: str, session_id: str, k: int = 5) -> list[Document]:
    """Pure dense semantic search (kept for backward compatibility)."""
    return get_vectorstore(session_id).similarity_search(query, k=k)


def hybrid_search(
    query: str,
    session_id: str,
    k: int = 4,
    prune_sentences: bool = True,
) -> list[Document]:
    """
    Full 3-stage retrieval pipeline:
    1. Hybrid Candidate Retrieval: Qdrant Dense (15) + BM25 Lexical (15) -> RRF Fusion (15).
    2. Local Cross-Encoder Reranking: FlashRank scores (query, chunk) pairs and filters noise.
    3. Extractive Sentence Pruning: Strips boilerplate to boost Contextual Relevancy.
    """
    fetch_k = 15  # broad candidate pool for RRF + FlashRank

    # Branch 1: Dense Semantic Search (Qdrant)
    dense_results: List[Document] = get_vectorstore(session_id).similarity_search(
        query, k=fetch_k
    )

    # Branch 2: BM25 Keyword Search
    bm25_results: List[Document] = []
    with _bm25_lock:
        entry = _get_bm25_entry(session_id)

    if entry is not None:
        bm25_index, all_docs = entry
        query_tokens = _tokenize(query)
        bm25_scores = bm25_index.get_scores(query_tokens)
        ranked_indices = [
            i
            for i in sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
            if bm25_scores[i] > 0.0
        ][:fetch_k]
        bm25_results = [all_docs[i] for i in ranked_indices]

    # Stage 1: Reciprocal Rank Fusion
    if bm25_results:
        fused = _reciprocal_rank_fusion([dense_results, bm25_results])
    else:
        fused = dense_results

    # Stage 2: Cross-Encoder Reranking with FlashRank
    reranked = rerank_documents(query=query, docs=fused, top_k=k, min_score=0.20)

    # Stage 3: Sentence Pruning for Contextual Relevancy
    if prune_sentences:
        pruned_docs = []
        for d in reranked:
            pruned_text = prune_chunk_sentences(query, d.page_content)
            pruned_doc = Document(page_content=pruned_text, metadata=d.metadata.copy())
            pruned_docs.append(pruned_doc)
        return pruned_docs

    return reranked
