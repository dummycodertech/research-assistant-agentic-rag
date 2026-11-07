import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")

# ── Chunk settings ────────────────────────────────────────────────────────────
# 512 tokens ≈ 1800 chars for all-MiniLM-L6-v2 (max 256 tokens, but we chunk
# larger so the LLM context is richer). 150-char overlap keeps sentences intact
# across boundaries and improves BM25 recall on split terms.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
_md_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)


def _stamp_title(docs: list[Document], title: str) -> list[Document]:
    for doc in docs:
        doc.metadata["title"] = title
    return docs


def load_pdf(file_path: str) -> list[Document]:
    """Structure-aware PDF loader using pymupdf4llm markdown extraction with section & page header prepending."""
    title = Path(file_path).stem
    try:
        import pymupdf4llm
        pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        current_section = "Overview"
        docs = []
        for p in pages:
            page_num = p.get("metadata", {}).get("page_number", 1)
            text = p.get("text", "")
            if not text.strip():
                continue
            headers = re.findall(r"^(#{1,3}\s+.+)$", text, re.MULTILINE)
            if headers:
                current_section = headers[0].lstrip("#").strip()

            sub_chunks = _splitter.split_text(text)
            for c in sub_chunks:
                header = f"[Paper: {title} | Section: {current_section} | Page: {page_num}]"
                prepended = f"{header}\n{c}"
                docs.append(Document(
                    page_content=prepended,
                    metadata={"title": title, "section": current_section, "page": page_num, "source": file_path}
                ))
        if docs:
            return docs
    except Exception as e:
        print(f"⚠️ pymupdf4llm extraction fallback to standard PyMuPDFLoader: {e}")

    # Fallback: standard PyMuPDFLoader
    raw_docs = PyMuPDFLoader(file_path).load()
    split_docs = _splitter.split_documents(raw_docs)
    for d in split_docs:
        page_num = d.metadata.get("page", 1)
        d.metadata["title"] = title
        d.metadata["section"] = "General"
        d.page_content = f"[Paper: {title} | Section: General | Page: {page_num}]\n{d.page_content}"
    return split_docs


def load_text(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_title(_splitter.split_documents(docs), Path(file_path).stem)


def load_markdown(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_title(_md_splitter.split_documents(docs), Path(file_path).stem)


def load_webpage(url: str) -> list[Document]:
    docs = WebBaseLoader(url, requests_kwargs={"timeout": 30}).load()
    title = (docs[0].metadata.get("title") or url) if docs else url
    return _stamp_title(_splitter.split_documents(docs), title)


def _extract_arxiv_id(query: str) -> str | None:
    """Return bare ArXiv ID (no version suffix) if one appears in the query."""
    m = _ARXIV_ID_RE.search(query)
    if m:
        return re.sub(r"v\d+$", "", m.group(1))
    return None


def _arxiv_api_lookup(arxiv_id: str) -> str:
    """Fetch paper title by ID from the ArXiv Atom API."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        xml = resp.read().decode()
    titles = re.findall(r"<title>(.*?)</title>", xml, re.DOTALL)
    return titles[1].strip() if len(titles) > 1 else arxiv_id


def _arxiv_search(query: str) -> str:
    """Search ArXiv Atom API by title phrase and return the top result's bare paper ID."""
    phrase = query.strip('"')
    search_query = urllib.parse.quote(f'ti:"{phrase}"')
    url = f"https://export.arxiv.org/api/query?search_query={search_query}&max_results=1&sortBy=relevance"
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml = resp.read().decode()
    m = re.search(r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)</id>", xml)
    if not m:
        raise ValueError(f"No ArXiv paper found for: {query}")
    return re.sub(r"v\d+$", "", m.group(1))


def _load_arxiv_by_id(arxiv_id: str) -> list[Document]:
    """Download and chunk an ArXiv paper PDF by its bare ID using structure-aware parsing."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    with urllib.request.urlopen(pdf_url, timeout=60) as resp:
        pdf_bytes = resp.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        title = _arxiv_api_lookup(arxiv_id) or f"arXiv:{arxiv_id}"
        docs = load_pdf(tmp_path)
        if not docs:
            raise ValueError(f"Could not load PDF for ArXiv ID: {arxiv_id}")
        for d in docs:
            d.metadata["title"] = title
            # Re-stamp title into prepended header if needed
            if d.page_content.startswith("[Paper:"):
                d.page_content = re.sub(r"^\[Paper:\s*[^|]+", f"[Paper: {title}", d.page_content)
        return docs
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def load_arxiv(query: str) -> list[Document]:
    arxiv_id = _extract_arxiv_id(query) or _arxiv_search(query)
    return _load_arxiv_by_id(arxiv_id)


def load_document(source: str) -> list[Document]:
    """Dispatch to the appropriate loader based on URL prefix or file extension."""
    if source.startswith(("http://", "https://")):
        return load_webpage(source)
    ext = Path(source).suffix.lower()
    if ext == ".pdf":
        return load_pdf(source)
    if ext == ".txt":
        return load_text(source)
    if ext in (".md", ".markdown"):
        return load_markdown(source)
    raise ValueError(f"Unsupported file type: {ext}")
