# app.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
from typing import Dict, Any
from src.ocr_extraction import ocr_and_chunk_pages
from src.chunking import chunk_pages
from src.embeddings_store import EmbeddingsStore
from src.rag_chain import build_langchain_chains
from src.utils import logger, Config, detect_lang
from src.eval import evaluate_query
import uvicorn
import uuid

app = FastAPI(title="RAG - Megher Upor Bari", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data and chroma directories exist
os.makedirs("data", exist_ok=True)
os.makedirs(Config.CHROMA_PERSIST_DIR, exist_ok=True)

# Initialize embedding store (lazy load)
_store = None
_chains = None

def get_store():
    global _store
    if _store is None:
        _store = EmbeddingsStore(model_name=Config.EMBED_MODEL, persist_directory=Config.CHROMA_PERSIST_DIR, collection_name=Config.COLLECTION_NAME)
    return _store

def get_chains():
    global _chains
    if _chains is None:
        _chains = build_langchain_chains(chroma_persist_dir=Config.CHROMA_PERSIST_DIR, collection_name=Config.COLLECTION_NAME, backend=Config.RAG_BACKEND)
    return _chains

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload an image-based PDF and ingest (OCR -> clean -> chunk -> embed -> persist)
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
    save_path = os.path.join("data", f"{uuid.uuid4().hex}_{file.filename}")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    logger.info("Saved uploaded PDF to %s", save_path)

    # OCR
    pages = ocr_and_chunk_pages(save_path)
    if not pages:
        raise HTTPException(status_code=500, detail="OCR produced no pages/text.")
    # Chunk
    docs = chunk_pages(pages)
    if not docs:
        raise HTTPException(status_code=500, detail="Chunking produced no documents.")
    # Store embeddings
    store = get_store()
    store.add_documents(docs)

    return {"status": "success", "n_pages": len(pages), "n_chunks": len(docs)}

@app.post("/chat")
async def chat_endpoint(payload: Dict[str, Any]):
    """
    Payload example: {"query": "প্রশ্ন...", "lang": "auto", "chat": true/false}
    """
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="`query` is required.")
    lang = payload.get("lang", "auto")
    chat_mode = bool(payload.get("chat", True))
    top_k = int(payload.get("top_k", Config.TOP_K))

    # detect language if auto
    if lang == "auto":
        lang = detect_lang(query)
    # Build chains and store
    chains = get_chains()
    store = get_store()

    # Evaluate retrieval for debugging
    eval_res = evaluate_query(query, store, top_k=top_k, save_csv=False)

    # Choose chain
    if chat_mode:
        conv_chain = chains["conv_chain"]
        # LangChain conversational call expects {'question': q, 'chat_history': []}
        result = conv_chain({"question": query})
        answer = result.get("answer") or result.get("result") or ""
        source_docs = result.get("source_documents", [])
    else:
        qa_chain = chains["qa_chain"]
        result = qa_chain.run(query)
        # older RetrievalQA.run returns string, but we want sources; call with .run may not return sources — use predict if available
        # We will attempt to call the chain more robustly:
        try:
            res = qa_chain({"query": query})
            answer = res.get("result") or res.get("answer") or ""
            source_docs = res.get("source_documents", [])
        except Exception:
            # Fallback: return evaluation retrieval as sources and empty answer.
            answer = ""
            source_docs = []

    # Format sources
    sources = []
    for d in source_docs:
        meta = getattr(d, "metadata", {}) if d else {}
        page = meta.get("page")
        chunk_id = meta.get("chunk_id")
        snippet = (d.page_content[:800] + "...") if d and hasattr(d, "page_content") else ""
        sources.append({"page": page, "chunk_id": chunk_id, "snippet": snippet, "metadata": meta})

    return {"query": query, "answer": answer, "lang": lang, "sources": sources, "retrieval_eval": eval_res}

@app.get("/status")
async def status():
    store = get_store()
    stats = store.get_collection_stats()
    return {"status": "ok", "collection": stats}

@app.get("/sources")
async def list_sources(limit: int = 100):
    store = get_store()
    coll = store.collection
    try:
        docs = coll.get(include=["documents", "metadatas", "ids"], limit=limit)
        items = []
        for i, doc in enumerate(docs.get("documents", [])[:limit]):
            metadata = docs.get("metadatas", [])[i]
            _id = docs.get("ids", [])[i]
            items.append({"id": _id, "document_first_chars": doc[:200], "metadata": metadata})
    except Exception as e:
        logger.exception("Failed to list sources: %s", e)
        items = []
    return {"n_sources": len(items), "items": items}

@app.on_event("startup")
async def startup_event():
    logger.info("Starting RAG service. Backend=%s — Chroma persist dir=%s", Config.RAG_BACKEND, Config.CHROMA_PERSIST_DIR)
    # lazy init store and chains
    try:
        get_store()
        get_chains()
    except Exception as e:
        logger.warning("Startup initialization incomplete: %s", e)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
