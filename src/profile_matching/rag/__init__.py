"""Retrieval-Augmented Generation (RAG) layer: embeddings, vector store, ingestion."""

from __future__ import annotations

from profile_matching.rag.embeddings import get_embeddings
from profile_matching.rag.ingestion import ingest_resumes, load_resume_documents
from profile_matching.rag.vector_store import ResumeVectorStore, get_vector_store

__all__ = [
    "ResumeVectorStore",
    "get_embeddings",
    "get_vector_store",
    "ingest_resumes",
    "load_resume_documents",
]
