from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

import numpy as np
from docx import Document
from pypdf import PdfReader

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_PATH,
    DOCS_DIR,
    EMBED_BATCH_SIZE,
    SUPPORTED_EXTENSIONS,
)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".docx":
        doc = Document(path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    if suffix == ".pdf":
        reader = PdfReader(path)
        pages: list[str] = []
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(text)
        return "\n\n".join(pages)

    raise ValueError(f"Unsupported document type: {suffix}")


def _tail_overlap(text: str, overlap: int) -> str:
    if overlap <= 0 or not text:
        return ""

    tail = text[-overlap:]
    first_space = tail.find(" ")
    if first_space >= 0:
        tail = tail[first_space + 1 :]
    return tail.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        current = current.strip()
        if current:
            chunks.append(current)
            current = _tail_overlap(current, overlap)

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            flush()

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if len(candidate) > chunk_size and current:
                flush()
                candidate = f"{current} {word}".strip()
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return list(dict.fromkeys(chunk for chunk in chunks if chunk.strip()))


def document_paths() -> list[Path]:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in DOCS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _relative_name(path: Path) -> str:
    return path.relative_to(DOCS_DIR).as_posix()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            source TEXT PRIMARY KEY,
            file_hash TEXT NOT NULL,
            chunk_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_source
        ON chunks(source);
        """
    )

    document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if document_count == 0 and chunk_count > 0:
        conn.execute("DELETE FROM chunks")

    conn.commit()


def _normalized_vector(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def _embed_texts(embedding_client, texts: list[str]) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    total = len(texts)

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = embedding_client.generate_embeddings(batch)
        batch_vectors = [_normalized_vector(item.embedding) for item in response.data]

        if len(batch_vectors) != len(batch):
            raise RuntimeError("Embedding model returned an unexpected number of vectors.")

        vectors.extend(batch_vectors)
        print(f"[index] {min(start + len(batch), total)}/{total}")

    return vectors


def index_is_current() -> bool:
    paths = document_paths()

    if not DB_PATH.exists():
        return False

    try:
        with _connect() as conn:
            _ensure_schema(conn)
            stored = dict(conn.execute("SELECT source, file_hash FROM documents").fetchall())
    except sqlite3.Error:
        return False

    current = {_relative_name(path): _file_hash(path) for path in paths}
    return current == stored


def sync_index(embedding_client) -> dict[str, int]:
    """
    Synchronize SQLite with docs/ without replacing the database file.

    Only new or modified documents are embedded. Removed documents are deleted
    from SQLite. This avoids Windows file-lock problems and unnecessary reindexing.
    """
    paths = document_paths()
    current_paths = {_relative_name(path): path for path in paths}
    current_hashes = {source: _file_hash(path) for source, path in current_paths.items()}

    with _connect() as conn:
        _ensure_schema(conn)
        stored_hashes = dict(conn.execute("SELECT source, file_hash FROM documents").fetchall())

    removed_sources = sorted(set(stored_hashes) - set(current_hashes))
    changed_sources = sorted(
        source
        for source, file_hash in current_hashes.items()
        if stored_hashes.get(source) != file_hash
    )

    prepared: dict[str, list[str]] = {}
    for source in changed_sources:
        path = current_paths[source]
        try:
            text = read_document(path)
        except Exception as exc:
            raise RuntimeError(f"Could not read '{source}': {exc}") from exc

        chunks = chunk_text(text)
        if not chunks:
            raise RuntimeError(
                f"No readable text could be extracted from '{source}'. "
                "Scanned/image-only PDFs need OCR before using this project."
            )

        prepared[source] = chunks
        print(f"[doc]  {source}: {len(chunks)} chunks")

    new_vectors: dict[str, list[np.ndarray]] = {}
    for source in changed_sources:
        chunks = prepared[source]
        print(f"[index] Embedding {source}...")
        new_vectors[source] = _embed_texts(embedding_client, chunks)

    with _connect() as conn:
        _ensure_schema(conn)

        try:
            conn.execute("BEGIN IMMEDIATE")

            for source in removed_sources:
                conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
                conn.execute("DELETE FROM documents WHERE source = ?", (source,))
                print(f"[index] Removed: {source}")

            for source in changed_sources:
                chunks = prepared[source]
                vectors = new_vectors[source]

                conn.execute("DELETE FROM chunks WHERE source = ?", (source,))

                payload = [
                    (
                        source,
                        index,
                        content,
                        vector.tobytes(),
                        int(vector.size),
                    )
                    for index, (content, vector) in enumerate(zip(chunks, vectors), start=1)
                ]

                conn.executemany(
                    """
                    INSERT INTO chunks(source, chunk_index, content, embedding, embedding_dim)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    payload,
                )

                conn.execute(
                    """
                    INSERT INTO documents(source, file_hash, chunk_count)
                    VALUES (?, ?, ?)
                    ON CONFLICT(source) DO UPDATE SET
                        file_hash = excluded.file_hash,
                        chunk_count = excluded.chunk_count
                    """,
                    (source, current_hashes[source], len(chunks)),
                )

                print(f"[index] Updated: {source}")

            conn.commit()
        except Exception:
            conn.rollback()
            raise

        total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    return {
        "added_or_updated": len(changed_sources),
        "removed": len(removed_sources),
        "total_documents": len(current_hashes),
        "total_chunks": int(total_chunks),
    }


def rebuild_index(embedding_client) -> int:
    """Force a complete rebuild without replacing the SQLite database file."""
    with _connect() as conn:
        _ensure_schema(conn)
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        conn.commit()

    stats = sync_index(embedding_client)
    print(f"[index] Rebuilt: {stats['total_chunks']} chunks")
    return stats["total_chunks"]
