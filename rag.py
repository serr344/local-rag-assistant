from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager

from config import APP_NAME, CHAT_MODEL, DB_PATH, EMBEDDING_MODEL, TOP_K
from ingest import rebuild_index, sync_index


MIN_RELEVANCE_SCORE = 0.50


@dataclass(frozen=True)
class SearchResult:
    source: str
    chunk_index: int
    content: str
    score: float


class LocalRAG:
    def __init__(
        self,
        top_k: int = TOP_K,
        force_reindex: bool = False,
        min_relevance_score: float = MIN_RELEVANCE_SCORE,
    ):
        self.top_k = max(1, top_k)
        self.force_reindex = force_reindex
        self.min_relevance_score = float(min_relevance_score)

        self.embedding_model = None
        self.embedding_client = None
        self.chat_model = None
        self.chat_client = None

        self.sources: list[str] = []
        self.chunk_indices: list[int] = []
        self.contents: list[str] = []
        self.matrix = np.empty((0, 0), dtype=np.float32)

    @staticmethod
    def _download_progress(label: str):
        def callback(progress: float) -> None:
            print(f"\r[{label}] {progress:5.1f}%", end="", flush=True)

        return callback

    @staticmethod
    def _get_manager():
        try:
            FoundryLocalManager.initialize(Configuration(app_name=APP_NAME))
        except Exception as error:
            if "already been initialized" not in str(error).lower():
                raise

        return FoundryLocalManager.instance

    def _load_embedding_model(self, manager) -> None:
        if self.embedding_client is not None:
            return

        print(f"[model] Embedding: {EMBEDDING_MODEL}")
        self.embedding_model = manager.catalog.get_model(EMBEDDING_MODEL)
        self.embedding_model.download(self._download_progress("embedding download"))
        print()
        self.embedding_model.load()
        self.embedding_client = self.embedding_model.get_embedding_client()

    def _load_chat_model(self, manager) -> None:
        if self.chat_client is not None:
            return

        print(f"[model] Chat: {CHAT_MODEL}")
        self.chat_model = manager.catalog.get_model(CHAT_MODEL)
        self.chat_model.download(self._download_progress("chat download"))
        print()
        self.chat_model.load()
        self.chat_client = self.chat_model.get_chat_client()

    def start(self) -> None:
        manager = self._get_manager()

        self._load_embedding_model(manager)

        if self.force_reindex:
            rebuild_index(self.embedding_client)
        else:
            stats = sync_index(self.embedding_client)
            if stats["added_or_updated"] or stats["removed"]:
                print(
                    f"[index] Synced: {stats['added_or_updated']} updated, "
                    f"{stats['removed']} removed."
                )
            else:
                print("[index] Existing index is current.")

        self._load_index_into_memory()
        self._load_chat_model(manager)

    def _load_index_into_memory(self) -> None:
        self.sources.clear()
        self.chunk_indices.clear()
        self.contents.clear()
        self.matrix = np.empty((0, 0), dtype=np.float32)

        if not DB_PATH.exists():
            print("[index] No database yet.")
            return

        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            rows = conn.execute(
                """
                SELECT source, chunk_index, content, embedding, embedding_dim
                FROM chunks
                ORDER BY id
                """
            ).fetchall()

        if not rows:
            print("[index] No chunks loaded. Add a document to begin.")
            return

        vectors: list[np.ndarray] = []

        for source, chunk_index, content, blob, dim in rows:
            vector = np.frombuffer(blob, dtype=np.float32, count=dim).copy()
            vectors.append(vector)
            self.sources.append(source)
            self.chunk_indices.append(int(chunk_index))
            self.contents.append(content)

        dimensions = {vector.size for vector in vectors}
        if len(dimensions) != 1:
            raise RuntimeError(
                "Stored embeddings have inconsistent dimensions. Use Rebuild Index."
            )

        self.matrix = np.vstack(vectors).astype(np.float32, copy=False)
        print(f"[index] Loaded {len(rows)} chunks into RAM.")

    def refresh_index(self) -> dict[str, int]:
        """Sync changed documents without unloading or reloading either model."""
        if self.embedding_client is None:
            raise RuntimeError("The embedding model is not loaded.")

        stats = sync_index(self.embedding_client)
        self._load_index_into_memory()
        return stats

    def rebuild_all(self) -> int:
        """Force a full re-embedding while keeping the models loaded."""
        if self.embedding_client is None:
            raise RuntimeError("The embedding model is not loaded.")

        count = rebuild_index(self.embedding_client)
        self._load_index_into_memory()
        return count

    def search(self, question: str) -> list[SearchResult]:
        question = question.strip()
        if not question or self.matrix.size == 0:
            return []

        response = self.embedding_client.generate_embedding(question)
        query = np.asarray(response.data[0].embedding, dtype=np.float32)
        norm = float(np.linalg.norm(query))

        if norm == 0.0:
            return []

        query /= norm

        if self.matrix.shape[1] != query.size:
            raise RuntimeError(
                "Query embedding dimension does not match the stored index. "
                "Use Rebuild Index."
            )

        scores = self.matrix @ query
        k = min(self.top_k, scores.size)

        if k <= 0:
            return []

        if k == scores.size:
            indices = np.argsort(scores)[::-1]
        else:
            indices = np.argpartition(scores, -k)[-k:]
            indices = indices[np.argsort(scores[indices])[::-1]]

        return [
            SearchResult(
                source=self.sources[int(i)],
                chunk_index=self.chunk_indices[int(i)],
                content=self.contents[int(i)],
                score=float(scores[int(i)]),
            )
            for i in indices[:k]
        ]

    @staticmethod
    def _build_messages(
        question: str,
        results: list[SearchResult],
    ) -> list[dict[str, str]]:
        context = "\n\n".join(
            f"[SOURCE {number}: {item.source} | chunk {item.chunk_index}]\n{item.content}"
            for number, item in enumerate(results, start=1)
        )

        system_prompt = (
            "You are a local document question-answering assistant. "
            "Answer ONLY from the supplied CONTEXT. "
            "Never use outside knowledge, even if you already know the answer. "
            "Never guess or invent missing information. "
            "If the answer is not supported by the CONTEXT, say that the uploaded "
            "documents do not contain enough information to answer. "
            "Answer in the same language as the user's question. "
            "Be concise but complete.\n\n"
            f"CONTEXT:\n{context}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

    def answer(self, question: str) -> tuple[str, list[SearchResult]]:
        results = self.search(question)

        if not results:
            return "No relevant information was found in the uploaded documents.", []

        best_score = results[0].score

        if best_score < self.min_relevance_score:
            return (
                "The answer to this question is not present in the uploaded documents. "
                f"(best similarity: {best_score:.3f})",
                [],
            )

        messages = self._build_messages(question, results)
        parts: list[str] = []

        for chunk in self.chat_client.complete_streaming_chat(messages):
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                parts.append(content)

        generated_answer = "".join(parts).strip()
        if not generated_answer:
            generated_answer = "The local model returned an empty response."

        return generated_answer, results

    def close(self) -> None:
        if self.chat_model is not None:
            try:
                self.chat_model.unload()
            except Exception:
                pass
            finally:
                self.chat_model = None
                self.chat_client = None

        if self.embedding_model is not None:
            try:
                self.embedding_model.unload()
            except Exception:
                pass
            finally:
                self.embedding_model = None
                self.embedding_client = None
