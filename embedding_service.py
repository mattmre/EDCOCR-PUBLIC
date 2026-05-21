"""Embedding generation service for semantic document search.

Phase 10 — Persistent Intelligence Platform: generates text embeddings
for document chunks using sentence-transformers. Chunks can be stored
in the DocumentChunk model for later similarity search.

Configuration:
- ``ENABLE_EMBEDDINGS``: env var to enable/disable (default: false)
- ``EMBEDDING_MODEL``: model name (default: all-MiniLM-L6-v2)
- ``EMBEDDING_CHUNK_SIZE``: characters per chunk (default: 500)
- ``EMBEDDING_CHUNK_OVERLAP``: overlap between chunks (default: 50)

Usage::

    from embedding_service import ChunkEmbedder

    embedder = ChunkEmbedder()
    if embedder.is_available:
        chunks = embedder.chunk_text("long document text...", page_num=1)
        embedded = embedder.embed_chunks(chunks)
        # embedded: list of (chunk_text, embedding_vector, chunk_index, page_num)
"""

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENABLE_EMBEDDINGS = os.environ.get(
    'ENABLE_EMBEDDINGS', 'false'
).lower() in ('true', '1', 'yes')

EMBEDDING_MODEL = os.environ.get(
    'EMBEDDING_MODEL', 'all-MiniLM-L6-v2'
)

EMBEDDING_CHUNK_SIZE = int(os.environ.get('EMBEDDING_CHUNK_SIZE', '500'))
EMBEDDING_CHUNK_OVERLAP = int(os.environ.get('EMBEDDING_CHUNK_OVERLAP', '50'))

# ---------------------------------------------------------------------------
# Guarded sentence-transformers import
# ---------------------------------------------------------------------------

_ST_AVAILABLE = False
_SentenceTransformer = None

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# ChunkEmbedder
# ---------------------------------------------------------------------------


class ChunkEmbedder:
    """Generates embeddings for text chunks using sentence-transformers.

    Graceful degradation: returns empty results when sentence-transformers
    is not installed or ENABLE_EMBEDDINGS is false.
    """

    def __init__(self, model_name=None, chunk_size=None, chunk_overlap=None):
        """Initialize the embedder.

        Args:
            model_name: sentence-transformers model name. Defaults to
                EMBEDDING_MODEL env var or 'all-MiniLM-L6-v2'.
            chunk_size: characters per chunk. Defaults to EMBEDDING_CHUNK_SIZE.
            chunk_overlap: character overlap between consecutive chunks.
                Defaults to EMBEDDING_CHUNK_OVERLAP.
        """
        self._model_name = model_name or EMBEDDING_MODEL
        self._chunk_size = chunk_size or EMBEDDING_CHUNK_SIZE
        self._chunk_overlap = chunk_overlap or EMBEDDING_CHUNK_OVERLAP
        self._model = None
        self._load_attempted = False

        if not _ST_AVAILABLE:
            logger.info(
                "sentence-transformers not available; embedding generation "
                "will be skipped. Install sentence-transformers for embedding support."
            )

    @property
    def is_available(self):
        """Whether sentence-transformers is installed and enabled."""
        return _ST_AVAILABLE and ENABLE_EMBEDDINGS

    @property
    def model_name(self):
        """The configured embedding model name."""
        return self._model_name

    def _ensure_model(self):
        """Lazy-load the sentence-transformers model on first use."""
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True

        if not _ST_AVAILABLE:
            return False

        try:
            self._model = _SentenceTransformer(self._model_name)
            logger.info("Loaded embedding model: %s", self._model_name)
            return True
        except Exception as exc:
            logger.warning(
                "Failed to load embedding model '%s': %s",
                self._model_name,
                exc,
            )
            return False

    def chunk_text(self, text, page_num=0):
        """Split document text into overlapping chunks.

        Args:
            text: Full document/page text to split.
            page_num: Page number for attribution (1-based).

        Returns:
            List of dicts with keys: chunk_text, chunk_index, page_num.
        """
        if not text:
            return []

        chunks = []
        chunk_index = 0
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self._chunk_size, text_len)
            chunk = text[start:end]

            # Try to break at word boundary (look back up to 50 chars)
            if end < text_len:
                last_space = chunk.rfind(' ', max(0, len(chunk) - 50))
                if last_space > 0:
                    chunk = chunk[:last_space]
                    end = start + last_space

            if chunk.strip():
                chunks.append({
                    'chunk_text': chunk.strip(),
                    'chunk_index': chunk_index,
                    'page_num': page_num,
                })
                chunk_index += 1

            # If we've reached the end of the text, stop
            if end >= text_len:
                break

            # Advance with overlap
            step = max(1, end - start - self._chunk_overlap)
            start += step

        return chunks

    def embed_text(self, text):
        """Embed a single text string and return the embedding vector.

        Args:
            text: Text to embed.

        Returns:
            List of floats (embedding vector), or None on failure.
        """
        if not self.is_available:
            return None

        if not self._ensure_model():
            return None

        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as exc:
            logger.warning("Embedding generation failed: %s", exc)
            return None

    def embed_chunks(self, chunks):
        """Generate embeddings for a list of text chunks.

        Args:
            chunks: List of dicts from chunk_text() with at least
                'chunk_text', 'chunk_index', 'page_num' keys.

        Returns:
            List of dicts, each containing:
            - chunk_text: the text content
            - embedding: list of floats (the embedding vector)
            - chunk_index: position index
            - page_num: source page number
            - model: model name used for embedding
        """
        if not chunks:
            return []

        if not self.is_available:
            # Return chunks without embeddings
            return [
                {
                    'chunk_text': c['chunk_text'],
                    'embedding': None,
                    'chunk_index': c['chunk_index'],
                    'page_num': c['page_num'],
                    'model': '',
                }
                for c in chunks
            ]

        if not self._ensure_model():
            return [
                {
                    'chunk_text': c['chunk_text'],
                    'embedding': None,
                    'chunk_index': c['chunk_index'],
                    'page_num': c['page_num'],
                    'model': '',
                }
                for c in chunks
            ]

        texts = [c['chunk_text'] for c in chunks]
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
        except Exception as exc:
            logger.warning("Batch embedding generation failed: %s", exc)
            return [
                {
                    'chunk_text': c['chunk_text'],
                    'embedding': None,
                    'chunk_index': c['chunk_index'],
                    'page_num': c['page_num'],
                    'model': '',
                }
                for c in chunks
            ]

        results = []
        for i, chunk in enumerate(chunks):
            results.append({
                'chunk_text': chunk['chunk_text'],
                'embedding': embeddings[i].tolist(),
                'chunk_index': chunk['chunk_index'],
                'page_num': chunk['page_num'],
                'model': self._model_name,
            })

        return results

    def embed_and_store(self, text, job_id, page_num=0):
        """Chunk text, embed chunks, and prepare for DocumentChunk storage.

        This is a convenience method that combines chunk_text() and
        embed_chunks() into a single call, returning data ready for
        Django model creation.

        Args:
            text: Full text to chunk and embed.
            job_id: UUID of the parent Job.
            page_num: Page number (1-based).

        Returns:
            List of dicts ready for DocumentChunk.objects.create():
            - job_id, page_number, chunk_index, chunk_text,
              embedding_json, embedding_model, metadata_json
        """
        chunks = self.chunk_text(text, page_num)
        embedded = self.embed_chunks(chunks)

        records = []
        for item in embedded:
            records.append({
                'job_id': job_id,
                'page_number': item['page_num'],
                'chunk_index': item['chunk_index'],
                'chunk_text': item['chunk_text'],
                'embedding_json': item['embedding'],
                'embedding_model': item['model'],
                'metadata_json': {},
            })
        return records
