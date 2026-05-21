"""Django models for persistent extraction storage.

Phase 10 — Persistent Intelligence Platform: stores extracted entities,
form key-value pairs, and document chunks for semantic search.

These models provide queryable, indexed storage for OCR pipeline outputs
that were previously only available as sidecar JSON files.
"""

from django.db import models

from .encrypted_field import EncryptedTextField


class ExtractedEntity(models.Model):
    """Persisted entity extracted from a processed document.

    Stores entities from NER, structured extraction, PII detection, and
    other extraction modules. Each entity has spatial coordinates for
    forensic traceability back to the source document.
    """

    job = models.ForeignKey(
        'Job',
        on_delete=models.CASCADE,
        related_name='extracted_entities',
    )
    page_number = models.IntegerField()
    entity_type = models.CharField(
        max_length=100,
        db_index=True,
        help_text='Entity type (e.g., PERSON, DATE, AMOUNT, CASE_NUMBER)',
    )
    entity_text = EncryptedTextField()
    confidence = models.FloatField(default=0.0)

    # Spatial bounding box (optional — not all extractors provide coordinates)
    bbox_x1 = models.FloatField(null=True, blank=True)
    bbox_y1 = models.FloatField(null=True, blank=True)
    bbox_x2 = models.FloatField(null=True, blank=True)
    bbox_y2 = models.FloatField(null=True, blank=True)

    source_module = models.CharField(
        max_length=50,
        help_text='Extraction source (ner, extraction, pii, barcode, etc.)',
    )
    metadata_json = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['entity_type', 'entity_text']),
            models.Index(fields=['job', 'page_number']),
        ]
        ordering = ['job', 'page_number', 'entity_type']

    def __str__(self):
        return (
            f"{self.entity_type}: {self.entity_text[:50]} "
            f"(page {self.page_number}, {self.confidence:.2f})"
        )


class ExtractedFormValue(models.Model):
    """Key-value pair extracted from a form or document.

    Stores structured field extractions (e.g., invoice_number, date,
    total_amount) from the PaddleNLP UIE or regex extraction modules.
    """

    job = models.ForeignKey(
        'Job',
        on_delete=models.CASCADE,
        related_name='form_values',
    )
    page_number = models.IntegerField()
    field_key = models.CharField(max_length=255)
    field_value = EncryptedTextField()
    confidence = models.FloatField(default=0.0)
    source_module = models.CharField(
        max_length=50,
        help_text='Extraction source (extraction, form_kv, etc.)',
    )
    metadata_json = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['field_key']),
            models.Index(fields=['job', 'page_number']),
        ]
        ordering = ['job', 'page_number', 'field_key']

    def __str__(self):
        return (
            f"{self.field_key}={self.field_value[:50]} "
            f"(page {self.page_number}, {self.confidence:.2f})"
        )


class DocumentChunk(models.Model):
    """Document chunk for semantic search (pgvector ready).

    Stores text chunks with optional embedding vectors for similarity
    search. Embeddings are stored as JSON lists of floats until pgvector
    is available, at which point they can be migrated to a native vector
    column.
    """

    job = models.ForeignKey(
        'Job',
        on_delete=models.CASCADE,
        related_name='chunks',
    )
    page_number = models.IntegerField()
    chunk_index = models.IntegerField()
    chunk_text = models.TextField()
    contains_pii = models.BooleanField(
        default=False,
        help_text='Flag for selective future encryption of PII-bearing chunks',
    )

    # Embedding stored as JSON list of floats (pgvector-ready migration path)
    embedding_json = models.JSONField(null=True, blank=True)
    embedding_model = models.CharField(max_length=100, default='', blank=True)

    metadata_json = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['job', 'page_number']),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['job', 'page_number', 'chunk_index'],
                name='unique_chunk',
            ),
        ]
        ordering = ['job', 'page_number', 'chunk_index']

    def __str__(self):
        preview = self.chunk_text[:40] + '...' if len(self.chunk_text) > 40 else self.chunk_text
        return f"Chunk {self.chunk_index} (page {self.page_number}): {preview}"
