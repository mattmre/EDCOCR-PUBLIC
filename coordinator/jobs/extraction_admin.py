"""Django admin registration for Phase 10 extraction models."""

from django.contrib import admin

from .extraction_models import DocumentChunk, ExtractedEntity, ExtractedFormValue


@admin.register(ExtractedEntity)
class ExtractedEntityAdmin(admin.ModelAdmin):
    list_display = [
        'job',
        'page_number',
        'entity_type',
        'entity_text_short',
        'confidence',
        'source_module',
        'created_at',
    ]
    list_filter = ['entity_type', 'source_module']
    search_fields = ['entity_text', 'entity_type']
    raw_id_fields = ['job']
    readonly_fields = ['created_at']

    @admin.display(description='Text')
    def entity_text_short(self, obj):
        if len(obj.entity_text) > 60:
            return obj.entity_text[:60] + '...'
        return obj.entity_text


@admin.register(ExtractedFormValue)
class ExtractedFormValueAdmin(admin.ModelAdmin):
    list_display = [
        'job',
        'page_number',
        'field_key',
        'field_value_short',
        'confidence',
        'source_module',
        'created_at',
    ]
    list_filter = ['source_module']
    search_fields = ['field_key', 'field_value']
    raw_id_fields = ['job']
    readonly_fields = ['created_at']

    @admin.display(description='Value')
    def field_value_short(self, obj):
        if len(obj.field_value) > 60:
            return obj.field_value[:60] + '...'
        return obj.field_value


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = [
        'job',
        'page_number',
        'chunk_index',
        'chunk_text_short',
        'embedding_model',
        'has_embedding',
        'created_at',
    ]
    list_filter = ['embedding_model']
    search_fields = ['chunk_text']
    raw_id_fields = ['job']
    readonly_fields = ['created_at']

    @admin.display(description='Text')
    def chunk_text_short(self, obj):
        if len(obj.chunk_text) > 80:
            return obj.chunk_text[:80] + '...'
        return obj.chunk_text

    @admin.display(boolean=True, description='Embedded')
    def has_embedding(self, obj):
        return obj.embedding_json is not None
