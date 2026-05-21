import pytest

from jobs.models import Job, PageResult, PiiEntity


@pytest.fixture
def test_job(db):
    return Job.objects.create(status="processing", source_file="test_pii_doc.pdf")

@pytest.fixture
def test_page(db, test_job):
    return PageResult.objects.create(
        job=test_job, document_id="test_pii_doc", page_num=1
    )

@pytest.mark.django_db
def test_pii_entity_crud(test_job, test_page):
    """Test Create, Read, Update, Delete for PiiEntity."""
    # Create
    entity = PiiEntity.objects.create(
        job=test_job,
        page=test_page,
        entity_type="SSN",
        entity_value="***-**-9999",
        bounding_box=[10.5, 20.5, 30.5, 40.5],
        confidence=0.95
    )
    assert entity.id is not None
    
    # Read
    retrieved = PiiEntity.objects.get(id=entity.id)
    assert retrieved.entity_type == "SSN"
    assert retrieved.entity_value == "***-**-9999"
    
    # Update
    retrieved.confidence = 0.99
    retrieved.save()
    updated = PiiEntity.objects.get(id=entity.id)
    assert updated.confidence == 0.99
    
    # Delete
    entity_id = entity.id
    updated.delete()
    with pytest.raises(PiiEntity.DoesNotExist):
        PiiEntity.objects.get(id=entity_id)

@pytest.mark.django_db
def test_pii_spatial_accuracy(test_job, test_page):
    """Test that JSONB bounding box preserves exact spatial coordinates."""
    original_box = [12.3456789, 98.7654321, 50.555555, 100.111111]
    
    entity = PiiEntity.objects.create(
        job=test_job,
        page=test_page,
        entity_type="EMAIL",
        entity_value="contact@domain.com",
        bounding_box=original_box,
        confidence=0.9
    )
    
    # Retrieve from DB to ensure it was serialized and deserialized properly
    retrieved = PiiEntity.objects.get(id=entity.id)
    stored_box = retrieved.bounding_box
    
    assert len(stored_box) == 4
    
    # Check for extreme precision retention (standard float equality)
    for orig, stored in zip(original_box, stored_box):
        assert abs(orig - stored) < 1e-6, f"Precision loss: {orig} vs {stored}"
