import os
import sys
from pathlib import Path

import django

# Add coordinator to path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'coordinator'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coordinator.settings_test')
django.setup()

from django.core.management import call_command

call_command('migrate', verbosity=0)

from jobs.models import Job, PageResult, PiiEntity


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    # box1, box2: [x1, y1, x2, y2]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def run_validation():
    print("Synthesizing PII/PHI test corpus...")

    # Create mock job and page
    job = Job.objects.create(status="processing", source_file="doc_spatial_test.pdf")
    page = PageResult.objects.create(job=job, document_id="doc_spatial_test", page_num=1)
    
    # Mock data with high precision floats to test JSONField persistence
    mock_data = [
        {"type": "SSN", "value": "***-**-1234", "box": [10.123456789, 20.987654321, 50.111111111, 30.222222222]},
        {"type": "PHONE", "value": "555-1234", "box": [100.5, 200.75, 150.25, 210.125]},
        {"type": "EMAIL", "value": "test@example.com", "box": [5.0000001, 5.0000001, 15.0000001, 15.0000001]}
    ]
    
    for item in mock_data:
        PiiEntity.objects.create(
            job=job,
            page=page,
            entity_type=item["type"],
            entity_value=item["value"],
            bounding_box=item["box"],
            confidence=0.99
        )
    
    print("Data saved to DB. Retrieving and verifying spatial fidelity...")
    
    retrieved = PiiEntity.objects.filter(job=job).order_by('id')
    
    all_passed = True
    for original, stored in zip(mock_data, retrieved):
        iou = calculate_iou(original["box"], stored.bounding_box)
        print(f"Entity: {original['type']}")
        print(f"  Original: {original['box']}")
        print(f"  Stored:   {stored.bounding_box}")
        print(f"  Overlap Precision (IoU): {iou * 100:.6f}%")
        
        if iou < 0.99:
            print("  [FAIL] Precision loss detected! Overlap < 99%")
            all_passed = False
        else:
            print("  [PASS] Spatial fidelity maintained.")
    
    # Cleanup
    job.delete()
    
    if all_passed:
        print("\nSUCCESS: All spatial fidelity tests passed (>99% overlap).")
        sys.exit(0)
    else:
        print("\nFAILURE: Spatial fidelity tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    run_validation()
