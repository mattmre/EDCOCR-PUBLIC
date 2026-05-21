"""Shared fixtures for integration and end-to-end tests.

Provides realistic document data (text, OCR lines, structure data)
for exercising all five sidecar modules together.
"""

import importlib.machinery
import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

# Add project root to path so modules can be imported directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set permissive rate limits for the test suite.  The default 10/minute
# submit rate is too low when hundreds of tests share the same in-memory
# storage singleton.  This must be set before api.limits is imported
# because the rate string is evaluated at module-import time.
PERMISSIVE_RATE_LIMIT = "10000/minute"
os.environ.setdefault("OCR_SUBMIT_RATE_LIMIT", PERMISSIVE_RATE_LIMIT)
os.environ.setdefault("OCR_RATE_LIMIT", PERMISSIVE_RATE_LIMIT)
# FastAPI app startup now requires explicit insecure override when OCR_API_KEY
# is unset; keep test defaults explicit.
os.environ.setdefault("ALLOW_UNAUTHENTICATED", "true")
# Anonymous role defaults to "viewer" in production; set to "admin" here so
# existing tests that rely on admin-level anonymous access are unaffected.
# Tests that specifically verify anonymous role behavior patch this explicitly.
os.environ.setdefault("ANONYMOUS_ROLE", "admin")

# Pre-import slowapi and its dependency chain (limits → slowapi.extension)
# to prevent "partially initialized module" circular-import errors when
# the full test suite triggers concurrent/interleaved module loading.
try:
    import slowapi  # noqa: F401
    import slowapi.extension  # noqa: F401
except ImportError:
    pass

# Many tests import ocr_gpu_async for queue/state helpers without ever
# exercising the Tesseract OCR path. Provide a lightweight stub only when the
# dependency is missing so import-time module setup remains testable.
if importlib.util.find_spec("pytesseract") is None and "pytesseract" not in sys.modules:
    pytesseract_stub = ModuleType("pytesseract")
    pytesseract_stub.__spec__ = importlib.machinery.ModuleSpec(
        "pytesseract",
        loader=None,
    )
    pytesseract_stub.Output = SimpleNamespace(DICT="DICT")

    def _stub_get_tesseract_version():
        return "stubbed"

    def _stub_image_to_data(*_args, **_kwargs):
        raise RuntimeError("pytesseract is not installed in this test environment")

    pytesseract_stub.get_tesseract_version = _stub_get_tesseract_version
    pytesseract_stub.image_to_data = _stub_image_to_data
    sys.modules["pytesseract"] = pytesseract_stub


@pytest.fixture(autouse=True)
def _reset_job_manager_singleton():
    """Clear the JobManager singleton between tests.

    Without this, the singleton created by one test's DB fixture would
    persist into the next test with a stale session factory.

    worker threads are now non-daemon, so we must call shutdown()
    on any live singleton before clearing the reference to avoid leaking
    worker threads between tests.

    When API dependencies are not installed (e.g. in the sdk-tests CI job
    which only installs httpx/pydantic), the import is skipped gracefully
    so SDK-only tests can run without the full API stack.
    """
    try:
        import api.job_manager as _jm
    except ImportError:
        yield
        return

    def _drain_singleton() -> None:
        instance = _jm._manager_instance
        _jm._manager_instance = None
        if instance is None:
            return
        try:
            instance.shutdown(timeout=5.0)
        except Exception:
            pass

    _drain_singleton()
    yield
    _drain_singleton()


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    """Give each test a fresh SQLite database.

    This is the shared base fixture for API database isolation.  It patches
    both ``api.config.DB_PATH`` and ``api.database.DB_PATH`` to a per-test
    temporary file and resets the SQLAlchemy engine before and after each
    test.

    Test files that need *additional* patches (e.g. SOURCE_FOLDER,
    OUTPUT_FOLDER, EVENT_STORE_PATH) define their own ``_isolate_db``
    fixture which shadows this one.

    When API dependencies are not installed (e.g. in the sdk-tests CI job),
    the fixture yields immediately so SDK-only tests are not affected.
    """
    try:
        from api.database import get_engine, reset_engine
    except ImportError:
        yield
        return

    reset_engine()
    db_file = str(tmp_path / "test_jobs.db")
    with patch("api.config.DB_PATH", db_file), \
         patch("api.database.DB_PATH", db_file):
        reset_engine()
        get_engine(db_file)
        yield
        reset_engine()


@pytest.fixture
def output_dir(tmp_path):
    """Return a temporary directory for sidecar JSON output."""
    return tmp_path


@pytest.fixture
def sample_page_texts():
    """Realistic OCR text for a 3-page document.

    Page 1: Invoice (triggers classification + extraction: dates, amounts, email, phone)
    Page 2: Legal text (triggers NER: Case No., Bates Number, Exhibit)
    Page 3: Letter (triggers extraction: amounts, dates, emails)
    """
    return {
        1: (
            "INVOICE\n"
            "Invoice Number: INV-2024-00457\n"
            "Date: January 15, 2024\n"
            "Due Date: 02/15/2024\n\n"
            "Bill To: Acme Corporation\n"
            "123 Main Street, Suite 400\n"
            "Springfield, IL 62704\n\n"
            "Description                   Qty    Amount\n"
            "Legal consulting services      40    $5,000.00\n"
            "Document review                20    $2,500.00\n"
            "Court filing fees               3      $450.00\n\n"
            "Subtotal: $7,950.00\n"
            "Tax (8%): $636.00\n"
            "Total: $8,586.00\n"
            "Amount Due: $8,586.00\n\n"
            "Payment Terms: Net 30\n"
            "Contact: billing@lawfirm.com\n"
            "Phone: (555) 123-4567\n"
        ),
        2: (
            "IN THE UNITED STATES DISTRICT COURT\n"
            "FOR THE NORTHERN DISTRICT OF ILLINOIS\n\n"
            "Case No. 2024-CV-01234\n"
            "Docket No. 2024-56789\n\n"
            "SMITH ENTERPRISES, INC.,\n"
            "    Plaintiff,\n"
            "v.\n"
            "JONES HOLDINGS LLC,\n"
            "    Defendant.\n\n"
            "MEMORANDUM IN SUPPORT OF MOTION TO DISMISS\n\n"
            "Pursuant to Fed. R. Civ. P. 12(b)(6), Defendant hereby\n"
            "moves to dismiss the Complaint. See Exhibit A for the\n"
            "original agreement dated March 10, 2023. As noted in\n"
            "Exhibit B-2, the statute of limitations has expired.\n\n"
            "The relevant documents are stamped ABC001234 through\n"
            "ABC001290. Plaintiff's Exhibit 1 was produced on\n"
            "2023-06-15 as part of initial disclosures.\n\n"
            "Respectfully submitted,\n"
            "Jane Attorney, Esq.\n"
            "Bar No. 12345\n"
        ),
        3: (
            "Dear Mr. Johnson,\n\n"
            "Thank you for your letter dated December 5, 2023\n"
            "regarding the outstanding balance of $12,450.00 on\n"
            "account REF: 78901.\n\n"
            "We have reviewed the matter and can confirm that a\n"
            "payment of $6,225.00 was processed on 2024-01-10.\n"
            "The remaining balance of $6,225.00 is scheduled for\n"
            "payment on 2024-02-10.\n\n"
            "Please direct any questions to our accounts department\n"
            "at accounts@example.com or call (555) 987-6543.\n\n"
            "Sincerely,\n"
            "Robert Williams\n"
            "Chief Financial Officer\n"
            "Global Industries Inc.\n"
        ),
    }


@pytest.fixture
def sample_paddle_lines():
    """Synthetic PaddleOCR line tuples: (text, confidence, [x1, y1, x2, y2]).

    Page 1: 10 printed lines (high confidence ~0.92)
    Page 2: 5 printed + 5 handwritten (low confidence ~0.35)
    Page 3: 8 handwritten lines (low confidence)
    """
    page1_lines = [
        ("INVOICE", 0.95, [50, 10, 200, 40]),
        ("Invoice Number: INV-2024-00457", 0.93, [50, 50, 400, 80]),
        ("Date: January 15, 2024", 0.92, [50, 90, 350, 120]),
        ("Bill To: Acme Corporation", 0.91, [50, 140, 380, 170]),
        ("123 Main Street, Suite 400", 0.94, [50, 180, 400, 210]),
        ("Legal consulting services  40  $5,000.00", 0.90, [50, 250, 500, 280]),
        ("Document review  20  $2,500.00", 0.93, [50, 290, 500, 320]),
        ("Subtotal: $7,950.00", 0.92, [50, 360, 350, 390]),
        ("Total: $8,586.00", 0.94, [50, 400, 300, 430]),
        ("Payment Terms: Net 30", 0.91, [50, 460, 350, 490]),
    ]

    page2_lines = [
        # 5 printed lines (high confidence)
        ("IN THE UNITED STATES DISTRICT COURT", 0.96, [100, 10, 600, 45]),
        ("Case No. 2024-CV-01234", 0.94, [100, 60, 400, 90]),
        ("SMITH ENTERPRISES, INC., Plaintiff", 0.93, [100, 110, 500, 140]),
        ("v. JONES HOLDINGS LLC, Defendant", 0.92, [100, 150, 500, 180]),
        ("MEMORANDUM IN SUPPORT OF MOTION", 0.95, [100, 200, 550, 235]),
        # 5 handwritten lines (low confidence)
        ("See attached notes", 0.32, [50, 300, 350, 345]),
        ("Reviewed by J. Smith 1/20/24", 0.28, [50, 360, 450, 410]),
        ("Approved for filing", 0.35, [50, 430, 320, 470]),
        ("Rush - expedite", 0.30, [50, 490, 280, 535]),
        ("Confidential", 0.38, [50, 555, 250, 595]),
    ]

    page3_lines = [
        ("Dear Mr Johnson", 0.30, [40, 20, 350, 65]),
        ("Thank you for your letter", 0.28, [40, 80, 450, 130]),
        ("regarding the balance of $12,450", 0.33, [40, 150, 500, 200]),
        ("We confirm payment of $6,225", 0.35, [40, 220, 480, 265]),
        ("processed on January 10", 0.31, [40, 285, 400, 330]),
        ("remaining balance scheduled", 0.29, [40, 350, 430, 395]),
        ("Please contact accounts dept", 0.32, [40, 420, 460, 465]),
        ("Sincerely Robert Williams", 0.27, [40, 510, 420, 560]),
    ]

    return {1: page1_lines, 2: page2_lines, 3: page3_lines}


@pytest.fixture
def sample_structure_data():
    """Synthetic layout/table/form data for 3 pages.

    Page 1: Table with monetary content (invoice layout)
    Page 2: Title + text blocks (legal document layout)
    Page 3: Text blocks only (letter layout)
    """
    return {
        1: {
            "layout_regions": [
                {"type": "title", "bbox": [50, 10, 200, 40], "text": "INVOICE"},
                {"type": "text", "bbox": [50, 50, 400, 210], "text": "Bill To: Acme Corporation"},
                {"type": "table", "bbox": [50, 230, 500, 400], "text": ""},
                {"type": "text", "bbox": [50, 410, 400, 500], "text": "Total: $8,586.00"},
            ],
            "tables": [
                {
                    "html": "<table><tr><td>Description</td><td>Amount</td></tr></table>",
                    "bbox": [50, 230, 500, 400],
                },
            ],
            "form_fields": [],
        },
        2: {
            "layout_regions": [
                {"type": "title", "bbox": [100, 10, 600, 45], "text": "UNITED STATES DISTRICT COURT"},
                {"type": "title", "bbox": [100, 55, 500, 85], "text": "NORTHERN DISTRICT OF ILLINOIS"},
                {"type": "text", "bbox": [100, 100, 500, 180], "text": "Case details"},
                {"type": "text", "bbox": [100, 190, 550, 400], "text": "Memorandum text"},
                {"type": "text", "bbox": [100, 410, 500, 500], "text": "Conclusion"},
            ],
            "tables": [],
            "form_fields": [],
        },
        3: {
            "layout_regions": [
                {"type": "text", "bbox": [40, 20, 500, 300], "text": "Letter body"},
                {"type": "text", "bbox": [40, 310, 500, 500], "text": "Closing"},
            ],
            "tables": [],
            "form_fields": [],
        },
    }
