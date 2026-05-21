# LayoutLMv3 Fine-Tuning Operator Guide

## 1. Overview

### What is LayoutLMv3?

LayoutLMv3 is a multimodal transformer model from Microsoft that jointly models text, layout (bounding boxes), and visual features for document understanding tasks. Unlike pure text models, LayoutLMv3 understands where words appear on a page, making it ideal for structured document extraction.

In EDCOCR, LayoutLMv3 is used exclusively for **token classification** (BIO tagging) -- not text generation. This preserves the pipeline's CTC-safe (zero hallucination) guarantee: the model labels existing OCR text rather than generating new text.

### The 7 Modules

| Module | Purpose |
|---|---|
| `layoutlm_labels.py` | Label configuration -- 4 built-in label sets (default, forensic, receipt, form) with BIO expansion, custom JSON support, and environment-driven selection |
| `layoutlm_data.py` | Data loading -- JSONL parser with label validation, HuggingFace dataset creation with sub-word alignment |
| `layoutlm_finetune.py` | Training -- HuggingFace Trainer with optional LoRA adapters via `peft`, seqeval metrics, full CLI |
| `layoutlm_evaluate.py` | Evaluation -- Entity-level precision/recall/F1 via seqeval, per-entity breakdown, JSON reports |
| `layoutlm_model_registry.py` | Model versioning -- File-based registry (manifest.json), version tracking, active model selection |
| `layoutlm_calibration.py` | Confidence calibration -- Temperature scaling, Platt scaling, isotonic regression for well-calibrated entity confidence scores |
| `layoutlm_summarization.py` | Extractive summarization -- TextRank, entity density, and layout position scoring (CTC-safe, no generation) |

### Use Cases

- **Forensic document NER**: Extract case numbers, Bates numbers, exhibit numbers, court names, attorney names, filing dates, and docket numbers from legal documents.
- **Invoice/receipt extraction**: Extract store names, dates, line items, totals, payment methods, and card numbers from financial documents.
- **Form field extraction**: Identify field labels, field values, checkboxes, and signature fields in structured forms.
- **General business NER**: Extract dates, amounts, person names, organizations, addresses, and reference numbers from any document type.

### Design Principles

All heavy ML imports (`torch`, `transformers`, `peft`, `datasets`, `seqeval`) are **lazy** -- imported inside functions, not at module level. This means:

- Modules can be imported and tested without GPU dependencies.
- The lightweight Python-only modules (`layoutlm_labels.py`, `layoutlm_calibration.py`, `layoutlm_summarization.py`) work without any ML stack installed.
- Production inference and training only load heavy dependencies when actually called.

---

## 2. Prerequisites

### Hardware Requirements

| Task | GPU VRAM | System RAM | Storage |
|---|---|---|---|
| Fine-tuning (LayoutLMv3-base) | 8 GB minimum | 16 GB | 5 GB per checkpoint |
| Fine-tuning with LoRA | 6 GB minimum | 16 GB | 1 GB per adapter |
| Inference only | 4 GB minimum | 8 GB | 1 GB per model |
| CPU-only inference (ONNX) | N/A | 16 GB | 1 GB per model |

LoRA (Low-Rank Adaptation) reduces VRAM requirements by training only small adapter matrices instead of the full model. This is recommended for most fine-tuning scenarios.

### Python Dependencies

Core dependencies for training:

```
torch>=2.0.0               # PyTorch (CUDA build for GPU training)
transformers>=4.30.0        # HuggingFace Transformers (LayoutLMv3)
datasets>=2.14.0            # HuggingFace Datasets
seqeval>=1.2.2              # Entity-level NER evaluation metrics
numpy>=1.24.0               # Numerical operations
```

Optional dependencies:

```
peft>=0.5.0                 # LoRA adapter support (parameter-efficient fine-tuning)
scikit-learn>=1.3.0         # Isotonic regression for confidence calibration
```

Install all training dependencies:

```bash
pip install torch transformers datasets seqeval peft
```

### Dataset Formats

LayoutLMv3 training requires annotated document pages in JSONL format. Each line represents one page with word-level bounding boxes and BIO labels. See Section 3 for the complete data format specification.

---

## 3. Data Preparation

### JSONL Format

Each line in a training JSONL file is a JSON object representing a single annotated page:

```json
{
  "doc_id": "invoice_001",
  "page_num": 1,
  "words": [
    {"text": "Invoice", "bbox": [10, 20, 100, 40], "label": "O"},
    {"text": "#12345",  "bbox": [105, 20, 200, 40], "label": "B-INVOICE_NUMBER"},
    {"text": "Date:",   "bbox": [10, 50, 60, 70],   "label": "O"},
    {"text": "2026-01-15", "bbox": [65, 50, 180, 70], "label": "B-DATE"}
  ],
  "image_path": "images/invoice_001_p1.png"
}
```

Field descriptions:

| Field | Type | Required | Description |
|---|---|---|---|
| `doc_id` | string | Recommended | Document identifier for tracking |
| `page_num` | integer | Recommended | 1-based page number within document |
| `words` | array | Yes | List of word objects (see below) |
| `words[].text` | string | Yes | The word text |
| `words[].bbox` | array of 4 ints | Yes | Bounding box `[x0, y0, x1, y1]` in pixel coordinates |
| `words[].label` | string | Yes | BIO tag: `"O"`, `"B-ENTITY_TYPE"`, or `"I-ENTITY_TYPE"` |
| `image_path` | string | Optional | Path to the page image (for multimodal features) |

### BIO Tagging Convention

Labels follow the standard BIO (Begin-Inside-Outside) scheme:

- `O` -- Outside any entity (non-entity word)
- `B-ENTITY_TYPE` -- Beginning of a new entity
- `I-ENTITY_TYPE` -- Inside (continuation) of the current entity

Example for a multi-word entity:

```
"John"   -> B-PERSON_NAME
"Q."     -> I-PERSON_NAME
"Public" -> I-PERSON_NAME
```

### Bounding Box Coordinates

Bounding boxes use pixel coordinates in `[x0, y0, x1, y1]` format:

- `x0, y0`: top-left corner
- `x1, y1`: bottom-right corner
- Coordinates are relative to the page image at OCR resolution (typically 300 DPI)

If bounding boxes are not available, use `[0, 0, 0, 0]` as a placeholder. The model will still learn from the text tokens, but layout-aware features will be degraded.

### Converting Custom Datasets

**From FUNSD format** (standard NER on forms):

FUNSD provides annotations as JSON with `form` entries containing `words` with `text` and `box` fields. Convert by mapping FUNSD labels to BIO tags and reformatting boxes:

```python
import json

def funsd_to_jsonl(funsd_json_path, output_jsonl_path, doc_id="funsd"):
    with open(funsd_json_path) as f:
        data = json.load(f)

    words = []
    for item in data.get("form", []):
        label = item.get("label", "other")
        for idx, word_info in enumerate(item.get("words", [])):
            bio_prefix = "B-" if idx == 0 and label != "other" else "I-"
            bio_label = f"{bio_prefix}{label.upper}" if label != "other" else "O"
            words.append({
                "text": word_info["text"],
                "bbox": word_info["box"],  # FUNSD uses [x0, y0, x1, y1]
                "label": bio_label,
            })

    page = {"doc_id": doc_id, "page_num": 1, "words": words}
    with open(output_jsonl_path, "a") as f:
        f.write(json.dumps(page) + "\n")
```

**From OCR pipeline output** (using EDCOCR's own output):

EDCOCR's PaddleOCR workers produce `(text, bbox, confidence)` tuples. These can be exported to JSONL for annotation. Words start with label `"O"` and are then manually or semi-automatically annotated.

### Data Splits

The `create_hf_dataset` function in `layoutlm_data.py` automatically splits data:

- Default: 80% train / 20% test (controlled by `--test-size`)
- Random split with configurable seed (`--seed 42`)
- When only 1 sample exists, it is duplicated into both splits

For production workflows, prepare separate JSONL files:

```
data/
  train.jsonl       # Training data (hundreds to thousands of pages)
  dev.jsonl         # Validation data (~10-20% of training size)
  test.jsonl        # Held-out test data (never seen during training)
```

### Label Configuration

Labels are defined in `layoutlm_labels.py`. Four built-in sets are available:

**default** (9 entity types, 19 BIO labels):
INVOICE_NUMBER, DATE, AMOUNT, PERSON_NAME, ORGANIZATION, ADDRESS, REFERENCE_NUMBER, PHONE_NUMBER, EMAIL

**forensic** (17 entity types, 35 BIO labels):
Default entities plus CASE_NUMBER, BATES_NUMBER, EXHIBIT_NUMBER, COURT_NAME, JUDGE_NAME, ATTORNEY_NAME, FILING_DATE, DOCKET_NUMBER

**receipt** (11 entity types, 23 BIO labels):
STORE_NAME, STORE_ADDRESS, DATE, TIME, ITEM_NAME, ITEM_PRICE, SUBTOTAL, TAX, TOTAL, PAYMENT_METHOD, CARD_NUMBER

**form** (8 entity types, 17 BIO labels):
FIELD_LABEL, FIELD_VALUE, CHECKBOX, SIGNATURE_FIELD, DATE_FIELD, PERSON_NAME, ORGANIZATION, ADDRESS

To define a custom label set, create a JSON file:

```json
{
  "name": "medical",
  "entity_types": [
    "PATIENT_NAME",
    "DATE_OF_BIRTH",
    "DIAGNOSIS_CODE",
    "MEDICATION",
    "DOSAGE",
    "PRESCRIBER_NAME"
  ],
  "type_map": {
    "PATIENT_NAME": "person_name",
    "DATE_OF_BIRTH": "date",
    "DIAGNOSIS_CODE": "reference_number",
    "MEDICATION": "medication",
    "DOSAGE": "amount",
    "PRESCRIBER_NAME": "person_name"
  }
}
```

Use it with `--label-set path/to/medical.json`.

---

## 4. Fine-Tuning

### CLI Invocation

The `layoutlm_finetune.py` module provides a full CLI. Run it directly:

```bash
python layoutlm_finetune.py \
    --dataset custom \
    --data-dir ./data \
    --output-dir ./models/forensic-v1 \
    --label-set forensic \
    --epochs 50 \
    --batch-size 2 \
    --learning-rate 5e-5
```

All CLI arguments and their defaults:

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `custom` | Dataset format (currently `custom` JSONL) |
| `--data-dir` | `./data` | Directory containing `.jsonl` training files |
| `--output-dir` | `./models/out` | Directory for model checkpoints and metadata |
| `--label-set` | `default` | Built-in label set name or path to JSON file |
| `--base-model` | `microsoft/layoutlmv3-base` | HuggingFace model to fine-tune (or set `LAYOUTLM_FINETUNE_MODEL` env var) |
| `--use-lora` | `false` | Enable LoRA adapters for parameter-efficient training |
| `--lora-rank` | `16` | LoRA rank (r parameter) |
| `--lora-alpha` | `32` | LoRA alpha scaling factor |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `2` | Per-device training batch size |
| `--learning-rate` | `5e-5` | Peak learning rate |
| `--test-size` | `0.2` | Fraction of data for evaluation split |
| `--seed` | `42` | Random seed for reproducibility |

### LoRA Adapter Usage

LoRA (Low-Rank Adaptation) trains small adapter matrices that are merged with the pre-trained model weights. Benefits:

- 10-100x fewer trainable parameters
- 30-50% less VRAM usage
- Faster training (fewer gradients to compute)
- Multiple adapters can be swapped at inference time

To enable LoRA:

```bash
python layoutlm_finetune.py \
    --data-dir ./data \
    --output-dir ./models/forensic-lora \
    --label-set forensic \
    --use-lora \
    --lora-rank 16 \
    --lora-alpha 32 \
    --epochs 30
```

The LoRA adapter targets the `query` and `value` attention projections with a dropout of 0.1. These defaults are effective for most document understanding tasks. Adjust `--lora-rank` for the trade-off:

| LoRA Rank | Trainable Params | Quality | Speed |
|---|---|---|---|
| 4 | ~100K | Good for simple label sets | Fastest |
| 8 | ~200K | Good general-purpose | Fast |
| 16 (default) | ~400K | Best for complex label sets | Moderate |
| 32 | ~800K | Highest capacity | Slower |

### Recommended Hyperparameters

For forensic document types with medium-sized datasets (500-2000 annotated pages):

```bash
python layoutlm_finetune.py \
    --data-dir ./data/forensic \
    --output-dir ./models/forensic-v1 \
    --label-set forensic \
    --use-lora \
    --lora-rank 16 \
    --lora-alpha 32 \
    --epochs 30 \
    --batch-size 2 \
    --learning-rate 3e-5 \
    --seed 42
```

Guidelines by dataset size:

| Annotated Pages | Epochs | Learning Rate | Batch Size | LoRA? |
|---|---|---|---|---|
| < 100 | 50-100 | 2e-5 | 1-2 | Yes (rank 8) |
| 100-500 | 30-50 | 3e-5 | 2 | Yes (rank 16) |
| 500-2000 | 20-30 | 5e-5 | 2-4 | Either |
| > 2000 | 10-20 | 5e-5 | 4-8 | Full fine-tuning |

### Checkpoint Strategy

The trainer saves a checkpoint at every epoch and loads the best model (by F1 score) at the end. Output structure:

```
models/forensic-v1/
  config.json              # Model configuration
  model.safetensors        # Model weights (or adapter_model.safetensors for LoRA)
  tokenizer.json           # Tokenizer configuration
  label_set.json           # Saved label set (entity types, BIO labels, mappings)
  training_metadata.json   # Training config, timing, and final metrics
  logs/                    # TensorBoard training logs
  checkpoint-*/            # Per-epoch checkpoints (can be pruned)
```

The `training_metadata.json` file records all training parameters, timing, and evaluation metrics for reproducibility and audit purposes.

---

## 5. Evaluation

### Running Evaluation

Evaluation is done programmatically via `layoutlm_evaluate.py`. There are two modes:

**Model-based evaluation** (runs inference on test data):

```python
from layoutlm_evaluate import evaluate_model
from layoutlm_labels import load_label_set
from layoutlm_data import load_custom_jsonl

ls = load_label_set("forensic")
test_pages = load_custom_jsonl("data/test.jsonl", ls)

results = evaluate_model(
    model_path="./models/forensic-v1",
    test_data=test_pages,
    label_set=ls,
    confidence_threshold=0.5,
    output_path="benchmark_results/eval_forensic.json")
```

**Offline evaluation** (from pre-computed predictions):

```python
from layoutlm_evaluate import evaluate_predictions
from layoutlm_labels import load_label_set

ls = load_label_set("forensic")

results = evaluate_predictions(
    true_labels=[["O", "B-CASE_NUMBER", "I-CASE_NUMBER", "O"]],
    pred_labels=[["O", "B-CASE_NUMBER", "I-CASE_NUMBER", "O"]],
    label_set=ls,
    output_path="benchmark_results/eval_offline.json")
```

### Understanding Seqeval Metrics

The evaluation uses `seqeval`, which computes **entity-level** (not token-level) metrics:

- **Precision**: Of all entities the model predicted, what fraction were correct?
- **Recall**: Of all entities in the ground truth, what fraction did the model find?
- **F1**: Harmonic mean of precision and recall.

An entity prediction is correct only when both the entity type **and** the exact span boundaries match the ground truth. This is stricter than token-level accuracy.

The JSON report includes:

- `overall`: Micro-averaged precision, recall, and F1 across all entity types.
- `per_entity`: Breakdown by entity type (CASE_NUMBER, DATE, etc.) with per-type precision, recall, F1, and support count.
- `averages`: Micro, macro, and weighted averages from seqeval's classification report.

Target metrics for production use:

| Use Case | Minimum F1 | Recommended F1 |
|---|---|---|
| General NER | 0.75 | 0.85+ |
| Forensic entities | 0.80 | 0.90+ |
| Receipt extraction | 0.85 | 0.92+ |
| Form fields | 0.80 | 0.88+ |

### Confidence Calibration

Raw softmax confidence scores from LayoutLMv3 tend to be **over-confident**. The `layoutlm_calibration.py` module provides post-hoc calibration to produce well-calibrated probabilities.

**Methods available:**

| Method | Env Var Value | Requirements | Best For |
|---|---|---|---|
| None (identity) | `none` | None | Baseline |
| Temperature scaling | `temperature` | None (pure Python) | Quick calibration, few parameters |
| Platt scaling | `platt` | None (pure Python) | Binary-style calibration |
| Isotonic regression | `isotonic` | scikit-learn, numpy | Best calibration quality |

**Usage:**

```python
from layoutlm_calibration import (
    CalibrationConfig,
    CalibrationMethod,
    ConfidenceCalibrator,
    calibrate_entity_confidence,
    compute_ece)

# Configure calibrator
config = CalibrationConfig(method=CalibrationMethod.TEMPERATURE_SCALING)
calibrator = ConfidenceCalibrator(config)

# Fit on validation data
calibrator.fit(
    validation_predictions=[
        {"confidence": 0.95, "logit": 2.5, "label": "B-DATE"},
        {"confidence": 0.80, "logit": 1.2, "label": "O"},
    ],
    validation_labels=["B-DATE", "B-AMOUNT"],  # ground truth
)

# Apply to extracted entities
calibrated = calibrate_entity_confidence(raw_entities, calibrator)

# Save/load calibration parameters
calibrator.save("calibration/forensic_temp.json")
calibrator.load("calibration/forensic_temp.json")

# Measure calibration quality (Expected Calibration Error)
ece = compute_ece(predictions=[0.9, 0.8, 0.7], labels=[1, 1, 0])
```

**Environment-driven defaults:**

```bash
export LAYOUTLM_CALIBRATION_METHOD=temperature
export LAYOUTLM_CALIBRATION_PATH=./calibration/forensic_temp.json
```

The `compute_ece` function returns the Expected Calibration Error (ECE). A perfectly calibrated model has ECE = 0.0. Values below 0.05 indicate good calibration.

---

## 6. Model Registry

### How It Works

The `layoutlm_model_registry.py` module provides a file-based registry for tracking trained model checkpoints. Models are stored in a `manifest.json` file inside the registry directory.

**Default registry location:** `./models/registry/` (configurable via `LAYOUTLM_REGISTRY_DIR` env var).

### Registering a New Model

After training and evaluation:

```python
from layoutlm_model_registry import ModelRegistry

registry = ModelRegistry  # or ModelRegistry("./models/my-registry")

entry = registry.register(
    model_dir="./models/forensic-v1",
    name="forensic-ner",
    version="1.0.0",
    label_set_name="forensic",
    metrics={"f1": 0.92, "precision": 0.91, "recall": 0.93},
    description="Forensic NER model trained on 1500 annotated pages")
```

The registry enforces uniqueness on the `(name, version)` pair. Attempting to register a duplicate raises `ValueError`.

### Retrieving Models

```python
# Get latest version of a model by name
latest = registry.get_model("forensic-ner")

# Get a specific version
v1 = registry.get_model("forensic-ner", version="1.0.0")

# List all registered models
all_models = registry.list_models
for model in all_models:
    print(f"{model.name} v{model.version} -- F1={model.metrics.get('f1', 'N/A')}")
```

### Active Model Selection

Set the active model via environment variable:

```bash
# Specific version
export LAYOUTLM_ACTIVE_MODEL=forensic-ner:1.0.0

# Latest version of a name
export LAYOUTLM_ACTIVE_MODEL=forensic-ner
```

Retrieve in code:

```python
active = registry.get_active_model
if active:
    print(f"Active model: {active.name} v{active.version} at {active.model_path}")
```

### Production Deployment Workflow

1. Train a model and evaluate it (Sections 4 and 5).
2. If F1 meets the target threshold, register it with an incremented version.
3. Update `LAYOUTLM_ACTIVE_MODEL` to point to the new version.
4. The pipeline loads the active model at startup.
5. Keep previous versions registered for rollback.

Current live-inference promotion resolves the registered model checkpoint path.
If a registry entry also carries `adapter_path`, that metadata is retained for
observability, but adapter composition is not performed automatically in this
pass.

---

## 7. Integration with OCR Pipeline

### Pipeline Architecture

LayoutLMv3 operates as an **optional enhancement layer** on top of the base OCR pipeline. The base pipeline (`ocr_gpu_async.py`) produces OCR text with bounding boxes via PaddleOCR. LayoutLMv3 then consumes that text + layout data to produce entity-level annotations.

The integration flow:

```
PaddleOCR -> (text, bbox, confidence) per word
    |
    v
LayoutLMv3 Token Classification -> BIO labels per word
    |
    v
Entity Assembly -> structured entities with types and spans
    |
    v
Calibration -> well-calibrated confidence scores
    |
    v
Output -> .entities.json sidecar files
```

### The AB Test Harness

The `scripts/ab_test_layout_engines.py` script provides a comparison framework for evaluating LayoutLMv3 against PP-StructureV3 on the same documents:

```bash
python scripts/ab_test_layout_engines.py \
    --input-dir ./test_documents \
    --output-dir ./ab_results \
    --engines both
```

Engine options:
- `layoutlm` -- LayoutLMv3 region detection + semantic extraction
- `ppstructure` -- PP-StructureV3 layout analysis
- `both` -- Run both and compare

### When to Use LayoutLMv3 vs Base OCR

| Scenario | Recommendation |
|---|---|
| Simple text extraction | Base OCR only (PaddleOCR + Tesseract fallback) |
| Named entity extraction on known doc types | LayoutLMv3 with domain-specific fine-tuning |
| Table structure detection | PP-StructureV3 (better for tables) |
| Form field extraction | LayoutLMv3 with `form` label set |
| Mixed document batches | Base OCR with optional LayoutLMv3 classification |
| Air-gapped deployment | Base OCR (LayoutLMv3 models need separate packaging) |

### Extractive Summarization

The `layoutlm_summarization.py` module provides CTC-safe document summarization using extracted text. It selects existing sentences rather than generating new text. Four scoring methods are available:

- **TextRank**: Graph-based sentence ranking using word overlap similarity.
- **Entity Density**: Scores sentences by count of recognized entities.
- **Layout Position**: Scores by document position (titles, headers, first/last paragraphs).
- **Combined** (default): Weighted combination of all three methods.

```python
from layoutlm_summarization import summarize_document, SummarizationConfig

config = SummarizationConfig(max_sentences=5)
summary = summarize_document(
    pages_text=["Page 1 text...", "Page 2 text..."],
    entities=[{"text": "John Smith", "type": "PERSON_NAME"}],
    config=config,
    document_id="doc_001")

for sent in summary.sentences:
    print(f"[Page {sent.page_num}, score={sent.score}] {sent.text}")
```

Environment variables for summarization:

```bash
export LAYOUTLM_SUMMARIZATION_METHOD=combined    # textrank, entity_density, layout_position, combined
export LAYOUTLM_SUMMARY_MAX_SENTENCES=5
```

---

## 8. Example End-to-End Workflow

This section walks through the complete lifecycle: preparing data, fine-tuning a model, evaluating it, registering it, and deploying it.

### Step 1: Prepare Training Data

Create a directory with annotated JSONL files:

```bash
mkdir -p data/forensic
```

Create `data/forensic/train.jsonl` with annotated pages. Each line is one page:

```json
{"doc_id":"case_001","page_num":1,"words":[{"text":"Case","bbox":[10,20,60,40],"label":"O"},{"text":"No.","bbox":[65,20,90,40],"label":"O"},{"text":"2026-CV-12345","bbox":[95,20,230,40],"label":"B-CASE_NUMBER"},{"text":"Filed:","bbox":[10,50,55,70],"label":"O"},{"text":"2026-01-15","bbox":[60,50,170,70],"label":"B-FILING_DATE"},{"text":"Presiding","bbox":[10,80,85,100],"label":"O"},{"text":"Judge:","bbox":[90,80,140,100],"label":"O"},{"text":"Hon.","bbox":[145,80,175,100],"label":"B-JUDGE_NAME"},{"text":"Sarah","bbox":[180,80,220,100],"label":"I-JUDGE_NAME"},{"text":"Chen","bbox":[225,80,265,100],"label":"I-JUDGE_NAME"}]}
```

Minimum recommended data volumes:

| Quality Target | Annotated Pages | Annotated Entities |
|---|---|---|
| Proof of concept | 50-100 | 500+ |
| Baseline model | 200-500 | 2,000+ |
| Production model | 1,000-5,000 | 10,000+ |

### Step 2: Fine-Tune the Model

```bash
python layoutlm_finetune.py \
    --data-dir ./data/forensic \
    --output-dir ./models/forensic-v1 \
    --label-set forensic \
    --use-lora \
    --lora-rank 16 \
    --lora-alpha 32 \
    --epochs 30 \
    --batch-size 2 \
    --learning-rate 3e-5 \
    --test-size 0.2 \
    --seed 42
```

Monitor training progress in the console output. The trainer reports loss and F1 at each epoch. Training automatically selects the best checkpoint by eval F1.

### Step 3: Evaluate the Model

```python
from layoutlm_evaluate import evaluate_model
from layoutlm_labels import load_label_set
from layoutlm_data import load_custom_jsonl

ls = load_label_set("forensic")
test_pages = load_custom_jsonl("data/forensic/test.jsonl", ls)

results = evaluate_model(
    model_path="./models/forensic-v1",
    test_data=test_pages,
    label_set=ls,
    confidence_threshold=0.5,
    output_path="benchmark_results/forensic_v1_eval.json")

print(f"Overall F1: {results['overall']['f1']:.4f}")
for entity_type, metrics in results["per_entity"].items:
    print(f"  {entity_type}: F1={metrics['f1']:.4f} (support={metrics['support']})")
```

### Step 4: Calibrate Confidence Scores

```python
from layoutlm_calibration import (
    CalibrationConfig,
    CalibrationMethod,
    ConfidenceCalibrator,
    compute_ece)

config = CalibrationConfig(method=CalibrationMethod.TEMPERATURE_SCALING)
calibrator = ConfidenceCalibrator(config)

# Fit on validation predictions (from Step 3 evaluation)
calibrator.fit(validation_predictions, validation_labels)

# Check calibration quality
ece_before = compute_ece(raw_confidences, correctness_labels)
ece_after = compute_ece(calibrated_confidences, correctness_labels)
print(f"ECE: {ece_before:.4f} -> {ece_after:.4f}")

# Save for production use
calibrator.save("calibration/forensic_v1_temp.json")
```

### Step 5: Register the Model

```python
from layoutlm_model_registry import ModelRegistry

registry = ModelRegistry("./models/registry")

entry = registry.register(
    model_dir="./models/forensic-v1",
    name="forensic-ner",
    version="1.0.0",
    label_set_name="forensic",
    metrics={
        "f1": results["overall"]["f1"],
        "precision": results["overall"]["precision"],
        "recall": results["overall"]["recall"],
    },
    description="Forensic NER model - 30 epochs LoRA rank 16, trained on 1500 pages")

print(f"Registered: {entry.name} v{entry.version}")
```

### Step 6: Deploy to Production

Set environment variables to activate the model:

```bash
# Model selection
export LAYOUTLM_ACTIVE_MODEL=forensic-ner:1.0.0
export LAYOUTLM_REGISTRY_DIR=./models/registry

# Label set
export LAYOUTLM_LABEL_SET=forensic

# Confidence calibration
export LAYOUTLM_CALIBRATION_METHOD=temperature
export LAYOUTLM_CALIBRATION_PATH=./calibration/forensic_v1_temp.json

# Summarization (optional)
export LAYOUTLM_SUMMARIZATION_METHOD=combined
export LAYOUTLM_SUMMARY_MAX_SENTENCES=5
```

Verify the deployment:

```python
from layoutlm_model_registry import ModelRegistry

registry = ModelRegistry
active = registry.get_active_model
assert active is not None, "No active model configured"
print(f"Active: {active.name} v{active.version} at {active.model_path}")
```

### Step 7: Iterate

When new annotated data becomes available or accuracy targets increase:

1. Prepare new training data (add to existing JSONL files or create new ones).
2. Fine-tune a new version (increment the version string).
3. Evaluate and compare against the previous version.
4. If the new version is better, register it and update `LAYOUTLM_ACTIVE_MODEL`.
5. Keep previous versions for rollback.

---

## Appendix A: Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `LAYOUTLM_FINETUNE_MODEL` | `microsoft/layoutlmv3-base` | Base model checkpoint for fine-tuning |
| `LAYOUTLM_LABEL_SET` | `default` | Built-in label set name |
| `LAYOUTLM_LABEL_CONFIG` | (empty) | Path to custom label set JSON (overrides `LAYOUTLM_LABEL_SET`) |
| `LAYOUTLM_REGISTRY_DIR` | `./models/registry` | Model registry directory |
| `LAYOUTLM_ACTIVE_MODEL` | (empty) | Active model as `name:version` or `name` (latest) |
| `LAYOUTLM_CALIBRATION_METHOD` | `none` | Calibration method: `none`, `temperature`, `platt`, `isotonic` |
| `LAYOUTLM_CALIBRATION_PATH` | (empty) | Path to saved calibration parameters JSON |
| `LAYOUTLM_SUMMARIZATION_METHOD` | `combined` | Summarization method: `textrank`, `entity_density`, `layout_position`, `combined` |
| `LAYOUTLM_SUMMARY_MAX_SENTENCES` | `5` | Maximum sentences in extractive summary |

## Appendix B: Troubleshooting

**`ImportError: torch is required`** -- Install PyTorch with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**`ImportError: peft is required for LoRA`** -- Install peft: `pip install peft`

**`FileNotFoundError: No .jsonl files found`** -- Ensure your `--data-dir` contains `.jsonl` files (not `.json` or subdirectories).

**`ValueError: Unknown label set`** -- Use one of `default`, `forensic`, `receipt`, `form`, or provide a path ending in `.json`.

**Out of memory during training** -- Reduce `--batch-size` to 1, enable `--use-lora`, or reduce `--lora-rank`. For large label sets (17+ entities), LoRA is strongly recommended.

**Low F1 score** -- Check data quality (are bounding boxes accurate?), increase training data volume, try different learning rates (2e-5 to 1e-4), or increase epochs.

**Calibration ECE is high (> 0.10)** -- Try isotonic regression (requires sklearn): set `LAYOUTLM_CALIBRATION_METHOD=isotonic` and fit on a larger validation set (500+ predictions).

## Appendix C: Test Coverage

The LayoutLMv3 modules are covered by 10 test files in `tests/`:

| Test File | Covers |
|---|---|
| `tests/test_layoutlm_labels.py` | Label sets, BIO expansion, custom JSON loading |
| `tests/test_layoutlm_data.py` | JSONL parsing, label validation, dataset creation |
| `tests/test_layoutlm_finetune.py` | CLI argument parsing, config construction, training flow |
| `tests/test_layoutlm_evaluate.py` | Model evaluation, offline evaluation, empty report handling |
| `tests/test_layoutlm_model_registry.py` | Registration, retrieval, versioning, active model |
| `tests/test_layoutlm_calibration.py` | Temperature/Platt/isotonic calibration, ECE, persistence |
| `tests/test_layoutlm_summarization.py` | TextRank, entity density, layout scoring, file-based API |
| `tests/test_layoutlm_structure.py` | LayoutLMv3 structure integration |
| `tests/test_layoutlm_worker.py` | Celery worker integration |
| `tests/test_layoutlm_dockerfile.py` | Docker image build configuration |

Run the full LayoutLMv3 test suite:

```bash
python -m pytest tests/test_layoutlm_*.py -v
```
