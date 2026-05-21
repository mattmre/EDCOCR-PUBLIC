# 11: ML Training and Model Customization

## Overview

EDCOCR supports LayoutLMv3 fine-tuning for domain-specific document understanding. All ML functionality is CTC-safe and uses token classification rather than text generation.

> [!IMPORTANT]
> Heavy ML imports are lazy so the module can be imported without GPU dependencies.

---

## LayoutLMv3 Architecture

```mermaid
flowchart LR
    A[Document Image] --> B[OCR Text + Bounding Boxes]
    A --> C[Visual Features]
    B --> D[LayoutLMv3 Token Classification]
    C --> D
    D --> E[BIO Label Predictions]
    E --> F[Entity Extraction]
```

## Label Sets

| Label Set | Entities | Use Case |
|---|---|---|
| `default` | INVOICE_NUMBER, DATE, AMOUNT, PERSON_NAME, ORGANIZATION, ADDRESS | General business documents |
| `forensic` | Default + CASE_NUMBER, BATES_NUMBER, EXHIBIT_NUMBER, COURT_NAME | Legal and forensic documents |
| `receipt` | STORE_NAME, DATE, TOTAL, PAYMENT_METHOD | Receipts and invoices |
| `form` | FIELD_LABEL, FIELD_VALUE, CHECKBOX, SIGNATURE_FIELD | Structured forms |

## Training Data Format

Training data uses JSONL format with `text`, `boxes`, `labels`, and `image_path`.

## Fine-Tuning Workflow

```mermaid
flowchart TD
    A[Prepare Data] --> B[Configure Label Set]
    B --> C[Train]
    C --> D[Evaluate]
    D --> E{F1 >= target?}
    E -->|Yes| F[Register Model]
    E -->|No| G[Adjust Hyperparams]
    G --> C
```

## Evaluation and Registry

- Evaluate with entity-level precision, recall, and F1.
- Register the model with a versioned path when performance is acceptable.

## Confidence Calibration

Supported approaches include temperature scaling, Platt scaling, and isotonic calibration.
