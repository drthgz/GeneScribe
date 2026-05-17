# GeneScribe - AI-Powered Clinical Genomic Variant Interpreter

Kaggle Gemma 4 Good Hackathon (Health and Sciences track, 2026).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Gemma%204-violet?logo=google)](https://ai.google.dev/gemma)
[![Tests](https://img.shields.io/badge/Tests-62%20passing-success)](tests/)
[![License](https://img.shields.io/badge/License-GPLv3-green)](LICENSE)

## Project Goal

GeneScribe reduces time-to-interpretation for genomic variants by combining:
1. Deterministic variant parsing and ACMG-guided prioritization.
2. LLM-assisted clinical interpretation for top candidates.
3. Clinician-facing report generation (HTML primary for PDF export, JSON for integration).

Pipeline:

`VCF -> ACMG-guided scoring -> Gemma interpretation -> Clinical report`

## Current Status (Checkpoint)

The current repository state is focused on reliable end-to-end demo execution and production-style report output:
1. VCF parsing, prioritization, cohort summary, pathway analysis, and report generation are implemented.
2. HTML report sections now render markdown-like model output correctly in both live and mock mode.
3. Report rendering includes HTML escaping for data-derived fields.
4. Notebook supports cache-based reruns for quota-safe iteration.
5. Test suite passes (62 tests).

## Repository Layout

```text
GeneScribe/
├── src/
│   ├── __init__.py
│   ├── variant_parser.py
│   ├── gemma_client.py              # Dual-mode: local + API
│   ├── genomic_analyzer.py
│   └── report_generator.py
├── data/
│   └── sample_variants.vcf
├── notebooks/
│   ├── genescribe_kaggle_demo.ipynb  # Kaggle notebook with local model auto-detect
│   └── genescribe_report.html        # generated artifact
├── tests/
│   ├── test_variant_parser.py
│   └── test_genomic_analyzer.py
├── requirements.txt                  # Includes optional transformers/torch
├── KAGGLE_LOCAL_MODEL_SETUP.md       # Setup guide for local model mode
├── NEXT_STEPS_TODO.md
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

```bash
export GOOGLE_API_KEY="your-key-here"
```

### 3. Run from Python

```python
from src.genomic_analyzer import GenomicAnalyzer
from src.report_generator import ReportGenerator

analyzer = GenomicAnalyzer(api_key="YOUR_KEY")
report = analyzer.analyze(
    vcf_path="data/sample_variants.vcf",
    patient_phenotypes="Family history of breast cancer, Lynch syndrome suspected",
)

generator = ReportGenerator()
generator.save_html(report, "genescribe_report.html")
generator.save_json(report, "genescribe_report.json")
```

### 4. Run tests

```bash
pytest -q
```

## Model and Hackathon Alignment

The codebase implements **dual-mode Gemma 4 support** to maximize reliability for hackathon submission:

### 1. **Local Model Mode** (Recommended) ⭐
- **How:** Attach Kaggle's [google/gemma-4 model](https://www.kaggle.com/models/google/gemma-4) as a notebook Input
- **Benefits:** No API quota, full reproducibility, uses Kaggle GPU, no secrets management
- **Auto-detection:** GemmaClient automatically detects local model in `/kaggle/input/gemma-4`
- **Loading:** Via HuggingFace `transformers` library

### 2. **API Mode** (Fallback)
- **How:** Set `GOOGLE_API_KEY` in Kaggle Secrets
- **When:** Use if local model unavailable
- **Model:** `gemma-4-9b-it` via Google Generative AI SDK
- **Note:** Subject to API quota limits

### 3. **Mock Mode** (Last Resort)
- **When:** Both local model and API key unavailable
- **Returns:** Placeholder interpretations (for offline testing)

**Priority:** Local > API > Mock  
**Status:** All modes fully integrated and tested.

For detailed setup instructions, see [KAGGLE_LOCAL_MODEL_SETUP.md](KAGGLE_LOCAL_MODEL_SETUP.md).

## Reports

Primary outputs:
1. HTML report for clinical review and PDF export.
2. JSON report for downstream systems (LIMS/EHR/analytics).

The report includes:
1. Summary statistics.
2. Complete variant table.
3. Cohort summary.
4. Top-priority variant interpretation cards.
5. Gene pathway analysis.
6. Trial information and references.

## Notes on SDK Lifecycle

`google-generativeai` currently emits a deprecation warning in runtime logs. Migration to `google.genai` is tracked in [NEXT_STEPS_TODO.md](NEXT_STEPS_TODO.md) and should be completed before post-hackathon hardening.

## Disclaimer

GeneScribe is for research and educational use. It is not a substitute for certified clinical laboratory interpretation. Clinical decisions require review by qualified medical genetics professionals.
