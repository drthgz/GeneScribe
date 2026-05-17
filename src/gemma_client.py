"""
gemma_client.py
===============
Client wrapper for Google's Gemma 4 model.
Supports both:
  - Local inference via HuggingFace transformers (Kaggle Inputs)
  - API-based inference via Google Generative AI SDK
Provides clinical-genomics-specific prompt templates and structured output
parsing for variant interpretation tasks.
"""

from __future__ import annotations

import os
import textwrap
import time
from pathlib import Path
from typing import Any

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_CONTEXT = """You are a board-certified clinical geneticist and bioinformatics
expert. You have deep knowledge of human genomics, population genetics, molecular
biology, and rare disease genetics. When interpreting genomic variants, you follow
ACMG/AMP variant classification guidelines (2015 & 2023 updates).

Always be precise, evidence-based, and communicate findings in a way that is
useful for both clinical professionals and informed patients. When uncertain,
clearly state the limitations of the interpretation."""

_VARIANT_INTERPRETATION_TEMPLATE = """\
Interpret the following genomic variant for a clinical report.

## Variant Details
- Chromosome/Position: {chrom}:{pos}
- Reference Allele: {ref}
- Alternate Allele(s): {alt}
- Gene: {gene}
- Molecular Consequence: {consequence}
- Predicted Impact: {impact}
- HGVS (coding): {hgvs_c}
- HGVS (protein): {hgvs_p}
- ClinVar Classification: {clinvar_sig}
- gnomAD Allele Frequency: {af_gnomad}
- Patient Genotype: {genotype}

## Additional Context
{context}

## Your Task
Provide a structured clinical interpretation with the following sections:

### 1. Variant Summary
Briefly describe the variant in plain language.

### 2. Gene Function & Disease Relevance
What is the known function of this gene? What diseases are associated with variants in this gene?

### 3. Clinical Significance Assessment
Based on available evidence (ClinVar classification, allele frequency, consequence type),
what is the likely clinical significance of this variant?
Use ACMG categories: Pathogenic / Likely Pathogenic / Variant of Uncertain Significance (VUS) / Likely Benign / Benign.

### 4. Inheritance Pattern
What is the expected inheritance pattern (autosomal dominant, autosomal recessive, X-linked, etc.)?

### 5. Recommendations
What follow-up actions would you recommend (additional testing, specialist referral, etc.)?

### 6. Patient-Friendly Explanation
In 2-3 sentences suitable for a patient without a medical background, explain what this finding means.
"""

_COHORT_SUMMARY_TEMPLATE = """\
You are reviewing a genomic analysis report for a patient with suspected genetic disease.

## Variant Cohort Summary
Total variants identified: {total_variants}
HIGH impact variants: {high_impact}
MODERATE impact variants: {moderate_impact}
LOW impact variants: {low_impact}
Pathogenic or likely pathogenic (ClinVar): {pathogenic_count}
Variants of Uncertain Significance: {vus_count}
Rare variants (AF < 0.01): {rare_count}

## Top Candidate Variants
{top_variants}

## Clinical Question
{clinical_question}

## Your Task
Provide a concise clinical summary and prioritized interpretation of the variant cohort.
Highlight which variants are most likely to be disease-causing and why.
Suggest a differential diagnosis if possible based on the gene set.
"""

_GENE_PATHWAY_TEMPLATE = """\
Given the following set of genes with potentially disease-causing variants:

Genes: {gene_list}

Identified diseases/phenotypes in the patient's clinical record: {phenotypes}

Tasks:
1. Identify any shared biological pathways or processes among these genes.
2. Suggest which gene(s) are most likely to explain the patient's phenotype.
3. Are there known gene-gene interactions (digenic inheritance, modifier genes)?
4. What additional genomic analyses would help clarify the diagnosis?
"""


# ── Client ────────────────────────────────────────────────────────────────────

class GemmaClient:
    """
    Wrapper for Gemma 4 model with dual inference modes.

    Modes (in priority order):
    1. Local inference: Load model from Kaggle Inputs or GENESCRIBE_MODEL_PATH
    2. API inference: Use Google Generative AI SDK (requires GOOGLE_API_KEY)
    3. Mock: Return placeholder responses when no inference backend available

    Parameters
    ----------
    api_key : str, optional
        Google AI API key. Falls back to the ``GOOGLE_API_KEY`` environment
        variable.
    model_name : str
        Gemma 4 model identifier (for API mode). Defaults to ``"gemma-4-9b-it"``.
    temperature : float
        Sampling temperature (0 = deterministic, 1 = creative).
    max_retries : int
        Number of times to retry on API errors.
    use_local : bool
        Force use of local model if available. Default True.
    local_model_path : str, optional
        Path to local Gemma 4 model. Auto-discovers from Kaggle Inputs if not set.
    """

    DEFAULT_MODEL = "gemma-4-9b-it"
    KAGGLE_INPUT_PATHS = [
        Path("/kaggle/input/gemma-4"),
        Path("/kaggle/input/gemma_4"),
        Path("/kaggle/input/google-gemma-4"),
    ]

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        max_retries: int = 3,
        use_local: bool = True,
        local_model_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self._model: Any = None
        self._tokenizer: Any = None
        self._mode: str = "mock"

        # ── Try local model first ─────────────────────────────────────────────
        if use_local and _TRANSFORMERS_AVAILABLE:
            local_path = self._find_local_model(local_model_path)
            if local_path:
                try:
                    self._load_local_model(local_path)
                    self._mode = "local"
                    print(f"[GemmaClient] Local model loaded from {local_path}")
                    return
                except Exception as e:
                    print(f"[GemmaClient] Failed to load local model: {e}. Falling back to API...")

        # ── Try API mode ──────────────────────────────────────────────────────
        if not _GENAI_AVAILABLE:
            print(
                "[GemmaClient] Warning: 'google-generativeai' not installed. "
                "Set GENESCRIBE_FORCE_LOCAL=1 and attach model in Kaggle Inputs, "
                "or calls will use mock responses."
            )
            return

        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            print(
                "[GemmaClient] Warning: No API key found. "
                "Set GOOGLE_API_KEY or attach Gemma 4 in Kaggle Inputs. "
                "Calls will return mock responses."
            )
            return

        try:
            genai.configure(api_key=key)
            system_instruction = textwrap.dedent(_SYSTEM_CONTEXT).strip()
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=2048,
                ),
            )
            self._mode = "api"
            print(f"[GemmaClient] API mode initialized (model: {self.model_name})")
        except Exception as e:
            print(f"[GemmaClient] Failed to initialize API client: {e}. Falling back to mock mode.")
            self._mode = "mock"

    @staticmethod
    def _find_local_model(custom_path: str | None = None) -> Path | None:
        """Discover local Gemma 4 model in Kaggle Inputs or custom path."""
        if custom_path:
            p = Path(custom_path)
            if p.exists() and (p / "config.json").exists():
                return p
        for kaggle_path in GemmaClient.KAGGLE_INPUT_PATHS:
            if kaggle_path.exists():
                config_file = kaggle_path / "config.json"
                if config_file.exists():
                    return kaggle_path
        env_path = os.environ.get("GENESCRIBE_MODEL_PATH", "").strip()
        if env_path:
            p = Path(env_path)
            if p.exists() and (p / "config.json").exists():
                return p
        return None

    def _load_local_model(self, model_path: Path) -> None:
        """Load Gemma 4 locally using transformers."""
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library not installed. Run: pip install transformers torch")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[GemmaClient] Loading model on {device}...")
        
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        print(f"[GemmaClient] Model loaded successfully on {device}")

    # ── Public API ────────────────────────────────────────────────────────────

    def interpret_variant(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        gene: str = "",
        consequence: str = "",
        impact: str = "",
        hgvs_c: str = "",
        hgvs_p: str = "",
        clinvar_sig: str = "",
        af_gnomad: float | None = None,
        genotype: str = "unknown",
        context: str = "",
    ) -> str:
        """
        Generate a clinical interpretation for a single genomic variant.

        Returns a formatted markdown string with the interpretation.
        """
        prompt = _VARIANT_INTERPRETATION_TEMPLATE.format(
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt=alt,
            gene=gene or "Unknown",
            consequence=consequence or "Not specified",
            impact=impact or "Not specified",
            hgvs_c=hgvs_c or "Not available",
            hgvs_p=hgvs_p or "Not available",
            clinvar_sig=clinvar_sig or "Not classified",
            af_gnomad=f"{af_gnomad:.6f}" if af_gnomad is not None else "Not available",
            genotype=genotype,
            context=context or "No additional clinical context provided.",
        )
        return self._generate(prompt)

    def summarize_cohort(
        self,
        total_variants: int,
        high_impact: int,
        moderate_impact: int,
        low_impact: int,
        pathogenic_count: int,
        vus_count: int,
        rare_count: int,
        top_variants: str,
        clinical_question: str = "What is the likely genetic diagnosis?",
    ) -> str:
        """Generate a summary interpretation for a cohort of variants."""
        prompt = _COHORT_SUMMARY_TEMPLATE.format(
            total_variants=total_variants,
            high_impact=high_impact,
            moderate_impact=moderate_impact,
            low_impact=low_impact,
            pathogenic_count=pathogenic_count,
            vus_count=vus_count,
            rare_count=rare_count,
            top_variants=top_variants,
            clinical_question=clinical_question,
        )
        return self._generate(prompt)

    def analyze_gene_pathways(
        self,
        gene_list: list[str],
        phenotypes: str = "",
    ) -> str:
        """Analyze shared pathways and prioritize candidate genes."""
        prompt = _GENE_PATHWAY_TEMPLATE.format(
            gene_list=", ".join(gene_list) if gene_list else "None identified",
            phenotypes=phenotypes or "Not provided",
        )
        return self._generate(prompt)

    def ask(self, question: str) -> str:
        """General-purpose genomics Q&A via Gemma 4."""
        return self._generate(question)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        """Send prompt to Gemma 4 and return the response text."""
        if self._mode == "mock":
            return self._mock_response(prompt)

        if self._mode == "local":
            return self._generate_local(prompt)

        if self._mode == "api":
            return self._generate_api(prompt)

        return self._mock_response(prompt)

    def _generate_local(self, prompt: str) -> str:
        """Generate response using local model."""
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=min(2048, len(inputs["input_ids"][0]) + 1024),
                    temperature=self.temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            response_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response_text
        except Exception as exc:
            print(f"[GemmaClient] Local inference failed: {exc}")
            return self._mock_response(prompt)

    def _generate_api(self, prompt: str) -> str:
        """Generate response using API."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._model.generate_content(prompt)
                return response.text
            except Exception as exc:  # noqa: BLE001
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    print(f"[GemmaClient] Attempt {attempt} failed: {exc}. Retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"[GemmaClient] All {self.max_retries} attempts failed: {exc}"
                    ) from exc
        return ""  # unreachable

    @staticmethod
    def _mock_response(prompt: str) -> str:
        """Return a structured mock response when the API is unavailable."""
        return textwrap.dedent("""\
            ### 1. Variant Summary
            [MOCK] This is a demonstration response generated without an active
            Gemma 4 API connection. Set your GOOGLE_API_KEY to receive real
            AI-generated interpretations.

            ### 2. Gene Function & Disease Relevance
            Gene function information would be provided here by Gemma 4,
            drawing on its knowledge of human genomics and rare disease genetics.

            ### 3. Clinical Significance Assessment
            Variant of Uncertain Significance (VUS) — pending live API analysis.

            ### 4. Inheritance Pattern
            To be determined — requires live API analysis.

            ### 5. Recommendations
            1. Connect to the Gemma 4 API by setting GOOGLE_API_KEY.
            2. Re-run the analysis pipeline.

            ### 6. Patient-Friendly Explanation
            We found a change in your genetic code. Our AI system needs to
            connect to Google's servers to provide a full explanation.
            Please ask your clinician for more details.
        """)
