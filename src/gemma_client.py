"""
gemma_client.py
===============
Client wrapper for Google's Gemma 4 model.
Supports both:
  - Local inference via HuggingFace transformers (Kaggle Inputs)
    - API-based inference via Google GenAI SDK (preferred)
        with legacy google-generativeai fallback
Provides clinical-genomics-specific prompt templates and structured output
parsing for variant interpretation tasks.
"""

from __future__ import annotations

import os
import re
import textwrap
import time
import warnings
from pathlib import Path
from typing import Any

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
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
    2. API inference: Use Google GenAI SDK (requires GOOGLE_API_KEY)
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
        Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it/1"),
        Path("/kaggle/input/gemma-4"),
        Path("/kaggle/input/gemma_4"),
        Path("/kaggle/input/google-gemma-4"),
        Path.cwd() / "kaggle" / "input" / "gemma-4",
        Path.cwd() / "kaggle" / "input" / "gemma_4",
        Path.cwd() / "kaggle" / "input" / "google-gemma-4",
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
        self._processor: Any = None
        self._tokenizer: Any = None
        self._api_sdk: Any = None
        self._api_client: Any = None
        self._api_backend: str = "none"
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
        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            print(
                "[GemmaClient] Warning: No API key found. "
                "Set GOOGLE_API_KEY or attach Gemma 4 in Kaggle Inputs. "
                "Calls will return mock responses."
            )
            return

        # API backend priority:
        # 1) google.genai (current)  2) google-generativeai (legacy fallback)
        try:
            from google import genai as genai_new  # type: ignore

            self._api_client = genai_new.Client(api_key=key)
            self._api_backend = "google_genai"
            self._mode = "api"
            print(f"[GemmaClient] API mode initialized via google.genai (model: {self.model_name})")
            return
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            print(f"[GemmaClient] google.genai init failed: {e}. Trying legacy SDK...")

        try:
            # Import legacy SDK only when actually needed.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import google.generativeai as genai_legacy  # type: ignore
            self._api_sdk = genai_legacy
            self._api_backend = "google_generativeai"
        except ImportError:
            print(
                "[GemmaClient] Warning: No supported Google API SDK found. "
                "Install google-genai (preferred) or google-generativeai (legacy), "
                "or continue with local model mode."
            )
            return

        try:
            self._api_sdk.configure(api_key=key)
            system_instruction = textwrap.dedent(_SYSTEM_CONTEXT).strip()
            self._model = self._api_sdk.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction,
                generation_config=self._api_sdk.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=2048,
                ),
            )
            self._mode = "api"
            print(f"[GemmaClient] API mode initialized via legacy SDK (model: {self.model_name})")
        except Exception as e:
            print(f"[GemmaClient] Failed to initialize API client: {e}. Falling back to mock mode.")
            self._mode = "mock"

    @staticmethod
    def _find_local_model(custom_path: str | None = None) -> Path | None:
        """Discover local Gemma 4 model in Kaggle Inputs or custom path."""
        def _resolve_model_dir(base: Path) -> Path | None:
            if (base / "config.json").exists():
                return base
            # Kaggle model downloads are often nested 1-3 levels below the input root.
            for config in base.rglob("config.json"):
                candidate = config.parent
                if (candidate / "generation_config.json").exists() or any(
                    candidate.glob("*.safetensors")
                ):
                    return candidate
            return None

        if custom_path:
            p = Path(custom_path)
            if p.exists():
                found = _resolve_model_dir(p)
                if found:
                    return found
        for kaggle_path in GemmaClient.KAGGLE_INPUT_PATHS:
            if kaggle_path.exists():
                found = _resolve_model_dir(kaggle_path)
                if found:
                    return found
        env_path = os.environ.get("GENESCRIBE_MODEL_PATH", "").strip()
        if env_path:
            p = Path(env_path)
            if p.exists():
                found = _resolve_model_dir(p)
                if found:
                    return found
        return None

    def _load_local_model(self, model_path: Path) -> None:
        """Load Gemma 4 locally using transformers."""
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library not installed. Run: pip install transformers torch")

        has_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if has_cuda else torch.float32
        print(f"[GemmaClient] Loading model on {'cuda' if has_cuda else 'cpu'}...")

        if not has_cuda:
            weight_bytes = sum(f.stat().st_size for f in model_path.glob("*.safetensors"))
            if weight_bytes > 3_000_000_000:
                print(
                    "[GemmaClient] Warning: Large local model on CPU-only runtime. "
                    "Loading may be slow or may run out of memory."
                )

        try:
            self._processor = AutoProcessor.from_pretrained(str(model_path))
            self._tokenizer = self._processor.tokenizer
        except Exception:
            self._processor = None
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map="auto" if has_cuda else None,
            low_cpu_mem_usage=True,
        )
        if not has_cuda:
            self._model = self._model.to("cpu")
        print("[GemmaClient] Model loaded successfully")

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
            if self._processor is not None:
                messages = [
                    {"role": "system", "content": _SYSTEM_CONTEXT},
                    {"role": "user", "content": prompt},
                ]
                text = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)
            else:
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

            input_len = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=self.temperature,
                    top_p=0.95,
                    top_k=64,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            response_text = self._tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

            if self._processor is not None and hasattr(self._processor, "parse_response"):
                try:
                    parsed = self._processor.parse_response(response_text)
                    if isinstance(parsed, str):
                        response_text = parsed
                    elif isinstance(parsed, dict):
                        for key in (
                            "final",
                            "final_answer",
                            "answer",
                            "response",
                            "text",
                            "content",
                        ):
                            value = parsed.get(key)
                            if isinstance(value, str) and value.strip():
                                response_text = value
                                break
                except Exception:
                    # Parsing is best-effort; raw decoded text is a safe fallback.
                    pass

            # Some local runs can still emit chat control tags; strip them for clean reports.
            response_text = re.sub(r"<\/?\w+\|>", "", response_text)
            return response_text.strip()
        except Exception as exc:
            print(f"[GemmaClient] Local inference failed: {exc}")
            return self._mock_response(prompt)

    def _generate_api(self, prompt: str) -> str:
        """Generate response using API."""
        for attempt in range(1, self.max_retries + 1):
            try:
                if self._api_backend == "google_genai":
                    response = self._api_client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={
                            "system_instruction": textwrap.dedent(_SYSTEM_CONTEXT).strip(),
                            "temperature": self.temperature,
                            "max_output_tokens": 2048,
                        },
                    )
                    response_text = getattr(response, "text", None)
                    if isinstance(response_text, str) and response_text.strip():
                        return response_text
                    return str(response)

                # Legacy fallback path.
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
