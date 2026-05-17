"""
genomic_analyzer.py
====================
Orchestrates the end-to-end genomic analysis pipeline:

  VCF file  →  VCFParser  →  filter/prioritize  →  GemmaClient  →  results

Usage
-----
>>> from src.genomic_analyzer import GenomicAnalyzer
>>> analyzer = GenomicAnalyzer(api_key="YOUR_KEY")
>>> results = analyzer.analyze("sample.vcf", patient_phenotypes="muscle weakness, elevated CK")
>>> print(results.summary)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .gemma_client import GemmaClient
from .variant_parser import VCFParser, Variant


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class VariantResult:
    """Holds a parsed Variant alongside its AI-generated interpretation."""
    variant: Variant
    interpretation: str = ""
    priority_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = self.variant.to_dict()
        d["interpretation"] = self.interpretation
        d["priority_score"] = self.priority_score
        return d


@dataclass
class AnalysisReport:
    """Top-level result of a complete VCF analysis run."""
    total_variants: int = 0
    filtered_variants: int = 0
    variant_results: list[VariantResult] = field(default_factory=list)
    cohort_summary: str = ""
    pathway_analysis: str = ""
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def high_impact_count(self) -> int:
        return sum(
            1 for vr in self.variant_results
            if vr.variant.impact.upper() == "HIGH"
        )

    @property
    def pathogenic_count(self) -> int:
        return sum(
            1 for vr in self.variant_results
            if "pathogenic" in vr.variant.clinvar_sig.lower()
        )

    @property
    def vus_count(self) -> int:
        return sum(
            1 for vr in self.variant_results
            if "uncertain" in vr.variant.clinvar_sig.lower()
        )

    @property
    def rare_count(self) -> int:
        return sum(
            1 for vr in self.variant_results
            if vr.variant.af_gnomad is None or vr.variant.af_gnomad < 0.01
        )

    @property
    def top_genes(self) -> list[str]:
        """Unique genes from the top-priority variants (sorted by score)."""
        seen: set[str] = set()
        genes: list[str] = []
        for vr in sorted(self.variant_results, key=lambda x: x.priority_score, reverse=True):
            g = vr.variant.gene
            if g and g not in seen:
                seen.add(g)
                genes.append(g)
        return genes

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {
                "total_variants": self.total_variants,
                "filtered_variants": self.filtered_variants,
                "high_impact_count": self.high_impact_count,
                "pathogenic_count": self.pathogenic_count,
                "vus_count": self.vus_count,
                "rare_count": self.rare_count,
                "top_genes": self.top_genes,
                "cohort_summary": self.cohort_summary,
                "pathway_analysis": self.pathway_analysis,
                "variants": [vr.to_dict() for vr in self.variant_results],
            },
            indent=indent,
        )


# ── Priority scoring ──────────────────────────────────────────────────────────

_IMPACT_SCORES = {"HIGH": 40, "MODERATE": 20, "LOW": 5, "MODIFIER": 1}
_CLINVAR_SCORES = {
    "pathogenic": 50,
    "likely pathogenic": 35,
    "uncertain significance": 10,
    "likely benign": -10,
    "benign": -20,
}
_CONSEQUENCE_SCORES = {
    "stop_gained": 40,
    "frameshift_variant": 40,
    "splice_acceptor_variant": 35,
    "splice_donor_variant": 35,
    "start_lost": 35,
    "stop_lost": 30,
    "missense_variant": 20,
    "inframe_insertion": 15,
    "inframe_deletion": 15,
    "synonymous_variant": 2,
    "intron_variant": 1,
    "intergenic_variant": 0,
}


def _score_variant(variant: Variant) -> float:
    """Compute a clinical priority score (higher = more likely pathogenic)."""
    score = 0.0

    # Impact
    score += _IMPACT_SCORES.get(variant.impact.upper(), 0)

    # Consequence
    for csq_key, csq_score in _CONSEQUENCE_SCORES.items():
        if csq_key in variant.consequence.lower():
            score += csq_score
            break

    # ClinVar
    for cv_key, cv_score in _CLINVAR_SCORES.items():
        if cv_key in variant.clinvar_sig.lower():
            score += cv_score
            break

    # Allele frequency (rare = more interesting)
    if variant.af_gnomad is None:
        score += 15  # absent from gnomAD → potentially very rare
    elif variant.af_gnomad < 0.0001:
        score += 20
    elif variant.af_gnomad < 0.001:
        score += 10
    elif variant.af_gnomad < 0.01:
        score += 5
    else:
        score -= 5  # common variant

    # Zygosity
    gt = variant.genotype
    if gt in ("1/1", "1|1"):
        score += 10  # homozygous alt
    elif gt in ("0/1", "0|1", "1|0"):
        score += 3   # heterozygous

    return score


# ── Analyzer ──────────────────────────────────────────────────────────────────

class GenomicAnalyzer:
    """
    Main analysis pipeline that ties together VCF parsing and Gemma 4
    interpretation.

    Parameters
    ----------
    api_key : str, optional
        Google AI API key. Falls back to the GOOGLE_API_KEY environment
        variable.
    model_name : str
        Gemma 4 model name (default: ``"gemma-4-9b-it"``).
    max_variants_to_interpret : int
        Maximum number of variants to individually interpret via the AI.
        This caps API usage. Variants are ranked by priority score first.
    min_priority_score : float
        Only variants above this score are sent for individual AI interpretation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = GemmaClient.DEFAULT_MODEL,
        max_variants_to_interpret: int = 10,
        min_priority_score: float = 20.0,
        client: GemmaClient | None = None,
    ) -> None:
        # Reuse an existing client when provided to avoid reloading large local models.
        self.client = client or GemmaClient(api_key=api_key, model_name=model_name)
        self.max_variants_to_interpret = max_variants_to_interpret
        self.min_priority_score = min_priority_score

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        vcf_path: str | Path,
        patient_phenotypes: str = "",
        apply_pass_filter: bool = True,
    ) -> AnalysisReport:
        """
        Run the full analysis pipeline on a VCF file.

        Parameters
        ----------
        vcf_path : str | Path
            Path to the VCF file (plain or gzip-compressed).
        patient_phenotypes : str
            Free-text description of patient's clinical features / HPO terms.
        apply_pass_filter : bool
            When True, skip variants that did not PASS quality filters.

        Returns
        -------
        AnalysisReport
            Structured result including individual interpretations and a
            cohort-level summary.
        """
        vcf_path = Path(vcf_path)
        print(f"[GenomicAnalyzer] Parsing {vcf_path.name} …")

        parser = VCFParser(vcf_path)
        all_variants = parser.parse()
        total = len(all_variants)
        print(f"[GenomicAnalyzer] Loaded {total} variants.")

        # Filter
        if apply_pass_filter:
            variants = [v for v in all_variants if v.is_pass]
        else:
            variants = all_variants

        # Score and rank
        scored: list[tuple[float, Variant]] = []
        for v in variants:
            scored.append((_score_variant(v), v))
        scored.sort(key=lambda x: x[0], reverse=True)

        print(f"[GenomicAnalyzer] {len(scored)} variants after filtering, ranked by priority.")

        # Interpret top variants
        report = AnalysisReport(
            total_variants=total,
            filtered_variants=len(scored),
        )

        interpret_candidates = [
            (score, v) for score, v in scored
            if score >= self.min_priority_score
        ][: self.max_variants_to_interpret]

        print(
            f"[GenomicAnalyzer] Sending {len(interpret_candidates)} variants to Gemma 4 …"
        )
        for score, variant in interpret_candidates:
            interp = self.client.interpret_variant(
                chrom=variant.chrom,
                pos=variant.pos,
                ref=variant.ref,
                alt=", ".join(variant.alt),
                gene=variant.gene,
                consequence=variant.consequence,
                impact=variant.impact,
                hgvs_c=variant.hgvs_c,
                hgvs_p=variant.hgvs_p,
                clinvar_sig=variant.clinvar_sig,
                af_gnomad=variant.af_gnomad,
                genotype=variant.genotype,
                context=f"Patient phenotypes: {patient_phenotypes}" if patient_phenotypes else "",
            )
            report.variant_results.append(
                VariantResult(variant=variant, interpretation=interp, priority_score=score)
            )

        # Remaining scored variants without individual AI interpretation
        interpreted_set = {id(vr.variant) for vr in report.variant_results}
        for score, variant in scored:
            if id(variant) not in interpreted_set:
                report.variant_results.append(
                    VariantResult(variant=variant, priority_score=score)
                )

        # Cohort-level summary
        top_variants_text = self._format_top_variants(report.variant_results[:5])
        report.cohort_summary = self.client.summarize_cohort(
            total_variants=total,
            high_impact=report.high_impact_count,
            moderate_impact=sum(
                1 for vr in report.variant_results
                if vr.variant.impact.upper() == "MODERATE"
            ),
            low_impact=sum(
                1 for vr in report.variant_results
                if vr.variant.impact.upper() == "LOW"
            ),
            pathogenic_count=report.pathogenic_count,
            vus_count=report.vus_count,
            rare_count=report.rare_count,
            top_variants=top_variants_text,
            clinical_question=(
                f"Patient presents with: {patient_phenotypes}. "
                "What is the most likely genetic diagnosis?"
                if patient_phenotypes
                else "What is the most likely genetic diagnosis?"
            ),
        )

        # Pathway analysis for top genes
        top_genes = report.top_genes[:8]
        if top_genes:
            report.pathway_analysis = self.client.analyze_gene_pathways(
                gene_list=top_genes,
                phenotypes=patient_phenotypes,
            )

        # Build summary DataFrame
        report.dataframe = pd.DataFrame(
            [vr.to_dict() for vr in report.variant_results]
        )
        if not report.dataframe.empty:
            report.dataframe["priority_score"] = [
                vr.priority_score for vr in report.variant_results
            ]
            report.dataframe.sort_values("priority_score", ascending=False, inplace=True)
            report.dataframe.reset_index(drop=True, inplace=True)

        print("[GenomicAnalyzer] Analysis complete.")
        return report

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_top_variants(variant_results: list[VariantResult]) -> str:
        lines = []
        for i, vr in enumerate(variant_results, 1):
            v = vr.variant
            lines.append(
                f"{i}. {v.chrom}:{v.pos} {v.ref}>{'/'.join(v.alt)} | "
                f"Gene: {v.gene or 'Unknown'} | "
                f"Consequence: {v.consequence or 'N/A'} | "
                f"Impact: {v.impact or 'N/A'} | "
                f"ClinVar: {v.clinvar_sig or 'Not classified'} | "
                f"AF: {v.af_gnomad if v.af_gnomad is not None else 'N/A'} | "
                f"Score: {vr.priority_score:.1f}"
            )
        return "\n".join(lines) if lines else "No high-priority variants identified."
