"""
Tests for src/genomic_analyzer.py (priority scoring & pipeline logic)
and src/report_generator.py (HTML / Markdown rendering).
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.variant_parser import Variant
from src.genomic_analyzer import (
    GenomicAnalyzer,
    AnalysisReport,
    VariantResult,
    _score_variant,
)
from src.report_generator import ReportGenerator, _escape_html


# ── Priority scoring tests ────────────────────────────────────────────────────

def _make_variant(**kwargs) -> Variant:
    """Helper to create a Variant with sensible defaults."""
    defaults = dict(
        chrom="chr1", pos=100, variant_id=".", ref="A", alt=["G"],
        qual=100.0, filter_status="PASS", info={},
    )
    defaults.update(kwargs)
    return Variant(**defaults)


class TestPriorityScoring:
    def test_pathogenic_high_impact_scores_highest(self) -> None:
        v = _make_variant(
            impact="HIGH",
            consequence="stop_gained",
            clinvar_sig="Pathogenic",
            af_gnomad=0.0000010,
            genotype="0/1",
        )
        score = _score_variant(v)
        assert score > 80  # HIGH (40) + stop_gained (40) + pathogenic (50) + rare (20)

    def test_benign_common_scores_low(self) -> None:
        v = _make_variant(
            impact="LOW",
            consequence="synonymous_variant",
            clinvar_sig="Benign",
            af_gnomad=0.45,
            genotype="0/1",
        )
        score = _score_variant(v)
        assert score < 10

    def test_high_impact_scores_more_than_low(self) -> None:
        high = _make_variant(impact="HIGH")
        low = _make_variant(impact="LOW")
        assert _score_variant(high) > _score_variant(low)

    def test_absent_from_gnomad_adds_points(self) -> None:
        absent = _make_variant(af_gnomad=None)
        common = _make_variant(af_gnomad=0.3)
        assert _score_variant(absent) > _score_variant(common)

    def test_homozygous_scores_more_than_heterozygous(self) -> None:
        hom = _make_variant(genotype="1/1")
        het = _make_variant(genotype="0/1")
        assert _score_variant(hom) > _score_variant(het)

    def test_vus_scores_between_pathogenic_and_benign(self) -> None:
        path = _make_variant(clinvar_sig="Pathogenic")
        vus = _make_variant(clinvar_sig="Uncertain_significance")
        benign = _make_variant(clinvar_sig="Benign")
        assert _score_variant(path) > _score_variant(vus) > _score_variant(benign)

    def test_frameshift_scores_same_as_stop_gained(self) -> None:
        fs = _make_variant(consequence="frameshift_variant")
        sg = _make_variant(consequence="stop_gained")
        assert _score_variant(fs) == _score_variant(sg)


# ── AnalysisReport tests ──────────────────────────────────────────────────────

def _make_report(*variants_and_scores) -> AnalysisReport:
    report = AnalysisReport(total_variants=len(variants_and_scores))
    for v, score in variants_and_scores:
        report.variant_results.append(VariantResult(variant=v, priority_score=score))
    return report


class TestAnalysisReport:
    def test_high_impact_count(self) -> None:
        v1 = _make_variant(impact="HIGH")
        v2 = _make_variant(impact="MODERATE")
        v3 = _make_variant(impact="HIGH")
        report = _make_report((v1, 50), (v2, 30), (v3, 60))
        assert report.high_impact_count == 2

    def test_pathogenic_count(self) -> None:
        v1 = _make_variant(clinvar_sig="Pathogenic")
        v2 = _make_variant(clinvar_sig="Likely pathogenic")
        v3 = _make_variant(clinvar_sig="Benign")
        report = _make_report((v1, 90), (v2, 75), (v3, 5))
        assert report.pathogenic_count == 2

    def test_vus_count(self) -> None:
        v1 = _make_variant(clinvar_sig="Uncertain significance")
        v2 = _make_variant(clinvar_sig="Pathogenic")
        report = _make_report((v1, 20), (v2, 80))
        assert report.vus_count == 1

    def test_rare_count_none_af(self) -> None:
        v = _make_variant(af_gnomad=None)
        report = _make_report((v, 50))
        assert report.rare_count == 1

    def test_rare_count_low_af(self) -> None:
        v = _make_variant(af_gnomad=0.005)
        report = _make_report((v, 30))
        assert report.rare_count == 1

    def test_rare_count_excludes_common(self) -> None:
        v = _make_variant(af_gnomad=0.3)
        report = _make_report((v, 10))
        assert report.rare_count == 0

    def test_top_genes_returns_unique_ordered(self) -> None:
        v1 = _make_variant(gene="BRCA1")
        v2 = _make_variant(gene="BRCA2")
        v3 = _make_variant(gene="BRCA1")  # duplicate
        report = _make_report((v1, 90), (v2, 70), (v3, 60))
        genes = report.top_genes
        assert genes[0] == "BRCA1"
        assert genes.count("BRCA1") == 1  # deduplicated

    def test_to_json_is_valid_json(self) -> None:
        import json
        v = _make_variant(gene="TP53", impact="HIGH", clinvar_sig="Pathogenic")
        report = _make_report((v, 80))
        data = json.loads(report.to_json())
        assert "variants" in data
        assert data["total_variants"] == 1


# ── GenomicAnalyzer pipeline tests ────────────────────────────────────────────

class TestGenomicAnalyzerPipeline:
    """These tests exercise the pipeline without making real API calls."""

    SAMPLE_VCF = Path(__file__).parent.parent / "data" / "sample_variants.vcf"

    def test_analyze_returns_report(self) -> None:
        if not self.SAMPLE_VCF.exists():
            pytest.skip("sample_variants.vcf not found")
        analyzer = GenomicAnalyzer(api_key=None)  # no key → mock responses
        report = analyzer.analyze(self.SAMPLE_VCF)
        assert isinstance(report, AnalysisReport)

    def test_total_variants_correct(self) -> None:
        if not self.SAMPLE_VCF.exists():
            pytest.skip("sample_variants.vcf not found")
        analyzer = GenomicAnalyzer(api_key=None)
        report = analyzer.analyze(self.SAMPLE_VCF)
        assert report.total_variants > 0

    def test_dataframe_has_columns(self) -> None:
        if not self.SAMPLE_VCF.exists():
            pytest.skip("sample_variants.vcf not found")
        analyzer = GenomicAnalyzer(api_key=None)
        report = analyzer.analyze(self.SAMPLE_VCF)
        if not report.dataframe.empty:
            for col in ("chrom", "pos", "ref", "alt", "gene", "priority_score"):
                assert col in report.dataframe.columns

    def test_pass_filter_applied(self) -> None:
        if not self.SAMPLE_VCF.exists():
            pytest.skip("sample_variants.vcf not found")
        analyzer = GenomicAnalyzer(api_key=None)
        # With filter
        report_filtered = analyzer.analyze(self.SAMPLE_VCF, apply_pass_filter=True)
        # Without filter
        report_all = analyzer.analyze(self.SAMPLE_VCF, apply_pass_filter=False)
        # With PASS filter we should have fewer or equal variants
        assert report_filtered.filtered_variants <= report_all.filtered_variants

    def test_variant_results_sorted_by_priority(self) -> None:
        if not self.SAMPLE_VCF.exists():
            pytest.skip("sample_variants.vcf not found")
        analyzer = GenomicAnalyzer(api_key=None)
        report = analyzer.analyze(self.SAMPLE_VCF)
        scores = [vr.priority_score for vr in report.variant_results]
        assert scores == sorted(scores, reverse=True)

    def test_cohort_summary_not_empty(self) -> None:
        if not self.SAMPLE_VCF.exists():
            pytest.skip("sample_variants.vcf not found")
        analyzer = GenomicAnalyzer(api_key=None)
        report = analyzer.analyze(self.SAMPLE_VCF)
        assert len(report.cohort_summary) > 0


# ── ReportGenerator tests ─────────────────────────────────────────────────────

class TestReportGenerator:
    def _sample_report(self) -> AnalysisReport:
        v1 = _make_variant(
            chrom="chr17", pos=43071077, ref="G", alt=["T"],
            gene="BRCA1", consequence="stop_gained", impact="HIGH",
            clinvar_sig="Pathogenic", af_gnomad=0.0000024,
            hgvs_c="c.5266G>T", hgvs_p="p.Glu1756Ter",
        )
        v2 = _make_variant(
            chrom="chr7", pos=117548628, ref="CTT", alt=["C"],
            gene="CFTR", consequence="frameshift_variant", impact="HIGH",
            clinvar_sig="Pathogenic", af_gnomad=0.013,
            hgvs_p="p.Phe508del",
        )
        report = AnalysisReport(total_variants=2, filtered_variants=2)
        report.variant_results = [
            VariantResult(variant=v1, interpretation="Test interpretation 1.", priority_score=95),
            VariantResult(variant=v2, interpretation="Test interpretation 2.", priority_score=80),
        ]
        report.cohort_summary = "Mock cohort summary from Gemma 4."
        report.pathway_analysis = "Mock pathway analysis from Gemma 4."
        return report

    def test_save_html_creates_file(self, tmp_path: Path) -> None:
        generator = ReportGenerator()
        report = self._sample_report()
        out = tmp_path / "report.html"
        result = generator.save_html(report, out)
        assert result.exists()
        content = result.read_text()
        assert "GeneScribe" in content
        assert "BRCA1" in content

    def test_html_contains_variant_data(self, tmp_path: Path) -> None:
        generator = ReportGenerator()
        report = self._sample_report()
        out = tmp_path / "report.html"
        generator.save_html(report, out)
        content = out.read_text()
        assert "chr17" in content
        assert "stop_gained" in content

    def test_save_markdown_creates_file(self, tmp_path: Path) -> None:
        generator = ReportGenerator()
        report = self._sample_report()
        out = tmp_path / "report.md"
        result = generator.save_markdown(report, out)
        assert result.exists()
        content = result.read_text()
        assert "GeneScribe" in content
        assert "BRCA1" in content

    def test_markdown_has_stats_table(self, tmp_path: Path) -> None:
        generator = ReportGenerator()
        report = self._sample_report()
        out = tmp_path / "report.md"
        generator.save_markdown(report, out)
        content = out.read_text()
        assert "Total Variants" in content
        assert "HIGH Impact" in content

    def test_save_json_creates_file(self, tmp_path: Path) -> None:
        import json
        generator = ReportGenerator()
        report = self._sample_report()
        out = tmp_path / "report.json"
        generator.save_json(report, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "variants" in data

    def test_escape_html_ampersand(self) -> None:
        assert _escape_html("A&B") == "A&amp;B"

    def test_escape_html_angle_brackets(self) -> None:
        assert _escape_html("<script>") == "&lt;script&gt;"

    def test_escape_html_quotes(self) -> None:
        result = _escape_html('"hello"')
        assert "&quot;" in result
