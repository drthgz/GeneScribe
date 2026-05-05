"""
Tests for src/variant_parser.py
"""

import sys
import textwrap
from pathlib import Path
import pytest

# Make the repo root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.variant_parser import VCFParser, Variant


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_VCF = Path(__file__).parent.parent / "data" / "sample_variants.vcf"

MINIMAL_VCF_CONTENT = textwrap.dedent("""\
    ##fileformat=VCFv4.2
    ##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
    ##INFO=<ID=CLNSIG,Number=.,Type=String,Description="ClinVar significance">
    ##INFO=<ID=ANN,Number=.,Type=String,Description="SnpEff annotation">
    ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
    #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
    chr1\t100\t.\tA\tG\t50\tPASS\tAF=0.001;CLNSIG=Pathogenic\tGT\t0/1
    chr2\t200\t.\tC\tT\t.\t.\tANN=T|missense_variant|MODERATE|BRCA1|ENSG000|.|.|.|.|c.100C>T|p.Arg34Cys|.|.|.|.;\tGT\t1/1
    chrX\t300\trs1\tG\tA,C\t30\tFAIL\t.\tGT\t0/1
""")


@pytest.fixture
def tmp_vcf(tmp_path: Path) -> Path:
    vcf_file = tmp_path / "test.vcf"
    vcf_file.write_text(MINIMAL_VCF_CONTENT)
    return vcf_file


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestVCFParserBasics:
    def test_parse_returns_list(self, tmp_vcf: Path) -> None:
        parser = VCFParser(tmp_vcf)
        variants = parser.parse()
        assert isinstance(variants, list)

    def test_correct_variant_count(self, tmp_vcf: Path) -> None:
        variants = VCFParser(tmp_vcf).parse()
        assert len(variants) == 3

    def test_chromosome_parsed(self, tmp_vcf: Path) -> None:
        variants = VCFParser(tmp_vcf).parse()
        assert variants[0].chrom == "chr1"
        assert variants[1].chrom == "chr2"
        assert variants[2].chrom == "chrX"

    def test_position_parsed(self, tmp_vcf: Path) -> None:
        variants = VCFParser(tmp_vcf).parse()
        assert variants[0].pos == 100
        assert variants[1].pos == 200

    def test_ref_alt_parsed(self, tmp_vcf: Path) -> None:
        variants = VCFParser(tmp_vcf).parse()
        assert variants[0].ref == "A"
        assert variants[0].alt == ["G"]

    def test_multi_allelic_alt(self, tmp_vcf: Path) -> None:
        variants = VCFParser(tmp_vcf).parse()
        assert variants[2].alt == ["A", "C"]

    def test_qual_parsed(self, tmp_vcf: Path) -> None:
        variants = VCFParser(tmp_vcf).parse()
        assert variants[0].qual == 50.0
        assert variants[1].qual is None  # "." → None

    def test_filter_parsed(self, tmp_vcf: Path) -> None:
        variants = VCFParser(tmp_vcf).parse()
        assert variants[0].filter_status == "PASS"
        assert variants[2].filter_status == "FAIL"

    def test_genotype_extracted(self, tmp_vcf: Path) -> None:
        variants = VCFParser(tmp_vcf).parse()
        assert variants[0].genotype == "0/1"
        assert variants[1].genotype == "1/1"


class TestVCFParserInfoAnnotation:
    def test_af_gnomad_parsed(self, tmp_vcf: Path) -> None:
        variants = VCFParser(tmp_vcf).parse()
        assert variants[0].af_gnomad == pytest.approx(0.001)

    def test_clinvar_sig_parsed(self, tmp_vcf: Path) -> None:
        variants = VCFParser(tmp_vcf).parse()
        assert "pathogenic" in variants[0].clinvar_sig.lower()

    def test_ann_consequence_parsed(self, tmp_vcf: Path) -> None:
        variants = VCFParser(tmp_vcf).parse()
        assert variants[1].consequence == "missense_variant"

    def test_ann_impact_parsed(self, tmp_vcf: Path) -> None:
        variants = VCFParser(tmp_vcf).parse()
        assert variants[1].impact == "MODERATE"

    def test_ann_gene_parsed(self, tmp_vcf: Path) -> None:
        variants = VCFParser(tmp_vcf).parse()
        assert variants[1].gene == "BRCA1"

    def test_ann_hgvs_c_parsed(self, tmp_vcf: Path) -> None:
        variants = VCFParser(tmp_vcf).parse()
        assert variants[1].hgvs_c == "c.100C>T"

    def test_ann_hgvs_p_parsed(self, tmp_vcf: Path) -> None:
        variants = VCFParser(tmp_vcf).parse()
        assert variants[1].hgvs_p == "p.Arg34Cys"


class TestVariantProperties:
    def test_is_snv_true(self) -> None:
        v = Variant("chr1", 100, ".", "A", ["G"], 50.0, "PASS", {})
        assert v.is_snv is True

    def test_is_snv_false_for_indel(self) -> None:
        v = Variant("chr1", 100, ".", "ATG", ["A"], 50.0, "PASS", {})
        assert v.is_snv is False

    def test_is_indel_true(self) -> None:
        v = Variant("chr1", 100, ".", "ATG", ["A"], 50.0, "PASS", {})
        assert v.is_indel is True

    def test_is_pass_true(self) -> None:
        v = Variant("chr1", 100, ".", "A", ["G"], 50.0, "PASS", {})
        assert v.is_pass is True

    def test_is_pass_false(self) -> None:
        v = Variant("chr1", 100, ".", "A", ["G"], 50.0, "FAIL", {})
        assert v.is_pass is False

    def test_to_dict_has_required_keys(self) -> None:
        v = Variant("chr1", 100, ".", "A", ["G"], 50.0, "PASS", {})
        d = v.to_dict()
        for key in ("chrom", "pos", "ref", "alt", "gene", "consequence", "impact"):
            assert key in d

    def test_str_representation(self) -> None:
        v = Variant("chr1", 100, ".", "A", ["G"], 50.0, "PASS", {}, gene="BRCA1")
        s = str(v)
        assert "chr1" in s
        assert "100" in s
        assert "BRCA1" in s


class TestVCFParserHeaderParsing:
    def test_header_fileformat(self, tmp_vcf: Path) -> None:
        parser = VCFParser(tmp_vcf)
        parser.parse()
        assert parser.header.fileformat == "VCFv4.2"

    def test_header_info_fields(self, tmp_vcf: Path) -> None:
        parser = VCFParser(tmp_vcf)
        parser.parse()
        assert "AF" in parser.header.info_fields
        assert "CLNSIG" in parser.header.info_fields

    def test_header_samples(self, tmp_vcf: Path) -> None:
        parser = VCFParser(tmp_vcf)
        parser.parse()
        assert "SAMPLE" in parser.header.samples

    def test_iter_variants_yields_same_count(self, tmp_vcf: Path) -> None:
        parser = VCFParser(tmp_vcf)
        count = sum(1 for _ in parser.iter_variants())
        assert count == 3


class TestSampleVCF:
    """Integration-style test using the provided sample VCF."""

    def test_sample_vcf_loads(self) -> None:
        if not SAMPLE_VCF.exists():
            pytest.skip("sample_variants.vcf not found")
        variants = VCFParser(SAMPLE_VCF).parse()
        assert len(variants) > 0

    def test_sample_vcf_has_pass_variants(self) -> None:
        if not SAMPLE_VCF.exists():
            pytest.skip("sample_variants.vcf not found")
        variants = VCFParser(SAMPLE_VCF).parse()
        pass_variants = [v for v in variants if v.is_pass]
        assert len(pass_variants) > 0

    def test_sample_vcf_has_genes(self) -> None:
        if not SAMPLE_VCF.exists():
            pytest.skip("sample_variants.vcf not found")
        variants = VCFParser(SAMPLE_VCF).parse()
        genes = [v.gene for v in variants if v.gene]
        assert len(genes) > 0

    def test_sample_vcf_has_clinvar_sigs(self) -> None:
        if not SAMPLE_VCF.exists():
            pytest.skip("sample_variants.vcf not found")
        variants = VCFParser(SAMPLE_VCF).parse()
        sigs = [v.clinvar_sig for v in variants if v.clinvar_sig]
        assert len(sigs) > 0

    def test_known_brca1_variant_present(self) -> None:
        """BRCA1 stop_gained variant should be in the sample file."""
        if not SAMPLE_VCF.exists():
            pytest.skip("sample_variants.vcf not found")
        variants = VCFParser(SAMPLE_VCF).parse()
        brca1_stop = [
            v for v in variants
            if v.gene == "BRCA1" and "stop_gained" in v.consequence
        ]
        assert len(brca1_stop) >= 1

    def test_cftr_delta_f508_present(self) -> None:
        """CFTR p.Phe508del (delta-F508) should be in the sample file."""
        if not SAMPLE_VCF.exists():
            pytest.skip("sample_variants.vcf not found")
        variants = VCFParser(SAMPLE_VCF).parse()
        cftr = [v for v in variants if v.gene == "CFTR"]
        assert len(cftr) >= 1
        assert any("508" in v.hgvs_p for v in cftr)
