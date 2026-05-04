"""
variant_parser.py
=================
Parses VCF (Variant Call Format) files and extracts structured variant data.
Supports VCF 4.x format. Handles both compressed and uncompressed files.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class Variant:
    """Represents a single genomic variant."""
    chrom: str
    pos: int
    variant_id: str
    ref: str
    alt: list[str]
    qual: float | None
    filter_status: str
    info: dict[str, str]
    genotype: str = "unknown"

    # Derived / annotated fields populated later
    gene: str = ""
    consequence: str = ""
    impact: str = ""
    af_gnomad: float | None = None   # allele frequency in gnomAD
    clinvar_sig: str = ""            # ClinVar clinical significance
    hgvs_c: str = ""                 # HGVS coding nomenclature
    hgvs_p: str = ""                 # HGVS protein nomenclature

    @property
    def is_snv(self) -> bool:
        """True if variant is a Single Nucleotide Variant."""
        return len(self.ref) == 1 and all(len(a) == 1 for a in self.alt)

    @property
    def is_indel(self) -> bool:
        """True if variant is an insertion or deletion."""
        return not self.is_snv

    @property
    def is_pass(self) -> bool:
        return self.filter_status.upper() in ("PASS", ".")

    def to_dict(self) -> dict:
        return {
            "chrom": self.chrom,
            "pos": self.pos,
            "id": self.variant_id,
            "ref": self.ref,
            "alt": ",".join(self.alt),
            "qual": self.qual,
            "filter": self.filter_status,
            "gene": self.gene,
            "consequence": self.consequence,
            "impact": self.impact,
            "af_gnomad": self.af_gnomad,
            "clinvar_sig": self.clinvar_sig,
            "hgvs_c": self.hgvs_c,
            "hgvs_p": self.hgvs_p,
            "genotype": self.genotype,
        }

    def __str__(self) -> str:
        alts = "/".join(self.alt)
        gene_info = f" [{self.gene}]" if self.gene else ""
        return f"{self.chrom}:{self.pos} {self.ref}>{alts}{gene_info}"


@dataclass
class VCFHeader:
    """Stores VCF header metadata."""
    fileformat: str = "VCFv4.2"
    reference: str = ""
    samples: list[str] = field(default_factory=list)
    contigs: list[str] = field(default_factory=list)
    info_fields: dict[str, str] = field(default_factory=dict)
    raw_lines: list[str] = field(default_factory=list)


# ── Parser ────────────────────────────────────────────────────────────────────

class VCFParser:
    """
    Lightweight VCF parser that handles VCF 4.x files without external
    dependencies beyond the standard library.

    Usage
    -----
    >>> parser = VCFParser("sample.vcf")
    >>> for variant in parser.parse():
    ...     print(variant)
    """

    # Known INFO keys that directly contain a gene symbol (not complex annotations)
    _GENE_KEYS = ("GENE", "SYMBOL")
    _CSQ_CONSEQUENCE = re.compile(r"[A-Z_]+variant")

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        self.header = VCFHeader()
        self._parsed = False

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self) -> list[Variant]:
        """Parse the VCF file and return a list of Variant objects."""
        variants: list[Variant] = []
        for variant in self._iter_variants():
            variants.append(variant)
        self._parsed = True
        return variants

    def iter_variants(self) -> Iterator[Variant]:
        """Lazily iterate over variants (memory-efficient for large files)."""
        yield from self._iter_variants()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _open(self):
        """Open plain or gzip-compressed VCF."""
        if self.filepath.suffix == ".gz":
            return gzip.open(self.filepath, "rt")
        return open(self.filepath, "r")

    def _iter_variants(self) -> Iterator[Variant]:
        with self._open() as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("##"):
                    self._parse_meta(line)
                elif line.startswith("#CHROM"):
                    self._parse_column_header(line)
                elif line.strip():
                    variant = self._parse_data_line(line)
                    if variant:
                        yield variant

    def _parse_meta(self, line: str) -> None:
        self.header.raw_lines.append(line)
        if line.startswith("##fileformat="):
            self.header.fileformat = line.split("=", 1)[1]
        elif line.startswith("##reference="):
            self.header.reference = line.split("=", 1)[1]
        elif line.startswith("##contig="):
            m = re.search(r"ID=([^,>]+)", line)
            if m:
                self.header.contigs.append(m.group(1))
        elif line.startswith("##INFO="):
            m = re.search(r"ID=([^,]+).*Description=\"([^\"]+)\"", line)
            if m:
                self.header.info_fields[m.group(1)] = m.group(2)

    def _parse_column_header(self, line: str) -> None:
        cols = line.lstrip("#").split("\t")
        # Columns: CHROM POS ID REF ALT QUAL FILTER INFO [FORMAT [SAMPLE …]]
        if len(cols) > 9:
            self.header.samples = cols[9:]

    def _parse_data_line(self, line: str) -> Variant | None:
        cols = line.split("\t")
        if len(cols) < 8:
            return None

        chrom, pos, vid, ref, alt_str, qual_str, flt, info_str = cols[:8]

        try:
            pos_int = int(pos)
        except ValueError:
            return None

        try:
            qual = float(qual_str) if qual_str not in (".", "") else None
        except ValueError:
            qual = None

        alt = [a for a in alt_str.split(",") if a != "."]
        info = self._parse_info(info_str)

        genotype = "unknown"
        if len(cols) >= 10:
            genotype = self._extract_genotype(cols[8], cols[9])

        variant = Variant(
            chrom=chrom,
            pos=pos_int,
            variant_id=vid,
            ref=ref,
            alt=alt if alt else ["."],
            qual=qual,
            filter_status=flt,
            info=info,
            genotype=genotype,
        )
        self._annotate_from_info(variant)
        return variant

    @staticmethod
    def _parse_info(info_str: str) -> dict[str, str]:
        """Parse the INFO column into a key-value dict."""
        result: dict[str, str] = {}
        if info_str in (".", ""):
            return result
        for item in info_str.split(";"):
            if "=" in item:
                k, v = item.split("=", 1)
                result[k] = v
            else:
                result[item] = "true"
        return result

    @staticmethod
    def _extract_genotype(fmt_str: str, sample_str: str) -> str:
        """Extract the GT field from FORMAT/SAMPLE columns."""
        fmt_fields = fmt_str.split(":")
        sample_fields = sample_str.split(":")
        try:
            gt_idx = fmt_fields.index("GT")
            return sample_fields[gt_idx]
        except (ValueError, IndexError):
            return sample_fields[0] if sample_fields else "unknown"

    def _annotate_from_info(self, variant: Variant) -> None:
        """Populate derived fields from common INFO annotations."""
        info = variant.info

        # Gene symbol
        for key in self._GENE_KEYS:
            if key in info:
                variant.gene = info[key].split("|")[0]
                break

        # Consequence / impact from VEP-style ANN field
        if "ANN" in info:
            parts = info["ANN"].split("|")
            if len(parts) > 1:
                variant.consequence = parts[1]
            if len(parts) > 2:
                variant.impact = parts[2]
            if len(parts) > 3 and not variant.gene:
                variant.gene = parts[3]
            if len(parts) > 9:
                variant.hgvs_c = parts[9]
            if len(parts) > 10:
                variant.hgvs_p = parts[10]

        # Simpler tags
        for key in ("CONSEQUENCE", "CSQ_CONSEQUENCE"):
            if key in info and not variant.consequence:
                variant.consequence = info[key]

        for key in ("IMPACT",):
            if key in info and not variant.impact:
                variant.impact = info[key]

        # allele frequency
        for key in ("AF", "AF_gnomAD", "gnomAD_AF", "MAX_AF"):
            if key in info:
                try:
                    variant.af_gnomad = float(info[key])
                except ValueError:
                    pass
                break

        # ClinVar significance
        for key in ("CLNSIG", "ClinSig", "CLINSIG"):
            if key in info:
                variant.clinvar_sig = info[key].replace("_", " ")
                break

        # HGVS
        if "HGVS_C" in info and not variant.hgvs_c:
            variant.hgvs_c = info["HGVS_C"]
        if "HGVS_P" in info and not variant.hgvs_p:
            variant.hgvs_p = info["HGVS_P"]
