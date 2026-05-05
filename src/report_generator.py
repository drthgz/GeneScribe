"""
report_generator.py
====================
Generates human-readable HTML and Markdown reports from an AnalysisReport.

Usage
-----
>>> from src.report_generator import ReportGenerator
>>> from src.genomic_analyzer import GenomicAnalyzer

>>> analyzer = GenomicAnalyzer(api_key="YOUR_KEY")
>>> report = analyzer.analyze("sample.vcf", patient_phenotypes="muscle weakness")

>>> generator = ReportGenerator()
>>> generator.save_html(report, "output_report.html")
>>> generator.save_markdown(report, "output_report.md")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .genomic_analyzer import AnalysisReport, VariantResult


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>GeneScribe Clinical Genomics Report</title>
  <style>
    :root {{
      --primary: #2563eb;
      --accent:  #7c3aed;
      --danger:  #dc2626;
      --warn:    #d97706;
      --ok:      #16a34a;
      --bg:      #f8fafc;
      --card:    #ffffff;
      --border:  #e2e8f0;
      --text:    #1e293b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 2rem; }}
    header {{ background: linear-gradient(135deg, var(--primary), var(--accent)); color: #fff; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; }}
    header h1 {{ font-size: 2rem; font-weight: 800; }}
    header p {{ opacity: 0.85; margin-top: 0.5rem; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
    .stat-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
    .stat-card .number {{ font-size: 2.5rem; font-weight: 700; }}
    .stat-card .label {{ font-size: 0.85rem; color: #64748b; margin-top: 0.25rem; }}
    .high {{ color: var(--danger); }}
    .moderate {{ color: var(--warn); }}
    .pathogenic {{ color: var(--danger); }}
    .section {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
    .section h2 {{ font-size: 1.25rem; font-weight: 700; margin-bottom: 1rem; border-bottom: 2px solid var(--primary); padding-bottom: 0.5rem; color: var(--primary); }}
    .variant-card {{ border: 1px solid var(--border); border-radius: 8px; padding: 1rem; margin-bottom: 1rem; position: relative; }}
    .variant-card .badge {{ display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 999px; margin: 2px; }}
    .badge-HIGH {{ background: #fef2f2; color: var(--danger); border: 1px solid #fca5a5; }}
    .badge-MODERATE {{ background: #fffbeb; color: var(--warn); border: 1px solid #fcd34d; }}
    .badge-LOW {{ background: #f0fdf4; color: var(--ok); border: 1px solid #86efac; }}
    .badge-pathogenic {{ background: #fef2f2; color: var(--danger); border: 1px solid #fca5a5; }}
    .badge-vus {{ background: #eff6ff; color: var(--primary); border: 1px solid #93c5fd; }}
    .badge-benign {{ background: #f0fdf4; color: var(--ok); border: 1px solid #86efac; }}
    .interp {{ background: #f8fafc; border-left: 4px solid var(--accent); padding: 0.75rem 1rem; margin-top: 0.75rem; border-radius: 0 6px 6px 0; white-space: pre-wrap; font-size: 0.9rem; line-height: 1.6; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    th {{ background: var(--primary); color: #fff; padding: 0.6rem 0.8rem; text-align: left; }}
    td {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--border); }}
    tr:hover td {{ background: #f1f5f9; }}
    .score-bar {{ height: 8px; border-radius: 4px; background: linear-gradient(90deg, var(--ok), var(--warn), var(--danger)); }}
    footer {{ text-align: center; color: #94a3b8; font-size: 0.8rem; margin-top: 2rem; }}
  </style>
</head>
<body>
  <header>
    <h1>🧬 GeneScribe Clinical Genomics Report</h1>
    <p>AI-Powered Variant Interpretation using Google Gemma 4 &nbsp;|&nbsp; Generated: {timestamp}</p>
    <p style="margin-top:0.5rem; font-weight:600;">Patient Phenotypes: {phenotypes}</p>
  </header>

  <!-- Stats -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="number">{total_variants}</div>
      <div class="label">Total Variants</div>
    </div>
    <div class="stat-card">
      <div class="number high">{high_impact}</div>
      <div class="label">HIGH Impact</div>
    </div>
    <div class="stat-card">
      <div class="number pathogenic">{pathogenic_count}</div>
      <div class="label">Pathogenic / Likely Path.</div>
    </div>
    <div class="stat-card">
      <div class="number moderate">{vus_count}</div>
      <div class="label">Variants of Uncertain Sig.</div>
    </div>
    <div class="stat-card">
      <div class="number">{rare_count}</div>
      <div class="label">Rare (AF &lt; 1%)</div>
    </div>
  </div>

  <!-- Cohort Summary -->
  <div class="section">
    <h2>🤖 Gemma 4 Cohort Summary</h2>
    <div class="interp">{cohort_summary}</div>
  </div>

  <!-- Top Variants -->
  <div class="section">
    <h2>🔬 Top Priority Variants</h2>
    {variant_cards}
  </div>

  <!-- Pathway Analysis -->
  <div class="section">
    <h2>🧪 Gene Pathway Analysis</h2>
    <div class="interp">{pathway_analysis}</div>
  </div>

  <!-- Full Variant Table -->
  <div class="section">
    <h2>📋 Complete Variant Table</h2>
    {variant_table}
  </div>

  <footer>
    GeneScribe v1.0 · Built for the Kaggle Gemma 4 Good Hackathon 2026 ·
    <em>Not a substitute for certified clinical laboratory interpretation.</em>
  </footer>
</body>
</html>
"""


# ── ReportGenerator ───────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Generates HTML and Markdown reports from an AnalysisReport.
    """

    def __init__(self, patient_name: str = "Anonymous", patient_id: str = "N/A") -> None:
        self.patient_name = patient_name
        self.patient_id = patient_id

    # ── Public API ────────────────────────────────────────────────────────────

    def save_html(
        self,
        report: AnalysisReport,
        output_path: str | Path = "genescribe_report.html",
        patient_phenotypes: str = "",
    ) -> Path:
        """Render and save an HTML report."""
        html = self._render_html(report, patient_phenotypes)
        path = Path(output_path)
        path.write_text(html, encoding="utf-8")
        print(f"[ReportGenerator] HTML report saved → {path}")
        return path

    def save_markdown(
        self,
        report: AnalysisReport,
        output_path: str | Path = "genescribe_report.md",
        patient_phenotypes: str = "",
    ) -> Path:
        """Render and save a Markdown report."""
        md = self._render_markdown(report, patient_phenotypes)
        path = Path(output_path)
        path.write_text(md, encoding="utf-8")
        print(f"[ReportGenerator] Markdown report saved → {path}")
        return path

    def save_json(
        self,
        report: AnalysisReport,
        output_path: str | Path = "genescribe_report.json",
    ) -> Path:
        """Save the raw analysis data as JSON."""
        path = Path(output_path)
        path.write_text(report.to_json(), encoding="utf-8")
        print(f"[ReportGenerator] JSON report saved → {path}")
        return path

    # ── HTML rendering ────────────────────────────────────────────────────────

    def _render_html(self, report: AnalysisReport, phenotypes: str) -> str:
        top_n = report.variant_results[:10]
        variant_cards = "".join(self._render_variant_card(vr) for vr in top_n)
        variant_table = self._render_variant_table(report.variant_results)

        return _HTML_TEMPLATE.format(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            phenotypes=phenotypes or "Not specified",
            total_variants=report.total_variants,
            high_impact=report.high_impact_count,
            pathogenic_count=report.pathogenic_count,
            vus_count=report.vus_count,
            rare_count=report.rare_count,
            cohort_summary=_escape_html(report.cohort_summary),
            pathway_analysis=_escape_html(report.pathway_analysis),
            variant_cards=variant_cards,
            variant_table=variant_table,
        )

    @staticmethod
    def _impact_badge(impact: str) -> str:
        cls = f"badge-{impact.upper()}"
        return f'<span class="badge {cls}">{impact or "Unknown"}</span>'

    @staticmethod
    def _clinvar_badge(sig: str) -> str:
        sig_lower = sig.lower()
        if "pathogenic" in sig_lower and "likely" not in sig_lower:
            css = "badge-pathogenic"
        elif "likely pathogenic" in sig_lower:
            css = "badge-pathogenic"
        elif "uncertain" in sig_lower:
            css = "badge-vus"
        elif "benign" in sig_lower:
            css = "badge-benign"
        else:
            css = ""
        return f'<span class="badge {css}">{sig or "Unclassified"}</span>'

    def _render_variant_card(self, vr: VariantResult) -> str:
        v = vr.variant
        alts = "/".join(v.alt)
        impact_badge = self._impact_badge(v.impact)
        clinvar_badge = self._clinvar_badge(v.clinvar_sig)
        af_text = f"{v.af_gnomad:.4%}" if v.af_gnomad is not None else "Not in gnomAD"
        interp_html = (
            f'<div class="interp">{_escape_html(vr.interpretation)}</div>'
            if vr.interpretation
            else '<p style="color:#94a3b8; font-style:italic;">No AI interpretation generated for this variant.</p>'
        )
        return f"""
        <div class="variant-card">
          <strong>{v.chrom}:{v.pos} &nbsp; {v.ref} → {alts}</strong>
          &nbsp;&nbsp; {impact_badge} {clinvar_badge}
          <span class="badge" style="background:#f1f5f9;color:#475569;border:1px solid #cbd5e1;">
            Score: {vr.priority_score:.0f}
          </span>
          <br/>
          <small>
            Gene: <strong>{v.gene or 'Unknown'}</strong> &nbsp;|&nbsp;
            Consequence: {v.consequence or 'N/A'} &nbsp;|&nbsp;
            gnomAD AF: {af_text} &nbsp;|&nbsp;
            Genotype: {v.genotype}
            {f'&nbsp;|&nbsp; HGVS: {v.hgvs_c}' if v.hgvs_c else ''}
            {f'&nbsp;|&nbsp; Protein: {v.hgvs_p}' if v.hgvs_p else ''}
          </small>
          {interp_html}
        </div>"""

    def _render_variant_table(self, variant_results: list[VariantResult]) -> str:
        rows = []
        for vr in variant_results:
            v = vr.variant
            alts = "/".join(v.alt)
            af_text = f"{v.af_gnomad:.4%}" if v.af_gnomad is not None else "—"
            impact_badge = self._impact_badge(v.impact)
            clinvar_badge = self._clinvar_badge(v.clinvar_sig)
            rows.append(f"""
            <tr>
              <td>{v.chrom}:{v.pos}</td>
              <td>{v.ref}</td>
              <td>{alts}</td>
              <td>{v.gene or '—'}</td>
              <td>{v.consequence or '—'}</td>
              <td>{impact_badge}</td>
              <td>{clinvar_badge}</td>
              <td>{af_text}</td>
              <td>{vr.priority_score:.0f}</td>
            </tr>""")

        return f"""
        <table>
          <thead>
            <tr>
              <th>Position</th><th>Ref</th><th>Alt</th><th>Gene</th>
              <th>Consequence</th><th>Impact</th><th>ClinVar</th>
              <th>gnomAD AF</th><th>Priority</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>"""

    # ── Markdown rendering ────────────────────────────────────────────────────

    def _render_markdown(self, report: AnalysisReport, phenotypes: str) -> str:
        lines: list[str] = []
        lines.append("# 🧬 GeneScribe Clinical Genomics Report")
        lines.append(f"> *Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Powered by Google Gemma 4*\n")
        lines.append(f"**Patient Phenotypes:** {phenotypes or 'Not specified'}")
        lines.append("")

        # Stats
        lines.append("## Summary Statistics\n")
        lines.append(f"| Metric | Count |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Variants | {report.total_variants} |")
        lines.append(f"| HIGH Impact | {report.high_impact_count} |")
        lines.append(f"| Pathogenic / Likely Pathogenic | {report.pathogenic_count} |")
        lines.append(f"| Variants of Uncertain Significance | {report.vus_count} |")
        lines.append(f"| Rare Variants (AF < 1%) | {report.rare_count} |")
        lines.append("")

        # Cohort summary
        lines.append("## 🤖 Gemma 4 Cohort Summary\n")
        lines.append(report.cohort_summary)
        lines.append("")

        # Top variants
        lines.append("## 🔬 Top Priority Variants\n")
        for i, vr in enumerate(report.variant_results[:10], 1):
            v = vr.variant
            alts = "/".join(v.alt)
            af_text = f"{v.af_gnomad:.4%}" if v.af_gnomad is not None else "Not in gnomAD"
            lines.append(f"### {i}. {v.chrom}:{v.pos} {v.ref}→{alts} [{v.gene or 'Unknown gene'}]")
            lines.append(f"- **Impact:** {v.impact or 'N/A'}")
            lines.append(f"- **Consequence:** {v.consequence or 'N/A'}")
            lines.append(f"- **ClinVar:** {v.clinvar_sig or 'Not classified'}")
            lines.append(f"- **gnomAD AF:** {af_text}")
            lines.append(f"- **Genotype:** {v.genotype}")
            if v.hgvs_c:
                lines.append(f"- **HGVS (coding):** `{v.hgvs_c}`")
            if v.hgvs_p:
                lines.append(f"- **HGVS (protein):** `{v.hgvs_p}`")
            lines.append(f"- **Priority Score:** {vr.priority_score:.0f}")
            if vr.interpretation:
                lines.append("\n**Gemma 4 Interpretation:**")
                lines.append("")
                lines.append(vr.interpretation)
            lines.append("")

        # Pathway analysis
        lines.append("## 🧪 Gene Pathway Analysis\n")
        lines.append(report.pathway_analysis)
        lines.append("")

        lines.append("---")
        lines.append("*GeneScribe v1.0 · Built for the Kaggle Gemma 4 Good Hackathon 2026*")
        lines.append("*Not a substitute for certified clinical laboratory interpretation.*")

        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _escape_html(text: str) -> str:
    """Minimal HTML entity escaping for safe embedding in HTML."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
