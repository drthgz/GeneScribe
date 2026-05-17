# GeneScribe Next Steps TODO

## P0 - Must Fix Before Submission

- [ ] Resolve license mismatch across project metadata.
  - Current issue: README badge says MIT, but LICENSE file is GPLv3.
  - Action: Decide intended license (MIT vs GPLv3), then align README badge/text and LICENSE file.
  - Done when: README and LICENSE are fully consistent.

- [ ] Harden HTML report rendering against untrusted VCF content.
  - Current issue: Several variant fields are injected into HTML without escaping.
  - Action: Escape all user/data-derived fields in variant cards and tables (chrom, ref, alt, gene, consequence, genotype, HGVS, etc.).
  - Done when: malicious strings like <script>alert(1)</script> render as plain text.

## P1 - High Value Reliability and Clarity

- [ ] Fix duplicated conditional logic in cohort clinical question.
  - Current issue: both branches of the patient phenotype conditional produce the same fallback question.
  - Action: simplify to one expression or use phenotype-aware branch with distinct fallback.
  - Done when: code path is clear and behavior intentional.

- [ ] Fix AF formatting bug for zero frequency in top-variant text.
  - Current issue: AF value 0.0 is rendered as N/A due to truthy-or fallback.
  - Action: use explicit None check instead of "or" fallback.
  - Done when: AF 0.0 is displayed correctly.

- [ ] Align dependency list with actual usage.
  - Current issue: requirements include packages not used by src/tests (e.g., biopython, pyvcf3), while docs claim no bioinformatics-specific runtime deps.
  - Action: either remove unused deps or add documented usage; split runtime vs notebook/dev requirements if needed.
  - Done when: requirements and README are consistent and minimal.

## P2 - Documentation and Polish

- [ ] Update stale repository naming/references in docs and notebook.
  - Current issue: legacy repo name/link appears in README structure and notebook summary cell.
  - Action: replace old naming with current GeneScribe project identity.
  - Done when: all public-facing docs point to the current repo/project name.

- [ ] Add targeted tests for newly fixed edge cases.
  - Suggested tests:
    - HTML escaping regression tests for dangerous strings in variant fields.
    - AF=0.0 display test in formatted top-variant summary.
    - License consistency check in CI (optional script test).
  - Done when: tests fail before fixes and pass after fixes.

## P3 - Nice to Have (Post-Hackathon)

- [ ] Introduce non-interactive CLI entrypoint.
  - Action: add scripts/CLI command for analyze + report generation.

- [ ] Add CI workflow.
  - Action: run tests on push/PR and optionally lint/type-check.

- [ ] Add reproducible environment file.
  - Action: provide pinned lock file or optional pyproject.toml-based setup.
