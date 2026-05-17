# GeneScribe Next Steps TODO

## Checkpoint Completed (May 17, 2026)

- [x] HTML report markdown rendering fixed for both `markdown`-installed and fallback modes.
- [x] HTML output hardened with escaping for data-derived variant fields.
- [x] AF formatting edge case fixed (`0.0` no longer displayed as N/A).
- [x] README synchronized with actual repository state and GPLv3 license.
- [x] Dependencies trimmed to match active runtime usage.
- [x] Notebook updated to keep Gemma 4 as the default model goal with an explicit fallback option.
- [x] Test suite passing (`62 passed`).

## P0 - Submission-Critical (Hackathon Alignment)

- [ ] Produce one final live-run artifact set with a Gemma 4 model.
  - Required output: final `genescribe_report.html` and `genescribe_report.json` generated from a live key run.
  - Rationale: confirms end-to-end compliance with Gemma 4 hackathon objective.

- [ ] Add a short "How this meets hackathon criteria" section in the notebook.
  - Include: societal impact (health), technical novelty, Gemma usage, and responsible-use disclaimer.

- [ ] Freeze the demo narrative for judging.
  - Ensure README and notebook use the same language for: problem, pipeline, outputs, and limitations.

## P1 - Reliability and Maintainability

- [ ] Migrate from `google-generativeai` to `google.genai`.
  - Current status: deprecation warning appears during tests.
  - Done when: client wrapper and notebook run without deprecation warning.

- [ ] Add regression tests for report safety/rendering.
  - Test 1: HTML escaping of malicious variant fields.
  - Test 2: markdown headings/lists render in fallback mode.
  - Test 3: AF formatting preserves `0.0`.

- [ ] Add a minimal CLI entry point for reproducible runs.
  - Example: `python -m src.cli --vcf data/sample_variants.vcf --out notebooks/`.

## P2 - Demo Polish

- [ ] Generate one PDF from the final HTML report and include it in `notebooks/` for reviewer convenience.
- [ ] Add one compact benchmark table (runtime + interpreted variant count) to notebook summary.
- [ ] Add a small architecture diagram (parser -> scorer -> Gemma -> report).

## P3 - Post-Hackathon

- [ ] Add CI workflow for tests on push/PR.
- [ ] Split requirements into runtime and dev/test extras.
- [ ] Evaluate optional support for local/private model serving for sensitive clinical data.
