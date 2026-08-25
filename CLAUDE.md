# clarity — contributor rules

- **Docs rule (mandatory):** any behavioral change ships the matching doc
  update in the SAME commit — README.md for user-facing behavior, docs/DESIGN.md
  for architecture/decision changes, docs/CALIBRATION.md for thresholds,
  docs/ROADMAP.md checkboxes when a phase item lands. Bump "Last updated" lines.
- **No LLM judges.** LLMs may verbalize computed evidence; they never decide
  whether text is AI. This is a research-backed hard rule (DESIGN.md#no-llm-judges).
- Never present detector output as proof. Labels are "likely", with caveats.
- Tests: model-free layers get unit tests (`pytest`, no downloads).
  Model-path changes require running `scripts/smoke.py` locally and pasting
  the result in the PR/commit body.
- Env: `uv venv -p 3.12` (torch has no cp314 wheels yet); `.venv/bin/python`
  directly, not `uv run`, for scripts that print (stdout-eating issue).
- Thresholds are pair-specific; never change defaults without CALIBRATION.md.
