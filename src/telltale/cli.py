"""CLI: `telltale <file>` or `telltale -` for stdin. Rich terminal heatmap or --json."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .detector import (
    DEFAULT_THRESHOLD_HIGH,
    DEFAULT_THRESHOLD_LOW,
    Report,
    analyze,
)
from .models import DEFAULT_OBSERVER, DEFAULT_PERFORMER, ModelPair

LABEL_STYLE = {"ai": "bold white on red", "uncertain": "black on yellow", "human": "default"}
LABEL_WORD = {"ai": "likely AI-generated", "uncertain": "uncertain", "human": "likely human"}


def render(report: Report, console: Console) -> None:
    verdict = LABEL_WORD[report.doc_label]
    header = (
        f"[bold]Document:[/bold] {verdict}  "
        f"(score {report.doc_score:.3f}; AI < {report.threshold_low}, "
        f"human > {report.threshold_high})"
    )
    caveats = []
    if not report.reliable:
        caveats.append(
            f"only {report.n_tokens} tokens — too short for a reliable verdict"
        )
    if report.truncated:
        caveats.append("text was truncated at the model's scoring window")
    if caveats:
        header += f"\n[yellow]⚠ {'; '.join(caveats)}[/yellow]"
    console.print(Panel(header, title="telltale"))

    body = Text()
    for s in report.sentences:
        body.append(s.text + " ", style=LABEL_STYLE[s.label])
    console.print(body)
    console.print()

    flagged = [s for s in report.sentences if s.signals]
    if flagged:
        console.print("[bold]Why flagged:[/bold]")
        for s in flagged:
            preview = s.text if len(s.text) <= 70 else s.text[:67] + "..."
            console.print(f'  [red]▌[/red] "{preview}" (score {s.score:.3f})')
            for g in s.signals:
                console.print(f"      • {g.detail}")
    console.print(
        "\n[dim]No detector is proof. Scores are calibrated probabilities, not "
        "verdicts; false positives disproportionately affect non-native writers. "
        "Always review the evidence.[/dim]"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="telltale", description="Evidence-first AI-text detector")
    p.add_argument("input", help="path to a text file, or '-' for stdin")
    p.add_argument("--observer", default=DEFAULT_OBSERVER, help="HF id of observer model")
    p.add_argument("--performer", default=DEFAULT_PERFORMER, help="HF id of performer model")
    p.add_argument("--threshold-low", type=float, default=DEFAULT_THRESHOLD_LOW)
    p.add_argument("--threshold-high", type=float, default=DEFAULT_THRESHOLD_HIGH)
    p.add_argument("--json", action="store_true", help="emit the full report as JSON")
    args = p.parse_args(argv)

    text = sys.stdin.read() if args.input == "-" else open(args.input).read()
    if not text.strip():
        print("error: empty input", file=sys.stderr)
        return 2

    console = Console(stderr=True)
    with console.status("loading model pair (first run downloads weights)..."):
        pair = ModelPair(observer_name=args.observer, performer_name=args.performer)
    with console.status("scoring..."):
        report = analyze(
            text, pair,
            threshold_low=args.threshold_low,
            threshold_high=args.threshold_high,
        )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        render(report, Console())
    return 0


if __name__ == "__main__":
    sys.exit(main())
