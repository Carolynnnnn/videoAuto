#!/usr/bin/env python3
"""Task-9 contrast verification for mapped GUI token pairs."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


DEFAULT_MAPPED_PAIRS = [
    {"text": "text_secondary", "background": "bg_dark", "threshold": 4.5},
    {"text": "text_secondary", "background": "bg_card", "threshold": 4.5},
    {"text": "text_secondary", "background": "bg_input", "threshold": 4.5},
    {"text": "text_muted", "background": "bg_dark", "threshold": 3.0},
    {"text": "text_muted", "background": "bg_card", "threshold": 3.0},
    {"text": "text_muted", "background": "bg_input", "threshold": 3.0},
]


def load_checker_module(checker_file: Path):
    spec = importlib.util.spec_from_file_location("task5_contrast_checker", checker_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load checker from {checker_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_mapped_pairs(pairs_file: Path | None):
    if pairs_file is None:
        return DEFAULT_MAPPED_PAIRS

    payload = json.loads(pairs_file.read_text())
    pairs = payload.get("pairs", [])
    for pair in pairs:
        pair.setdefault("threshold", 3.0)
    return pairs


def run_checks(tokens, mapped_pairs, checker_module):
    results = []
    for pair in mapped_pairs:
        threshold = float(pair["threshold"])
        check_result = checker_module.check_pairs(
            tokens,
            [{"text": pair["text"], "background": pair["background"]}],
            threshold=threshold,
        )[0]
        check_result["pair_id"] = f"{pair['text']} on {pair['background']}"
        results.append(check_result)
    return results


def summarize(results):
    failed = [row for row in results if not row.get("pass")]
    return {
        "total": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
    }


def print_failures(failures):
    print("Contrast threshold violations detected:")
    for row in failures:
        if row.get("error"):
            print(f"- {row['pair']}: {row['error']}")
            continue

        ratio = row.get("ratio")
        print(
            f"- {row['pair']}: {row['text_color']} on {row['bg_color']} "
            f"= {ratio:.2f} (threshold {row['threshold']:.2f})"
        )


def main():
    parser = argparse.ArgumentParser(description="Verify mapped GUI contrast ratios")
    parser.add_argument(
        "--style-file",
        default="src/gui/styles.py",
        help="Path to styles.py token file",
    )
    parser.add_argument(
        "--checker-file",
        default=(
            ".sisyphus/evidence/gui-page-review-hardening/"
            "task-5/scripts/contrast_checker.py"
        ),
        help="Path to task-5 contrast checker",
    )
    parser.add_argument(
        "--pairs-file",
        help="Optional JSON pairs file with text/background/threshold entries",
    )
    parser.add_argument("--output", required=True, help="Output report JSON path")
    parser.add_argument(
        "--secondary-threshold",
        type=float,
        help="Override threshold for text_secondary pairs",
    )
    parser.add_argument(
        "--muted-threshold",
        type=float,
        help="Override threshold for text_muted pairs",
    )
    parser.add_argument(
        "--only-failures",
        action="store_true",
        help="Write only failing rows in output",
    )
    args = parser.parse_args()

    checker_file = Path(args.checker_file).resolve()
    style_file = Path(args.style_file).resolve()
    pairs_file = Path(args.pairs_file).resolve() if args.pairs_file else None

    checker = load_checker_module(checker_file)
    tokens = checker.parse_style_tokens(style_file)
    mapped_pairs = load_mapped_pairs(pairs_file)

    for pair in mapped_pairs:
        if args.secondary_threshold is not None and pair["text"] == "text_secondary":
            pair["threshold"] = float(args.secondary_threshold)
        if args.muted_threshold is not None and pair["text"] == "text_muted":
            pair["threshold"] = float(args.muted_threshold)

    results = run_checks(tokens, mapped_pairs, checker)
    summary = summarize(results)
    failures = [row for row in results if not row.get("pass")]

    output_rows = failures if args.only_failures else results
    report = {
        "style_file": str(style_file),
        "checker_file": str(checker_file),
        "mapped_pairs": mapped_pairs,
        "results": output_rows,
        "summary": summary,
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"Contrast check complete: {summary['passed']} passed, "
        f"{summary['failed']} failed"
    )
    print(f"Report: {output_path}")

    if failures:
        print_failures(failures)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
