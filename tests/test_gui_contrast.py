import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "contrast" / "verify_gui_contrast.py"
STYLE_FILE = ROOT / "src" / "gui" / "styles.py"


def run_checker(tmp_path, *extra_args):
    report_path = tmp_path / "report.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--style-file",
        str(STYLE_FILE),
        "--output",
        str(report_path),
        *extra_args,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result, report_path


def test_gui_contrast_thresholds_pass(tmp_path):
    result, report_path = run_checker(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert report_path.exists()

    report = json.loads(report_path.read_text())
    assert report["summary"]["failed"] == 0
    assert report["summary"]["total"] >= 6


def test_gui_contrast_failure_output_has_pair_color_ratio(tmp_path):
    result, report_path = run_checker(tmp_path, "--muted-threshold", "3.80")

    assert result.returncode != 0
    assert report_path.exists()

    report = json.loads(report_path.read_text())
    assert report["summary"]["failed"] > 0

    combined_output = result.stdout + result.stderr
    assert "text_muted on bg_card" in combined_output
    assert "#66707c" in combined_output
    assert "#161b22" in combined_output
    assert re.search(r"=\s*\d+(?:\.\d+)?\s*\(threshold\s*\d+(?:\.\d+)?\)", combined_output)
