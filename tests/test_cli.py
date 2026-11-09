import subprocess
import sys


def test_cli_writes_utf8_json():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.cli",
            "总结这篇中文文章",
            "--max-candidates",
            "1",
        ],
        capture_output=True,
        check=True,
    )

    output = completed.stdout.decode("utf-8")
    assert "总结这篇中文文章" in output
    assert '"execution_mode"' in output

