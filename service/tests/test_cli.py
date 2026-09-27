import subprocess
import sys
from pathlib import Path

from app.main import run


def test_run_without_paths_prints_help(capfd):
    assert run([]) == 0
    stdout, _ = capfd.readouterr()
    assert "Remove a video background" in stdout


def test_main_py_can_be_executed_directly():
    container_main = Path("/app/main.py")
    main_py = container_main if container_main.exists() else Path(__file__).parents[1] / "app" / "main.py"
    result = subprocess.run(
        [sys.executable, str(main_py), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Remove a video background" in result.stdout


def test_display_aspect_option_is_in_help():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "app" / "main.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--display-aspect" in result.stdout
