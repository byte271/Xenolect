from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from xenolect import __version__
from xenolect.cli.main import app

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


def test_release_version_is_consistent() -> None:
    assert __version__ == "0.3.0"
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.3.0"' in pyproject
    readme = Path("README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Xenolect v0.3.0\n")
    assert "xenolect-0.3.0-py3-none-any.whl" in readme
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
    assert "## 0.3.0 - 2026-08-09" in changelog


def test_normal_cli_surface_does_not_expose_internal_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    output = _plain(result.stdout)
    for command in ("install", "kill", "ban", "status", "version"):
        assert command in output
    for internal in ("serve", "xpt", "compile", "eval", "mock", "gate"):
        assert internal not in output.lower()


def test_install_help_only_exposes_user_option() -> None:
    result = CliRunner().invoke(app, ["install", "--help"])
    assert result.exit_code == 0
    output = _plain(result.stdout)
    assert "--verbose" in output
    for internal in (
        "--base-url",
        "--model",
        "--api-key",
        "--deadline",
        "--max-generations",
        "--force",
    ):
        assert internal not in output


def test_release_metadata_declares_three_supported_desktop_platforms() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "Operating System :: Microsoft :: Windows" in pyproject
    assert "Operating System :: MacOS" in pyproject
    assert "Operating System :: POSIX :: Linux" in pyproject


def test_readme_is_truthful_about_finite_driver_grammar() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "144 representable Driver programs" in readme
    assert "does not synthesize arbitrary" in readme
    assert "does not invent application tools" in readme
