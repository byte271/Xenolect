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
    assert __version__ == "0.5.0"
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.5.0"' in pyproject
    readme = Path("README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Xenolect v0.5.0\n")
    assert "xenolect-0.5.0-py3-none-any.whl" in readme
    assert "releases/tag/v0.5.0" in readme
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
    assert "## 0.5.0 - 2026-08-09" in changelog


def test_release_workflow_is_version_bound_and_guarded() -> None:
    workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "workflow_run:" in workflow
    assert "github.event.workflow_run.conclusion == 'success'" in workflow
    assert "github.event.workflow_run.event == 'push'" in workflow
    assert "github.event.workflow_run.head_repository.full_name == github.repository" in workflow
    assert "github.event.workflow_run.head_branch == 'main'" in workflow
    assert "ref: ${{ github.event.workflow_run.head_sha }}" in workflow
    assert 'release_commit_message="$(git log -1 --pretty=%B)"' in workflow
    assert 'tomllib.load(open("pyproject.toml", "rb"))' in workflow
    assert "contents: write" in workflow
    assert 'gh release create "v${VERSION}"' in workflow
    assert '--target "$VERIFIED_SHA"' in workflow
    assert "python -m build" in workflow
    assert "Smoke-test wheel" in workflow
    assert "Delete fully merged agent branches" in workflow
    assert 'startswith("agent/")' in workflow
    assert '--state open' in workflow
    assert '--state merged' in workflow
    assert '--json headRefOid' in workflow
    assert 'select(.headRefOid == \"$branch_sha\")' in workflow
    assert 'if [ "$merged_head" != "$branch_sha" ]' in workflow
    assert 'git/refs/heads/${branch}' in workflow


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


def test_user_and_certification_docs_state_the_bounded_product_boundary() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    certification = Path("docs/CERTIFICATION.md").read_text(encoding="utf-8")
    assert "universal or arbitrary protocol synthesis" in readme
    assert "automatic creation or execution of application tools" in readme
    assert "Driver IR remains v0.2" in certification
    assert "does not add provider rules" in certification
