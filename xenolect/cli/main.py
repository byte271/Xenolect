"""Friendly product CLI for Xenolect."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import typer
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from xenolect import __version__

app = typer.Typer(
    help="Connect local models to OpenAI-style chat and tool-calling apps.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


@dataclass(frozen=True)
class ModelChoice:
    endpoint: object
    model: str


def _banner(title: str) -> None:
    logo = Text("XENOLECT", style="bold cyan")
    logo.append(f"  {title}", style="bold white")
    console.print(Panel(Align.left(logo), border_style="cyan", padding=(0, 1)))


def _friendly_model_name(model: str) -> str:
    leaf = model.rsplit("/", 1)[-1]
    parts = [part for part in re.split(r"[-_:]+", leaf) if part and part.lower() != "latest"]
    pretty: list[str] = []
    for part in parts:
        if re.fullmatch(r"\d+(?:\.\d+)?[bkmg]", part, flags=re.IGNORECASE):
            pretty.append(part.upper())
        elif re.fullmatch(r"ctx\d+[km]?", part, flags=re.IGNORECASE):
            pretty.append(part[:3].capitalize() + part[3:].upper())
        else:
            pretty.append(part[:1].upper() + part[1:])
    return " ".join(pretty) or model


def _driver_summary(driver) -> str:
    transforms = "+".join(t.value for t in driver.schema_transforms) or "none"
    return (
        f"tools={driver.tool_encoding.value} · parser={driver.parser.value} · "
        f"results={driver.tool_result_encoding.value} · schema={transforms}"
    )


def _source_label(base_url: str, index: int, total: int) -> str:
    try:
        parsed = urlparse(base_url)
        port = parsed.port
    except ValueError:
        port = None
    if total == 1:
        return "Local model server"
    return f"Local server {index}" + (f" · port {port}" if port else "")


def _environment_check(*, verbose: bool) -> Path:
    from xenolect.storage.registry import xenolect_home

    home = xenolect_home()
    py_ok = sys.version_info >= (3, 11)
    if not py_ok:
        console.print(Panel("Python 3.11 or newer is required.", title="Setup problem", border_style="red"))
        raise typer.Exit(code=2)

    try:
        home.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".xenolect-check-", dir=home, delete=True):
            pass
    except OSError as exc:
        console.print(Panel(f"Xenolect cannot write its local data folder:\n{exc}", title="Setup problem", border_style="red"))
        raise typer.Exit(code=2) from exc

    if verbose:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row("Python", f"[green]✓[/green] {sys.version.split()[0]}")
        table.add_row("Data folder", f"[green]✓[/green] {home}")
        table.add_row("Compatibility ABI", "[green]✓[/green] tool-abi-v0")
        console.print(table)
    else:
        console.print("[green]✓[/green] Your computer is ready for Xenolect.")
    return home


def _endpoint_from_prompt(value: str) -> str:
    from xenolect.endpoints.discovery import normalize_endpoint

    try:
        return normalize_endpoint(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _flatten_choices(endpoints) -> list[ModelChoice]:
    choices: list[ModelChoice] = []
    for endpoint in endpoints:
        for model in endpoint.models:
            choices.append(ModelChoice(endpoint=endpoint, model=model))
    return choices


def _choose_model(choices: list[ModelChoice], *, requested: str | None, verbose: bool) -> ModelChoice:
    if requested:
        matches = [choice for choice in choices if choice.model == requested]
        if not matches:
            raise RuntimeError(f"model {requested!r} was not found on the detected server")
        if len(matches) == 1:
            return matches[0]
        # Same model id on multiple endpoints: ask instead of guessing.
        choices = matches

    if not choices:
        raise RuntimeError("no models were found")
    if len(choices) == 1:
        choice = choices[0]
        console.print(f"Found [bold]{_friendly_model_name(choice.model)}[/bold].")
        return choice

    endpoint_order: list[str] = []
    for choice in choices:
        if choice.endpoint.base_url not in endpoint_order:
            endpoint_order.append(choice.endpoint.base_url)

    table = Table(title="Choose a model", border_style="cyan", header_style="bold")
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Model")
    if len(endpoint_order) > 1:
        table.add_column("Source", style="dim")
    if verbose:
        table.add_column("Model id", style="dim")

    for i, choice in enumerate(choices, 1):
        row = [str(i), _friendly_model_name(choice.model)]
        if len(endpoint_order) > 1:
            idx = endpoint_order.index(choice.endpoint.base_url) + 1
            row.append(_source_label(choice.endpoint.base_url, idx, len(endpoint_order)))
        if verbose:
            row.append(choice.model)
        table.add_row(*row)
    console.print(table)

    while True:
        selected = IntPrompt.ask("Select a model", default=1)
        if 1 <= selected <= len(choices):
            return choices[selected - 1]
        console.print("[red]Choose one of the listed numbers.[/red]")


def _interactive_discovery(
    *,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    verbose: bool,
    home: Path | None = None,
) -> ModelChoice:
    from xenolect.endpoints.discovery import candidate_base_urls, inspect_openai_endpoint, scan_openai_endpoints
    from xenolect.storage.registry import DriverRegistry

    registry = DriverRegistry(home)
    banned_pairs = {(item.base_url, item.model) for item in registry.list_banned()}
    interactive = sys.stdin.isatty()
    endpoints = []

    def available_choices() -> list[ModelChoice]:
        return [
            choice
            for choice in _flatten_choices(endpoints)
            if (choice.endpoint.base_url.rstrip("/"), choice.model) not in banned_pairs
        ]

    if base_url:
        url = _endpoint_from_prompt(base_url)
        with console.status("[cyan]Checking your model server…[/cyan]", spinner="dots"):
            try:
                endpoints = [inspect_openai_endpoint(base_url=url, api_key=api_key, timeout=1.8)]
            except Exception as exc:
                if not interactive:
                    raise RuntimeError(f"could not connect to the model server: {exc}") from exc
                if verbose:
                    console.print(f"[yellow]Nothing usable at {url}: {exc}[/yellow]")
    else:
        with console.status("[cyan]Looking for local models…[/cyan]", spinner="dots"):
            endpoints = scan_openai_endpoints(
                base_urls=candidate_base_urls(),
                api_key=api_key,
                timeout=1.0,
            )

    while not available_choices():
        raw_choices = _flatten_choices(endpoints)
        if not interactive:
            if raw_choices:
                raise RuntimeError("all detected models are banned in Xenolect")
            raise RuntimeError("no local model was found; start your model server and run Xenolect again")
        console.print()
        if raw_choices:
            console.print("[yellow]Every model found here is currently banned.[/yellow]")
            console.print("[dim]Run `xenolect ban` to restore one, or try another server.[/dim]")
        else:
            console.print("[yellow]No local model was found.[/yellow]")
        answer = Prompt.ask(
            "Enter another [bold]port[/bold] or [bold]server address[/bold] (q to quit)",
            default="11434",
        ).strip()
        if answer.lower() in {"q", "quit", "exit"}:
            raise typer.Exit(code=1)
        url = _endpoint_from_prompt(answer)
        with console.status("[cyan]Checking that address…[/cyan]", spinner="dots"):
            try:
                endpoints = [inspect_openai_endpoint(base_url=url, api_key=api_key, timeout=1.8)]
            except Exception as exc:
                endpoints = []
                if verbose:
                    console.print(f"[dim]{exc}[/dim]")

    return _choose_model(available_choices(), requested=model, verbose=verbose)


def _copy_to_clipboard(text: str) -> bool:
    commands: list[list[str]] = []
    if sys.platform == "win32":
        commands = [["clip.exe"]]
    elif sys.platform == "darwin":
        commands = [["pbcopy"]]
    elif sys.platform.startswith("linux"):
        if shutil.which("wl-copy"):
            commands = [["wl-copy"]]
        elif shutil.which("xclip"):
            commands = [["xclip", "-selection", "clipboard"]]
        elif shutil.which("xsel"):
            commands = [["xsel", "--clipboard", "--input"]]
    for command in commands:
        try:
            subprocess.run(
                command,
                input=text,
                text=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


def _show_ready(*, model: str, local_url: str, verbose: bool, result, service_state) -> None:
    name = _friendly_model_name(model)
    lines = Text()
    lines.append("✓ Model connected\n", style="green")
    lines.append("✓ Compatibility prepared\n", style="green")
    lines.append("✓ Verified\n", style="green")
    lines.append("✓ Xenolect is running", style="green")
    from xenolect.service import autostart_label

    if service_state.autostart_enabled:
        lines.append(f"\n✓ {autostart_label()} enabled", style="green")
    else:
        lines.append(
            f"\n! {autostart_label()} not enabled; Xenolect is running now",
            style="yellow",
        )
    console.print(Panel(lines, title=f"{name} is ready", border_style="green"))

    if verbose:
        details = Table(show_header=False, box=None, padding=(0, 1))
        details.add_row("Model id", model)
        details.add_row("Local API", local_url)
        details.add_row("Driver", result.installed.driver_hash)
        try:
            details.add_row("Driver program", _driver_summary(result.installed.load()))
        except Exception:
            pass
        details.add_row("Generations", str(result.generations))
        details.add_row("Time", f"{result.elapsed_s:.1f}s")
        if service_state.service_version:
            details.add_row("Service", service_state.service_version)
        console.print(details)

    next_steps = Text()
    next_steps.append("In your app, choose ", style="white")
    next_steps.append("OpenAI-compatible", style="bold")
    next_steps.append(" or ", style="white")
    next_steps.append("Custom OpenAI", style="bold")
    next_steps.append(".\n\n")
    next_steps.append("Address  ", style="dim")
    next_steps.append(local_url + "\n", style="bold cyan")
    next_steps.append("Model    ", style="dim")
    next_steps.append(model + "\n", style="bold cyan")
    next_steps.append("API key  ", style="dim")
    next_steps.append("Any non-empty value", style="bold cyan")
    next_steps.append("  (only if the app requires one)\n")
    next_steps.append("\nYou do not need to run another Xenolect command.", style="white")
    console.print(Panel(next_steps, title="Use it", border_style="cyan"))

    if sys.stdin.isatty() and _copy_to_clipboard(local_url):
        console.print("[dim]Connection address copied to your clipboard.[/dim]")


@app.callback()
def main() -> None:
    """Xenolect CLI."""


@app.command("install")
def install_cmd(
    base_url: Optional[str] = typer.Option(None, help="Model server URL or port", hidden=True),
    model: Optional[str] = typer.Option(None, help="Model id; otherwise choose interactively", hidden=True),
    api_key: Optional[str] = typer.Option(None, envvar="XENOLECT_API_KEY", hidden=True),
    deadline: float = typer.Option(300.0, min=1.0, hidden=True),
    max_generations: int = typer.Option(12, min=3, hidden=True),
    force: bool = typer.Option(False, "--force", help="Rebuild compatibility for this model", hidden=True),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show technical details"),
) -> None:
    """Find a model, prepare compatibility, and make Xenolect ready to use."""
    _banner("Install")
    home = _environment_check(verbose=verbose)

    try:
        choice = _interactive_discovery(
            base_url=base_url,
            model=model,
            api_key=api_key,
            verbose=verbose,
            home=home,
        )
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(Panel(str(exc), title="Could not find a model", border_style="red"))
        raise typer.Exit(code=2) from exc

    selected_model = choice.model
    endpoint = choice.endpoint
    console.print()
    console.print(f"Preparing [bold]{_friendly_model_name(selected_model)}[/bold]…")

    from xenolect.compiler.install import READY, ResolvedTarget, install_target

    target = ResolvedTarget.from_discovery(endpoint, selected_model)
    with console.status("[bold cyan]Setting up compatibility…[/bold cyan]", spinner="dots"):
        try:
            result = install_target(
                target,
                api_key=api_key,
                deadline_s=deadline,
                max_generations=max_generations,
                force=force,
            )
        except Exception as exc:
            console.print(Panel(str(exc), title="Setup failed", border_style="red"))
            raise typer.Exit(code=2) from exc

    if result.status != READY or result.installed is None:
        if verbose:
            reason = result.reason
        elif result.status == "ENDPOINT_TOO_SLOW_FOR_CERTIFICATION":
            reason = "This model is responding too slowly to finish setup within five minutes."
        elif result.status == "BUDGET_EXHAUSTED":
            reason = "Xenolect could not finish compatibility setup within the safe setup budget."
        else:
            reason = "Xenolect could not verify compatibility with this model yet."
        console.print(Panel(reason, title="Model not ready", border_style="red"))
        if result.report_path is not None:
            console.print(f"Diagnostic report: [bold]{result.report_path}[/bold]")
        if verbose:
            console.print(f"[dim]status={result.status} generations={result.generations} time={result.elapsed_s:.1f}s[/dim]")
        raise typer.Exit(code=3)

    from xenolect.service import ServiceError, ensure_background_service

    try:
        with console.status("[cyan]Starting Xenolect…[/cyan]", spinner="dots"):
            service_state = ensure_background_service(home=home, enable_autostart=True)
    except ServiceError as exc:
        console.print(
            Panel(
                "Your model is prepared, but Xenolect could not start automatically.\n"
                f"{exc}\n\nRun `xenolect install` again after fixing the issue.",
                title="Almost ready",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=4) from exc

    _show_ready(
        model=selected_model,
        local_url=service_state.base_url,
        verbose=verbose,
        result=result,
        service_state=service_state,
    )


@app.command("kill")
def kill_cmd() -> None:
    """Stop Xenolect and disable automatic startup."""
    _banner("Kill")
    from xenolect.service import ServiceError, stop_background_service

    try:
        state = stop_background_service(disable_autostart=True)
    except ServiceError as exc:
        console.print(Panel(str(exc), title="Could not stop Xenolect", border_style="red"))
        raise typer.Exit(code=2) from exc

    message = Text()
    message.append("✓ Xenolect is stopped.\n", style="green")
    from xenolect.service import autostart_label

    if state.autostart_enabled:
        message.append(f"! {autostart_label()} could not be disabled.\n", style="yellow")
    else:
        message.append(f"✓ {autostart_label()} is off.\n", style="green")
    message.append("\nYour prepared models were kept. Run ", style="white")
    message.append("xenolect install", style="bold cyan")
    message.append(" whenever you want to start again.", style="white")
    console.print(Panel(message, border_style="green"))


@app.command("ban")
def ban_cmd() -> None:
    """Ban a prepared model, or restore one that is already banned."""
    _banner("Ban")
    from xenolect.storage.registry import DriverRegistry, RegistryError

    registry = DriverRegistry()
    try:
        installed = registry.list(include_banned=True)
        banned_entries = registry.list_banned()
    except RegistryError as exc:
        console.print(Panel(str(exc), title="Could not read models", border_style="red"))
        raise typer.Exit(code=2) from exc

    banned_pairs = {(item.base_url, item.model) for item in banned_entries}
    by_key: dict[tuple[str, str], tuple[str, str, bool]] = {}
    for item in installed:
        key = (item.base_url, item.model)
        by_key[key] = (item.base_url, item.model, key in banned_pairs)
    for item in banned_entries:
        key = (item.base_url, item.model)
        by_key.setdefault(key, (item.base_url, item.model, True))

    choices = sorted(by_key.values(), key=lambda value: (_friendly_model_name(value[1]).lower(), value[0]))
    if not choices:
        console.print("No model is prepared yet. Run [bold]xenolect install[/bold] first.")
        return

    duplicate_ids = len({model_id for _, model_id, _ in choices}) != len(choices)
    table = Table(title="Models", border_style="cyan", header_style="bold")
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Model")
    if duplicate_ids:
        table.add_column("Source", style="dim")
    table.add_column("Status")
    for index, (base_url, model_id, banned) in enumerate(choices, 1):
        row = [str(index), _friendly_model_name(model_id)]
        if duplicate_ids:
            row.append(base_url)
        row.append("[red]Banned[/red]" if banned else "[green]Available[/green]")
        table.add_row(*row)
    console.print(table)

    while True:
        selected = IntPrompt.ask("Select a model", default=1)
        if 1 <= selected <= len(choices):
            break
        console.print("[red]Choose one of the listed numbers.[/red]")

    base_url, model_id, is_banned = choices[selected - 1]
    name = _friendly_model_name(model_id)
    if is_banned:
        if not Confirm.ask(f"Restore {name}?", default=True):
            return
        registry.unban(base_url, model_id)
        console.print(Panel(f"[green]✓[/green] {name} is available again.", border_style="green"))
    else:
        if not Confirm.ask(f"Ban {name}?", default=False):
            return
        registry.ban(base_url, model_id)
        console.print(
            Panel(
                f"[green]✓[/green] {name} is banned.\n\n"
                "Xenolect will no longer advertise or use this model. "
                "Run [bold]xenolect ban[/bold] again if you want to restore it.",
                border_style="green",
            )
        )


@app.command("status")
def status_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show technical details"),
) -> None:
    """Show whether Xenolect is ready."""
    _banner("Status")
    from xenolect.service import current_service_state
    from xenolect.storage.registry import DriverRegistry, RegistryError

    try:
        items = DriverRegistry().list()
        service = current_service_state()
    except (RegistryError, RuntimeError) as exc:
        console.print(Panel(str(exc), title="Status problem", border_style="red"))
        raise typer.Exit(code=2) from exc

    outdated_service = service.running and service.service_version != __version__
    if outdated_service:
        state = "[yellow]● Running · update needed[/yellow]"
    else:
        state = "[bold green]● Running[/bold green]" if service.running else "[yellow]○ Not running[/yellow]"
    console.print(state)
    if outdated_service:
        console.print("Run [bold]xenolect install[/bold] once to restart the background service with this version.")
    if not items:
        console.print("No model is prepared yet. Run [bold]xenolect install[/bold].")
        return

    table = Table(title="Prepared models", border_style="cyan")
    table.add_column("Model")
    if verbose:
        table.add_column("Endpoint", style="dim")
        table.add_column("Driver", style="dim")
        table.add_column("Driver program", style="dim")
    for item in items:
        row = [_friendly_model_name(item.model)]
        if verbose:
            try:
                program = _driver_summary(item.load())
            except Exception:
                program = "unreadable"
            row.extend([item.base_url, item.driver_hash, program])
        table.add_row(*row)
    console.print(table)
    try:
        banned = DriverRegistry().list_banned()
    except RegistryError:
        banned = []
    if banned:
        console.print(f"[dim]{len(banned)} model(s) banned · run `xenolect ban` to manage them[/dim]")
    if service.running:
        console.print(f"[dim]Connection address: {service.base_url}[/dim]")
        if verbose:
            from xenolect.service import autostart_label

            startup_state = "enabled" if service.autostart_enabled else "not enabled"
            console.print(f"[dim]{autostart_label()}: {startup_state}[/dim]")
            if service.service_version:
                console.print(f"[dim]Background service: {service.service_version}[/dim]")


@app.command("version")
def version_cmd() -> None:
    """Show the installed Xenolect version."""
    console.print(f"xenolect {__version__}")


if __name__ == "__main__":
    app()
