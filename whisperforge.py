#!/usr/bin/env python3
"""
WhisperForge — Batch Audio/Video Transcription
Powered by faster-whisper
"""

import argparse
import sys
import time
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn,
        TextColumn, TimeElapsedColumn, MofNCompleteColumn,
    )
    from rich.text import Text
    from rich import box
    from rich.columns import Columns
except ImportError:
    print("ERROR: 'rich' not installed.  pip install rich")
    sys.exit(1)

console = Console()

VERSION              = "2.0"
DEFAULT_MODEL_SIZE   = "large-v3"
DEFAULT_LANGUAGE     = "de"
SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg",
}


# ── UI helpers ─────────────────────────────────────────────────────────────────

def print_banner():
    title = Text.assemble(
        ("  🎙  ", ""),
        ("WhisperForge", "bold cyan"),
        (f"  v{VERSION}", "dim cyan"),
        ("  ", ""),
    )
    subtitle = Text.assemble(
        ("  Batch Audio / Video  →  Text  ", "white"),
        ("·  faster-whisper", "dim"),
        ("  ", ""),
    )
    content = Text("\n").join([title, subtitle])
    console.print(Panel(content, border_style="cyan", box=box.DOUBLE_EDGE,
                        expand=False, padding=(0, 2)))
    console.print()


def print_config(args, root_path: Path, output_dir: Path | None, extensions: set[str]):
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="dim", justify="right")
    grid.add_column(style="cyan")
    grid.add_row("root",       str(root_path))
    grid.add_row("output",     str(output_dir) if output_dir else "[dim](next to source)[/]")
    grid.add_row("model",      args.model)
    grid.add_row("language",   args.language)
    grid.add_row("format",     args.format)
    grid.add_row("extensions", "  ".join(sorted(extensions)))
    console.print(Panel(grid, title="[bold dim]Configuration[/]", border_style="dim",
                        box=box.ROUNDED, expand=False))
    console.print()


# ── Core logic ─────────────────────────────────────────────────────────────────

def find_media_files(root: Path, extensions: set[str]) -> list[Path]:
    if not root.exists():
        console.print(f"[bold red]✗  Directory not found:[/] {root}")
        sys.exit(1)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions)


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def to_srt_timestamp(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list) -> str:
    blocks = []
    for i, seg in enumerate(segments, 1):
        start = to_srt_timestamp(seg.start)
        end   = to_srt_timestamp(seg.end)
        blocks.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}")
    return "\n\n".join(blocks)


def transcribe_file(model, path: Path, language: str, fmt: str, batched: bool) -> tuple[str, int]:
    kwargs = dict(language=language, beam_size=5)
    if batched:
        kwargs["batch_size"] = 8
    segments, _ = model.transcribe(str(path), **kwargs)
    segments = list(segments)
    if fmt == "srt":
        text  = segments_to_srt(segments)
        words = sum(len(seg.text.split()) for seg in segments)
    else:
        text  = " ".join(seg.text.strip() for seg in segments).strip()
        words = len(text.split())
    return text, words


def resolve_output_path(src: Path, suffix: str, output_dir: Path | None) -> Path:
    if output_dir:
        return output_dir / src.with_suffix(suffix).name
    return src.with_suffix(suffix)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print_banner()

    parser = argparse.ArgumentParser(
        description="WhisperForge — batch-transcribe audio/video with faster-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Supported: " + "  ".join(sorted(SUPPORTED_EXTENSIONS)),
    )
    parser.add_argument("--root-dir",   required=True,  metavar="DIR",
                        help="Folder to scan for media files")
    parser.add_argument("--output-dir", default=None,   metavar="DIR",
                        help="Save transcripts here (default: next to source file)")
    parser.add_argument("--model",      default=DEFAULT_MODEL_SIZE, metavar="SIZE",
                        choices=["tiny", "base", "small", "medium", "large",
                                 "large-v2", "large-v3", "distil-large-v3"],
                        help=f"Whisper model size (default: {DEFAULT_MODEL_SIZE})")
    parser.add_argument("--language",   default=DEFAULT_LANGUAGE,   metavar="LANG",
                        help=f"ISO 639-1 language code (default: {DEFAULT_LANGUAGE})")
    parser.add_argument("--format",     default="txt",  choices=["txt", "srt"],
                        help="Output format: txt (plain) or srt (subtitles) — default: txt")
    parser.add_argument("--extensions", default=None,   metavar="EXT,EXT",
                        help="Comma-separated extensions to scan (default: all supported)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="List discovered files without transcribing")
    parser.add_argument("--force",      action="store_true",
                        help="Re-transcribe even if output file already exists")
    args = parser.parse_args()

    root_path  = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    suffix     = f".{args.format}"
    extensions = (
        {f".{e.strip().lstrip('.')}" for e in args.extensions.split(",")}
        if args.extensions else SUPPORTED_EXTENSIONS
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print_config(args, root_path, output_dir, extensions)

    # ── Discover files ─────────────────────────────────────────────────────────
    with console.status("[cyan]Scanning for media files …[/]"):
        files = find_media_files(root_path, extensions)

    if not files:
        console.print(Panel("[yellow]No media files found — nothing to do.[/]",
                            border_style="yellow", box=box.ROUNDED))
        sys.exit(0)

    done_count    = sum(1 for f in files
                        if resolve_output_path(f, suffix, output_dir).exists())
    pending_count = len(files) - done_count

    console.print(
        f"  Found [bold cyan]{len(files)}[/] file(s)  "
        f"[green]({done_count} done)[/]  "
        f"[yellow]({pending_count} pending)[/]\n"
    )

    table = Table(
        box=box.ROUNDED, border_style="cyan",
        header_style="bold cyan", show_lines=False,
    )
    table.add_column("#",      style="dim",   justify="right", width=4)
    table.add_column("File",   style="white")
    table.add_column("Output", style="dim",   overflow="fold")
    table.add_column("Status", justify="center", width=11)

    for i, f in enumerate(files, 1):
        out = resolve_output_path(f, suffix, output_dir)
        try:
            label = str(f.relative_to(root_path))
        except ValueError:
            label = str(f)
        out_label = str(out.relative_to(root_path)) if not output_dir else out.name
        status = (Text("✓  done",    style="bold green")
                  if out.exists() else
                  Text("·  pending", style="yellow"))
        table.add_row(str(i), label, out_label, status)

    console.print(table)
    console.print()

    if args.dry_run:
        console.print("[dim]--dry-run: exiting without transcribing.[/]")
        sys.exit(0)

    pending = [
        (f, resolve_output_path(f, suffix, output_dir))
        for f in files
        if args.force or not resolve_output_path(f, suffix, output_dir).exists()
    ]

    if not pending:
        console.print(Panel(
            "[green]All files already transcribed.[/]  Pass [cyan]--force[/] to redo them.",
            border_style="green", box=box.ROUNDED,
        ))
        sys.exit(0)

    # ── Load model ─────────────────────────────────────────────────────────────
    try:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
    except ImportError:
        console.print("[bold red]✗  faster-whisper not installed.[/]  pip install faster-whisper")
        sys.exit(1)

    import torch
    if torch.cuda.is_available():
        device, compute = "cuda", "float16"
        device_label    = torch.cuda.get_device_name(0)
        batched         = True
        console.print(f"  [bold green]✔  GPU[/]  [dim]—[/]  [cyan]{device_label}[/]")
    else:
        device, compute = "cpu", "int8"
        batched         = False
        console.print("  [bold yellow]⚠  No CUDA GPU detected — running on CPU [dim](slow!)[/dim][/]")

    with console.status(
        f"[cyan]Loading model [bold]{args.model}[/] …  "
        f"[dim](first run downloads weights, be patient)[/][/]"
    ):
        model = WhisperModel(args.model, device=device, compute_type=compute)
        if batched:
            model = BatchedInferencePipeline(model=model)

    console.print(f"  [green]✔  Model [bold]{args.model}[/] ready[/]\n")

    # ── Transcribe ─────────────────────────────────────────────────────────────
    total_start = time.time()
    failed      = []
    total_words = 0

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[cyan]{task.description}", justify="left"),
        BarColumn(bar_width=28, style="dim cyan", complete_style="cyan"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        overall = progress.add_task(f"[dim]{pending[0][0].name}[/]", total=len(pending))

        for src, out in pending:
            progress.update(overall, description=f"[white]{src.name}[/]")
            t0 = time.time()
            try:
                text, words = transcribe_file(model, src, args.language, args.format, batched)
                out.write_text(text, encoding="utf-8-sig")
                elapsed      = time.time() - t0
                total_words += words
                progress.console.print(
                    f"  [bold green]✔[/]  [white]{src.name}[/]  "
                    f"[dim]·  {format_duration(elapsed)}  ·  {words:,} words[/]"
                )
            except Exception as exc:
                elapsed = time.time() - t0
                progress.console.print(
                    f"  [bold red]✗[/]  [white]{src.name}[/]  "
                    f"[dim red]failed after {format_duration(elapsed)}:[/]  {exc}"
                )
                failed.append((src, str(exc)))
            progress.advance(overall)

    # ── Summary ────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    ok            = len(pending) - len(failed)

    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="dim",        justify="right")
    grid.add_column(style="bold white")
    grid.add_row("Duration",    format_duration(total_elapsed))
    grid.add_row("Transcribed", f"[bold green]{ok}[/] / {len(pending)}")
    grid.add_row("Total words", f"{total_words:,}")
    if failed:
        grid.add_row("Failed", f"[bold red]{len(failed)}[/]")

    border = "green" if not failed else "yellow"
    console.print()
    console.print(Panel(grid, title="[bold]  Session Summary  [/]",
                        border_style=border, box=box.ROUNDED, expand=False))

    if failed:
        console.print()
        err_table = Table(box=box.SIMPLE, header_style="bold red", show_header=True)
        err_table.add_column("File",  style="white")
        err_table.add_column("Error", style="dim red")
        for path, err in failed:
            err_table.add_row(path.name, err)
        console.print(err_table)

    console.print()


if __name__ == "__main__":
    main()
