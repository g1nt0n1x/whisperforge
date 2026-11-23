#!/usr/bin/env python3
"""
WhisperForge — Batch Audio/Video Transcription
Powered by faster-whisper
"""

import argparse
import glob
import sys
import time
from collections import Counter
from pathlib import Path

try:
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn,
        TextColumn, TimeElapsedColumn, MofNCompleteColumn,
    )
    from rich.text import Text
    from rich import box
except ImportError:
    print("ERROR: 'rich' not installed.  pip install -r requirements.txt")
    sys.exit(1)

console = Console()

VERSION              = "2.1"
DEFAULT_MODEL_SIZE   = "large-v3"
DEFAULT_LANGUAGE     = "de"
DEFAULT_BATCH_SIZE   = 8
SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus",
}
FALLBACK_MODELS = [
    "tiny", "base", "small", "medium", "large", "large-v2", "large-v3",
    "large-v3-turbo", "turbo", "distil-large-v3",
]


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


MAX_CONFIG_INPUTS = 5


def print_config(args, output_dir: Path | None, extensions: set[str]):
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="dim", justify="right")
    grid.add_column(style="cyan")
    shown = args.inputs[:MAX_CONFIG_INPUTS]
    for i, raw in enumerate(shown):
        grid.add_row("input" if i == 0 else "", escape(raw))
    if len(args.inputs) > len(shown):
        grid.add_row("", f"[dim]… and {len(args.inputs) - len(shown)} more[/]")
    grid.add_row("output",     escape(str(output_dir)) if output_dir else "[dim](next to source)[/]")
    grid.add_row("model",      args.model)
    grid.add_row("language",   escape(args.language))
    grid.add_row("format",     args.format)
    grid.add_row("extensions", "  ".join(sorted(extensions)))
    if args.prompt:
        grid.add_row("prompt", escape(args.prompt))
    console.print(Panel(grid, title="[bold dim]Configuration[/]", border_style="dim",
                        box=box.ROUNDED, expand=False))
    console.print()


# ── Core logic ─────────────────────────────────────────────────────────────────

def available_models() -> list[str]:
    try:
        from faster_whisper.utils import available_models as fw_models
        return list(fw_models())
    except Exception:
        return FALLBACK_MODELS


def parse_extensions(raw: str | None) -> set[str]:
    if not raw:
        return set(SUPPORTED_EXTENSIONS)
    exts = {f".{e.strip().lstrip('.').lower()}" for e in raw.split(",") if e.strip().lstrip(".")}
    if not exts:
        console.print("[bold red]✗  --extensions contains no valid entries.[/]")
        sys.exit(2)
    return exts


def fail(message: str, code: int = 1):
    """Print an error and exit. `message` is rich markup; escape user data in it."""
    console.print(f"[bold red]✗[/]  {message}")
    sys.exit(code)


def find_media_files(root: Path, extensions: set[str]) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.suffix.lower() in extensions and p.is_file())


def collect_sources(raw_inputs: list[str], extensions: set[str]) -> list[tuple[Path, Path]]:
    """Resolve CLI inputs to sorted (media file, base folder) pairs.

    Folders are scanned recursively and act as the base, so --output-dir mirrors
    their subfolders. A single file's base is its own folder. Wildcards are expanded
    here because cmd.exe and PowerShell pass them through literally.
    """
    found: dict[Path, Path] = {}

    def add_folder(folder: Path):
        for f in find_media_files(folder, extensions):
            found.setdefault(f, folder)

    for raw in raw_inputs:
        path = Path(raw).expanduser()
        if not path.exists() and any(c in raw for c in "*?["):
            matches = [Path(m).resolve() for m in glob.glob(str(path), recursive=True)]
            if not matches:
                console.print(f"  [yellow]⚠  Nothing matches[/] {escape(raw)}")
            for m in matches:
                if m.is_dir():
                    add_folder(m)
                elif m.suffix.lower() in extensions:
                    found.setdefault(m, m.parent)
            continue

        path = path.resolve()
        if path.is_dir():
            add_folder(path)
        elif path.is_file():
            if path.suffix.lower() not in extensions | SUPPORTED_EXTENSIONS:
                fail(f"Unsupported file type: {escape(str(path))}  "
                     f"[dim](add it with --extensions {escape(path.suffix.lstrip('.'))})[/]", 2)
            found.setdefault(path, path.parent)
        else:
            fail(f"Not found: {escape(raw)}")
    return sorted(found.items())


def plan_jobs(sources: list[tuple[Path, Path]], suffix: str,
              output_dir: Path | None) -> list[tuple[Path, Path, Path]]:
    """Return (source, base, output) triples, making sure no two sources share an output."""
    jobs = [(src, base, resolve_output_path(src, base, suffix, output_dir))
            for src, base in sources]

    # talk.mp4 and talk.mp3 in one folder would both become talk.txt and overwrite each other.
    # Name those after the full source file instead: talk.mp4.txt, talk.mp3.txt.
    counts = Counter(out for _, _, out in jobs)
    jobs = [(src, base, out.with_name(src.name + suffix) if counts[out] > 1 else out)
            for src, base, out in jobs]

    counts = Counter(out for _, _, out in jobs)
    clashes = sorted({out for out, n in counts.items() if n > 1})
    if clashes:
        lines = "\n".join(f"    {escape(str(c))}" for c in clashes)
        fail(f"Several inputs would write the same transcript:\n{lines}\n"
             f"  [dim]Transcribe them separately or drop --output-dir.[/]", 2)
    for src, _, out in jobs:
        if out == src:
            fail(f"Transcript would overwrite its own source: {escape(str(src))}", 2)
    return jobs


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
    # Round once on the total, so 1.9996 s becomes 00:00:02,000 instead of 00:00:01,1000.
    total_ms = max(0, int(round(seconds * 1000)))
    h, rest  = divmod(total_ms, 3_600_000)
    m, rest  = divmod(rest, 60_000)
    s, ms    = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


SRT_MAX_CHARS    = 84     # ~2 lines of 42 characters
SRT_MAX_SECONDS  = 6.0
SRT_MIN_CHARS    = 20     # don't end a cue on punctuation before this many chars


def build_cues(segments: list) -> list[tuple[float, float, str]]:
    """Split segments into subtitle-sized cues using word timestamps.

    Segments from the batched pipeline can be ~30 s long, far too long for subtitles.
    """
    cues = []
    for seg in segments:
        words = getattr(seg, "words", None)
        if not words:
            cues.append((seg.start, seg.end, seg.text.strip()))
            continue
        cur, start = "", None
        for w in words:
            token = w.word
            if start is not None and (len(cur + token) > SRT_MAX_CHARS
                                      or w.end - start > SRT_MAX_SECONDS):
                cues.append((start, prev_end, cur.strip()))
                cur, start = "", None
            if start is None:
                start = w.start
            cur, prev_end = cur + token, w.end
            if cur.rstrip().endswith((".", "?", "!")) and len(cur.strip()) >= SRT_MIN_CHARS:
                cues.append((start, prev_end, cur.strip()))
                cur, start = "", None
        if start is not None and cur.strip():
            cues.append((start, prev_end, cur.strip()))
    return cues


def segments_to_srt(segments: list) -> str:
    blocks = [f"{i}\n{to_srt_timestamp(s)} --> {to_srt_timestamp(e)}\n{t}"
              for i, (s, e, t) in enumerate(build_cues(segments), 1)]
    return "\n\n".join(blocks) + "\n"


def transcribe_file(model, path: Path, args, batched: bool, on_progress) -> tuple[str, int]:
    kwargs = dict(
        language=None if args.language == "auto" else args.language,
        beam_size=5,
        vad_filter=True,               # skip silence: faster and fewer hallucinations
        initial_prompt=args.prompt or None,
        word_timestamps=args.format == "srt",   # needed to split into short subtitle cues
    )
    if batched:
        kwargs["batch_size"] = args.batch_size
    segments_iter, info = model.transcribe(str(path), **kwargs)

    segments = []
    for seg in segments_iter:          # generator: decoding happens while iterating
        segments.append(seg)
        on_progress(seg.end, info.duration)

    if args.format == "srt":
        text = segments_to_srt(segments)
    else:
        text = " ".join(seg.text.strip() for seg in segments).strip() + "\n"
    words = sum(len(seg.text.split()) for seg in segments)
    return text, words


def resolve_output_path(src: Path, base: Path, suffix: str, output_dir: Path | None) -> Path:
    if output_dir:
        # Mirror the folder structure so equal names in different subfolders don't collide.
        return output_dir / src.relative_to(base).with_suffix(suffix)
    return src.with_suffix(suffix)


def write_atomic(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8-sig")
    tmp.replace(path)


def load_model(model_name: str):
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    # On Windows, importing torch (if installed) registers its bundled cuBLAS/cuDNN DLLs,
    # which CTranslate2 needs for CUDA. It is optional otherwise.
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = None

    import ctranslate2
    if ctranslate2.get_cuda_device_count() > 0:
        try:
            with console.status(f"[cyan]Loading model [bold]{model_name}[/] on GPU …  "
                                f"[dim](first run downloads weights, be patient)[/][/]"):
                model = WhisperModel(model_name, device="cuda", compute_type="float16")
            name = torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else "CUDA"
            console.print(f"  [bold green]✔  GPU[/]  [dim]—[/]  [cyan]{name}[/]")
            return BatchedInferencePipeline(model=model), True
        except Exception as exc:
            console.print(f"  [bold yellow]⚠  GPU init failed, falling back to CPU:[/] [dim]{exc}[/]")
    else:
        console.print("  [bold yellow]⚠  No CUDA GPU detected — running on CPU [dim](slow!)[/dim][/]")

    with console.status(f"[cyan]Loading model [bold]{model_name}[/] on CPU …[/]"):
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
    return model, False


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="WhisperForge — batch-transcribe audio/video with faster-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Supported: " + "  ".join(sorted(SUPPORTED_EXTENSIONS)) + "\n\n"
               "Examples:\n"
               "  whisperforge interview.mp4\n"
               "  whisperforge a.mp3 b.wav --format srt\n"
               "  whisperforge \"D:\\Videos\" --output-dir \"D:\\Transcripts\"\n"
               "  whisperforge \"D:\\Podcasts\\*.mp3\" --language en",
    )
    parser.add_argument("inputs",       nargs="*",      metavar="PATH",
                        help="Media files and/or folders (scanned recursively); wildcards like *.mp3 work")
    parser.add_argument("--root-dir",   action="append", default=[], metavar="DIR",
                        help="Folder to scan; same as passing it as PATH (can be repeated)")
    parser.add_argument("--output-dir", default=None,   metavar="DIR",
                        help="Save transcripts here, mirroring subfolders (default: next to source)")
    parser.add_argument("--model",      default=DEFAULT_MODEL_SIZE, metavar="SIZE",
                        choices=available_models(),
                        help=f"Whisper model (default: {DEFAULT_MODEL_SIZE})")
    parser.add_argument("--language",   default=DEFAULT_LANGUAGE,   metavar="LANG",
                        help=f"ISO 639-1 code, or 'auto' to detect (default: {DEFAULT_LANGUAGE})")
    parser.add_argument("--format",     default="txt",  choices=["txt", "srt"],
                        help="Output format: txt (plain) or srt (subtitles) — default: txt")
    parser.add_argument("--extensions", default=None,   metavar="EXT,EXT",
                        help="Comma-separated extensions to scan (default: all supported)")
    parser.add_argument("--prompt",     default=None,   metavar="TEXT",
                        help="Initial prompt with names/jargon to improve recognition")
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int, metavar="N",
                        help=f"GPU batch size; lower it if VRAM runs out (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--dry-run",    action="store_true",
                        help="List discovered files without transcribing")
    parser.add_argument("--force",      action="store_true",
                        help="Re-transcribe even if output file already exists")
    # Intermixed so options may follow the paths: whisperforge a.mp3 --format srt b.mp3
    args = parser.parse_intermixed_args()
    args.inputs = args.root_dir + args.inputs
    if not args.inputs:
        parser.error("give at least one file or folder to transcribe")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    return args


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print_banner()
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    suffix     = f".{args.format}"
    extensions = parse_extensions(args.extensions)

    print_config(args, output_dir, extensions)

    # ── Discover files ─────────────────────────────────────────────────────────
    with console.status("[cyan]Scanning for media files …[/]"):
        sources = collect_sources(args.inputs, extensions)

    if not sources:
        console.print(Panel("[yellow]No media files found — nothing to do.[/]",
                            border_style="yellow", box=box.ROUNDED))
        sys.exit(0)

    jobs    = plan_jobs(sources, suffix, output_dir)
    done    = {src for src, _, out in jobs if out.exists()}
    pending = [(src, out) for src, _, out in jobs if args.force or src not in done]

    console.print(
        f"  Found [bold cyan]{len(jobs)}[/] file(s)  "
        f"[green]({len(jobs) - len(pending)} done)[/]  "
        f"[yellow]({len(pending)} to transcribe)[/]\n"
    )

    table = Table(box=box.ROUNDED, border_style="cyan",
                  header_style="bold cyan", show_lines=False)
    table.add_column("#",      style="dim",   justify="right", width=4)
    table.add_column("File",   style="white")
    table.add_column("Output", style="dim",   overflow="fold")
    table.add_column("Status", justify="center", width=11)

    for i, (src, base, out) in enumerate(jobs, 1):
        # Text objects, not markup strings: file names may contain [brackets].
        out_label = Text(str(out.relative_to(output_dir or base)))
        if src not in done:
            status = Text("·  pending", style="yellow")
        elif args.force:
            status = Text("↻  redo",    style="yellow")
        else:
            status = Text("✓  done",    style="bold green")
        table.add_row(str(i), Text(str(src.relative_to(base))), out_label, status)

    console.print(table)
    console.print()

    if args.dry_run:
        console.print("[dim]--dry-run: exiting without transcribing.[/]")
        sys.exit(0)

    if not pending:
        console.print(Panel(
            "[green]All files already transcribed.[/]  Pass [cyan]--force[/] to redo them.",
            border_style="green", box=box.ROUNDED,
        ))
        sys.exit(0)

    # ── Load model ─────────────────────────────────────────────────────────────
    try:
        model, batched = load_model(args.model)
    except ImportError:
        console.print("[bold red]✗  faster-whisper not installed.[/]  pip install -r requirements.txt")
        sys.exit(1)

    console.print(f"  [green]✔  Model [bold]{args.model}[/] ready[/]\n")

    # ── Transcribe ─────────────────────────────────────────────────────────────
    total_start = time.time()
    failed      = []
    ok          = 0
    total_words = 0
    interrupted = False

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[cyan]{task.description}", justify="left"),
        BarColumn(bar_width=28, style="dim cyan", complete_style="cyan"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        overall = progress.add_task("[bold]Total[/]", total=len(pending))
        current = progress.add_task("", total=1, visible=False)

        def on_progress(pos: float, duration: float):
            progress.update(current, completed=min(pos, duration), total=max(duration, 1e-6))

        try:
            for src, out in pending:
                name = escape(src.name)
                progress.reset(current, total=1, description=f"[white]{name}[/]", visible=True)
                t0 = time.time()
                try:
                    text, words = transcribe_file(model, src, args, batched, on_progress)
                    write_atomic(out, text)
                    ok          += 1
                    total_words += words
                    progress.console.print(
                        f"  [bold green]✔[/]  [white]{name}[/]  "
                        f"[dim]·  {format_duration(time.time() - t0)}  ·  {words:,} words[/]"
                    )
                except Exception as exc:
                    progress.console.print(
                        f"  [bold red]✗[/]  [white]{name}[/]  "
                        f"[dim red]failed after {format_duration(time.time() - t0)}:[/]  "
                        f"{escape(str(exc))}"
                    )
                    failed.append((src, str(exc)))
                progress.advance(overall)
        except KeyboardInterrupt:
            interrupted = True
        progress.update(current, visible=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start

    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="dim",        justify="right")
    grid.add_column(style="bold white")
    grid.add_row("Duration",    format_duration(total_elapsed))
    grid.add_row("Transcribed", f"[bold green]{ok}[/] / {len(pending)}")
    grid.add_row("Total words", f"{total_words:,}")
    if failed:
        grid.add_row("Failed", f"[bold red]{len(failed)}[/]")
    if interrupted:
        grid.add_row("Stopped", "[bold yellow]interrupted (Ctrl+C)[/]")

    border = "green" if not failed and not interrupted else "yellow"
    console.print()
    console.print(Panel(grid, title="[bold]  Session Summary  [/]",
                        border_style=border, box=box.ROUNDED, expand=False))

    if failed:
        console.print()
        err_table = Table(box=box.SIMPLE, header_style="bold red", show_header=True)
        err_table.add_column("File",  style="white")
        err_table.add_column("Error", style="dim red")
        for path, err in failed:
            err_table.add_row(Text(path.name), Text(err))
        console.print(err_table)

    console.print()
    if interrupted:
        sys.exit(130)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl+C outside the transcription loop (scanning, model download/loading)
        console.print("\n[yellow]Interrupted.[/]")
        sys.exit(130)
