# WhisperForge

**Batch audio and video transcription from the command line.**

WhisperForge transcribes single media files, or recursively scans whole folders, and turns each one into plain text or SRT subtitles — locally, offline, with no API keys required. It uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a CTranslate2-optimised reimplementation of OpenAI Whisper that runs up to 4× faster at the same accuracy. On a CUDA GPU it batches inference automatically; on CPU it falls back gracefully.

The terminal UI is built with [rich](https://github.com/Textualize/rich): a live progress bar, colour-coded file table, and a summary panel greet you at every run.

---

## Requirements

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is optional but strongly recommended for large batches or big models. On Windows, install the CUDA build of PyTorch so faster-whisper finds the cuBLAS/cuDNN libraries:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

If no GPU is found, or the GPU fails to initialise, WhisperForge falls back to the CPU automatically. CPU transcription works but is significantly slower.

---

## Quick start

```bash
python whisperforge.py "E:\Videos\lecture.mp4"     # one file
python whisperforge.py "E:\Videos"                 # a whole folder, recursively
```

Pass any mix of files, folders and wildcards (`"E:\Podcasts\*.mp3"`); options may come before or after them.

Each file gets a transcript saved alongside it with the same stem:
`lecture.mp4` → `lecture.txt`. If two files in one folder share a stem (`talk.mp3` and `talk.mp4`), both keep their full name instead (`talk.mp3.txt`, `talk.mp4.txt`) so neither overwrites the other.

---

## Options

| Flag | Default | Description |
|---|---|---|
| `PATH …` | *(at least one)* | Media files, folders (scanned recursively) or wildcards to transcribe |
| `--root-dir DIR` | — | Same as passing a folder as `PATH`; kept for older scripts, can be repeated |
| `--language LANG` | `de` | ISO 639-1 language code (`en`, `fr`, `de`, …), or `auto` to detect |
| `--model SIZE` | `large-v3` | Whisper model to use (see table below) |
| `--format` | `txt` | Output format: `txt` (plain text) or `srt` (subtitles) |
| `--output-dir DIR` | next to source | Write transcripts to one folder, mirroring the subfolder structure of scanned folders |
| `--extensions EXT,EXT` | all supported | Limit which file types are picked up from folders and wildcards |
| `--prompt TEXT` | — | Names or jargon that help recognition, e.g. `"Fahrgemeinschaft, Franken, App"` |
| `--batch-size N` | `8` | GPU batch size; lower it if you run out of VRAM |
| `--dry-run` | off | Show what would be transcribed without doing any work |
| `--force` | off | Re-transcribe files that already have output |

---

## Examples

```bash
# A single file as subtitles
python whisperforge.py "E:\Videos\interview.mp4" --format srt

# A few specific files, English, faster model
python whisperforge.py intro.mp3 outro.wav --language en --model medium

# Generate SRT subtitle files into a dedicated folder
python whisperforge.py "E:\Videos" --format srt --output-dir "E:\Subtitles"

# Podcasts — MP3 only, English
python whisperforge.py "E:\Podcasts\*.mp3" --language en

# Preview what would run without touching anything
python whisperforge.py "E:\Videos" --dry-run

# Force a full redo of everything
python whisperforge.py "E:\Videos" --force
```

---

## Supported formats

`mp4` `mkv` `avi` `mov` `webm` `mp3` `wav` `m4a` `flac` `ogg` `opus`

---

## Models

Models are downloaded automatically on first use and cached locally (~/.cache/huggingface).

| Model | VRAM | Speed | Notes |
|---|---|---|---|
| `tiny` | ~1 GB | ★★★★★ | Rough drafts, quick checks |
| `base` | ~1 GB | ★★★★☆ | — |
| `small` | ~2 GB | ★★★☆☆ | — |
| `medium` | ~5 GB | ★★☆☆☆ | Good balance of speed and accuracy |
| `large-v2` | ~5 GB | ★☆☆☆☆ | — |
| `large-v3` | ~5 GB | ★☆☆☆☆ | Best accuracy — default |
| `large-v3-turbo` / `turbo` | ~3 GB | ★★★★☆ | Close to large-v3, several times faster |
| `distil-large-v3` | ~3 GB | ★★★☆☆ | English only |

VRAM figures are for float16 on the GPU (what WhisperForge uses); `large-v3` runs fine on an 8 GB card. Run `python whisperforge.py --help` for the full list of model names.

---

## Output formats

**Plain text** (`--format txt`):
```
La religion, ça doit être privé pour tout le monde. Il ne faut jamais poser
cette question à quelqu'un, c'est quoi ta religion ?
```

**SubRip subtitles** (`--format srt`):
```
1
00:00:00,000 --> 00:00:03,500
La religion, ça doit être privé pour tout le monde.

2
00:00:03,500 --> 00:00:07,200
Il ne faut jamais poser cette question à quelqu'un.
```

SRT cues are built from word timestamps and split at sentence ends, at most ~84 characters or 6 seconds per cue.

All output files are UTF-8 encoded with a BOM for seamless Windows compatibility. Files are written atomically, so an interrupted run never leaves a half-written transcript that would be skipped next time.

Silence is skipped with voice activity detection (VAD), which speeds things up and reduces hallucinated text. The exit code is `0` on success, `1` if any file failed and `130` if interrupted with Ctrl+C.
