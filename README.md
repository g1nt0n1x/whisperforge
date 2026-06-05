# WhisperForge

**Batch audio and video transcription from the command line.**

WhisperForge recursively scans a folder for media files and transcribes each one to plain text or SRT subtitles — locally, offline, with no API keys required. It uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a CTranslate2-optimised reimplementation of OpenAI Whisper that runs up to 4× faster at the same accuracy. On a CUDA GPU it batches inference automatically; on CPU it falls back gracefully.

The terminal UI is built with [rich](https://github.com/Textualize/rich): a live progress bar, colour-coded file table, and a summary panel greet you at every run.

---

## Requirements

```bash
pip install faster-whisper torch rich
```

A CUDA-capable GPU is optional but strongly recommended for large batches or big models. CPU transcription works but is significantly slower.

---

## Quick start

```bash
python whisperforge.py --root-dir "E:\Videos"
```

Each file gets a transcript saved alongside it with the same stem:
`lecture.mp4` → `lecture.txt`

---

## Options

| Flag | Default | Description |
|---|---|---|
| `--root-dir DIR` | *(required)* | Folder to scan recursively for media files |
| `--language LANG` | `de` | ISO 639-1 language code (`en`, `fr`, `de`, …) |
| `--model SIZE` | `large-v3` | Whisper model to use (see table below) |
| `--format` | `txt` | Output format: `txt` (plain text) or `srt` (subtitles) |
| `--output-dir DIR` | next to source | Write all transcripts to a single folder |
| `--extensions EXT,EXT` | all supported | Limit which file types are scanned |
| `--dry-run` | off | Show what would be transcribed without doing any work |
| `--force` | off | Re-transcribe files that already have output |

---

## Examples

```bash
# English videos, faster model
python whisperforge.py --root-dir "E:\Videos" --language en --model medium

# Generate SRT subtitle files into a dedicated folder
python whisperforge.py --root-dir "E:\Videos" --format srt --output-dir "E:\Subtitles"

# Podcasts — MP3 only, English
python whisperforge.py --root-dir "E:\Podcasts" --extensions mp3 --language en

# Preview what would run without touching anything
python whisperforge.py --root-dir "E:\Videos" --dry-run

# Force a full redo of everything
python whisperforge.py --root-dir "E:\Videos" --force
```

---

## Supported formats

`mp4` `mkv` `avi` `mov` `webm` `mp3` `wav` `m4a` `flac` `ogg`

---

## Models

Models are downloaded automatically on first use and cached locally (~/.cache/huggingface).

| Model | VRAM | Speed | Notes |
|---|---|---|---|
| `tiny` | ~1 GB | ★★★★★ | Rough drafts, quick checks |
| `base` | ~1 GB | ★★★★☆ | — |
| `small` | ~2 GB | ★★★☆☆ | — |
| `medium` | ~5 GB | ★★☆☆☆ | Good balance of speed and accuracy |
| `large-v2` | ~10 GB | ★☆☆☆☆ | — |
| `large-v3` | ~10 GB | ★☆☆☆☆ | Best accuracy — default |
| `distil-large-v3` | ~6 GB | ★★★☆☆ | Near large-v3 quality at lower cost |

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

All output files are UTF-8 encoded with a BOM for seamless Windows compatibility.
