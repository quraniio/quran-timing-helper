# Quran Timing Helper

**By the [qurani.io](https://qurani.io) team**

A command-line tool that generates precise, word-level timing files for Quran recitations. Give it an MP3 of a surah and it produces a JSON file mapping every single word of the Quran to its exact start and end time in the audio — the same technology that powers word-by-word highlighting in the Qurani app.

**Get the Qurani app:**

- iOS — <https://apps.apple.com/app/id6765754562>
- Android — <https://play.google.com/store/apps/details?id=io.qurani.app>
- Huawei AppGallery — <https://appgallery.huawei.com/app/C118056685>

---

## How it works

The tool combines two complementary models:

1. **OpenAI Whisper (large-v3)** transcribes the recitation with word timestamps and the transcript is matched against the authentic Quran text.
2. **wav2vec2 CTC forced alignment** (Arabic character-level) recovers and verifies any words Whisper mistimed, dropped, or compressed — openings, isti'adha/basmala intros, long madd words, repeated verses, and under-transcribed passages are all handled by dedicated recovery passes.

Every timestamp in the output is a real, measured position in the audio. The tool never fabricates timings, and a built-in validator flags any surah whose alignment quality is not good enough to ship.

## Requirements

- Python 3.10 or newer
- ffmpeg
- Roughly 10 GB of free disk space for the models (downloaded automatically on first run)

The tool has been developed and battle-tested on **macOS (Apple Silicon)**. It also runs on Windows and Linux — installation steps for all three platforms are below.

## Installation

### macOS

```bash
# 1. Install Homebrew if you don't have it: https://brew.sh
brew install ffmpeg python@3.11

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, pass `--device mps` for GPU acceleration.

### Windows

```powershell
# 1. Install Python 3.11+ from https://python.org (check "Add to PATH")
# 2. Install ffmpeg, e.g. with winget:
winget install ffmpeg

# 3. Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch first (see <https://pytorch.org/get-started/locally/>), then pass `--device cuda`.

### Linux (Debian/Ubuntu)

```bash
sudo apt update && sudo apt install -y ffmpeg python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pass `--device cuda` if an NVIDIA GPU is available, otherwise `--device cpu`.

## Quick start

Align a single surah from a local MP3:

```bash
python quran_forced_align_full.py \
  --chapter 1 \
  --audio path/to/001.mp3 \
  --out timings/001.json \
  --model large-v3 \
  --device mps        # or cuda / cpu
```

Align a whole reciter hosted online (files named `001.mp3` … `114.mp3`):

```bash
python quran_forced_align_full.py \
  --all \
  --reciter-url "https://example.com/audio/reciter-name" \
  --out-dir timings/reciter-name \
  --model large-v3 \
  --device mps
```

Batch-process many reciters from a list you host anywhere:

```bash
python quran_forced_align_full.py \
  --batch-reciter-list-url "https://example.com/reciters.txt" \
  --batch-base-url "https://example.com/audio" \
  --model large-v3 \
  --device mps
```

`reciters.txt` is a plain text file with one folder name per line (lines starting with `#` are ignored). You can also pass folders inline with `--batch-reciter-folders name1 name2` or from a local file with `--batch-reciter-file`.

The Quran word database (`quran_words.json`) is downloaded and built automatically on first run.

## Validating results

After a batch run, validate everything that was generated:

```bash
python quran_forced_align_full.py --validate-existing-timings
```

Each surah is graded **ok / needs_review / needs_fix** based on the measured alignment quality. A companion dashboard (`quran_dashboard.py`) renders the reports as a browsable HTML page.

## Command reference

Every command-line option is listed below, grouped by purpose. Run `python quran_forced_align_full.py --help` to see the same list in your terminal. Options prefixed with `--no-` disable a feature that is **on by default** — the tool's recovery and quality passes are all enabled out of the box, so a typical run needs only a handful of the options in the first few sections.

Most-used options: `--chapter`, `--audio`, `--all`, `--reciter-url`, `--out`, `--out-dir`, `--resume`, `--model`, `--device`, `--batch-reciter-list-url`, `--validate-existing-timings`, `--fix-existing-bad`.

### Input & mode selection

| Option | Value | Default | Description |
|---|---|---|---|
| `--chapter` | int | — | Surah number (1–114) to align a single chapter. |
| `--audio` | str | — | Path or URL to the audio (MP3/WAV) for the chosen chapter. |
| `--audio-pattern` | str | — | Filename/URL pattern for locating per-chapter audio. |
| `--all` | flag | — | Process all 114 chapters (use with --reciter-url and --out-dir). |
| `--reciter-url` | str | — | Base reciter URL, pattern URL, or one MP3 URL like .../001.mp3. With no --chapter, it generates all 114. |
| `--out` | str | timing.json | — |
| `--out-dir` | str | — | Output directory for per-chapter JSON when processing many chapters. |
| `--resume` | flag | — | Skip chapters whose output JSON already exists. |
| `--summary-file` | str | — | Write a JSON summary for all generated chapters. |

### Quran word database

| Option | Value | Default | Description |
|---|---|---|---|
| `--quran-words` | str | quran_words.json | — |
| `--download-quran-words-only` | flag | — | Only generate quran_words.json then exit. |
| `--force-quran-words` | flag | — | Regenerate quran_words.json even if it exists. |
| `--raw-kmaslesa` | str | raw_kmaslesa_pages.json | — |
| `--quran-words-mismatches` | str | quran_words_mismatches.json | — |
| `--quran-words-sleep` | float | 0.1 | — |

### Model & device

| Option | Value | Default | Description |
|---|---|---|---|
| `--model` | str | large-v3 | — |
| `--model-file` | str | — | Path to a local Whisper model checkpoint. |
| `--device` | [auto, mps, cpu, cuda] | auto | Compute device. |
| `--openai-fp16` | flag | — | Force fp16 for OpenAI Whisper. Usually only for CUDA. |
| `--no-openai-fp16` | flag | — | Disable fp16 for OpenAI Whisper. |
| `--asr-backend` | [openai-whisper, hf-transformers] | openai-whisper | ASR backend. Use hf-transformers for Quran-specific Hugging Face Whisper models. |
| `--hf-model` | str | IJyad/whisper-large-v3-Tarteel | Hugging Face ASR model id for --asr-backend hf-transformers. |
| `--hf-chunk-length-s` | int | 30 | HF ASR chunk length in seconds. |
| `--hf-stride-length-s` | int | 5 | HF ASR stride length in seconds. |
| `--hf-torch-dtype` | [auto, float32, float16, bfloat16] | auto | Torch dtype for the Hugging Face ASR backend. |
| `--hf-trust-remote-code` | flag | — | Allow executing custom model code from the HF checkpoint. |
| `--no-hf-fix-timestamp-config` | flag | — | Disable v36 auto patch for missing Whisper no_timestamps_token_id in Hugging Face checkpoints. |
| `--hf-pass-language-kwargs` | flag | — | Pass language/task to HF generate(). Default off because some Quran checkpoints have outdated generation_config. |
| `--no-hf-fix-alignment-heads` | flag | — | Disable v38 auto patch for missing Whisper alignment_heads in Hugging Face checkpoints. |
| `--hf-alignment-heads-from` | str | auto | Source model to copy alignment_heads from, e.g. openai/whisper-large-v3-turbo, or auto. |

### Alignment core

| Option | Value | Default | Description |
|---|---|---|---|
| `--min-score` | float | 0.55 | — |
| `--lookahead` | int | 15 | — |
| `--alignment` | [dp, sequence] | dp | Alignment strategy. |
| `--complete-output` | flag | — | Force complete JSON output. This is enabled by default unless --strict is used. |
| `--strict` | flag | — | Do not complete missing timings. Leave missing segments visible and allow non-complete JSON. |
| `--warn-below` | float | 0.85 | — |
| `--interpolate-missing` | flag | — | Interpolate timings for words with no match. |
| `--no-auto-retry-alignment` | flag | — | Disable v21 automatic alignment retry variants. |
| `--no-strip-recitation-extras` | flag | — | Do not strip leading isti'adha/basmala or trailing outro from ASR before alignment. |

### Batch processing

| Option | Value | Default | Description |
|---|---|---|---|
| `--batch-reciter-folders` | str | — | Run specific reciter folders, e.g. reciter-one reciter-two. |
| `--batch-reciter-file` | str | — | Text file with one reciter folder per line. |
| `--batch-reciter-list-url` | str | — | Direct URL to a plain-text reciter folder list (one folder name per line, # comments allowed). The tool downloads the list and runs every folder against --batch-base-url. |
| `--batch-base-url` | str | https://qurani.io/quransound | — |
| `--batch-out-root` | str | timings | Root folder for clean timing output: <out-root>/<reciter>/001.json..114.json (no debug files). |
| `--batch-log-root` | str | logs | — |
| `--batch-debug-root` | str | debug_logs | Root folder for ALL debug/diagnostic files in batch mode: per-reciter logs, summary.json, and *.debug.json/*.opening_*.json side-files go under <debug-root>/<reciter>/. Keeps the timing output folders clean. Default: debug_logs. |

### Validation

| Option | Value | Default | Description |
|---|---|---|---|
| `--validate-existing-timings` | flag | — | Validate already generated timings under timings/<reciter>/*.json. |
| `--validate-timings-root` | str | timings | — |
| `--validate-report` | str | logs/timings_validation_report.json | — |
| `--validate-fail-list` | str | logs/timings_needs_fix.txt | — |
| `--validate-reciter-folders` | str | — | Validate only these reciter folders under --validate-timings-root. |
| `--validate-reciter-list-url` | str | — | Direct URL to a plain-text reciter folder list; only these folders are validated. |
| `--validate-min-match` | float | 0.9955 | — |
| `--validate-max-estimated` | int | 0 | — |
| `--validate-severe-match` | float | 0.90 | v63: effective match rate below which a structurally-sound file is reported RED / needs_fix (severe under-transcription: most of the surah is interpolated, not really aligned) instead of mild YELLOW / needs_review. Cleanly separates genuinely-wrong surahs from borderline ones just under the ideal bar. Set 0 to disable and keep all quality shortfalls yellow. Default: 0.90. |
| `--validate-min-word-ms` | int | 20 | — |
| `--validate-max-word-ms` | int | 12000 | — |
| `--validate-require-debug` | flag | — | Mark files without .debug.json as needs_fix instead of needs_review. |
| `--validate-debug-root` | str | debug_logs | Folder where batch mode wrote *.debug.json / *.alignment_quality.json side-files (<debug-root>/<reciter>/<chapter>.debug.json). The validator falls back here when they are not next to the timing JSON, so quality checks work on clean batch outputs. Default: debug_logs. |
| `--validate-structural-only` | flag | True | Default: validate JSON structure/timing only, do not fail files because debug/log quality data is missing. |
| `--validate-quality` | flag | — | Enable strict quality validation using debug files and optionally fresh logs. |
| `--validate-use-debug-quality` | flag | True | When --validate-quality is used, read .debug.json quality data if present. |
| `--no-validate-use-debug-quality` | flag | — | — |
| `--validate-use-logs` | flag | False | Use logs/*.log quality data. Enable only when logs are fresh and match these outputs. |
| `--no-validate-use-logs` | flag | — | — |
| `--validate-log-root` | str | logs | Folder containing old batch logs like reciter-B_v19_all.log. |

### Fixing & repair

| Option | Value | Default | Description |
|---|---|---|---|
| `--fix-existing-bad` | flag | — | After validation, rerun files marked needs_fix. |
| `--fix-include-review` | flag | — | Also rerun files marked needs_review. |
| `--fix-backup-existing` | flag | True | Backup old file family before rebuilding. |
| `--no-fix-backup-existing` | flag | — | — |
| `--fix-report` | str | logs/timings_fix_report.json | — |
| `--no-fix-revalidate` | flag | — | Do not re-validate after fixing. By default the tool re-validates the timings once the fix pass finishes so the report, needs_fix list, and dashboard reflect the newly repaired files (reds clear and the count drops). |
| `--fix-override-manual` | flag | False | Re-align surahs even if they were hand-corrected in the dashboard (i.e. have a <chapter>.manual_lock.json marker). By default such files are treated as ok/locked and skipped so re-runs never reproduce the old result or overwrite manual edits. Use this only when you deliberately want to discard manual fixes and let the model re-align them. |
| `--repair-existing-timings` | flag | — | Repair existing final JSON in-place without rerunning Whisper. |
| `--no-repair-use-debug` | flag | — | Do not rebuild final JSON from .debug.json during --repair-existing-timings. |
| `--no-repair-backup-existing` | flag | — | Do not create .repair_backup files before in-place repair. |
| `--repair-existing-report` | str | logs/timings_repair_existing_report.json | — |

### Quality gating

| Option | Value | Default | Description |
|---|---|---|---|
| `--quality-min-match` | float | 0.995 | Mark output as needs_review below this match rate. |
| `--quality-max-estimated` | int | 0 | Mark output as needs_review if estimated timings exceed this count. |
| `--fail-on-low-quality` | flag | — | Raise an error instead of saving when quality is below threshold. |
| `--no-recitation-anomaly-check` | flag | — | Disable the per-reciter recitation verification pass that flags repeated (re-recited) words and phrases/ayat. Default: enabled for every reciter; it NEVER changes timestamps and NEVER corrects text, it only reports repeats. |
| `--fail-on-recitation-anomalies` | flag | — | Treat any detected repeated word or repeated phrase/ayah as needs_review (and a hard error together with --fail-on-low-quality). |
| `--recitation-repeat-similarity` | float | 0.80 | A single extra (unmatched) ASR token at least this similar to an adjacent matched word is flagged as a repeated word. |
| `--recitation-phrase-similarity` | float | 0.80 | A consecutive run of extra (unmatched) ASR tokens whose average per-position similarity to an adjacent matched run is at least this is flagged as a repeated phrase/ayah. |

### Madd (elongation) handling

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-madd-fix` | flag | — | Disable v18 madd-only end extension. |
| `--madd-mode` | [strong, broad] | broad | strong = clear Uthmani madd signs only; broad = long-vowel letters too. |
| `--madd-max-extend-ms` | int | 1800 | Max end extension inside the same verse. |
| `--madd-verse-end-max-extend-ms` | int | 3500 | Max end extension for final word before next verse. |
| `--madd-min-gap-ms` | int | 80 | — |
| `--madd-pause-min-ms` | int | 90 | — |
| `--no-madd-audio-pauses` | flag | — | Do not use audio silence to stop verse-ending madd. |

### Whisper prompt & intro handling

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-quran-initial-prompt` | flag | — | Disable Quran text initial_prompt for Whisper. |
| `--initial-prompt-words` | int | 100 | — |
| `--no-initial-prompt-basmala` | flag | — | Do not include bismillah in Whisper initial_prompt. |
| `--initial-prompt-istiatha` | flag | True | Add isti'adha to Whisper initial prompt for reciters that start with it. Default: ON (most reciters open with أعوذ بالله). |
| `--no-initial-prompt-istiatha` | flag | — | Do not add isti'adha to the Whisper initial prompt. |
| `--intro-false-match-penalty` | int | 1000000 | — |
| `--no-penalize-intro-false-matches` | flag | — | — |
| `--strict-intro-false-match-penalty` | flag | — | Use old strict penalty for original ASR intro false matches even in muqatta'at/opening-safe chapters. |
| `--no-intro-gap-repair` | flag | — | Disable redistribution of leading estimated words after basmala/intro. |
| `--intro-gap-min-word-ms` | int | 90 | — |
| `--no-intro-gap-audio-pauses` | flag | — | Disable audio silence detection inside leading intro gap. |
| `--intro-gap-pause-min-ms` | int | 140 | — |
| `--no-opening-safe-mode` | flag | — | Disable v28 opening-safe mode for basmala/isti'adha/muqatta'at beginnings. |
| `--opening-anchor-min-score` | float | 0.88 | — |
| `--no-allow-opening-estimates` | flag | — | Do not accept opening-region estimates for quality scoring. |
| `--no-opening-estimate-cap` | flag | — | v64: disable the opening-estimate intro budget cap and restore the old behaviour of forgiving EVERY opening-region estimate for quality scoring. By default only a plausible intro's worth of opening estimates (isti'adha + basmala + huruf muqatta'at + buffer) is forgiven; beyond that the excess is treated as genuine dropped recitation and counted against quality (QUALITY WARNING -> needs_review) instead of being hidden behind a fake effective_match_rate=100%. Default: cap enabled. |
| `--opening-estimate-forgive-buffer` | int | 4 | v64: extra opening-region estimated words allowed on top of the isti'adha + basmala + huruf-muqatta'at intro word budget before the excess is counted against quality. Absorbs madd/anchor lag at the opening. Default: 4. |
| `--opening-stable-anchor-words` | int | 5 | v31: close opening only after this many consecutive reliable words. |
| `--opening-stable-anchor-max-scan-words` | int | 80 | v31: search this many initial words for the stable opening anchor. |
| `--no-opening-absorb-early-estimates` | flag | — | v40: disable absorbing early estimated words into the opening island. |
| `--no-opening-keep-real-timed` | flag | — | v56: disable keeping real (Whisper-measured) opening word timestamps; redistribute the whole leading island statically as before. |
| `--opening-absorb-max-scan-words` | int | 80 | v40: absorb estimated words inside this initial word window into the opening island. |

### Muqatta'at (opening letters) handling

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-deterministic-muqattaat-bracket` | flag | — | v57: disable the deterministic muqatta'at bracket. By default, for chapters known to open with huruf muqatta'at, the opening is anchored on the known muqatta'at word count and the muqatta'at is bracketed over [intro_end, first real word] WITHOUT verifying it acoustically. Disabling falls back to the v56 score-anchor clamp. |
| `--opening-muqattaat-madd-weight` | int | 7 | v32: extra timing weight for opening huruf muqatta'at such as حم. |
| `--opening-muqattaat-min-ms` | int | 900 | v32: minimum first-span duration for opening muqatta'at when possible. |
| `--opening-muqattaat-mode` | [auto, off, weighted] | auto | v33: auto trusts audio silences first; weighted forces extra duration; off disables extra muqatta'at weighting. |
| `--no-muqattaat-asr-fold` | flag | — | v44: disable folding pronounced muqatta'at letter names (e.g. 'حا ميم') back to the Uthmani token before alignment. Default: enabled, and only ever runs on muqatta'at chapters. |
| `--muqattaat-asr-fold-max-scan` | int | 25 | v44: scan this many leading ASR tokens to locate and fold the opening muqatta'at letter names. Default: 25. |

### Optional ASR chunking (v43, off by default)

| Option | Value | Default | Description |
|---|---|---|---|
| `--asr-external-chunking` | flag | — | v43: enable experimental external ASR chunking. Default is OFF. |
| `--no-asr-external-chunking` | flag | — | v43: force whole-WAV ASR in one call. This is the default. |
| `--asr-opening-chunk-s` | float | 10.0 | v43 optional chunking: first opening chunk duration in seconds. Default: 10 seconds. |
| `--asr-chunk-s` | float | 30.0 | v43 optional chunking: body core chunk duration in seconds after the opening chunk. Default: 30 seconds. |
| `--asr-chunk-boundary-mode` | [silence, fixed] | silence | v43 optional chunking: choose body chunk boundaries by nearby silence or fixed duration. |
| `--asr-chunk-overlap-s` | float | 1.5 | v43 optional chunking: add this much left/right audio context to chunks, then trim overlap words. |
| `--asr-chunk-boundary-search-ms` | int | 2500 | v43 optional chunking: search window around target body chunk boundary for a silence. |
| `--asr-chunk-min-silence-ms` | int | 180 | v43 optional chunking: minimum silence length used for silence-aware chunk boundaries. |
| `--asr-prompt-every-chunk` | flag | — | v43 optional chunking: pass Quran initial_prompt to every ASR chunk, not only the opening chunk. |

### Character-projection repair

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-char-projection-repair` | flag | — | Disable v24 char-stream repair for estimated words. |
| `--char-projection-min-ratio` | float | 0.55 | — |
| `--char-projection-min-chars` | int | 2 | — |
| `--char-projection-min-word-ms` | int | 70 | — |
| `--char-projection-max-shift-ms` | int | 2500 | — |

### Opening recovery (dropped/mistimed intros)

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-opening-gap-recover` | flag | — | v46: disable re-transcription recovery of a dropped surah opening. By default, if Whisper leaves a large leading gap with ZERO tokens (e.g. it skipped حمٓ + the first verses), the gap clip is re-transcribed without the Quran initial_prompt to recover REAL word timestamps. Default: enabled. |
| `--opening-gap-recover-min-ms` | int | 4000 | v46: minimum leading inter-word ASR gap (ms) that triggers opening re-transcription recovery. Default: 4000. |
| `--opening-gap-recover-scan-ms` | int | 90000 | v46: only look for the dropped-opening gap within this many ms from the start. Default: 90000. |
| `--opening-gap-recover-window-ms` | int | 30000 | v46: max sub-window (ms) per re-transcription clip; longer gaps are split into overlapping windows so each stays within one Whisper decode window. Default: 30000. |
| `--no-opening-leading-gap-recover` | flag | — | v47: disable LEADING-gap opening recovery. By default, when Whisper drops the muqatta'at recited from the very start (so its first transcribed word lands well after the opening, e.g. طسٓ in An-Naml with no transcribed basmala before it), the region [0..first_asr_word] is re-transcribed as its own clip (no Quran initial_prompt) to recover REAL opening timestamps instead of statically estimating them from 0ms. Generic across all muqatta'at surahs/reciters. Default: enabled. |
| `--no-opening-onset-snap` | flag | — | v46: disable snapping the first recovered opening word's start to the real audio onset. By default, after recovery the first word's start is pulled back to the end of the last silence region before it, because Whisper's DTW clips the soft onset of a long madd (e.g. حمٓ). This uses a measured silence boundary (real audio), not static estimation. Default: enabled. |
| `--opening-onset-snap-min-ms` | int | 200 | v46: only snap the recovered opening onset when Whisper's start is at least this many ms later than the measured audio onset. Default: 200. |
| `--opening-onset-snap-max-ms` | int | 4000 | v46: never pull the recovered opening onset back by more than this many ms (safety bound against over-extension). Default: 4000. |
| `--no-opening-real-timestamp-enforce` | flag | — | v47: disable opening REAL-timestamp enforcement. By default, after the opening-safe leading repair, any surah-opening word still STATICALLY estimated (because the stable anchor landed a few words in, e.g. طسٓ/تِلۡكَ/ءَايَٰتُ before ٱلۡقُرۡءَانِ in An-Naml) is re-mapped onto the REAL recovered ASR words already present in the opening recovery window, taking their measured Whisper timestamps (no re-transcription). Generic across all muqatta'at surahs/reciters. Default: enabled. |
| `--opening-real-timestamp-enforce-min-score` | float | 0.6 | v47: minimum fuzzy similarity to map a still-estimated opening word onto a recovered real ASR word during opening REAL-timestamp enforcement. Default: 0.6. |
| `--opening-real-timestamp-enforce-max-scan` | int | 25 | v47: scan at most this many leading words for the estimated opening island during opening REAL-timestamp enforcement. Default: 25. |

### Interior word recovery (re-transcription)

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-estimated-word-recover` | flag | — | v47: disable interior estimated-word recovery. By default, any mid-surah word the aligner had to ESTIMATE (interpolate/split) gets a targeted re-transcription of just the gap between its real-timed neighbours (no Quran initial_prompt); if Whisper surfaces the dropped word there, its REAL timestamp replaces the interpolation. Generic across all surahs/reciters; opening-region estimates are excluded (handled by opening gap recovery). Default: enabled. |
| `--estimated-word-recover-pad-ms` | int | 600 | v47: outward acoustic-context padding (ms) added on each side of the interior gap clip before re-transcription; recovered words are still clamped to the gap between real neighbours. Default: 600. |
| `--estimated-word-recover-max-window-ms` | int | 12000 | v47: skip interior gaps wider than this (ms) — too wide to be a single dropped word, so the safe interpolation is kept. Default: 12000. |
| `--estimated-word-recover-min-score` | float | 0.62 | v47: minimum fuzzy similarity for a re-transcribed gap word to be accepted as the missing Quran word. Default: 0.62. |
| `--estimated-word-recover-min-word-ms` | int | 80 | v47: minimum duration (ms) assigned to a recovered interior word. Default: 80. |
| `--estimated-word-recover-max-runs` | int | 60 | v47: safety cap on how many interior estimated-word gaps to re-transcribe per chapter. Default: 60. |

### Under-transcription recovery

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-underrun-recover` | flag | — | v60: disable severe under-transcription recovery. By default, when the prompted Whisper pass covers far fewer words than the surah has (e.g. a short surah transcribed as 5 of 14 words, or a long surah missing a large fraction), the WHOLE clip is re-transcribed WITHOUT the Quran initial_prompt and whichever transcription ALIGNS more real Quran words is kept (a longer-but-hallucinated transcript cannot win). Generic across all surahs/reciters; the second pass runs only on the poor-coverage minority. Default: enabled. |
| `--underrun-recover-min-coverage` | float | 0.85 | v60: coverage threshold (ASR words / Quran words) below which the no-prompt under-transcription recovery pass is triggered. Default: 0.85. |

### CTC forced-alignment recovery

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-ctc-word-recover` | flag | — | v59: disable CTC forced-alignment recovery of interior estimated words. By default, any mid-surah word still ESTIMATED after Whisper recovery is recovered by force-aligning its KNOWN Quran text against the gap audio with a char-level wav2vec2 CTC model — forced alignment never drops a word, so merged/short function words (أَمۡ, هُوَ, إِلَّا) get REAL measured spans. Generic across all surahs/reciters; opening-region estimates are excluded; unmappable words keep their safe interpolation. Needs torchaudio>=2.1 + transformers (degrades gracefully with a hint if missing). Default: enabled. |
| `--ctc-model` | str | jonatasgrosman/wav2vec2-large-xlsr-53-arabic | v59: Hugging Face wav2vec2 CTC model (native Arabic char vocab) used for interior forced-alignment recovery. Default: jonatasgrosman/wav2vec2-large-xlsr-53-arabic. |
| `--ctc-recover-min-score` | float | 0.0 | v81: minimum average CTC acoustic score for a force-aligned interior word to replace its interpolation. Default dropped from 0.10 to 0.0 (matching the underrun full-surah CTC path): interior gaps are bracketed by two REAL anchors and force-align KNOWN correct text, so slow-mujawwad madd words with low emission scores are still positionally sound; a per-gap degenerate guard reverts low-score words to safe interpolation when the run collapses to minimum-width spans. Default: 0.0. |
| `--ctc-recover-pad-ms` | int | 150 | v59: small symmetric acoustic-context padding (ms) added around the interior gap before CTC alignment; recovered spans are still clamped to the gap between real neighbours. Default: 150. |
| `--ctc-recover-max-window-ms` | int | 15000 | v59: interior gaps wider than this (ms) are aligned by CHUNKED CTC (see --ctc-recover-chunk-ms) instead of a single pass; a gap at or below this width aligns in one pass. Default: 15000. |
| `--ctc-recover-chunk-ms` | int | 45000 | v63: when an interior estimated-word gap bracketed by two REAL anchor timestamps is wider than --ctc-recover-max-window-ms, it is no longer skipped: it is split into sequential sub-windows of at most this many ms (using the interpolated word positions as cut points) and each chunk is force-aligned. Recovers the large under-transcribed stretches Whisper drops on long surahs while keeping every CTC forward pass memory-safe. Generic; healthy chapters are untouched; words a chunk cannot confidently place keep their safe interpolation. Default: 45000. |
| `--ctc-recover-min-word-ms` | int | 80 | v59: minimum duration (ms) assigned to a CTC-recovered interior word. Default: 80. |
| `--ctc-recover-max-runs` | int | 200 | v59: safety cap on how many interior estimated-word gaps to CTC-align per chapter (raised in v63 so heavily fragmented under-transcribed surahs are fully covered). Default: 200. |

### Duplicate-ayah CTC recovery

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-dup-ayat-ctc` | flag | — | v67: disable duplicate consecutive-ayah CTC realignment. By default, when two near-identical adjacent ayat (e.g. Ash-Sharh 94:5 فَإِنَّ مَعَ ٱلۡعُسۡرِ يُسۡرًا / 94:6 إِنَّ مَعَ ٱلۡعُسۡرِ يُسۡرًا) are collapsed by Whisper into a single recitation, the DP tie spreads that one copy across BOTH ayat and one whole copy is left ESTIMATED with fake timestamps that slide onto the next ayah. This pass detects such regions from the KNOWN Quran text (strictly consecutive ayat, identical up to a leading conjunction) and force-aligns the FULL region against the audio between its two OUTER real anchors, where every copy is present. It adopts the re-alignment only when forced alignment confidently placed every region word. Generic across all surahs/reciters; fires only when a duplicate-ayah region still holds an estimated word; degrades gracefully if CTC deps are missing. Default: enabled. |
| `--no-nonadjacent-dup-ayat-ctc` | flag | — | v74: disable NON-adjacent duplicate-ayah CTC realignment. By default, two ayat with identical recited text separated by a small number of DIFFERENT intervening ayat (e.g. Al-Kafirun 109:3 وَلَآ أَنتُمۡ عَٰبِدُونَ مَآ أَعۡبُدُ == 109:5, with 109:4 between them) — which Whisper collapses and drops one copy of, exactly like the adjacent case but invisible to the adjacent detector — are force-aligned as one span (first copy..last copy, grown outward across any contiguous interior estimates hugging it) between the span's two OUTER real anchors, adopting the re-alignment only when forced alignment placed EVERY word. Detection is from the KNOWN Quran text (strictly consecutive real verse keys, identical up to a leading conjunction), so a whole-surah refrain (e.g. Ar-Rahman) never forms a giant span. Generic across all surahs/reciters; fires only when such a span still holds an estimated word; degrades gracefully if CTC deps are missing. Default: enabled. |
| `--nonadjacent-dup-max-intervening-ayat` | int | 1 | v74: maximum number of DIFFERENT ayat allowed between the two identical copies for the non-adjacent duplicate-ayah pass (Al-Kafirun 109:3/109:5 have exactly 1). Kept small so a whole-surah refrain never forms a giant span. Default: 1. |
| `--nonadjacent-dup-max-span-words` | int | 40 | v74: skip a non-adjacent duplicate-ayah span wider than this many words (safety cap so only compact repeated-ayah regions are force-aligned as a whole). Default: 40. |

### Full-surah & opening CTC fallback

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-underrun-ctc` | flag | — | v61: disable the full-surah CTC fallback for severely under-transcribed clips. By default, when Whisper structurally drops most of a surah (so the majority of words are still MISSING/ESTIMATED after every other repair), the KNOWN whole-surah text is force-aligned against the audio with the wav2vec2 CTC model — forced alignment never drops a token, so every word (openings, madd, function words) gets a REAL measured span. A standard isti'adha/basmala prefix absorbs the leading intro. Generic across all surahs/reciters; healthy chapters are untouched; degrades gracefully if CTC deps are missing. Default: enabled. |
| `--underrun-ctc-min-bad-fraction` | float | 0.5 | v61: trigger the full-surah CTC fallback only when at least this fraction of surah words still lack a real timestamp (MISSING or ESTIMATED) after all other repairs. High by default so only catastrophic under-transcription triggers it. Default: 0.5. |
| `--underrun-ctc-max-duration-ms` | int | 120000 | v61: skip the full-surah CTC fallback for clips longer than this (ms) — too long to force-align in one CTC pass without excessive memory. Default: 120000 (2 min). |
| `--underrun-ctc-min-score` | float | 0.0 | v61: minimum average CTC acoustic score for a full-surah force-aligned word span to be accepted; below this the word keeps its prior (estimated/missing) state. This pass fires ONLY on catastrophic under-transcription where the text is KNOWN and forced alignment never drops a token, so the default is 0.0 (accept every forced-aligned span, bounded by the monotonic clamp + min-word-ms) to guarantee full coverage. Raise it to reject low-confidence spans. Default: 0.0. |
| `--underrun-ctc-min-word-ms` | int | 80 | v61: minimum duration (ms) assigned to a full-surah CTC-recovered word span. Default: 80. |
| `--underrun-ctc-force-min-mean-score` | float | 0.15 | v84: when the full-surah CTC pass runs as an ESCALATION (opening window squeeze) rather than via the catastrophic bad-fraction trigger, require the whole-surah forced alignment to place every word with at least this MEAN acoustic score before adopting anything. Default: 0.15. |
| `--no-opening-ctc` | flag | — | v66: disable the opening-scoped CTC recovery. By default, a LONG surah that aligns fine everywhere EXCEPT a dropped/mis-timed OPENING (first verse or two) — which the full-surah CTC never reaches because the clip is neither >=50% bad nor short enough — has just its opening window force-aligned with the wav2vec2 CTC model (isti'adha/basmala prefix + the leading estimated surah words, against 0..first-real-word). Fires only when the leading estimated run exceeds a plausible intro budget (isti'adha + basmala + muqatta'at + buffer), i.e. genuine dropped opening recitation. Generic; healthy openings untouched; degrades gracefully if CTC deps are missing. Default: enabled. |
| `--opening-ctc-pad-ms` | int | 1500 | v66: pad (ms) added after the first real word's start when sizing the opening CTC audio window, so a word straddling that boundary is not clipped. Default: 1500. |
| `--opening-ctc-inflated-intro-ms` | int | 12000 | v68/v75: intro span (ms) at/above which the stripped basmala/isti'adha block is treated as INFLATED by Whisper (a long post-basmala pause stretches the intro toward ~30000ms; or, v75, the basmala and the entire first ayah get merged into one ~14s block). A real isti'adha+basmala never approaches this, and the anchor+demote guards (not the duration) prevent false fires, so above it any surah word mis-aligned inside the intro is re-aligned by the opening CTC. Default: 12000. |
| `--opening-ctc-intro-overlap-tol-ms` | int | 200 | v68: tolerance (ms) subtracted from the inflated intro end when deciding whether an opening surah word is mis-aligned inside the intro (start < intro_end - tol) and when locating the post-intro resume anchor. Default: 200. |
| `--opening-ctc-anchor-run` | int | 3 | v68: number of consecutive real, monotonic, at/after-intro-end words required to accept a post-intro resume anchor for the inflated-intro guard, so a lone fluke timestamp is not used as the re-alignment boundary. Default: 3. |
| `--opening-ctc-intro-scan-limit` | int | 120 | v68: how many leading words to scan for the post-intro resume anchor in the inflated-intro guard. Raised from 60 to 120 so a genuine anchor that appears late (very fast recitation or unusual tokenization of a long opening) is not missed, which would leave the artifact unrepaired. Default: 120. |
| `--opening-ctc-prefix-hypothesis-margin` | float | 0.02 | v83: a SHORTER intro-prefix hypothesis (basmala-only / no intro) must beat the longer one's mean opening score by at least this margin to be selected; on a near-tie the longer, more conservative prefix wins. Default: 0.02. |
| `--opening-ctc-boundary-tol-ms` | int | 200 | v84: tolerance (ms) for detecting a window-end squeeze in the opening CTC pass — if any hypothesis places its last opening word within this distance of the window end AND the strict gate rejects, the first-real anchor bounding the window is flagged suspect and the full-surah CTC escalation fires. Default: 200. |
| `--opening-ctc-credible-prefix-score` | float | 0.5 | v84: a window-end squeeze only counts as anchor-suspect evidence when it comes from a CREDIBLE hypothesis: the one finally selected as best, or one whose intro PREFIX force-aligned with at least this mean acoustic score (positive evidence that prefix matches the actual recited intro). A random losing hypothesis touching the boundary never triggers escalation. Default: 0.5. |
| `--opening-ctc-small-lead-min-score` | float | 0.05 | v81: strict per-word score floor for the opening CTC pass when the leading estimated run is within the intro budget (non-intro-vocab small lead). Unlike the >budget case there is no positive evidence the words exist in the audio, so the alignment is adopted only if ALL words place with at least this confidence and pass the degenerate-shape checks; otherwise safe interpolations are kept. Default: 0.05. |

### Output & debugging

| Option | Value | Default | Description |
|---|---|---|---|
| `--debug` | flag | — | Write debug side-files and verbose diagnostics. |
| `--debug-dir` | str | — | Write all debug side-files (*.debug.json, *.opening_*.json, etc.) to this folder instead of next to the timing JSON. In batch mode this is set automatically to <batch-debug-root>/<reciter>/. |
| `--minify` | flag | — | Write compact JSON output with no extra whitespace. |
| `--insecure-ssl` | flag | — | Disable TLS certificate verification for downloads (not recommended). |
| `--ca-file` | str | — | Path to a custom CA bundle for HTTPS downloads. |

## Output format

Each output JSON contains one entry per Quran word:

```json
{
  "verse_key": "1:1",
  "position": 1,
  "start_ms": 440,
  "end_ms": 980
}
```

Timestamps are milliseconds from the start of the audio file.

## License

Free to use in free products only — see [LICENSE](LICENSE). In short: you may use, modify, and redistribute this tool at no cost, provided anything you build with it is offered completely free of charge, with no paid access and no in-app purchases.

---

# مساعد توقيتات القرآن — Quran Timing Helper

**من فريق [qurani.io](https://qurani.io)**

أداة سطر أوامر تولّد ملفات توقيت دقيقة على مستوى الكلمة لتلاوات القرآن الكريم. أعطها ملف MP3 لسورة، وتحصل على ملف JSON يحدد لكل كلمة من كلمات المصحف زمن بدايتها ونهايتها في التسجيل بدقة — وهي التقنية نفسها التي تشغّل الإبراز الكلمة-بكلمة في تطبيق قرآني.

**حمّل تطبيق قرآني:**

- آيفون — <https://apps.apple.com/app/id6765754562>
- أندرويد — <https://play.google.com/store/apps/details?id=io.qurani.app>
- هواوي — <https://appgallery.huawei.com/app/C118056685>

## كيف تعمل الأداة

1. نموذج **Whisper large-v3** يفرّغ التلاوة بطوابع زمنية لكل كلمة، ثم يُطابَق التفريغ مع النص القرآني الصحيح.
2. محاذاة قسرية بنموذج **wav2vec2 CTC** عربي تسترجع وتصحح أي كلمات أخطأ Whisper في توقيتها أو أسقطها — الافتتاحيات، الاستعاذة والبسملة، كلمات المدود الطويلة، الآيات المكررة، والمقاطع ناقصة التفريغ، لكل منها مسار معالجة مخصص.

كل طابع زمني في الناتج هو موضع حقيقي مقاس من الصوت. الأداة لا تختلق توقيتات أبدًا، ويوجد مدقق مدمج يعلّم أي سورة لم تبلغ جودة محاذاتها المستوى المطلوب.

## المتطلبات

- بايثون 3.10 أو أحدث
- ffmpeg
- نحو 10 جيجابايت مساحة فارغة للنماذج (تُحمَّل تلقائيًا عند أول تشغيل)

طُوِّرت الأداة واختُبرت بكثافة على **أجهزة ماك (Apple Silicon)**، وتعمل كذلك على ويندوز ولينكس — خطوات التثبيت للأنظمة الثلاثة موضحة في القسم الإنجليزي أعلاه.

## البدء السريع

محاذاة سورة واحدة من ملف محلي:

```bash
python quran_forced_align_full.py \
  --chapter 1 \
  --audio path/to/001.mp3 \
  --out timings/001.json \
  --model large-v3 \
  --device mps
```

محاذاة قارئ كامل مستضاف على الإنترنت (ملفات باسم `001.mp3` حتى `114.mp3`):

```bash
python quran_forced_align_full.py \
  --all \
  --reciter-url "https://example.com/audio/reciter-name" \
  --out-dir timings/reciter-name \
  --model large-v3 \
  --device mps
```

معالجة دفعة قرّاء من قائمة تستضيفها في أي مكان:

```bash
python quran_forced_align_full.py \
  --batch-reciter-list-url "https://example.com/reciters.txt" \
  --batch-base-url "https://example.com/audio" \
  --model large-v3 \
  --device mps
```

ملف `reciters.txt` هو ملف نصي بسيط فيه اسم مجلد قارئ في كل سطر (الأسطر التي تبدأ بـ `#` تُتجاهل).

## التحقق من النتائج

```bash
python quran_forced_align_full.py --validate-existing-timings
```

تُقيَّم كل سورة بدرجة **ok / needs_review / needs_fix** حسب جودة المحاذاة المقاسة، ويمكن عرض التقارير بصفحة HTML عبر `quran_dashboard.py`.

## مرجع الأوامر

فيما يلي كل خيارات سطر الأوامر مقسّمة حسب الوظيفة. شغّل `python quran_forced_align_full.py --help` لرؤية القائمة نفسها في الطرفية. الخيارات التي تبدأ بـ `--no-` تُعطّل ميزة **مفعّلة افتراضياً** — فكل مسارات الاسترجاع والجودة تعمل تلقائياً، وبالتالي التشغيل المعتاد يحتاج فقط عدداً قليلاً من خيارات الأقسام الأولى.

أكثر الخيارات استخداماً: `--chapter` و`--audio` و`--all` و`--reciter-url` و`--out` و`--out-dir` و`--resume` و`--model` و`--device` و`--batch-reciter-list-url` و`--validate-existing-timings` و`--fix-existing-bad`.

> شرح الأعمدة: **Option** الخيار · **Value** نوع القيمة أو الاختيارات · **Default** القيمة الافتراضية · **Description** الشرح.

### اختيار المدخلات والوضع

| Option | Value | Default | Description |
|---|---|---|---|
| `--chapter` | int | — | رقم السورة (1–114) لمحاذاة سورة واحدة. |
| `--audio` | str | — | مسار أو رابط الملف الصوتي (MP3/WAV) للسورة المختارة. |
| `--audio-pattern` | str | — | نمط اسم/رابط الملف لتحديد صوت كل سورة. |
| `--all` | flag | — | معالجة كل السور الـ114 (مع ‎--reciter-url و‎--out-dir). |
| `--reciter-url` | str | — | Base reciter URL, pattern URL, or one MP3 URL like .../001.mp3. With no --chapter, it generates all 114. |
| `--out` | str | timing.json | — |
| `--out-dir` | str | — | مجلد إخراج ملفات JOSN لكل سورة عند معالجة عدة سور. |
| `--resume` | flag | — | Skip chapters whose output JSON already exists. |
| `--summary-file` | str | — | Write a JSON summary for all generated chapters. |

### قاعدة بيانات كلمات القرآن

| Option | Value | Default | Description |
|---|---|---|---|
| `--quran-words` | str | quran_words.json | — |
| `--download-quran-words-only` | flag | — | Only generate quran_words.json then exit. |
| `--force-quran-words` | flag | — | Regenerate quran_words.json even if it exists. |
| `--raw-kmaslesa` | str | raw_kmaslesa_pages.json | — |
| `--quran-words-mismatches` | str | quran_words_mismatches.json | — |
| `--quran-words-sleep` | float | 0.1 | — |

### النموذج والجهاز

| Option | Value | Default | Description |
|---|---|---|---|
| `--model` | str | large-v3 | — |
| `--model-file` | str | — | مسار نسخة نموذج Whisper محلية. |
| `--device` | [auto, mps, cpu, cuda] | auto | جهاز الحوسبة. |
| `--openai-fp16` | flag | — | Force fp16 for OpenAI Whisper. Usually only for CUDA. |
| `--no-openai-fp16` | flag | — | Disable fp16 for OpenAI Whisper. |
| `--asr-backend` | [openai-whisper, hf-transformers] | openai-whisper | ASR backend. Use hf-transformers for Quran-specific Hugging Face Whisper models. |
| `--hf-model` | str | IJyad/whisper-large-v3-Tarteel | Hugging Face ASR model id for --asr-backend hf-transformers. |
| `--hf-chunk-length-s` | int | 30 | HF ASR chunk length in seconds. |
| `--hf-stride-length-s` | int | 5 | HF ASR stride length in seconds. |
| `--hf-torch-dtype` | [auto, float32, float16, bfloat16] | auto | نوع torch dtype لخلفية ASR من Hugging Face. |
| `--hf-trust-remote-code` | flag | — | السماح بتنفيذ كود النموذج المخصص من نسخة HF. |
| `--no-hf-fix-timestamp-config` | flag | — | Disable v36 auto patch for missing Whisper no_timestamps_token_id in Hugging Face checkpoints. |
| `--hf-pass-language-kwargs` | flag | — | Pass language/task to HF generate(). Default off because some Quran checkpoints have outdated generation_config. |
| `--no-hf-fix-alignment-heads` | flag | — | Disable v38 auto patch for missing Whisper alignment_heads in Hugging Face checkpoints. |
| `--hf-alignment-heads-from` | str | auto | Source model to copy alignment_heads from, e.g. openai/whisper-large-v3-turbo, or auto. |

### المحاذاة الأساسية

| Option | Value | Default | Description |
|---|---|---|---|
| `--min-score` | float | 0.55 | — |
| `--lookahead` | int | 15 | — |
| `--alignment` | [dp, sequence] | dp | استراتيجية المحاذاة. |
| `--complete-output` | flag | — | Force complete JSON output. This is enabled by default unless --strict is used. |
| `--strict` | flag | — | Do not complete missing timings. Leave missing segments visible and allow non-complete JSON. |
| `--warn-below` | float | 0.85 | — |
| `--interpolate-missing` | flag | — | تقدير توقيتات الكلمات غير المطابَقة بالاستيفاء. |
| `--no-auto-retry-alignment` | flag | — | Disable v21 automatic alignment retry variants. |
| `--no-strip-recitation-extras` | flag | — | Do not strip leading isti'adha/basmala or trailing outro from ASR before alignment. |

### المعالجة بالدفعات

| Option | Value | Default | Description |
|---|---|---|---|
| `--batch-reciter-folders` | str | — | Run specific reciter folders, e.g. reciter-one reciter-two. |
| `--batch-reciter-file` | str | — | Text file with one reciter folder per line. |
| `--batch-reciter-list-url` | str | — | Direct URL to a plain-text reciter folder list (one folder name per line, # comments allowed). The tool downloads the list and runs every folder against --batch-base-url. |
| `--batch-base-url` | str | https://qurani.io/quransound | — |
| `--batch-out-root` | str | timings | Root folder for clean timing output: <out-root>/<reciter>/001.json..114.json (no debug files). |
| `--batch-log-root` | str | logs | — |
| `--batch-debug-root` | str | debug_logs | Root folder for ALL debug/diagnostic files in batch mode: per-reciter logs, summary.json, and *.debug.json/*.opening_*.json side-files go under <debug-root>/<reciter>/. Keeps the timing output folders clean. Default: debug_logs. |

### التحقق من النتائج

| Option | Value | Default | Description |
|---|---|---|---|
| `--validate-existing-timings` | flag | — | Validate already generated timings under timings/<reciter>/*.json. |
| `--validate-timings-root` | str | timings | — |
| `--validate-report` | str | logs/timings_validation_report.json | — |
| `--validate-fail-list` | str | logs/timings_needs_fix.txt | — |
| `--validate-reciter-folders` | str | — | Validate only these reciter folders under --validate-timings-root. |
| `--validate-reciter-list-url` | str | — | Direct URL to a plain-text reciter folder list; only these folders are validated. |
| `--validate-min-match` | float | 0.9955 | — |
| `--validate-max-estimated` | int | 0 | — |
| `--validate-severe-match` | float | 0.90 | v63: effective match rate below which a structurally-sound file is reported RED / needs_fix (severe under-transcription: most of the surah is interpolated, not really aligned) instead of mild YELLOW / needs_review. Cleanly separates genuinely-wrong surahs from borderline ones just under the ideal bar. Set 0 to disable and keep all quality shortfalls yellow. Default: 0.90. |
| `--validate-min-word-ms` | int | 20 | — |
| `--validate-max-word-ms` | int | 12000 | — |
| `--validate-require-debug` | flag | — | Mark files without .debug.json as needs_fix instead of needs_review. |
| `--validate-debug-root` | str | debug_logs | Folder where batch mode wrote *.debug.json / *.alignment_quality.json side-files (<debug-root>/<reciter>/<chapter>.debug.json). The validator falls back here when they are not next to the timing JSON, so quality checks work on clean batch outputs. Default: debug_logs. |
| `--validate-structural-only` | flag | True | Default: validate JSON structure/timing only, do not fail files because debug/log quality data is missing. |
| `--validate-quality` | flag | — | Enable strict quality validation using debug files and optionally fresh logs. |
| `--validate-use-debug-quality` | flag | True | When --validate-quality is used, read .debug.json quality data if present. |
| `--no-validate-use-debug-quality` | flag | — | — |
| `--validate-use-logs` | flag | False | Use logs/*.log quality data. Enable only when logs are fresh and match these outputs. |
| `--no-validate-use-logs` | flag | — | — |
| `--validate-log-root` | str | logs | Folder containing old batch logs like reciter-B_v19_all.log. |

### الإصلاح والترميم

| Option | Value | Default | Description |
|---|---|---|---|
| `--fix-existing-bad` | flag | — | After validation, rerun files marked needs_fix. |
| `--fix-include-review` | flag | — | Also rerun files marked needs_review. |
| `--fix-backup-existing` | flag | True | Backup old file family before rebuilding. |
| `--no-fix-backup-existing` | flag | — | — |
| `--fix-report` | str | logs/timings_fix_report.json | — |
| `--no-fix-revalidate` | flag | — | Do not re-validate after fixing. By default the tool re-validates the timings once the fix pass finishes so the report, needs_fix list, and dashboard reflect the newly repaired files (reds clear and the count drops). |
| `--fix-override-manual` | flag | False | Re-align surahs even if they were hand-corrected in the dashboard (i.e. have a <chapter>.manual_lock.json marker). By default such files are treated as ok/locked and skipped so re-runs never reproduce the old result or overwrite manual edits. Use this only when you deliberately want to discard manual fixes and let the model re-align them. |
| `--repair-existing-timings` | flag | — | Repair existing final JSON in-place without rerunning Whisper. |
| `--no-repair-use-debug` | flag | — | Do not rebuild final JSON from .debug.json during --repair-existing-timings. |
| `--no-repair-backup-existing` | flag | — | Do not create .repair_backup files before in-place repair. |
| `--repair-existing-report` | str | logs/timings_repair_existing_report.json | — |

### بوابة الجودة

| Option | Value | Default | Description |
|---|---|---|---|
| `--quality-min-match` | float | 0.995 | Mark output as needs_review below this match rate. |
| `--quality-max-estimated` | int | 0 | Mark output as needs_review if estimated timings exceed this count. |
| `--fail-on-low-quality` | flag | — | Raise an error instead of saving when quality is below threshold. |
| `--no-recitation-anomaly-check` | flag | — | Disable the per-reciter recitation verification pass that flags repeated (re-recited) words and phrases/ayat. Default: enabled for every reciter; it NEVER changes timestamps and NEVER corrects text, it only reports repeats. |
| `--fail-on-recitation-anomalies` | flag | — | Treat any detected repeated word or repeated phrase/ayah as needs_review (and a hard error together with --fail-on-low-quality). |
| `--recitation-repeat-similarity` | float | 0.80 | A single extra (unmatched) ASR token at least this similar to an adjacent matched word is flagged as a repeated word. |
| `--recitation-phrase-similarity` | float | 0.80 | A consecutive run of extra (unmatched) ASR tokens whose average per-position similarity to an adjacent matched run is at least this is flagged as a repeated phrase/ayah. |

### معالجة المدود

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-madd-fix` | flag | — | Disable v18 madd-only end extension. |
| `--madd-mode` | [strong, broad] | broad | strong = clear Uthmani madd signs only; broad = long-vowel letters too. |
| `--madd-max-extend-ms` | int | 1800 | Max end extension inside the same verse. |
| `--madd-verse-end-max-extend-ms` | int | 3500 | Max end extension for final word before next verse. |
| `--madd-min-gap-ms` | int | 80 | — |
| `--madd-pause-min-ms` | int | 90 | — |
| `--no-madd-audio-pauses` | flag | — | Do not use audio silence to stop verse-ending madd. |

### موجِّه Whisper ومعالجة الافتتاحية

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-quran-initial-prompt` | flag | — | Disable Quran text initial_prompt for Whisper. |
| `--initial-prompt-words` | int | 100 | — |
| `--no-initial-prompt-basmala` | flag | — | Do not include bismillah in Whisper initial_prompt. |
| `--initial-prompt-istiatha` | flag | True | Add isti'adha to Whisper initial prompt for reciters that start with it. Default: ON (most reciters open with أعوذ بالله). |
| `--no-initial-prompt-istiatha` | flag | — | Do not add isti'adha to the Whisper initial prompt. |
| `--intro-false-match-penalty` | int | 1000000 | — |
| `--no-penalize-intro-false-matches` | flag | — | — |
| `--strict-intro-false-match-penalty` | flag | — | Use old strict penalty for original ASR intro false matches even in muqatta'at/opening-safe chapters. |
| `--no-intro-gap-repair` | flag | — | Disable redistribution of leading estimated words after basmala/intro. |
| `--intro-gap-min-word-ms` | int | 90 | — |
| `--no-intro-gap-audio-pauses` | flag | — | Disable audio silence detection inside leading intro gap. |
| `--intro-gap-pause-min-ms` | int | 140 | — |
| `--no-opening-safe-mode` | flag | — | Disable v28 opening-safe mode for basmala/isti'adha/muqatta'at beginnings. |
| `--opening-anchor-min-score` | float | 0.88 | — |
| `--no-allow-opening-estimates` | flag | — | Do not accept opening-region estimates for quality scoring. |
| `--no-opening-estimate-cap` | flag | — | v64: disable the opening-estimate intro budget cap and restore the old behaviour of forgiving EVERY opening-region estimate for quality scoring. By default only a plausible intro's worth of opening estimates (isti'adha + basmala + huruf muqatta'at + buffer) is forgiven; beyond that the excess is treated as genuine dropped recitation and counted against quality (QUALITY WARNING -> needs_review) instead of being hidden behind a fake effective_match_rate=100%. Default: cap enabled. |
| `--opening-estimate-forgive-buffer` | int | 4 | v64: extra opening-region estimated words allowed on top of the isti'adha + basmala + huruf-muqatta'at intro word budget before the excess is counted against quality. Absorbs madd/anchor lag at the opening. Default: 4. |
| `--opening-stable-anchor-words` | int | 5 | v31: close opening only after this many consecutive reliable words. |
| `--opening-stable-anchor-max-scan-words` | int | 80 | v31: search this many initial words for the stable opening anchor. |
| `--no-opening-absorb-early-estimates` | flag | — | v40: disable absorbing early estimated words into the opening island. |
| `--no-opening-keep-real-timed` | flag | — | v56: disable keeping real (Whisper-measured) opening word timestamps; redistribute the whole leading island statically as before. |
| `--opening-absorb-max-scan-words` | int | 80 | v40: absorb estimated words inside this initial word window into the opening island. |

### معالجة الحروف المقطّعة

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-deterministic-muqattaat-bracket` | flag | — | v57: disable the deterministic muqatta'at bracket. By default, for chapters known to open with huruf muqatta'at, the opening is anchored on the known muqatta'at word count and the muqatta'at is bracketed over [intro_end, first real word] WITHOUT verifying it acoustically. Disabling falls back to the v56 score-anchor clamp. |
| `--opening-muqattaat-madd-weight` | int | 7 | v32: extra timing weight for opening huruf muqatta'at such as حم. |
| `--opening-muqattaat-min-ms` | int | 900 | v32: minimum first-span duration for opening muqatta'at when possible. |
| `--opening-muqattaat-mode` | [auto, off, weighted] | auto | v33: auto trusts audio silences first; weighted forces extra duration; off disables extra muqatta'at weighting. |
| `--no-muqattaat-asr-fold` | flag | — | v44: disable folding pronounced muqatta'at letter names (e.g. 'حا ميم') back to the Uthmani token before alignment. Default: enabled, and only ever runs on muqatta'at chapters. |
| `--muqattaat-asr-fold-max-scan` | int | 25 | v44: scan this many leading ASR tokens to locate and fold the opening muqatta'at letter names. Default: 25. |

### تقطيع ASR الاختياري (مطفأ افتراضياً)

| Option | Value | Default | Description |
|---|---|---|---|
| `--asr-external-chunking` | flag | — | v43: enable experimental external ASR chunking. Default is OFF. |
| `--no-asr-external-chunking` | flag | — | v43: force whole-WAV ASR in one call. This is the default. |
| `--asr-opening-chunk-s` | float | 10.0 | v43 optional chunking: first opening chunk duration in seconds. Default: 10 seconds. |
| `--asr-chunk-s` | float | 30.0 | v43 optional chunking: body core chunk duration in seconds after the opening chunk. Default: 30 seconds. |
| `--asr-chunk-boundary-mode` | [silence, fixed] | silence | v43 optional chunking: choose body chunk boundaries by nearby silence or fixed duration. |
| `--asr-chunk-overlap-s` | float | 1.5 | v43 optional chunking: add this much left/right audio context to chunks, then trim overlap words. |
| `--asr-chunk-boundary-search-ms` | int | 2500 | v43 optional chunking: search window around target body chunk boundary for a silence. |
| `--asr-chunk-min-silence-ms` | int | 180 | v43 optional chunking: minimum silence length used for silence-aware chunk boundaries. |
| `--asr-prompt-every-chunk` | flag | — | v43 optional chunking: pass Quran initial_prompt to every ASR chunk, not only the opening chunk. |

### ترميم إسقاط الحروف

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-char-projection-repair` | flag | — | Disable v24 char-stream repair for estimated words. |
| `--char-projection-min-ratio` | float | 0.55 | — |
| `--char-projection-min-chars` | int | 2 | — |
| `--char-projection-min-word-ms` | int | 70 | — |
| `--char-projection-max-shift-ms` | int | 2500 | — |

### استرجاع الافتتاحية

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-opening-gap-recover` | flag | — | v46: disable re-transcription recovery of a dropped surah opening. By default, if Whisper leaves a large leading gap with ZERO tokens (e.g. it skipped حمٓ + the first verses), the gap clip is re-transcribed without the Quran initial_prompt to recover REAL word timestamps. Default: enabled. |
| `--opening-gap-recover-min-ms` | int | 4000 | v46: minimum leading inter-word ASR gap (ms) that triggers opening re-transcription recovery. Default: 4000. |
| `--opening-gap-recover-scan-ms` | int | 90000 | v46: only look for the dropped-opening gap within this many ms from the start. Default: 90000. |
| `--opening-gap-recover-window-ms` | int | 30000 | v46: max sub-window (ms) per re-transcription clip; longer gaps are split into overlapping windows so each stays within one Whisper decode window. Default: 30000. |
| `--no-opening-leading-gap-recover` | flag | — | v47: disable LEADING-gap opening recovery. By default, when Whisper drops the muqatta'at recited from the very start (so its first transcribed word lands well after the opening, e.g. طسٓ in An-Naml with no transcribed basmala before it), the region [0..first_asr_word] is re-transcribed as its own clip (no Quran initial_prompt) to recover REAL opening timestamps instead of statically estimating them from 0ms. Generic across all muqatta'at surahs/reciters. Default: enabled. |
| `--no-opening-onset-snap` | flag | — | v46: disable snapping the first recovered opening word's start to the real audio onset. By default, after recovery the first word's start is pulled back to the end of the last silence region before it, because Whisper's DTW clips the soft onset of a long madd (e.g. حمٓ). This uses a measured silence boundary (real audio), not static estimation. Default: enabled. |
| `--opening-onset-snap-min-ms` | int | 200 | v46: only snap the recovered opening onset when Whisper's start is at least this many ms later than the measured audio onset. Default: 200. |
| `--opening-onset-snap-max-ms` | int | 4000 | v46: never pull the recovered opening onset back by more than this many ms (safety bound against over-extension). Default: 4000. |
| `--no-opening-real-timestamp-enforce` | flag | — | v47: disable opening REAL-timestamp enforcement. By default, after the opening-safe leading repair, any surah-opening word still STATICALLY estimated (because the stable anchor landed a few words in, e.g. طسٓ/تِلۡكَ/ءَايَٰتُ before ٱلۡقُرۡءَانِ in An-Naml) is re-mapped onto the REAL recovered ASR words already present in the opening recovery window, taking their measured Whisper timestamps (no re-transcription). Generic across all muqatta'at surahs/reciters. Default: enabled. |
| `--opening-real-timestamp-enforce-min-score` | float | 0.6 | v47: minimum fuzzy similarity to map a still-estimated opening word onto a recovered real ASR word during opening REAL-timestamp enforcement. Default: 0.6. |
| `--opening-real-timestamp-enforce-max-scan` | int | 25 | v47: scan at most this many leading words for the estimated opening island during opening REAL-timestamp enforcement. Default: 25. |

### استرجاع الكلمات الداخلية بإعادة التفريغ

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-estimated-word-recover` | flag | — | v47: disable interior estimated-word recovery. By default, any mid-surah word the aligner had to ESTIMATE (interpolate/split) gets a targeted re-transcription of just the gap between its real-timed neighbours (no Quran initial_prompt); if Whisper surfaces the dropped word there, its REAL timestamp replaces the interpolation. Generic across all surahs/reciters; opening-region estimates are excluded (handled by opening gap recovery). Default: enabled. |
| `--estimated-word-recover-pad-ms` | int | 600 | v47: outward acoustic-context padding (ms) added on each side of the interior gap clip before re-transcription; recovered words are still clamped to the gap between real neighbours. Default: 600. |
| `--estimated-word-recover-max-window-ms` | int | 12000 | v47: skip interior gaps wider than this (ms) — too wide to be a single dropped word, so the safe interpolation is kept. Default: 12000. |
| `--estimated-word-recover-min-score` | float | 0.62 | v47: minimum fuzzy similarity for a re-transcribed gap word to be accepted as the missing Quran word. Default: 0.62. |
| `--estimated-word-recover-min-word-ms` | int | 80 | v47: minimum duration (ms) assigned to a recovered interior word. Default: 80. |
| `--estimated-word-recover-max-runs` | int | 60 | v47: safety cap on how many interior estimated-word gaps to re-transcribe per chapter. Default: 60. |

### استرجاع النقص في التفريغ

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-underrun-recover` | flag | — | v60: disable severe under-transcription recovery. By default, when the prompted Whisper pass covers far fewer words than the surah has (e.g. a short surah transcribed as 5 of 14 words, or a long surah missing a large fraction), the WHOLE clip is re-transcribed WITHOUT the Quran initial_prompt and whichever transcription ALIGNS more real Quran words is kept (a longer-but-hallucinated transcript cannot win). Generic across all surahs/reciters; the second pass runs only on the poor-coverage minority. Default: enabled. |
| `--underrun-recover-min-coverage` | float | 0.85 | v60: coverage threshold (ASR words / Quran words) below which the no-prompt under-transcription recovery pass is triggered. Default: 0.85. |

### استرجاع بالمحاذاة القسرية CTC

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-ctc-word-recover` | flag | — | v59: disable CTC forced-alignment recovery of interior estimated words. By default, any mid-surah word still ESTIMATED after Whisper recovery is recovered by force-aligning its KNOWN Quran text against the gap audio with a char-level wav2vec2 CTC model — forced alignment never drops a word, so merged/short function words (أَمۡ, هُوَ, إِلَّا) get REAL measured spans. Generic across all surahs/reciters; opening-region estimates are excluded; unmappable words keep their safe interpolation. Needs torchaudio>=2.1 + transformers (degrades gracefully with a hint if missing). Default: enabled. |
| `--ctc-model` | str | jonatasgrosman/wav2vec2-large-xlsr-53-arabic | v59: Hugging Face wav2vec2 CTC model (native Arabic char vocab) used for interior forced-alignment recovery. Default: jonatasgrosman/wav2vec2-large-xlsr-53-arabic. |
| `--ctc-recover-min-score` | float | 0.0 | v81: minimum average CTC acoustic score for a force-aligned interior word to replace its interpolation. Default dropped from 0.10 to 0.0 (matching the underrun full-surah CTC path): interior gaps are bracketed by two REAL anchors and force-align KNOWN correct text, so slow-mujawwad madd words with low emission scores are still positionally sound; a per-gap degenerate guard reverts low-score words to safe interpolation when the run collapses to minimum-width spans. Default: 0.0. |
| `--ctc-recover-pad-ms` | int | 150 | v59: small symmetric acoustic-context padding (ms) added around the interior gap before CTC alignment; recovered spans are still clamped to the gap between real neighbours. Default: 150. |
| `--ctc-recover-max-window-ms` | int | 15000 | v59: interior gaps wider than this (ms) are aligned by CHUNKED CTC (see --ctc-recover-chunk-ms) instead of a single pass; a gap at or below this width aligns in one pass. Default: 15000. |
| `--ctc-recover-chunk-ms` | int | 45000 | v63: when an interior estimated-word gap bracketed by two REAL anchor timestamps is wider than --ctc-recover-max-window-ms, it is no longer skipped: it is split into sequential sub-windows of at most this many ms (using the interpolated word positions as cut points) and each chunk is force-aligned. Recovers the large under-transcribed stretches Whisper drops on long surahs while keeping every CTC forward pass memory-safe. Generic; healthy chapters are untouched; words a chunk cannot confidently place keep their safe interpolation. Default: 45000. |
| `--ctc-recover-min-word-ms` | int | 80 | v59: minimum duration (ms) assigned to a CTC-recovered interior word. Default: 80. |
| `--ctc-recover-max-runs` | int | 200 | v59: safety cap on how many interior estimated-word gaps to CTC-align per chapter (raised in v63 so heavily fragmented under-transcribed surahs are fully covered). Default: 200. |

### استرجاع الآيات المكررة CTC

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-dup-ayat-ctc` | flag | — | v67: disable duplicate consecutive-ayah CTC realignment. By default, when two near-identical adjacent ayat (e.g. Ash-Sharh 94:5 فَإِنَّ مَعَ ٱلۡعُسۡرِ يُسۡرًا / 94:6 إِنَّ مَعَ ٱلۡعُسۡرِ يُسۡرًا) are collapsed by Whisper into a single recitation, the DP tie spreads that one copy across BOTH ayat and one whole copy is left ESTIMATED with fake timestamps that slide onto the next ayah. This pass detects such regions from the KNOWN Quran text (strictly consecutive ayat, identical up to a leading conjunction) and force-aligns the FULL region against the audio between its two OUTER real anchors, where every copy is present. It adopts the re-alignment only when forced alignment confidently placed every region word. Generic across all surahs/reciters; fires only when a duplicate-ayah region still holds an estimated word; degrades gracefully if CTC deps are missing. Default: enabled. |
| `--no-nonadjacent-dup-ayat-ctc` | flag | — | v74: disable NON-adjacent duplicate-ayah CTC realignment. By default, two ayat with identical recited text separated by a small number of DIFFERENT intervening ayat (e.g. Al-Kafirun 109:3 وَلَآ أَنتُمۡ عَٰبِدُونَ مَآ أَعۡبُدُ == 109:5, with 109:4 between them) — which Whisper collapses and drops one copy of, exactly like the adjacent case but invisible to the adjacent detector — are force-aligned as one span (first copy..last copy, grown outward across any contiguous interior estimates hugging it) between the span's two OUTER real anchors, adopting the re-alignment only when forced alignment placed EVERY word. Detection is from the KNOWN Quran text (strictly consecutive real verse keys, identical up to a leading conjunction), so a whole-surah refrain (e.g. Ar-Rahman) never forms a giant span. Generic across all surahs/reciters; fires only when such a span still holds an estimated word; degrades gracefully if CTC deps are missing. Default: enabled. |
| `--nonadjacent-dup-max-intervening-ayat` | int | 1 | v74: maximum number of DIFFERENT ayat allowed between the two identical copies for the non-adjacent duplicate-ayah pass (Al-Kafirun 109:3/109:5 have exactly 1). Kept small so a whole-surah refrain never forms a giant span. Default: 1. |
| `--nonadjacent-dup-max-span-words` | int | 40 | v74: skip a non-adjacent duplicate-ayah span wider than this many words (safety cap so only compact repeated-ayah regions are force-aligned as a whole). Default: 40. |

### المحاذاة القسرية للسورة الكاملة والافتتاحية

| Option | Value | Default | Description |
|---|---|---|---|
| `--no-underrun-ctc` | flag | — | v61: disable the full-surah CTC fallback for severely under-transcribed clips. By default, when Whisper structurally drops most of a surah (so the majority of words are still MISSING/ESTIMATED after every other repair), the KNOWN whole-surah text is force-aligned against the audio with the wav2vec2 CTC model — forced alignment never drops a token, so every word (openings, madd, function words) gets a REAL measured span. A standard isti'adha/basmala prefix absorbs the leading intro. Generic across all surahs/reciters; healthy chapters are untouched; degrades gracefully if CTC deps are missing. Default: enabled. |
| `--underrun-ctc-min-bad-fraction` | float | 0.5 | v61: trigger the full-surah CTC fallback only when at least this fraction of surah words still lack a real timestamp (MISSING or ESTIMATED) after all other repairs. High by default so only catastrophic under-transcription triggers it. Default: 0.5. |
| `--underrun-ctc-max-duration-ms` | int | 120000 | v61: skip the full-surah CTC fallback for clips longer than this (ms) — too long to force-align in one CTC pass without excessive memory. Default: 120000 (2 min). |
| `--underrun-ctc-min-score` | float | 0.0 | v61: minimum average CTC acoustic score for a full-surah force-aligned word span to be accepted; below this the word keeps its prior (estimated/missing) state. This pass fires ONLY on catastrophic under-transcription where the text is KNOWN and forced alignment never drops a token, so the default is 0.0 (accept every forced-aligned span, bounded by the monotonic clamp + min-word-ms) to guarantee full coverage. Raise it to reject low-confidence spans. Default: 0.0. |
| `--underrun-ctc-min-word-ms` | int | 80 | v61: minimum duration (ms) assigned to a full-surah CTC-recovered word span. Default: 80. |
| `--underrun-ctc-force-min-mean-score` | float | 0.15 | v84: when the full-surah CTC pass runs as an ESCALATION (opening window squeeze) rather than via the catastrophic bad-fraction trigger, require the whole-surah forced alignment to place every word with at least this MEAN acoustic score before adopting anything. Default: 0.15. |
| `--no-opening-ctc` | flag | — | v66: disable the opening-scoped CTC recovery. By default, a LONG surah that aligns fine everywhere EXCEPT a dropped/mis-timed OPENING (first verse or two) — which the full-surah CTC never reaches because the clip is neither >=50% bad nor short enough — has just its opening window force-aligned with the wav2vec2 CTC model (isti'adha/basmala prefix + the leading estimated surah words, against 0..first-real-word). Fires only when the leading estimated run exceeds a plausible intro budget (isti'adha + basmala + muqatta'at + buffer), i.e. genuine dropped opening recitation. Generic; healthy openings untouched; degrades gracefully if CTC deps are missing. Default: enabled. |
| `--opening-ctc-pad-ms` | int | 1500 | v66: pad (ms) added after the first real word's start when sizing the opening CTC audio window, so a word straddling that boundary is not clipped. Default: 1500. |
| `--opening-ctc-inflated-intro-ms` | int | 12000 | v68/v75: intro span (ms) at/above which the stripped basmala/isti'adha block is treated as INFLATED by Whisper (a long post-basmala pause stretches the intro toward ~30000ms; or, v75, the basmala and the entire first ayah get merged into one ~14s block). A real isti'adha+basmala never approaches this, and the anchor+demote guards (not the duration) prevent false fires, so above it any surah word mis-aligned inside the intro is re-aligned by the opening CTC. Default: 12000. |
| `--opening-ctc-intro-overlap-tol-ms` | int | 200 | v68: tolerance (ms) subtracted from the inflated intro end when deciding whether an opening surah word is mis-aligned inside the intro (start < intro_end - tol) and when locating the post-intro resume anchor. Default: 200. |
| `--opening-ctc-anchor-run` | int | 3 | v68: number of consecutive real, monotonic, at/after-intro-end words required to accept a post-intro resume anchor for the inflated-intro guard, so a lone fluke timestamp is not used as the re-alignment boundary. Default: 3. |
| `--opening-ctc-intro-scan-limit` | int | 120 | v68: how many leading words to scan for the post-intro resume anchor in the inflated-intro guard. Raised from 60 to 120 so a genuine anchor that appears late (very fast recitation or unusual tokenization of a long opening) is not missed, which would leave the artifact unrepaired. Default: 120. |
| `--opening-ctc-prefix-hypothesis-margin` | float | 0.02 | v83: a SHORTER intro-prefix hypothesis (basmala-only / no intro) must beat the longer one's mean opening score by at least this margin to be selected; on a near-tie the longer, more conservative prefix wins. Default: 0.02. |
| `--opening-ctc-boundary-tol-ms` | int | 200 | v84: tolerance (ms) for detecting a window-end squeeze in the opening CTC pass — if any hypothesis places its last opening word within this distance of the window end AND the strict gate rejects, the first-real anchor bounding the window is flagged suspect and the full-surah CTC escalation fires. Default: 200. |
| `--opening-ctc-credible-prefix-score` | float | 0.5 | v84: a window-end squeeze only counts as anchor-suspect evidence when it comes from a CREDIBLE hypothesis: the one finally selected as best, or one whose intro PREFIX force-aligned with at least this mean acoustic score (positive evidence that prefix matches the actual recited intro). A random losing hypothesis touching the boundary never triggers escalation. Default: 0.5. |
| `--opening-ctc-small-lead-min-score` | float | 0.05 | v81: strict per-word score floor for the opening CTC pass when the leading estimated run is within the intro budget (non-intro-vocab small lead). Unlike the >budget case there is no positive evidence the words exist in the audio, so the alignment is adopted only if ALL words place with at least this confidence and pass the degenerate-shape checks; otherwise safe interpolations are kept. Default: 0.05. |

### الإخراج والتشخيص

| Option | Value | Default | Description |
|---|---|---|---|
| `--debug` | flag | — | كتابة ملفات التشخيص الجانبية والمخرجات المفصّلة. |
| `--debug-dir` | str | — | Write all debug side-files (*.debug.json, *.opening_*.json, etc.) to this folder instead of next to the timing JSON. In batch mode this is set automatically to <batch-debug-root>/<reciter>/. |
| `--minify` | flag | — | كتابة JSON مضغوط دون مسافات إضافية. |
| `--insecure-ssl` | flag | — | تعطيل التحقق من شهادة TLS للتنزيلات (غير مستحسن). |
| `--ca-file` | str | — | مسار حزمة شهادات CA مخصّصة لتنزيلات HTTPS. |

## الترخيص

مجاني للاستخدام في المنتجات المجانية فقط — راجع ملف [LICENSE](LICENSE). باختصار: يمكنك استخدام الأداة وتعديلها وإعادة توزيعها دون مقابل، بشرط أن يكون أي منتج تبنيه بها مجانيًا بالكامل، دون أي وصول مدفوع ودون أي مشتريات داخل التطبيق.
