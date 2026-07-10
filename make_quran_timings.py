#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Quran Timing Helper — word-level Quran recitation timing generator.
# Built and maintained by the qurani.io team (https://qurani.io).
#
# Get the Qurani app:
#   iOS:     https://apps.apple.com/app/id6765754562
#   Android: https://play.google.com/store/apps/details?id=io.qurani.app
#   Huawei:  https://appgallery.huawei.com/app/C118056685

TOOL_NAME = "Quran Timing Helper"
TOOL_SITE = "https://qurani.io"
SCRIPT_VERSION = "v84_opening_squeeze_escalates_full_surah_ctc"


# v60: screen-debug gate. When False (default) the noisy internal-state dumps
# (alignment block/region boundaries, selected silence boundaries, etc.) are
# hidden from the console. Enabling --debug turns them back on. This only
# affects on-screen prints, never the side-file debug dumps or the timing JSON.
DEBUG_SCREEN = False


def dprint(*a, **k):
    """Print only when on-screen debug output is enabled (--debug)."""
    if DEBUG_SCREEN:
        print(*a, **k)


# Global ASR model caches
try:
    _WHISPER_MODEL_CACHE
except NameError:
    _WHISPER_MODEL_CACHE = {}

_HF_ASR_PIPELINE_CACHE = {}
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import bisect
import contextlib
import copy
import json
import re
import shutil
import sys
import ssl
import time
import subprocess
import tempfile
import unicodedata
from difflib import SequenceMatcher
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import torch
import whisper


ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]"
)


def configure_ssl(use_certifi=True, insecure_ssl=False, ca_file=None):
    if insecure_ssl:
        ssl._create_default_https_context = ssl._create_unverified_context
        os.environ["PYTHONHTTPSVERIFY"] = "0"
        print("WARNING: SSL verification disabled.")
        return

    if ca_file:
        ca_file = str(Path(ca_file).expanduser().resolve())
        os.environ["SSL_CERT_FILE"] = ca_file
        os.environ["REQUESTS_CA_BUNDLE"] = ca_file
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=ca_file)
        return

    if use_certifi:
        try:
            import certifi
            ca = certifi.where()
            os.environ.setdefault("SSL_CERT_FILE", ca)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
            ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=ca)
        except Exception:
            pass


def require_binary(name):
    if shutil.which(name) is None:
        raise RuntimeError(f"Missing required command: {name}")


def run(cmd):
    subprocess.run(cmd, check=True)


def pad3(n):
    return str(int(n)).zfill(3)


def is_url(value):
    return value.startswith("http://") or value.startswith("https://")


# Some reciter CDNs sit behind a WAF that blocks the default python-requests
# User-Agent. Send a descriptive UA containing "qurani" so the request is allowed.
HTTP_HEADERS = {
    "User-Agent": "qurani-forced-align/1.0 (+https://qurani.io)",
}


def download_file(url, output_path, insecure_ssl=False):
    with requests.get(url, stream=True, timeout=240, verify=(not insecure_ssl), headers=HTTP_HEADERS) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        downloaded = 0
        last_pct = -1
        total_mb = total / (1024 * 1024)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                mb = downloaded / (1024 * 1024)
                if total > 0:
                    pct = int(downloaded * 100 / total)
                    if pct != last_pct:
                        last_pct = pct
                        print(f"\r  Downloading: {pct:3d}%  ({mb:.1f}/{total_mb:.1f} MB)",
                              end="", flush=True)
                else:
                    print(f"\r  Downloading: {mb:.1f} MB", end="", flush=True)
        print()


def copy_or_download_audio(source, output_path, insecure_ssl=False):
    if is_url(source):
        download_file(source, output_path, insecure_ssl=insecure_ssl)
    else:
        src = Path(source).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"Audio file not found: {source}")
        shutil.copyfile(src, output_path)


def convert_to_wav(input_path, wav_path):
    # mp3float's "overread, skip ..." messages are logged at ffmpeg's ERROR level
    # (not warning), so -loglevel error does NOT hide them. Use -loglevel fatal to
    # suppress these harmless decoder complaints; real failures still surface via the
    # non-zero exit code (run() uses check=True).
    run([
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "fatal",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(wav_path),
    ])


def ffprobe_duration_ms(audio_path):
    out = subprocess.check_output([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]).decode().strip()

    return int(round(float(out) * 1000))


def normalize_arabic(text):
    text = unicodedata.normalize("NFKD", text)
    text = ARABIC_DIACRITICS_RE.sub("", text)

    replacements = {
        "ٱ": "ا", "أ": "ا", "إ": "ا", "آ": "ا",
        "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه",
        "ـ": "",
        "ۦ": "", "ۥ": "", "ۢ": "", "ۭ": "", "ۜ": "",
        "ۗ": "", "ۖ": "", "ۚ": "", "ۛ": "", "ۙ": "",
        "ۘ": "", "ۡ": "", "۞": "",
    }

    for a, b in replacements.items():
        text = text.replace(a, b)

    text = re.sub(r"[^\u0600-\u06FF]", "", text)
    return text.strip()


def similarity(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()



def patch_whisper_mps_dtw():
    """
    openai-whisper timing.py does:
        x.double().cpu().numpy()

    On Apple MPS this fails because MPS does not support float64.
    Fix:
        move tensor to CPU first, then convert to double:
        x.cpu().double().numpy()
    """
    try:
        import whisper.timing as timing

        if getattr(timing, "_qurani_mps_dtw_patched", False):
            return

        def dtw_fixed(x):
            if getattr(x, "is_cuda", False):
                return timing.dtw_cuda(x)

            return timing.dtw_cpu(x.cpu().double().numpy())

        timing.dtw = dtw_fixed
        timing._qurani_mps_dtw_patched = True
        print("Patched Whisper DTW for Apple MPS float64 issue.")
    except Exception as e:
        print(f"WARNING: Could not patch Whisper DTW: {e}")


def save_json(path, data, minify=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: serialize to a temp file in the same dir, fsync, then
    # os.replace() onto the final path. An interrupted run (Ctrl-C, crash, lost
    # connection) can never leave a truncated/half-written JSON that --resume
    # would then wrongly treat as a completed chapter and skip.
    tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            if minify:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            else:
                json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass




# ---------------------------------------------------------------------------
# Integrated quran_words.json downloader/builder
# ---------------------------------------------------------------------------

KMA_URL = "https://cdn.jsdelivr.net/npm/@kmaslesa/holy-quran-word-by-word-full-data@1.0.6/data.json"
QURAN_JSON_CHAPTER_URL = "https://cdn.jsdelivr.net/npm/quran-json@3.1.2/dist/chapters/{chapter}.json"

MANUAL_VERSE_WORD_FIXES = {
    "2:72": [
        "وَإِذۡ", "قَتَلۡتُمۡ", "نَفۡسٗا", "فَٱدَّـٰرَٰٔتُمۡ",
        "فِيهَاۖ", "وَٱللَّهُ", "مُخۡرِجٞ", "مَّا", "كُنتُمۡ", "تَكۡتُمُونَ",
    ],
    "2:181": [
        "فَمَنۢ", "بَدَّلَهُۥ", "بَعۡدَمَا", "سَمِعَهُۥ",
        "فَإِنَّمَآ", "إِثۡمُهُۥ", "عَلَى", "ٱلَّذِينَ", "يُبَدِّلُونَهُۥٓۚ",
        "إِنَّ", "ٱللَّهَ", "سَمِيعٌ", "عَلِيمٞ",
    ],
    "8:6": [
        "يُجَٰدِلُونَكَ", "فِي", "ٱلۡحَقِّ", "بَعۡدَمَا", "تَبَيَّنَ",
        "كَأَنَّمَا", "يُسَاقُونَ", "إِلَى", "ٱلۡمَوۡتِ", "وَهُمۡ", "يَنظُرُونَ",
    ],
    "13:37": [
        "وَكَذَٰلِكَ", "أَنزَلۡنَٰهُ", "حُكۡمًا", "عَرَبِيّٗاۚ", "وَلَئِنِ",
        "ٱتَّبَعۡتَ", "أَهۡوَآءَهُم", "بَعۡدَمَا", "جَآءَكَ", "مِنَ",
        "ٱلۡعِلۡمِ", "مَا", "لَكَ", "مِنَ", "ٱللَّهِ", "مِن", "وَلِيّٖ",
        "وَلَا", "وَاقٖ",
    ],
    "15:7": [
        "لَّوۡ", "مَا", "تَأۡتِينَا", "بِٱلۡمَلَـٰٓئِكَةِ",
        "إِن", "كُنتَ", "مِنَ", "ٱلصَّـٰدِقِينَ",
    ],
    "27:20": [
        "وَتَفَقَّدَ", "ٱلطَّيۡرَ", "فَقَالَ", "مَا", "لِيَ", "لَآ",
        "أَرَى", "ٱلۡهُدۡهُدَ", "أَمۡ", "كَانَ", "مِنَ", "ٱلۡغَآئِبِينَ",
    ],
    "36:22": [
        "وَمَا", "لِيَ", "لَآ", "أَعۡبُدُ", "ٱلَّذِي", "فَطَرَنِي",
        "وَإِلَيۡهِ", "تُرۡجَعُونَ",
    ],
    "37:130": [
        "سَلَٰمٌ", "عَلَىٰٓ", "إِلۡيَاسِينَ",
    ],
    "37:164": [
        "وَمَا", "مِنَّآ", "إِلَّا", "لَهُۥ", "مَقَامٞ", "مَّعۡلُومٞ",
    ],
    "41:47": [
        "إِلَيۡهِ", "يُرَدُّ", "عِلۡمُ", "ٱلسَّاعَةِۚ", "وَمَا", "تَخۡرُجُ",
        "مِن", "ثَمَرَٰتٖ", "مِّنۡ", "أَكۡمَامِهَا", "وَمَا", "تَحۡمِلُ",
        "مِنۡ", "أُنثَىٰ", "وَلَا", "تَضَعُ", "إِلَّا", "بِعِلۡمِهِۦۚ",
        "وَيَوۡمَ", "يُنَادِيهِمۡ", "أَيۡنَ", "شُرَكَآءِي", "قَالُوٓاْ",
        "ءَاذَنَّـٰكَ", "مَا", "مِنَّا", "مِن", "شَهِيدٖ",
    ],
}


def get_json(url, insecure_ssl=False):
    r = requests.get(url, timeout=300, verify=(not insecure_ssl), headers=HTTP_HEADERS)
    r.raise_for_status()
    return r.json()


def get_json_stream(url, raw_out, insecure_ssl=False):
    print(f"Downloading: {url}")
    raw_path = Path(raw_out)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=300, verify=(not insecure_ssl), headers=HTTP_HEADERS) as r:
        r.raise_for_status()

        with open(raw_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"Raw saved: {raw_path}")

    with open(raw_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_verse_text(text):
    text = str(text or "")
    text = re.sub(r"[\u06DD\u06DE۝۞]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_verse_words(text):
    text = clean_verse_text(text)
    if not text:
        return []
    return [w.strip() for w in text.split() if w.strip()]


def extract_kmaslesa_positions(data):
    by_verse = {}

    if not isinstance(data, list):
        raise ValueError("Expected kmaslesa data.json to be a list of pages")

    for page in data:
        for ayah in page.get("ayahs", []):
            for word in ayah.get("words", []):
                if word.get("char_type_name") != "word":
                    continue

                verse_key = word.get("parentAyahVerseKey")
                position = word.get("position")

                if not verse_key or position is None:
                    continue

                by_verse.setdefault(verse_key, set()).add(int(position))

    return {
        verse_key: sorted(list(positions))
        for verse_key, positions in by_verse.items()
    }


def extract_chapter_verses_from_quran_json(chapter_data, chapter):
    verses = chapter_data.get("verses") or chapter_data.get("ayahs") or []
    out = {}

    for idx, verse in enumerate(verses, start=1):
        ayah_no = (
            verse.get("id")
            or verse.get("verse_number")
            or verse.get("verseNumber")
            or verse.get("number")
            or verse.get("ayah")
            or idx
        )

        try:
            ayah_no = int(ayah_no)
        except Exception:
            ayah_no = idx

        text = (
            verse.get("text")
            or verse.get("arabic")
            or verse.get("uthmani")
            or verse.get("text_uthmani")
        )

        if not text:
            continue

        out[f"{chapter}:{ayah_no}"] = split_verse_words(text)

    return out


def download_all_quran_text(insecure_ssl=False, sleep_seconds=0.1):
    verse_words = {}

    for chapter in range(1, 115):
        url = QURAN_JSON_CHAPTER_URL.format(chapter=chapter)
        print(f"Downloading Quran text chapter {chapter}/114")
        chapter_data = get_json(url, insecure_ssl=insecure_ssl)
        verse_words.update(extract_chapter_verses_from_quran_json(chapter_data, chapter))
        if sleep_seconds:
            time.sleep(sleep_seconds)

    return verse_words


def build_quran_words(kma_positions, quran_text_words):
    result = {}
    mismatches = []

    for verse_key in sorted(
        kma_positions.keys(),
        key=lambda k: (int(k.split(":")[0]), int(k.split(":")[1]))
    ):
        chapter = verse_key.split(":")[0]
        positions = kma_positions[verse_key]
        text_words = quran_text_words.get(verse_key, [])

        if verse_key in MANUAL_VERSE_WORD_FIXES:
            text_words = MANUAL_VERSE_WORD_FIXES[verse_key]

        if len(text_words) != len(positions):
            mismatches.append({
                "verse_key": verse_key,
                "positions_count": len(positions),
                "text_words_count": len(text_words),
                "text_words": text_words,
            })

        result.setdefault(chapter, [])

        # Keep positions from the word-position source, but text from quran-json/manual fixes.
        for i, position in enumerate(positions):
            if i < len(text_words):
                text = text_words[i]
            else:
                text = ""

            if text:
                result[chapter].append({
                    "verse_key": verse_key,
                    "position": int(position),
                    "text": text,
                })

    return result, mismatches


def validate_generated_quran_words(result):
    total_chapters = len(result)
    total_words = sum(len(v) for v in result.values())

    errors = []

    if total_chapters != 114:
        errors.append(f"Expected 114 chapters, got {total_chapters}")

    if total_words < 77000:
        errors.append(f"Expected about 77429 words, got {total_words}")

    # Validate known manual-fix counts.
    for verse_key, words in MANUAL_VERSE_WORD_FIXES.items():
        ch = verse_key.split(":")[0]
        got = [w for w in result.get(ch, []) if w.get("verse_key") == verse_key]
        if len(got) != len(words):
            errors.append(f"Manual fix count failed {verse_key}: got {len(got)}, expected {len(words)}")

    return errors, {"chapters": total_chapters, "words": total_words}


def generate_quran_words_json(
    output="quran_words.json",
    raw_kmaslesa="raw_kmaslesa_pages.json",
    mismatches_path="quran_words_mismatches.json",
    insecure_ssl=False,
    sleep_seconds=0.1,
):
    print("Generating quran_words.json inside quran_forced_align_full.py ...")

    kma_data = get_json_stream(
        KMA_URL,
        raw_out=raw_kmaslesa,
        insecure_ssl=insecure_ssl,
    )

    print("Extracting word positions from kmaslesa data...")
    kma_positions = extract_kmaslesa_positions(kma_data)
    print(f"Verses with positions: {len(kma_positions)}")

    print("Downloading normal Uthmani Quran text from quran-json...")
    quran_text_words = download_all_quran_text(
        insecure_ssl=insecure_ssl,
        sleep_seconds=sleep_seconds,
    )

    result, mismatches = build_quran_words(kma_positions, quran_text_words)
    validation_errors, stats = validate_generated_quran_words(result)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if mismatches_path:
        mismatch_path = Path(mismatches_path)
        mismatch_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mismatch_path, "w", encoding="utf-8") as f:
            json.dump(mismatches, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path.resolve()}")
    print(f"Chapters: {stats['chapters']}")
    print(f"Words: {stats['words']}")
    print(f"Raw mismatches after manual fixes: {len(mismatches)}")

    if validation_errors:
        print("WARNING: quran_words validation issues:")
        for err in validation_errors:
            print(f"  - {err}")
    else:
        print("quran_words validation: OK")

    return out_path


def ensure_quran_words_file(args):
    path = Path(args.quran_words)

    if path.exists() and not getattr(args, "force_quran_words", False):
        return str(path)

    return str(generate_quran_words_json(
        output=str(path),
        raw_kmaslesa=args.raw_kmaslesa,
        mismatches_path=args.quran_words_mismatches,
        insecure_ssl=args.insecure_ssl,
        sleep_seconds=args.quran_words_sleep,
    ))


# ---------------------------------------------------------------------------
# Integrated batch reciter runner
# ---------------------------------------------------------------------------

class TeeText:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


@contextlib.contextmanager
def tee_to_log(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    with open(log_path, "a", encoding="utf-8") as log:
        sys.stdout = TeeText(old_stdout, log)
        sys.stderr = TeeText(old_stderr, log)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def fetch_reciter_folders_from_url(url, insecure_ssl=False):
    """Download a plain-text reciter folder list from a direct URL.

    One folder name per line; blank lines and lines starting with '#'
    are ignored. Keeps the tool fully generic: no reciter names are
    baked into the code."""
    r = requests.get(url, timeout=120, verify=(not insecure_ssl), headers=HTTP_HEADERS)
    r.raise_for_status()
    folders = []
    for line in r.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        folders.append(line.rstrip("/"))
    if not folders:
        raise SystemExit(f"Reciter list URL returned no folder names: {url}")
    return folders


def load_reciter_folders_from_args(args):
    folders = []

    if getattr(args, "batch_reciter_list_url", None):
        folders.extend(fetch_reciter_folders_from_url(
            args.batch_reciter_list_url,
            insecure_ssl=getattr(args, "insecure_ssl", False),
        ))

    if getattr(args, "batch_reciter_folders", None):
        for item in args.batch_reciter_folders:
            for part in re.split(r"[\s,]+", str(item).strip()):
                if part:
                    folders.append(part.rstrip("/"))

    if getattr(args, "batch_reciter_file", None):
        with open(args.batch_reciter_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                folders.append(line.rstrip("/"))

    # Deduplicate but preserve order.
    out = []
    seen = set()
    for folder in folders:
        if folder not in seen:
            seen.add(folder)
            out.append(folder)

    return out


def run_all_chapters_for_args(args):
    if not args.audio_pattern or not args.out_dir:
        raise SystemExit("--all requires --audio-pattern or --reciter-url, and --out-dir")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    summary = {
        "audio_pattern": args.audio_pattern,
        "out_dir": args.out_dir,
        "model": args.model,
        "device": args.device_resolved,
        "alignment": args.alignment,
        "complete_output": args.complete_output,
        "chapters": [],
        "failed": [],
    }

    for chapter_id in range(1, 115):
        try:
            item = process_chapter(args, chapter_id)
            summary["chapters"].append(item)
        except Exception as e:
            print(f"FAILED chapter {chapter_id}: {e}")
            summary["failed"].append({
                "chapter_id": chapter_id,
                "error": str(e),
                "status": "failed",
            })

    summary["ok_count"] = sum(1 for x in summary["chapters"] if x and x.get("status") == "ok")
    summary["skipped_count"] = sum(1 for x in summary["chapters"] if x and x.get("status") == "skipped_existing")
    summary["failed_count"] = len(summary["failed"])
    summary["total_estimated_timings"] = sum(int(x.get("estimated_timings", 0)) for x in summary["chapters"] if x)
    summary["total_final_missing_segments"] = sum(int(x.get("final_missing_segments", 0)) for x in summary["chapters"] if x)
    summary["total_recitation_repeated_words"] = sum(int((x.get("recitation_anomalies") or {}).get("repeated_words", 0)) for x in summary["chapters"] if x)
    summary["total_recitation_repeated_phrases"] = sum(int((x.get("recitation_anomalies") or {}).get("repeated_phrases", 0)) for x in summary["chapters"] if x)
    summary["total_recitation_repeated"] = sum(int((x.get("recitation_anomalies") or {}).get("repeated", 0)) for x in summary["chapters"] if x)
    summary["chapters_with_recitation_anomalies"] = [
        x.get("chapter_id") for x in summary["chapters"]
        if x and int((x.get("recitation_anomalies") or {}).get("total", 0))
    ]

    if args.summary_file:
        Path(args.summary_file).parent.mkdir(parents=True, exist_ok=True)
        save_json(args.summary_file, summary, minify=False)
        print("")
        print("=" * 72)
        print(f"Summary saved: {args.summary_file}")
        print(f"OK: {summary['ok_count']}")
        print(f"Skipped: {summary['skipped_count']}")
        print(f"Failed: {summary['failed_count']}")
        print(f"Total estimated timings: {summary['total_estimated_timings']}")
        print(f"Total final missing segments: {summary['total_final_missing_segments']}")

    return summary


def run_batch_reciters(args):
    reciters = load_reciter_folders_from_args(args)

    if not reciters:
        raise SystemExit("No reciters selected. Use --batch-reciter-folders, --batch-reciter-file, or --batch-reciter-list-url.")

    base_url = str(args.batch_base_url).rstrip("/")
    out_root = Path(args.batch_out_root)
    debug_root = Path(getattr(args, "batch_debug_root", None) or args.batch_log_root)
    log_root = debug_root

    out_root.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)

    failed_reciters = []
    batch_summary = {
        "script_version": SCRIPT_VERSION,
        "base_url": base_url,
        "out_root": str(out_root),
        "reciters_total": len(reciters),
        "reciters": [],
        "failed_reciters": [],
    }

    print("=" * 72)
    print(f"Batch reciters: {len(reciters)}")
    print(f"Base URL: {base_url}")
    print(f"Output root: {out_root}")
    print(f"Log root: {log_root}")
    print("=" * 72)

    for reciter in reciters:
        reciter = reciter.strip().rstrip("/")
        if not reciter:
            continue

        reciter_url = f"{base_url}/{reciter}"
        out_dir = out_root / reciter
        reciter_debug_dir = debug_root / reciter
        reciter_debug_dir.mkdir(parents=True, exist_ok=True)
        log_path = reciter_debug_dir / f"{reciter}.log"
        summary_file = reciter_debug_dir / "summary.json"

        child = argparse.Namespace(**vars(args))
        child.reciter_url = reciter_url
        child.audio_pattern = derive_audio_pattern_from_reciter_url(reciter_url)
        child.all = True
        child.chapter = None
        child.audio = None
        child.out_dir = str(out_dir)
        child.summary_file = str(summary_file)
        # Output folder stays clean (only 001.json..114.json). Debug side-files +
        # per-chapter logs land under debug_logs/<reciter>/ so a failed surah can
        # be diagnosed and re-run on its own.
        child.debug_dir = str(reciter_debug_dir)
        child.debug = True
        # Always resume in batch: completed surahs are skipped, so an interrupted
        # download/run continues where it stopped instead of restarting.
        child.resume = True

        print("")
        print("#" * 72)
        print(f"START RECITER: {reciter}")
        print(f"URL: {reciter_url}")
        print(f"OUT: {out_dir}")
        print(f"LOG: {log_path}")
        print("#" * 72)

        status = "ok"
        error = None

        try:
            with tee_to_log(log_path):
                summary = run_all_chapters_for_args(child)
            if summary.get("failed_count"):
                status = "failed"
                error = f"{summary.get('failed_count')} failed chapters"
        except Exception as e:
            status = "failed"
            error = str(e)
            print(f"FAILED RECITER {reciter}: {error}")

        item = {
            "reciter": reciter,
            "url": reciter_url,
            "out_dir": str(out_dir),
            "summary_file": str(summary_file),
            "log_file": str(log_path),
            "status": status,
        }

        if error:
            item["error"] = error
            failed_reciters.append(reciter)

        batch_summary["reciters"].append(item)

    batch_summary["failed_reciters"] = failed_reciters
    batch_summary["failed_count"] = len(failed_reciters)
    batch_summary["ok_count"] = len(reciters) - len(failed_reciters)

    save_json(log_root / "batch_summary.json", batch_summary, minify=False)

    with open(log_root / "failed_reciters.txt", "w", encoding="utf-8") as f:
        for reciter in failed_reciters:
            f.write(reciter + "\n")

    print("")
    print("=" * 72)
    print("BATCH FINISHED")
    print(f"OK reciters: {batch_summary['ok_count']}")
    print(f"Failed reciters: {batch_summary['failed_count']}")
    print(f"Batch summary: {log_root / 'batch_summary.json'}")
    print(f"Failed list: {log_root / 'failed_reciters.txt'}")
    print("=" * 72)

    if failed_reciters:
        raise SystemExit(1)


def load_quran_words(quran_words_path, chapter_id):
    with open(quran_words_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "quran_words.json must be object/dict. "
            "Run: python download_quran_words_from_jsdelivr.py --output quran_words.json"
        )

    words = data.get(str(chapter_id))

    if not words:
        found = ", ".join(list(data.keys())[:20])
        raise ValueError(
            f"Chapter {chapter_id} not found in {quran_words_path}. "
            f"Detected chapters: {found or 'none'}"
        )

    result = []

    for index, w in enumerate(words, start=1):
        if "verse_key" not in w or "position" not in w or "text" not in w:
            raise ValueError(
                f"Invalid item in quran_words chapter {chapter_id}, item {index}. "
                "Need verse_key, position, text."
            )

        result.append({
            "verse_key": str(w["verse_key"]),
            "position": int(w["position"]),
            "text": str(w["text"]),
            "norm": normalize_arabic(str(w["text"])),
        })

    return result


def pick_device(device):
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("WARNING: --device cuda requested but CUDA is not available. Falling back to cpu.")
                return "cpu"
        except Exception:
            print("WARNING: Could not import torch for CUDA check. Falling back to cpu.")
            return "cpu"
        return "cuda"

    return device

def build_quran_initial_prompt(quran_words, chapter_id=None, prompt_words=80, include_basmala=True, include_istiatha=False):
    """
    Give Whisper a text hint, not timings.

    This helps cases like Surah 27 where the audio starts with basmala
    and Whisper skips/garbles the short opening verses, then starts at verse 4.
    """
    parts = []

    try:
        ch = int(chapter_id) if chapter_id is not None else None
    except Exception:
        ch = None

    if include_istiatha:
        parts.append("أعوذ بالله من الشيطان الرجيم")

    if include_basmala and ch != 9 and not quran_starts_with_basmala(quran_words, chapter_id=ch):
        parts.append("بسم الله الرحمن الرحيم")

    for w in quran_words[:max(0, int(prompt_words))]:
        t = str(w.get("text", "")).strip()
        if t:
            parts.append(t)

    prompt = " ".join(parts).strip()
    return prompt or None



def run_whisper(wav_path, model_name, device, model_file=None, initial_prompt=None, fp16=None):
    if model_file:
        model_source = str(Path(model_file).expanduser().resolve())
    else:
        model_source = model_name

    cache_key = (str(model_source), str(device))

    if cache_key in _WHISPER_MODEL_CACHE:
        model = _WHISPER_MODEL_CACHE[cache_key]
        print(f"Using cached Whisper model: {model_source} on {device}")
    else:
        if model_file:
            print(f"Loading Whisper model from local file: {model_source}")
        else:
            print(f"Loading Whisper model: {model_name} on {device}")

        try:
            model = whisper.load_model(model_source, device=device)
        except Exception as e:
            print("")
            print("ERROR while loading/downloading Whisper model.")
            print("Try:")
            print("  python -m pip install -U certifi")
            print("  /Applications/Python\\ 3.12/Install\\ Certificates.command")
            print("  or add --insecure-ssl")
            print("")
            raise e

        _WHISPER_MODEL_CACHE[cache_key] = model

    if device == "mps":
        patch_whisper_mps_dtw()

    print("Transcribing audio with word_timestamps=True ...")

    if fp16 is None:
        fp16 = (str(device).lower() == "cuda")

    transcribe_kwargs = dict(
        language="ar",
        task="transcribe",
        word_timestamps=True,
        condition_on_previous_text=False,
        fp16=bool(fp16),
        verbose=False,
    )

    if bool(fp16):
        print("OpenAI Whisper fp16 enabled for faster CUDA inference.")

    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt
        print(f"Using Quran initial prompt words/chars: {len(initial_prompt.split())}/{len(initial_prompt)}")

    result = model.transcribe(
        str(wav_path),
        **transcribe_kwargs,
    )

    return result


def hf_device_arg(device):
    if device == "cuda":
        return 0
    if device == "mps":
        try:
            import torch
            return torch.device("mps")
        except Exception:
            return "mps"
    return -1


def hf_torch_dtype_arg(device, dtype_name="auto"):
    try:
        import torch
    except Exception:
        return None

    dtype_name = (dtype_name or "auto").lower().strip()

    if dtype_name == "auto":
        if device == "cuda":
            return torch.float16
        # MPS word timestamp generation is safer in float32.
        return torch.float32

    if dtype_name in ("float16", "fp16", "half"):
        return torch.float16
    if dtype_name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if dtype_name in ("float32", "fp32", "full"):
        return torch.float32

    return torch.float32


def seconds_to_ms(value):
    if value is None:
        return None
    try:
        return int(round(float(value) * 1000))
    except Exception:
        return None


def split_chunk_text_to_words(text_value):
    text_value = (text_value or "").strip()
    if not text_value:
        return []
    return [x.strip() for x in re.split(r"\s+", text_value) if x.strip()]


def convert_hf_asr_result_to_whisper_like(hf_result, duration_ms=None):
    """
    Convert Hugging Face transformers ASR pipeline result into the same shape
    expected by extract_asr_words(): {"segments": [{"words": [...]}]}.

    Preferred: return_timestamps="word" chunks.
    Fallback: split timestamped chunks by text proportion if a model returns
    segment chunks instead of word chunks.
    """
    chunks = []
    if isinstance(hf_result, dict):
        chunks = hf_result.get("chunks") or []

    words_out = []

    for ci, ch in enumerate(chunks):
        if not isinstance(ch, dict):
            continue

        txt = (ch.get("text") or "").strip()
        ts = ch.get("timestamp")
        start_s = None
        end_s = None

        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
            start_s, end_s = ts[0], ts[1]

        start_ms = seconds_to_ms(start_s)
        end_ms = seconds_to_ms(end_s)

        if start_ms is None or end_ms is None:
            continue

        parts = split_chunk_text_to_words(txt)
        if not parts:
            continue

        # Word timestamp mode normally returns one word per chunk.
        if len(parts) == 1:
            words_out.append({
                "word": parts[0],
                "start": start_ms / 1000.0,
                "end": end_ms / 1000.0,
                "start_ms": start_ms,
                "end_ms": max(end_ms, start_ms + 1),
                "probability": 1.0,
                "hf_chunk_index": ci,
            })
        else:
            # Fallback for segment-level timestamps: split proportionally by
            # normalized character length. This is only a rescue path.
            weights = [max(1, len(normalize_arabic(p) or p)) for p in parts]
            total = max(1, sum(weights))
            cur = start_ms
            span = max(1, end_ms - start_ms)
            acc = 0
            for wi, p in enumerate(parts):
                acc += weights[wi]
                nxt = start_ms + round(span * acc / total)
                if wi == len(parts) - 1:
                    nxt = end_ms
                nxt = max(nxt, cur + 1)
                words_out.append({
                    "word": p,
                    "start": cur / 1000.0,
                    "end": nxt / 1000.0,
                    "start_ms": cur,
                    "end_ms": nxt,
                    "probability": 0.80,
                    "hf_chunk_index": ci,
                    "hf_segment_split": True,
                })
                cur = nxt

    if not words_out and isinstance(hf_result, dict):
        # Last-resort text-only fallback. Avoid claiming good timestamps.
        # Spread words across whole duration so structural output exists, then
        # alignment quality will reveal whether this is useful.
        txt = hf_result.get("text") or ""
        parts = split_chunk_text_to_words(txt)
        if parts and duration_ms:
            span = max(1, int(duration_ms))
            for i, p in enumerate(parts):
                s = round(span * i / len(parts))
                e = round(span * (i + 1) / len(parts))
                words_out.append({
                    "word": p,
                    "start": s / 1000.0,
                    "end": max(e, s + 1) / 1000.0,
                    "start_ms": s,
                    "end_ms": max(e, s + 1),
                    "probability": 0.50,
                    "hf_text_only_fallback": True,
                })

    return {"segments": [{"id": 0, "words": words_out}]}



def patch_hf_whisper_timestamp_generation_config(pipe, verbose=True):
    """
    Some Quran fine-tuned Whisper checkpoints on Hugging Face miss
    generation_config.no_timestamps_token_id. Transformers refuses
    return_timestamps without it. We can recover it from the tokenizer:
    <|notimestamps|>.
    """
    patched = {
        "applied": False,
        "no_timestamps_token_id": None,
        "source": None,
        "warnings": [],
    }

    try:
        tokenizer = getattr(pipe, "tokenizer", None)
        model = getattr(pipe, "model", None)
        gen_config = getattr(model, "generation_config", None)
        model_config = getattr(model, "config", None)

        if tokenizer is None or model is None or gen_config is None:
            patched["warnings"].append("missing tokenizer/model/generation_config")
            return patched

        current = getattr(gen_config, "no_timestamps_token_id", None)
        if current is not None:
            patched["no_timestamps_token_id"] = int(current)
            patched["source"] = "already_present"
            patched["applied"] = False
            if verbose:
                print(f"HF timestamp config OK: no_timestamps_token_id={current}")
            return patched

        token_id = None
        for tok in ("<|notimestamps|>", "<|notimestamps|>"):
            try:
                val = tokenizer.convert_tokens_to_ids(tok)
                if val is not None and int(val) >= 0 and val != tokenizer.unk_token_id:
                    token_id = int(val)
                    patched["source"] = tok
                    break
            except Exception:
                pass

        if token_id is None:
            try:
                vocab = tokenizer.get_vocab()
                if "<|notimestamps|>" in vocab:
                    token_id = int(vocab["<|notimestamps|>"])
                    patched["source"] = "tokenizer_vocab"
            except Exception:
                pass

        if token_id is None:
            patched["warnings"].append("could_not_find_<|notimestamps|>_token")
            if verbose:
                print("WARNING: Could not patch HF timestamp config; <|notimestamps|> token not found.")
            return patched

        setattr(gen_config, "no_timestamps_token_id", token_id)
        try:
            setattr(model_config, "no_timestamps_token_id", token_id)
        except Exception:
            pass

        # Keep generation language/task aligned for Quran Arabic without
        # passing `language=` into generate(). Some fine-tuned checkpoints have
        # an old generation_config that crashes when `language` is passed as a
        # generate kwarg. forced_decoder_ids is the compatible route.
        try:
            forced_ids = None
            if hasattr(tokenizer, "get_decoder_prompt_ids"):
                for lang in ("ar", "arabic"):
                    try:
                        forced_ids = tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
                        if forced_ids:
                            break
                    except Exception:
                        forced_ids = None

            if forced_ids:
                setattr(gen_config, "forced_decoder_ids", forced_ids)
                try:
                    setattr(model_config, "forced_decoder_ids", forced_ids)
                except Exception:
                    pass
                patched["forced_decoder_ids"] = forced_ids
                patched["forced_decoder_ids_source"] = "tokenizer.get_decoder_prompt_ids"
                if verbose:
                    print(f"HF generation config patched: forced_decoder_ids={forced_ids}")
        except Exception as e:
            patched["warnings"].append(f"forced_decoder_ids_patch_failed: {e}")

        patched["applied"] = True
        patched["no_timestamps_token_id"] = token_id

        if verbose:
            print(f"HF timestamp config patched: no_timestamps_token_id={token_id}")

        return patched

    except Exception as e:
        patched["warnings"].append(str(e))
        if verbose:
            print(f"WARNING: HF timestamp config patch failed: {e}")
        return patched




def guess_hf_alignment_heads_source_model(model_id):
    mid = (model_id or "").lower()

    if "large-v3-turbo" in mid or "v3-turbo" in mid or "turbo" in mid:
        return "openai/whisper-large-v3-turbo"

    if "large-v3" in mid or "large_v3" in mid:
        return "openai/whisper-large-v3"

    if "large-v2" in mid or "large_v2" in mid:
        return "openai/whisper-large-v2"

    if "large" in mid:
        return "openai/whisper-large-v3"

    if "medium" in mid:
        return "openai/whisper-medium"

    if "small" in mid:
        return "openai/whisper-small"

    if "base" in mid:
        return "openai/whisper-base"

    if "tiny" in mid:
        return "openai/whisper-tiny"

    return None


def patch_hf_whisper_alignment_heads(pipe, model_id=None, alignment_heads_from="auto", verbose=True):
    """
    Some Quran fine-tuned Whisper checkpoints are missing generation_config.alignment_heads.
    Transformers needs alignment_heads for token/word timestamps. Copy them from the
    matching original Whisper checkpoint generation_config.
    """
    patched = {
        "applied": False,
        "alignment_heads_from": None,
        "alignment_heads_count": None,
        "warnings": [],
    }

    try:
        model = getattr(pipe, "model", None)
        gen_config = getattr(model, "generation_config", None)
        model_config = getattr(model, "config", None)

        if model is None or gen_config is None:
            patched["warnings"].append("missing model/generation_config")
            return patched

        current = getattr(gen_config, "alignment_heads", None)
        if current:
            try:
                patched["alignment_heads_count"] = len(current)
            except Exception:
                patched["alignment_heads_count"] = None
            patched["alignment_heads_from"] = "already_present"
            if verbose:
                print(f"HF alignment_heads OK: count={patched['alignment_heads_count']}")
            return patched

        src = alignment_heads_from
        if not src or src == "auto":
            src = guess_hf_alignment_heads_source_model(model_id)

        if not src or src == "none":
            patched["warnings"].append("no_alignment_heads_source_model")
            if verbose:
                print("WARNING: No HF alignment_heads source model selected.")
            return patched

        try:
            from transformers import GenerationConfig
            ref_config = GenerationConfig.from_pretrained(src)
            heads = getattr(ref_config, "alignment_heads", None)
        except Exception as e:
            patched["warnings"].append(f"load_generation_config_failed_from_{src}: {e}")
            heads = None

        if not heads:
            patched["warnings"].append(f"source_model_has_no_alignment_heads: {src}")
            if verbose:
                print(f"WARNING: Could not patch alignment_heads from {src}; none found.")
            return patched

        setattr(gen_config, "alignment_heads", heads)
        try:
            setattr(model_config, "alignment_heads", heads)
        except Exception:
            pass

        patched["applied"] = True
        patched["alignment_heads_from"] = src
        try:
            patched["alignment_heads_count"] = len(heads)
        except Exception:
            patched["alignment_heads_count"] = None

        if verbose:
            print(
                "HF alignment_heads patched: "
                f"from={src}, count={patched['alignment_heads_count']}"
            )

        return patched

    except Exception as e:
        patched["warnings"].append(str(e))
        if verbose:
            print(f"WARNING: HF alignment_heads patch failed: {e}")
        return patched



def run_hf_transformers_asr(
    wav_path,
    model_id,
    device,
    duration_ms=None,
    chunk_length_s=30,
    stride_length_s=5,
    dtype_name="auto",
    trust_remote_code=False,
    fix_timestamp_config=True,
    pass_language_kwargs=False,
    fix_alignment_heads=True,
    alignment_heads_from="auto",
):
    global _HF_ASR_PIPELINE_CACHE
    try:
        _HF_ASR_PIPELINE_CACHE
    except NameError:
        _HF_ASR_PIPELINE_CACHE = {}

    try:
        import torch
        from transformers import pipeline
    except Exception as e:
        raise RuntimeError(
            "Missing Hugging Face dependencies. Install:\n"
            "  pip install -U transformers accelerate sentencepiece soundfile librosa\n"
            f"Original import error: {e}"
        )

    cache_key = (
        str(model_id),
        str(device),
        str(dtype_name),
        int(chunk_length_s or 0),
        int(stride_length_s or 0),
        bool(trust_remote_code),
        bool(fix_timestamp_config),
        bool(pass_language_kwargs),
        bool(fix_alignment_heads),
        str(alignment_heads_from),
    )

    if cache_key in _HF_ASR_PIPELINE_CACHE:
        pipe = _HF_ASR_PIPELINE_CACHE[cache_key]
        print(f"Using cached HF ASR model: {model_id} on {device}")
    else:
        dtype = hf_torch_dtype_arg(device, dtype_name=dtype_name)
        dev_arg = hf_device_arg(device)
        print(f"Loading HF ASR model: {model_id} on {device}")
        print(f"HF device arg: {dev_arg}, torch_dtype: {dtype}")

        pipe_kwargs = dict(
            task="automatic-speech-recognition",
            model=model_id,
            device=dev_arg,
            trust_remote_code=bool(trust_remote_code),
            ignore_warning=True,
        )

        if dtype is not None:
            pipe_kwargs["torch_dtype"] = dtype

        try:
            pipe = pipeline(**pipe_kwargs)
        except Exception as e:
            print("")
            print("ERROR while loading Hugging Face ASR model.")
            print(f"Model: {model_id}")
            print("Try one of these:")
            print("  pip install -U transformers accelerate sentencepiece soundfile librosa")
            print("  --hf-torch-dtype float32")
            print("  --device cpu   # if MPS has a model-loading issue")
            print("  --hf-model naazimsnh02/whisper-large-v3-turbo-ar-quran")
            print("")
            raise e

        if fix_timestamp_config:
            patch_hf_whisper_timestamp_generation_config(pipe, verbose=True)

        if fix_alignment_heads:
            patch_hf_whisper_alignment_heads(
                pipe,
                model_id=model_id,
                alignment_heads_from=alignment_heads_from,
                verbose=True,
            )

        _HF_ASR_PIPELINE_CACHE[cache_key] = pipe

    if fix_timestamp_config:
        patch_hf_whisper_timestamp_generation_config(pipe, verbose=False)

    if fix_alignment_heads:
        patch_hf_whisper_alignment_heads(
            pipe,
            model_id=model_id,
            alignment_heads_from=alignment_heads_from,
            verbose=False,
        )

    print("Transcribing audio with Hugging Face transformers return_timestamps='word' ...")

    call_kwargs = {
        "return_timestamps": "word",
    }

    # v37: Do not pass language/task as generate kwargs by default.
    # Some Quran fine-tuned Whisper checkpoints have an older generation_config
    # and crash with: "not compatible with the `language` argument to generate".
    # We set forced_decoder_ids on generation_config instead.
    if pass_language_kwargs:
        call_kwargs["generate_kwargs"] = {
            "task": "transcribe",
            "language": "ar",
        }

    if chunk_length_s and int(chunk_length_s) > 0:
        call_kwargs["chunk_length_s"] = int(chunk_length_s)

    if stride_length_s and int(stride_length_s) > 0:
        call_kwargs["stride_length_s"] = int(stride_length_s)

    try:
        hf_result = pipe(str(wav_path), **call_kwargs)
    except Exception as e:
        print(f"WARNING: HF word timestamps failed: {e}")
        print("Retrying HF ASR with segment timestamps fallback...")
        fallback_kwargs = dict(call_kwargs)
        fallback_kwargs["return_timestamps"] = True
        hf_result = pipe(str(wav_path), **fallback_kwargs)

    return convert_hf_asr_result_to_whisper_like(hf_result, duration_ms=duration_ms)


def run_asr_backend(
    wav_path,
    args,
    initial_prompt=None,
    duration_ms=None,
):
    backend = getattr(args, "asr_backend", "openai-whisper")

    if backend == "openai-whisper":
        return run_whisper(
            wav_path=wav_path,
            model_name=args.model,
            device=args.device_resolved,
            model_file=args.model_file,
            initial_prompt=initial_prompt,
            fp16=getattr(args, "openai_fp16", None),
        )

    if backend == "hf-transformers":
        if initial_prompt:
            print("NOTE: Quran initial_prompt is not passed to HF transformers backend; using model fine-tuning instead.")

        return run_hf_transformers_asr(
            wav_path=wav_path,
            model_id=args.hf_model,
            device=args.device_resolved,
            duration_ms=duration_ms,
            chunk_length_s=args.hf_chunk_length_s,
            stride_length_s=args.hf_stride_length_s,
            dtype_name=args.hf_torch_dtype,
            trust_remote_code=args.hf_trust_remote_code,
            fix_timestamp_config=not args.no_hf_fix_timestamp_config,
            pass_language_kwargs=args.hf_pass_language_kwargs,
            fix_alignment_heads=not args.no_hf_fix_alignment_heads,
            alignment_heads_from=args.hf_alignment_heads_from,
        )

    raise ValueError(f"Unknown ASR backend: {backend}")



def extract_wav_segment(input_wav, output_wav, start_ms, end_ms):
    start_ms = max(0, int(start_ms))
    end_ms = max(start_ms + 1, int(end_ms))
    duration_ms = end_ms - start_ms

    run([
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_ms / 1000.0:.3f}",
        "-i", str(input_wav),
        "-t", f"{duration_ms / 1000.0:.3f}",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_wav),
    ])


def read_wav_pcm16_mono_samples(wav_path):
    import wave
    import numpy as _np

    with wave.open(str(wav_path), "rb") as wf:
        sr = int(wf.getframerate())
        channels = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        # The script creates pcm_s16le WAVs, but keep this explicit for safety.
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width={sampwidth}")

    samples = _np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    return samples.astype("float32"), sr


def nearest_silence_boundary_ms(target_ms, silence_regions, min_ms, max_ms, search_ms=2500):
    target_ms = int(target_ms)
    min_ms = int(min_ms)
    max_ms = int(max_ms)
    search_ms = int(max(0, search_ms or 0))

    if not silence_regions or search_ms <= 0:
        return int(max(min_ms, min(max_ms, target_ms))), None

    candidates = []
    for r in silence_regions:
        mid = int(r.get("mid_ms", (int(r.get("start_ms", 0)) + int(r.get("end_ms", 0))) / 2))
        if mid < min_ms or mid > max_ms:
            continue
        dist = abs(mid - target_ms)
        if dist <= search_ms:
            # prefer longer silence when distances are close
            dur = int(r.get("duration_ms", 0))
            candidates.append((dist, -dur, mid, r))

    if not candidates:
        return int(max(min_ms, min(max_ms, target_ms))), None

    candidates.sort()
    return int(candidates[0][2]), candidates[0][3]


def build_external_asr_chunk_plan(
    duration_ms,
    opening_chunk_s=10.0,
    chunk_s=30.0,
    wav_path=None,
    boundary_mode="silence",
    overlap_s=1.5,
    boundary_search_ms=2500,
    min_silence_ms=180,
):
    """
    v42:
    Build ASR chunks with:
    - core intervals that cover the surah without overlap.
    - input intervals that include overlap context.
    - optional silence-aware body boundaries to avoid cutting words/ayahs.
    """
    duration_ms = max(1, int(duration_ms or 0))
    opening_ms = max(1000, int(round(float(opening_chunk_s) * 1000)))
    chunk_ms = max(5000, int(round(float(chunk_s) * 1000)))
    overlap_ms = max(0, int(round(float(overlap_s or 0.0) * 1000)))
    boundary_search_ms = max(0, int(boundary_search_ms or 0))
    min_silence_ms = max(60, int(min_silence_ms or 180))

    silence_regions = []
    if str(boundary_mode or "silence").lower() == "silence" and wav_path is not None:
        try:
            samples, sr = read_wav_pcm16_mono_samples(wav_path)
            silence_regions = detect_silence_regions_ms(
                samples,
                sr,
                0,
                duration_ms,
                min_silence_ms=min_silence_ms,
            )
        except Exception as e:
            print(f"WARNING: silence-aware ASR chunking failed; falling back to fixed chunks: {e}")
            silence_regions = []

    plan = []
    first_core_end = min(duration_ms, opening_ms)
    plan.append({
        "index": 0,
        "kind": "opening",
        "core_start_ms": 0,
        "core_end_ms": first_core_end,
        "input_start_ms": 0,
        "input_end_ms": min(duration_ms, first_core_end + overlap_ms),
        "boundary_mode": str(boundary_mode or "silence"),
        "boundary_silence": None,
    })

    cur = first_core_end
    idx = 1

    while cur < duration_ms:
        target = min(duration_ms, cur + chunk_ms)
        min_end = min(duration_ms, cur + max(5000, int(chunk_ms * 0.55)))
        max_end = min(duration_ms, cur + int(chunk_ms * 1.45))

        if target >= duration_ms:
            core_end = duration_ms
            chosen_silence = None
        elif silence_regions:
            core_end, chosen_silence = nearest_silence_boundary_ms(
                target_ms=target,
                silence_regions=silence_regions,
                min_ms=min_end,
                max_ms=max_end,
                search_ms=boundary_search_ms,
            )
            core_end = max(cur + 250, min(duration_ms, int(core_end)))
        else:
            core_end = target
            chosen_silence = None

        if core_end <= cur:
            core_end = min(duration_ms, cur + chunk_ms)

        input_start = max(0, cur - overlap_ms)
        input_end = min(duration_ms, core_end + overlap_ms)

        if core_end - cur >= 250:
            plan.append({
                "index": idx,
                "kind": "body",
                "core_start_ms": int(cur),
                "core_end_ms": int(core_end),
                "input_start_ms": int(input_start),
                "input_end_ms": int(input_end),
                "boundary_mode": str(boundary_mode or "silence"),
                "boundary_silence": chosen_silence,
            })
            idx += 1

        cur = core_end

    return plan


def build_external_asr_chunk_plan_fixed(duration_ms, opening_chunk_s=10.0, chunk_s=30.0):
    # Backward-compatible simple fixed plan, mainly for tests and debugging.
    return build_external_asr_chunk_plan(
        duration_ms=duration_ms,
        opening_chunk_s=opening_chunk_s,
        chunk_s=chunk_s,
        wav_path=None,
        boundary_mode="fixed",
        overlap_s=0.0,
        boundary_search_ms=0,
    )


def shift_whisper_like_result_times(result, offset_ms, chunk_index=None, chunk_kind=None):
    offset_s = float(offset_ms) / 1000.0
    shifted = copy.deepcopy(result or {})
    segments = shifted.get("segments") or []

    for seg in segments:
        if not isinstance(seg, dict):
            continue

        if "start" in seg:
            try:
                seg["start"] = float(seg["start"]) + offset_s
            except Exception:
                pass
        if "end" in seg:
            try:
                seg["end"] = float(seg["end"]) + offset_s
            except Exception:
                pass

        seg["external_chunk_index"] = chunk_index
        seg["external_chunk_kind"] = chunk_kind

        for w in seg.get("words", []) or []:
            if not isinstance(w, dict):
                continue
            if "start" in w:
                try:
                    w["start"] = float(w["start"]) + offset_s
                except Exception:
                    pass
            if "end" in w:
                try:
                    w["end"] = float(w["end"]) + offset_s
                except Exception:
                    pass
            if "start_ms" in w:
                try:
                    w["start_ms"] = int(w["start_ms"]) + int(offset_ms)
                except Exception:
                    pass
            if "end_ms" in w:
                try:
                    w["end_ms"] = int(w["end_ms"]) + int(offset_ms)
                except Exception:
                    pass
            w["external_chunk_index"] = chunk_index
            w["external_chunk_kind"] = chunk_kind

    return shifted


def word_mid_ms_from_whisper_word(w):
    if "start_ms" in w and "end_ms" in w:
        try:
            return int((int(w["start_ms"]) + int(w["end_ms"])) / 2)
        except Exception:
            pass

    st = None
    en = None
    try:
        if "start" in w:
            st = float(w["start"]) * 1000.0
        if "end" in w:
            en = float(w["end"]) * 1000.0
    except Exception:
        pass

    if st is not None and en is not None:
        return int((st + en) / 2)
    if st is not None:
        return int(st)
    if en is not None:
        return int(en)
    return None


def trim_whisper_like_result_to_core(result, core_start_ms, core_end_ms):
    """
    Remove words that came only from overlap context, avoiding duplicates across chunks.
    A word is kept if its midpoint is inside [core_start_ms, core_end_ms).
    """
    core_start_ms = int(core_start_ms)
    core_end_ms = int(core_end_ms)
    trimmed = copy.deepcopy(result or {})
    new_segments = []

    for seg in trimmed.get("segments", []) or []:
        if not isinstance(seg, dict):
            continue

        words = []
        for w in seg.get("words", []) or []:
            if not isinstance(w, dict):
                continue
            mid = word_mid_ms_from_whisper_word(w)
            if mid is None:
                words.append(w)
                continue
            if core_start_ms <= mid < core_end_ms:
                words.append(w)

        if words:
            seg["words"] = words
            # reset segment span to retained words for cleaner downstream handling
            try:
                if "start" in words[0]:
                    seg["start"] = float(words[0]["start"])
                if "end" in words[-1]:
                    seg["end"] = float(words[-1]["end"])
            except Exception:
                pass
            seg["core_start_ms"] = core_start_ms
            seg["core_end_ms"] = core_end_ms
            new_segments.append(seg)

    trimmed["segments"] = new_segments
    return trimmed


def count_result_words(result):
    total = 0
    for seg in (result or {}).get("segments", []) or []:
        if isinstance(seg, dict):
            total += len(seg.get("words", []) or [])
    return total


def merge_whisper_like_results(results):
    merged_segments = []
    for res in results:
        for seg in (res or {}).get("segments", []) or []:
            if isinstance(seg, dict):
                seg = dict(seg)
                seg["id"] = len(merged_segments)
                merged_segments.append(seg)
    return {"segments": merged_segments}


def run_asr_backend_chunked(
    wav_path,
    args,
    initial_prompt=None,
    duration_ms=None,
    tmpdir=None,
):
    """
    v41 external chunking:
    - First 10 seconds are transcribed as a dedicated opening chunk.
    - The rest of the surah is transcribed in fixed-size chunks.
    This prevents the ASR backend from reading the whole surah in one pass.
    """
    if duration_ms is None:
        duration_ms = ffprobe_duration_ms(wav_path)

    opening_s = float(getattr(args, "asr_opening_chunk_s", 10.0) or 10.0)
    chunk_s = float(getattr(args, "asr_chunk_s", 30.0) or 30.0)

    boundary_mode = str(getattr(args, "asr_chunk_boundary_mode", "silence") or "silence")
    overlap_s = float(getattr(args, "asr_chunk_overlap_s", 1.5) or 0.0)
    boundary_search_ms = int(getattr(args, "asr_chunk_boundary_search_ms", 2500) or 0)
    min_silence_ms = int(getattr(args, "asr_chunk_min_silence_ms", 180) or 180)

    plan = build_external_asr_chunk_plan(
        duration_ms=duration_ms,
        opening_chunk_s=opening_s,
        chunk_s=chunk_s,
        wav_path=wav_path,
        boundary_mode=boundary_mode,
        overlap_s=overlap_s,
        boundary_search_ms=boundary_search_ms,
        min_silence_ms=min_silence_ms,
    )

    print(
        "ASR external chunking enabled: "
        f"opening={opening_s:g}s, body_chunk={chunk_s:g}s, chunks={len(plan)}, "
        f"boundary={boundary_mode}, overlap={overlap_s:g}s"
    )

    if tmpdir is None:
        tmpdir = Path(wav_path).parent
    else:
        tmpdir = Path(tmpdir)

    shifted_results = []
    chunk_report = []

    for item in plan:
        ci = int(item["index"])
        kind = item["kind"]
        core_start_ms = int(item.get("core_start_ms", item.get("start_ms", 0)))
        core_end_ms = int(item.get("core_end_ms", item.get("end_ms", 0)))
        input_start_ms = int(item.get("input_start_ms", core_start_ms))
        input_end_ms = int(item.get("input_end_ms", core_end_ms))
        seg_wav = tmpdir / f"asr_chunk_{ci:03d}_{kind}_{input_start_ms}_{input_end_ms}.wav"

        boundary_silence = item.get("boundary_silence")
        boundary_note = ""
        if boundary_silence:
            boundary_note = f", boundary_silence_mid={boundary_silence.get('mid_ms')}ms"

        print(
            f"ASR chunk {ci + 1}/{len(plan)} "
            f"({kind}): core {core_start_ms/1000:.3f}s -> {core_end_ms/1000:.3f}s, "
            f"input {input_start_ms/1000:.3f}s -> {input_end_ms/1000:.3f}s"
            f"{boundary_note}"
        )

        extract_wav_segment(wav_path, seg_wav, input_start_ms, input_end_ms)

        chunk_args = copy.copy(args)
        # Guard against future accidental recursion if run_asr_backend ever
        # grows its own chunk switch.
        setattr(chunk_args, "no_asr_external_chunking", True)

        chunk_prompt = None
        if ci == 0 or getattr(args, "asr_prompt_every_chunk", False):
            chunk_prompt = initial_prompt

        raw_result = run_asr_backend(
            wav_path=seg_wav,
            args=chunk_args,
            initial_prompt=chunk_prompt,
            duration_ms=(input_end_ms - input_start_ms),
        )

        shifted = shift_whisper_like_result_times(
            raw_result,
            offset_ms=input_start_ms,
            chunk_index=ci,
            chunk_kind=kind,
        )
        trimmed = trim_whisper_like_result_to_core(
            shifted,
            core_start_ms=core_start_ms,
            core_end_ms=core_end_ms,
        )
        shifted_results.append(trimmed)

        raw_wc = count_result_words(shifted)
        wc = count_result_words(trimmed)
        chunk_report.append({
            "index": ci,
            "kind": kind,
            "core_start_ms": core_start_ms,
            "core_end_ms": core_end_ms,
            "input_start_ms": input_start_ms,
            "input_end_ms": input_end_ms,
            "core_duration_ms": core_end_ms - core_start_ms,
            "input_duration_ms": input_end_ms - input_start_ms,
            "boundary_silence": boundary_silence,
            "raw_words": raw_wc,
            "words": wc,
            "trimmed_overlap_words": max(0, raw_wc - wc),
        })
        print(f"ASR chunk {ci + 1}/{len(plan)} words: {wc} (raw={raw_wc}, trimmed={max(0, raw_wc - wc)})")

    merged = merge_whisper_like_results(shifted_results)
    merged["external_chunk_report"] = chunk_report
    merged["external_chunking"] = {
        "enabled": True,
        "opening_chunk_s": opening_s,
        "chunk_s": chunk_s,
        "chunks": len(plan),
        "boundary_mode": boundary_mode,
        "overlap_s": overlap_s,
        "boundary_search_ms": boundary_search_ms,
        "min_silence_ms": min_silence_ms,
    }
    return merged



def extract_asr_words(whisper_result):
    words = []

    for segment in whisper_result.get("segments", []):
        for w in segment.get("words", []):
            text = str(w.get("word", "")).strip()

            if not text:
                continue

            if "start" not in w or "end" not in w:
                continue

            norm = normalize_arabic(text)

            if not norm:
                continue

            start_ms = int(round(float(w["start"]) * 1000))
            end_ms = int(round(float(w["end"]) * 1000))

            if end_ms <= start_ms:
                continue

            words.append({
                "word": text,
                "norm": norm,
                "start_ms": start_ms,
                "end_ms": end_ms,
            })

    return words


def recover_dropped_opening_asr(
    asr_words,
    args,
    wav_path,
    tmpdir,
    chapter_id=None,
    duration_ms=None,
):
    """Recover a surah opening that Whisper dropped from its transcription.

    Whisper (both openai-whisper and faster-whisper) can SKIP the surah
    opening when it sits inside the first 30s decode window right after the
    basmala - e.g. the long madd on huruf muqatta'at like حمٓ. The dropped
    region shows up as a large time gap between two consecutive ASR words
    near the start, with REAL recitation audio inside it but ZERO ASR
    tokens. The old behaviour then "estimated" those opening words by static
    redistribution across the gap, which violates the hard requirement that
    the opening get REAL Whisper word timestamps.

    This re-transcribes ONLY the dropped gap as its own clip, with NO Quran
    initial_prompt (the prompt + preceding basmala is what makes Whisper skip
    the opening; clipping to start at the opening reliably recovers it). The
    recovered words carry their REAL timestamps (madd preserved); we offset
    them by the clip start and splice them back into ``asr_words``. No
    static/estimated timings are introduced. If the gap is genuine silence,
    nothing is recovered and ``asr_words`` is returned unchanged.
    """
    report = {
        "attempted": False,
        "recovered": 0,
        "leading": False,
        "gap_start_ms": None,
        "gap_end_ms": None,
        "windows": [],
        "words": [],
    }

    if getattr(args, "no_opening_gap_recover", False):
        return asr_words, report

    if not asr_words:
        return asr_words, report

    min_gap_ms = int(getattr(args, "opening_gap_recover_min_ms", 4000))
    scan_ms = int(getattr(args, "opening_gap_recover_scan_ms", 90000))
    window_ms = max(5000, int(getattr(args, "opening_gap_recover_window_ms", 30000)))

    # Locate the dropped opening region. Two distinct cases:
    #
    #   (A) LEADING drop: Whisper emitted its FIRST word only well AFTER the
    #       opening, because it skipped the muqatta'at recited from the very
    #       start (e.g. طسٓ in An-Naml, or a حمٓ with no transcribed basmala
    #       before it). The dropped region is [0 .. first_asr_word.start] and
    #       has NO left ASR anchor, so the inter-word scan below can never see
    #       it - the old behaviour then statically estimated those opening
    #       words from 0ms. This case is checked FIRST because it is the
    #       earliest possible region.
    #
    #   (B) INTER-WORD drop: the opening sits in a large gap between two
    #       consecutive ASR words (e.g. حمٓ right after a transcribed basmala).
    #
    # Both are recovered by the same clip re-transcription below; only the gap
    # bounds and the splice index differ.
    gap_i = None
    leading = False
    gap_start_ms = None
    gap_end_ms = None

    leading_enabled = not getattr(args, "no_opening_leading_gap_recover", False)
    first_start = int(asr_words[0]["start_ms"])

    if leading_enabled and min_gap_ms <= first_start <= scan_ms:
        gap_i = 0
        leading = True
        gap_start_ms = 0
        gap_end_ms = first_start
    else:
        if len(asr_words) < 2:
            return asr_words, report
        for i in range(1, len(asr_words)):
            prev_end = int(asr_words[i - 1]["end_ms"])
            if prev_end > scan_ms:
                break
            gap = int(asr_words[i]["start_ms"]) - prev_end
            if gap >= min_gap_ms:
                gap_i = i
                break

        if gap_i is None:
            return asr_words, report

        gap_start_ms = int(asr_words[gap_i - 1]["end_ms"])
        gap_end_ms = int(asr_words[gap_i]["start_ms"])

    report["attempted"] = True
    report["leading"] = leading
    report["gap_start_ms"] = gap_start_ms
    report["gap_end_ms"] = gap_end_ms

    print(
        f"Opening gap recovery: detected dropped region "
        f"{gap_start_ms}ms..{gap_end_ms}ms ({gap_end_ms - gap_start_ms}ms) "
        f"between ASR words "
        f"[{'<surah start>' if leading else asr_words[gap_i - 1]['word']}] and "
        f"[{asr_words[gap_i]['word'] if gap_i < len(asr_words) else '<surah end>'}]; "
        f"re-transcribing that clip without the Quran initial_prompt."
    )

    # Split the gap into <= window_ms sub-windows. Whisper internally windows
    # at 30s, so keeping each clip within one window avoids re-dropping a long
    # opening. A small overlap keeps a word straddling a boundary recoverable.
    overlap_ms = 1000
    spans = []
    s = gap_start_ms
    while s < gap_end_ms:
        e = min(gap_end_ms, s + window_ms)
        spans.append((s, e))
        if e >= gap_end_ms:
            break
        s = e - overlap_ms

    recovered = []
    for (clip_start_ms, clip_end_ms) in spans:
        clip_path = str(Path(tmpdir) / f"opening_gap_{clip_start_ms}_{clip_end_ms}.wav")
        try:
            extract_wav_segment(wav_path, clip_path, clip_start_ms, clip_end_ms)
        except Exception as e:
            print(f"  Opening gap recovery: failed to extract clip {clip_start_ms}..{clip_end_ms}ms: {e}")
            continue

        try:
            clip_result = run_asr_backend(
                wav_path=clip_path,
                args=args,
                initial_prompt=None,
                duration_ms=clip_end_ms - clip_start_ms,
            )
        except Exception as e:
            print(f"  Opening gap recovery: ASR failed on clip {clip_start_ms}..{clip_end_ms}ms: {e}")
            continue

        clip_words = extract_asr_words(clip_result)
        win_recovered = 0
        for w in clip_words:
            start_ms = clip_start_ms + int(w["start_ms"])
            end_ms = clip_start_ms + int(w["end_ms"])
            # Keep strictly inside the dropped region so we never duplicate the
            # already-transcribed words on either side of the gap.
            if end_ms <= gap_start_ms or start_ms >= gap_end_ms:
                continue
            start_ms = max(gap_start_ms, start_ms)
            end_ms = min(gap_end_ms, end_ms)
            if end_ms <= start_ms:
                continue
            recovered.append({
                "word": w["word"],
                "norm": w["norm"],
                "start_ms": start_ms,
                "end_ms": end_ms,
            })
            win_recovered += 1
        report["windows"].append({
            "clip_start_ms": clip_start_ms,
            "clip_end_ms": clip_end_ms,
            "recovered": win_recovered,
        })

    if not recovered:
        print("  Opening gap recovery: no words recovered (region may be genuine silence); leaving ASR unchanged.")
        return asr_words, report

    # Sort and dedupe duplicates produced by overlapping windows.
    recovered.sort(key=lambda x: (x["start_ms"], x["end_ms"]))
    deduped = []
    for w in recovered:
        if deduped:
            last = deduped[-1]
            mid = (w["start_ms"] + w["end_ms"]) // 2
            if mid <= last["end_ms"] and w["norm"] == last["norm"]:
                # Same word captured twice across an overlap; keep the earlier.
                continue
        deduped.append(w)

    # Snap the FIRST recovered opening word's start back to the true audio
    # speech onset. Whisper's DTW clips the soft onset of a long sustained madd
    # (e.g. the meem madd in حمٓ), so it places the word START late and cuts off
    # the beginning of the madd. The end of the LAST silence region before the
    # recovered start is the real speech onset; pulling the start back to it
    # RESPECTS the madd using REAL audio evidence (a measured silence boundary),
    # not static redistribution. The word identity and end stay from Whisper;
    # only the clipped onset is corrected, and the original Whisper start is kept
    # in the report for transparency.
    report["onset_snap"] = {"applied": False}
    if not getattr(args, "no_opening_onset_snap", False) and deduped:
        snap_min_ms = int(getattr(args, "opening_onset_snap_min_ms", 200))
        snap_max_ms = int(getattr(args, "opening_onset_snap_max_ms", 4000))
        first = deduped[0]
        whisper_start_ms = int(first["start_ms"])
        sil = []
        try:
            _samples, _sr = read_wav_pcm16_mono_samples(wav_path)
            sil = detect_silence_regions_ms(
                _samples, _sr, gap_start_ms, whisper_start_ms, min_silence_ms=140
            )
        except Exception as e:
            print(f"  Opening gap recovery: onset snap skipped (audio read failed: {e}).")
        # End of the last silence region before the recovered start is the real
        # onset of the speech run leading into the first opening word.
        onset_ms = None
        for r in sil:
            if int(r["end_ms"]) <= whisper_start_ms:
                onset_ms = int(r["end_ms"])
        if onset_ms is not None:
            onset_ms = max(gap_start_ms, onset_ms)
            delta = whisper_start_ms - onset_ms
            if snap_min_ms <= delta <= snap_max_ms and onset_ms < int(first["end_ms"]):
                first["start_ms"] = onset_ms
                report["onset_snap"] = {
                    "applied": True,
                    "word": first["word"],
                    "whisper_start_ms": whisper_start_ms,
                    "snapped_start_ms": onset_ms,
                    "delta_ms": delta,
                }
                print(
                    f"Opening gap recovery: snapped first word [{first['word']}] start "
                    f"{whisper_start_ms}ms -> {onset_ms}ms to the real audio onset "
                    f"(recovered {delta}ms of clipped madd)."
                )

    # Right-anchor dedupe: if the clip re-transcribed a trailing token that is
    # the SAME word as the next original ASR word (asr_words[gap_i]) and ends
    # right at the gap boundary, drop it so we never insert a short duplicate of
    # the following real word just before it. Applies to both leading and
    # inter-word gaps.
    if deduped and gap_i < len(asr_words):
        right_norm = asr_words[gap_i].get("norm")
        boundary_ms = 200
        while (
            deduped
            and right_norm
            and deduped[-1]["norm"] == right_norm
            and int(deduped[-1]["end_ms"]) >= gap_end_ms - boundary_ms
        ):
            deduped.pop()

    if not deduped:
        print("  Opening gap recovery: recovered tokens were boundary duplicates; leaving ASR unchanged.")
        return asr_words, report

    # Splice recovered words into the gap and keep global time order.
    new_words = asr_words[:gap_i] + deduped + asr_words[gap_i:]
    new_words.sort(key=lambda x: (int(x["start_ms"]), int(x["end_ms"])))

    report["recovered"] = len(deduped)
    report["words"] = deduped
    print(
        f"Opening gap recovery: recovered {len(deduped)} REAL ASR words in "
        f"{gap_start_ms}ms..{gap_end_ms}ms (first: [{deduped[0]['word']}] @ "
        f"{deduped[0]['start_ms']}ms, last: [{deduped[-1]['word']}] @ {deduped[-1]['end_ms']}ms)."
    )
    return new_words, report


def map_region_fuzzy(q_region, a_region, min_score):
    # Conservative monotonic fuzzy mapping inside a non-equal global region.
    mapped_pairs = {}
    used_a = set()

    # Pass 1: exact normalized matches.
    for qi, qw in enumerate(q_region):
        for ai, aw in enumerate(a_region):
            if ai in used_a:
                continue
            if qw["norm"] and qw["norm"] == aw["norm"]:
                mapped_pairs[qi] = ai
                used_a.add(ai)
                break

    # Pass 2: fuzzy nearest, preserving order.
    last_a = -1
    for qi, qw in enumerate(q_region):
        if qi in mapped_pairs:
            last_a = max(last_a, mapped_pairs[qi])
            continue

        best_ai = None
        best_score = 0.0
        for ai, aw in enumerate(a_region):
            if ai in used_a or ai <= last_a:
                continue
            score = similarity(qw["norm"], aw["norm"])
            if score > best_score:
                best_score = score
                best_ai = ai

        if best_ai is not None and best_score >= min_score:
            mapped_pairs[qi] = best_ai
            used_a.add(best_ai)
            last_a = best_ai

    return mapped_pairs


# Module-level cache of RAW fuzzy similarity ratios, keyed by (quran_norm,
# asr_norm). The raw ratio is min_score-independent, so the same cache is reused
# safely across every alignment pass (all min_score variants, both ASR sources)
# and across chapters (the Quran vocabulary is shared). Only fully-computed
# ratios are stored; pruned pairs (cheaply rejected below the current min_score)
# are NEVER cached, because their rejection is min_score-dependent and a later
# lower-threshold pass must be free to recompute them. Bounded to cap memory on
# the longest surahs.
_SIMILARITY_RATIO_CACHE = {}
_SIMILARITY_RATIO_CACHE_MAX = 6_000_000


def dp_global_align(quran_words, asr_words, min_score=0.35):
    """
    Needleman-Wunsch style global alignment using fuzzy similarity.

    This is stronger than SequenceMatcher for long Quran surahs because it can
    align repeated words like "ما" and regions with small ASR/Quran differences.

    Performance: the inner cell loop is the hottest path in the whole pipeline
    (O(n*m) over surahs up to ~6000x6000). Two optimizations make it far faster
    while producing BIT-IDENTICAL output to the naive version:
      1. A shared raw-ratio cache avoids recomputing SequenceMatcher.ratio() for
         repeated (quran, asr) word pairs (the Quran vocabulary is highly
         repetitive, and every pass re-sees the same pairs).
      2. SequenceMatcher.real_quick_ratio()/quick_ratio() are O(1)/O(len) upper
         bounds on ratio(); when an upper bound is below min_score the true ratio
         is guaranteed below min_score too, so the costly ratio() is skipped and
         the pair is treated as a mismatch exactly as before.
    """
    n = len(quran_words)
    m = len(asr_words)

    # Score design:
    # - exact/fuzzy match has positive reward depending on similarity
    # - gaps are allowed but penalized
    # - low similarity pairs are treated as mismatches and usually avoided
    gap_penalty = -38
    low_match_penalty = -45

    prev = np.zeros(m + 1, dtype=np.int32)
    curr = np.zeros(m + 1, dtype=np.int32)

    # 0 diag, 1 up/delete Quran word, 2 left/skip ASR word
    trace = np.zeros((n + 1, m + 1), dtype=np.uint8)

    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + gap_penalty
        trace[0, j] = 2

    ratio_cache = _SIMILARITY_RATIO_CACHE
    if len(ratio_cache) > _SIMILARITY_RATIO_CACHE_MAX:
        ratio_cache.clear()
    ratio_get = ratio_cache.get
    # Per-call match_score cache. min_score is fixed within a single call, so the
    # thresholded match_score for a given (qn, an) pair is well-defined and can be
    # reused for every repeated occurrence of that pair in the matrix. This also
    # means a pruned (dissimilar) pair only pays the quick-ratio check once per
    # pass instead of once per cell. Freed when the call returns.
    score_cache = {}
    score_get = score_cache.get
    asr_norms = [w["norm"] for w in asr_words]
    SM = SequenceMatcher

    for i in range(1, n + 1):
        curr[0] = prev[0] + gap_penalty
        trace[i, 0] = 1

        qn = quran_words[i - 1]["norm"]

        for j in range(1, m + 1):
            an = asr_norms[j - 1]
            key = (qn, an)

            # Sentinel -999 means "match_score not yet computed this pass".
            match_score = score_get(key, -999)
            if match_score == -999:
                # raw ratio (>= 0); -1.0 sentinel means "not in shared cache".
                sim = ratio_get(key, -1.0)
                if sim < 0.0:
                    if not qn or not an:
                        # Original path: similarity()==0.0 then thresholded.
                        # Faithful for every min_score (incl. <= 0).
                        match_score = 0 if 0.0 >= min_score else low_match_penalty
                    elif qn == an:
                        ratio_cache[key] = 1.0
                        match_score = int(round(1.0 * 100))
                    else:
                        sm = SM(None, qn, an)
                        if sm.real_quick_ratio() < min_score or sm.quick_ratio() < min_score:
                            # ratio() <= quick_ratio() < min_score -> guaranteed
                            # mismatch; identical to the naive path, but cheaper.
                            match_score = low_match_penalty
                        else:
                            sim = sm.ratio()
                            ratio_cache[key] = sim
                            match_score = int(round(sim * 100)) if sim >= min_score else low_match_penalty
                else:
                    match_score = int(round(sim * 100)) if sim >= min_score else low_match_penalty
                score_cache[key] = match_score

            diag = prev[j - 1] + match_score
            up = prev[j] + gap_penalty
            left = curr[j - 1] + gap_penalty

            if diag >= up and diag >= left:
                curr[j] = diag
                trace[i, j] = 0
            elif up >= left:
                curr[j] = up
                trace[i, j] = 1
            else:
                curr[j] = left
                trace[i, j] = 2

        prev, curr = curr, prev

    pairs = {}
    i, j = n, m

    while i > 0 or j > 0:
        direction = trace[i, j]

        if i > 0 and j > 0 and direction == 0:
            score = similarity(quran_words[i - 1]["norm"], asr_words[j - 1]["norm"])
            if score >= min_score:
                pairs[i - 1] = {
                    "asr_index": j - 1,
                    "score": round(score, 3),
                    "method": "dp_global",
                }
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or direction == 1):
            i -= 1
        else:
            j -= 1

    return pairs


def map_region_fuzzy(q_region, a_region, min_score):
    # Conservative monotonic fuzzy mapping inside a non-equal region.
    mapped_pairs = {}
    used_a = set()

    for qi, qw in enumerate(q_region):
        for ai, aw in enumerate(a_region):
            if ai in used_a:
                continue
            if qw["norm"] and qw["norm"] == aw["norm"]:
                mapped_pairs[qi] = ai
                used_a.add(ai)
                break

    last_a = -1
    for qi, qw in enumerate(q_region):
        if qi in mapped_pairs:
            last_a = max(last_a, mapped_pairs[qi])
            continue

        best_ai = None
        best_score = 0.0

        for ai, aw in enumerate(a_region):
            if ai in used_a or ai <= last_a:
                continue
            score = similarity(qw["norm"], aw["norm"])
            if score > best_score:
                best_score = score
                best_ai = ai

        if best_ai is not None and best_score >= min_score:
            mapped_pairs[qi] = best_ai
            used_a.add(best_ai)
            last_a = best_ai

    return mapped_pairs


def sequence_matcher_align(quran_words, asr_words, min_score=0.40):
    q_norm = [w["norm"] for w in quran_words]
    a_norm = [w["norm"] for w in asr_words]

    sm = SequenceMatcher(None, q_norm, a_norm, autojunk=False)
    opcodes = sm.get_opcodes()

    mapped_by_q = {}
    exact_blocks = 0
    fuzzy_regions = 0
    fuzzy_matched = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            exact_blocks += 1
            length = min(i2 - i1, j2 - j1)
            for offset in range(length):
                qi = i1 + offset
                ai = j1 + offset
                mapped_by_q[qi] = {
                    "asr_index": ai,
                    "score": 1.0,
                    "method": "exact_global",
                }
            continue

        q_region = quran_words[i1:i2]
        a_region = asr_words[j1:j2]

        if q_region and a_region:
            fuzzy_regions += 1
            local_pairs = map_region_fuzzy(q_region, a_region, min_score=min_score)
            for local_qi, local_ai in local_pairs.items():
                qi = i1 + local_qi
                ai = j1 + local_ai
                mapped_by_q[qi] = {
                    "asr_index": ai,
                    "score": round(similarity(quran_words[qi]["norm"], asr_words[ai]["norm"]), 3),
                    "method": "fuzzy_region",
                }
                fuzzy_matched += 1

    dprint(f"Global exact blocks: {exact_blocks}")
    dprint(f"Fuzzy regions: {fuzzy_regions}")
    dprint(f"Fuzzy matched: {fuzzy_matched}")

    return mapped_by_q


def complete_missing_inside_main(mapped_words, min_word_ms=120, default_word_ms=300):
    """
    Make output complete inside the main script, before saving JSON.

    This does not change Quran text or word positions. It only assigns timing
    to missing words using nearby matched timings, and marks debug metadata.
    """
    result = [dict(x) for x in mapped_words]
    n = len(result)
    i = 0
    completed = 0

    while i < n:
        if result[i].get("start_ms") is not None and result[i].get("end_ms") is not None:
            i += 1
            continue

        start = i
        while i < n and (result[i].get("start_ms") is None or result[i].get("end_ms") is None):
            i += 1
        end = i - 1
        indices = list(range(start, end + 1))
        count = len(indices)

        prev_idx = None
        next_idx = None

        for j in range(start - 1, -1, -1):
            if result[j].get("start_ms") is not None and result[j].get("end_ms") is not None:
                prev_idx = j
                break

        for j in range(end + 1, n):
            if result[j].get("start_ms") is not None and result[j].get("end_ms") is not None:
                next_idx = j
                break

        if prev_idx is not None and next_idx is not None:
            left = int(result[prev_idx]["end_ms"])
            right = int(result[next_idx]["start_ms"])

            if right - left >= count * min_word_ms:
                cursor = left
                span = right - left
                for pos, idx in enumerate(indices):
                    s = cursor
                    e = right if pos == count - 1 else left + round(span * (pos + 1) / count)
                    result[idx]["start_ms"] = int(s)
                    result[idx]["end_ms"] = int(e)
                    result[idx]["estimated"] = True
                    result[idx]["method"] = "main_gap_complete"
                    cursor = e
                    completed += 1
            else:
                # No real gap; split the longer neighbor when possible.
                prev_dur = int(result[prev_idx]["end_ms"] - result[prev_idx]["start_ms"])
                next_dur = int(result[next_idx]["end_ms"] - result[next_idx]["start_ms"])

                if next_dur >= (count + 1) * min_word_ms:
                    s0 = int(result[next_idx]["start_ms"])
                    e0 = int(result[next_idx]["end_ms"])
                    span = e0 - s0
                    cursor = s0

                    for pos, idx in enumerate(indices):
                        e = s0 + round(span * (pos + 1) / (count + 1))
                        result[idx]["start_ms"] = int(cursor)
                        result[idx]["end_ms"] = int(e)
                        result[idx]["estimated"] = True
                        result[idx]["method"] = "main_split_next"
                        cursor = e
                        completed += 1

                    result[next_idx]["start_ms"] = int(cursor)
                    result[next_idx]["split_adjusted"] = True

                elif prev_dur >= (count + 1) * min_word_ms:
                    s0 = int(result[prev_idx]["start_ms"])
                    e0 = int(result[prev_idx]["end_ms"])
                    span = e0 - s0

                    # Keep the previous word first part, then missing words after it.
                    new_prev_end = s0 + round(span / (count + 1))
                    result[prev_idx]["end_ms"] = int(new_prev_end)
                    result[prev_idx]["split_adjusted"] = True

                    cursor = new_prev_end
                    for pos, idx in enumerate(indices):
                        e = e0 if pos == count - 1 else s0 + round(span * (pos + 2) / (count + 1))
                        result[idx]["start_ms"] = int(cursor)
                        result[idx]["end_ms"] = int(e)
                        result[idx]["estimated"] = True
                        result[idx]["method"] = "main_split_prev"
                        cursor = e
                        completed += 1
                else:
                    cursor = left
                    for idx in indices:
                        result[idx]["start_ms"] = int(cursor)
                        result[idx]["end_ms"] = int(cursor + default_word_ms)
                        result[idx]["estimated"] = True
                        result[idx]["method"] = "main_estimated_after_prev"
                        cursor += default_word_ms
                        completed += 1

        elif prev_idx is not None:
            cursor = int(result[prev_idx]["end_ms"])
            for idx in indices:
                result[idx]["start_ms"] = int(cursor)
                result[idx]["end_ms"] = int(cursor + default_word_ms)
                result[idx]["estimated"] = True
                result[idx]["method"] = "main_estimated_after_prev"
                cursor += default_word_ms
                completed += 1

        elif next_idx is not None:
            cursor = max(0, int(result[next_idx]["start_ms"]) - count * default_word_ms)
            for idx in indices:
                result[idx]["start_ms"] = int(cursor)
                result[idx]["end_ms"] = int(cursor + default_word_ms)
                result[idx]["estimated"] = True
                result[idx]["method"] = "main_estimated_before_next"
                cursor += default_word_ms
                completed += 1

    print(f"Completed missing inside main script: {completed}")
    return result


def map_quran_to_audio(quran_words, asr_words, min_score=0.55, lookahead=15, alignment="dp", complete_output=False):
    if alignment == "sequence":
        mapped_by_q = sequence_matcher_align(quran_words, asr_words, min_score=min_score)
    else:
        mapped_by_q = dp_global_align(quran_words, asr_words, min_score=min_score)
        print("Alignment method: dp_global")

    mapped = []
    matched = 0

    for qi, q_word in enumerate(quran_words):
        hit = mapped_by_q.get(qi)
        if hit:
            ai = hit["asr_index"]
            a = asr_words[ai]
            mapped.append({
                **q_word,
                "start_ms": a["start_ms"],
                "end_ms": a["end_ms"],
                "score": hit["score"],
                "estimated": False,
                "asr_word": a["word"],
                "asr_index": ai,
                "method": hit["method"],
            })
            matched += 1
        else:
            mapped.append({
                **q_word,
                "start_ms": None,
                "end_ms": None,
                "score": 0,
                "estimated": False,
                "asr_word": None,
                "method": "missing",
            })

    if complete_output:
        mapped = complete_missing_inside_main(mapped)

    return mapped, matched

def interpolate_missing(mapped):
    result = [dict(x) for x in mapped]
    n = len(result)
    i = 0

    while i < n:
        if result[i]["start_ms"] is not None:
            i += 1
            continue

        start_missing = i

        while i < n and result[i]["start_ms"] is None:
            i += 1

        end_missing = i - 1
        prev_idx = start_missing - 1
        next_idx = i if i < n else None

        if prev_idx >= 0 and next_idx is not None:
            gap_start = result[prev_idx]["end_ms"]
            gap_end = result[next_idx]["start_ms"]

            if gap_end <= gap_start:
                continue

            count = end_missing - start_missing + 1
            step = max(1, (gap_end - gap_start) // (count + 1))

            for offset, idx in enumerate(range(start_missing, end_missing + 1), start=1):
                s = gap_start + step * offset
                e = min(s + step, gap_end)

                result[idx]["start_ms"] = int(s)
                result[idx]["end_ms"] = int(e)
                result[idx]["estimated"] = True

    return result



def repair_missing_by_splitting_next(mapped):
    """
    Fix cases where Whisper merges a short word with the next word.

    Example in Al-Fatihah:
      word 8: وَلَا        -> missing
      word 9: ٱلضَّآلِّينَ -> has a long segment covering "ولا الضالين"

    If a missing block is directly before a matched word and there is no positive
    timing gap to interpolate, split the next matched word's interval among the
    missing word(s) and the next word using normalized text length weights.
    """
    result = [dict(x) for x in mapped]
    n = len(result)
    i = 0

    while i < n:
        if result[i].get("start_ms") is not None:
            i += 1
            continue

        start_missing = i
        while i < n and result[i].get("start_ms") is None:
            i += 1
        end_missing = i - 1

        next_idx = i if i < n else None
        prev_idx = start_missing - 1

        if next_idx is None:
            continue

        next_start = result[next_idx].get("start_ms")
        next_end = result[next_idx].get("end_ms")

        if next_start is None or next_end is None:
            continue

        # Only split if regular interpolation cannot work:
        # no prev, or prev end is >= next start, or the next segment is unusually long.
        prev_end = result[prev_idx].get("end_ms") if prev_idx >= 0 else None
        no_positive_gap = prev_end is None or prev_end >= next_start

        total_duration = int(next_end - next_start)
        missing_count = end_missing - start_missing + 1

        if total_duration <= 0:
            continue

        # Minimum duration check to avoid damaging short normal words.
        if total_duration < 250 * (missing_count + 1):
            continue

        if not no_positive_gap:
            continue

        group_indices = list(range(start_missing, end_missing + 1)) + [next_idx]

        weights = []
        for idx in group_indices:
            norm = result[idx].get("norm") or normalize_arabic(result[idx].get("text", ""))
            weights.append(max(1, len(norm)))

        weight_sum = max(1, sum(weights))

        cursor = int(next_start)

        for pos, idx in enumerate(group_indices):
            if pos == len(group_indices) - 1:
                s = cursor
                e = int(next_end)
            else:
                dur = max(120, int(round(total_duration * weights[pos] / weight_sum)))
                s = cursor
                e = min(int(next_end), s + dur)
                cursor = e

            result[idx]["start_ms"] = int(s)
            result[idx]["end_ms"] = int(e)

            if idx != next_idx:
                result[idx]["estimated"] = True
                result[idx]["split_from_next"] = True
            else:
                result[idx]["estimated"] = bool(result[idx].get("estimated", False))
                result[idx]["split_adjusted"] = True

    return result




# ---------------------------------------------------------------------------
# v18 Madd-only post process
# ---------------------------------------------------------------------------

MADD_STRONG_CHARS = set("ٰٓۥۦآ")
MADD_BROAD_CHARS = set("ٰٓۥۦآاىوي")


def has_madd_candidate(word_text, mode="broad"):
    """
    Detect words likely to contain madd/long-vowel stretch.

    strong:
      only clear Uthmani madd signs/dagger alif/maddah.
    broad:
      also long-vowel letters ا و ي ى.
      Does NOT include ٱ so normal definite article does not trigger by itself.
    """
    text = str(word_text or "")

    if mode == "strong":
        return any(ch in text for ch in MADD_STRONG_CHARS)

    return any(ch in text for ch in MADD_BROAD_CHARS)


def load_wav_mono_float(wav_path):
    import wave
    import numpy as _np

    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise RuntimeError("Madd audio pause detection expects 16-bit PCM WAV.")

    audio = _np.frombuffer(frames, dtype=_np.int16).astype("float32")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sr


def find_first_silence_start_ms(samples, sr, start_ms, end_ms, min_silence_ms=90):
    """
    Find first sustained low-energy region between old word end and next word start.

    Used for verse-ending madd:
      extend final word until the recitation sound dies away,
      but do not highlight through a long silence.
    """
    import numpy as _np

    start_ms = int(max(0, start_ms))
    end_ms = int(max(start_ms, end_ms))

    if end_ms - start_ms < min_silence_ms:
        return None

    s = int(sr * start_ms / 1000)
    e = int(sr * end_ms / 1000)
    region = samples[s:e]

    if len(region) < int(sr * min_silence_ms / 1000):
        return None

    frame_ms = 25
    hop_ms = 10
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))

    rms = []
    starts = []

    for i in range(0, max(1, len(region) - frame + 1), hop):
        chunk = region[i:i + frame]
        if len(chunk) == 0:
            continue
        rms.append(float(_np.sqrt(_np.mean(chunk * chunk))))
        starts.append(i)

    if not rms:
        return None

    rms = _np.array(rms, dtype="float32")
    p10 = float(_np.percentile(rms, 10))
    p20 = float(_np.percentile(rms, 20))
    med = float(_np.percentile(rms, 50))

    threshold = max(12.0, min(p20 * 1.20, med * 0.35), p10 * 1.55)
    low = rms <= threshold

    need = max(1, int(round(min_silence_ms / hop_ms)))

    run_start = None
    for idx, is_low in enumerate(low):
        if is_low and run_start is None:
            run_start = idx
        elif (not is_low) and run_start is not None:
            if idx - run_start >= need:
                return int(start_ms + starts[run_start] * 1000 / sr)
            run_start = None

    if run_start is not None and len(low) - run_start >= need:
        return int(start_ms + starts[run_start] * 1000 / sr)

    return None


def apply_madd_only_extension(
    mapped_words,
    wav_path=None,
    mode="broad",
    max_extend_ms=1800,
    verse_end_max_extend_ms=3500,
    min_gap_ms=80,
    pause_min_ms=90,
    use_audio_pauses=True,
):
    """
    Extend only end_ms for madd/long-vowel words.

    Hard rules:
    - NEVER changes start_ms.
    - NEVER changes word order.
    - NEVER remaps Quran text to ASR.
    - Only moves end_ms forward.
    - Does not pass the next word start.
    """
    result = [dict(w) for w in mapped_words]
    changes = []

    samples = None
    sr = None

    if wav_path and use_audio_pauses:
        try:
            samples, sr = load_wav_mono_float(wav_path)
        except Exception as e:
            print(f"WARNING: Madd audio pause detection disabled: {e}")
            samples, sr = None, None

    n = len(result)

    for i, w in enumerate(result):
        if w.get("start_ms") is None or w.get("end_ms") is None:
            continue

        text = w.get("text", "")
        if not has_madd_candidate(text, mode=mode):
            continue

        old_end = int(w["end_ms"])
        start = int(w["start_ms"])

        next_idx = None
        for j in range(i + 1, n):
            if result[j].get("start_ms") is not None and result[j].get("end_ms") is not None:
                next_idx = j
                break

        if next_idx is None:
            continue

        next_start = int(result[next_idx]["start_ms"])

        if next_start <= old_end + min_gap_ms:
            continue

        same_verse = result[next_idx].get("verse_key") == w.get("verse_key")
        max_ext = int(max_extend_ms if same_verse else verse_end_max_extend_ms)

        limit = min(next_start, old_end + max_ext)
        new_end = limit

        if not same_verse and samples is not None and sr is not None:
            silence_start = find_first_silence_start_ms(
                samples=samples,
                sr=sr,
                start_ms=old_end,
                end_ms=next_start,
                min_silence_ms=pause_min_ms,
            )
            if silence_start is not None and silence_start > old_end + min_gap_ms:
                new_end = min(new_end, int(silence_start))

        new_end = max(old_end, min(int(new_end), next_start))

        if new_end > old_end + min_gap_ms:
            result[i]["end_ms"] = int(new_end)
            result[i]["madd_extended"] = True
            result[i]["madd_old_end_ms"] = old_end
            result[i]["method"] = str(result[i].get("method", "")) + "+madd_only_extend"

            changes.append({
                "index": i,
                "verse_key": w.get("verse_key"),
                "position": w.get("position"),
                "text": text,
                "same_verse_next": same_verse,
                "start_ms": start,
                "old_end_ms": old_end,
                "new_end_ms": int(new_end),
                "next_word": result[next_idx].get("text"),
                "next_start_ms": next_start,
                "extended_by_ms": int(new_end - old_end),
                "mode": mode,
            })

    print(f"Madd-only end extensions: {len(changes)}")
    return result, changes



def build_output(chapter_id, duration_ms, mapped_words, interpolate=False):
    if interpolate:
        mapped_words = interpolate_missing(mapped_words)
        mapped_words = repair_missing_by_splitting_next(mapped_words)

    verses = {}

    for w in mapped_words:
        verses.setdefault(w["verse_key"], []).append(w)

    verse_timings = []

    for verse_key, words in verses.items():
        valid = [
            w for w in words
            if w["start_ms"] is not None and w["end_ms"] is not None
        ]

        if valid:
            timestamp_from = min(w["start_ms"] for w in valid)
            timestamp_to = max(w["end_ms"] for w in valid)
        else:
            timestamp_from = 0
            timestamp_to = 0

        segments = []

        for w in words:
            if w["start_ms"] is None or w["end_ms"] is None:
                segments.append([int(w["position"])])
            else:
                segments.append([
                    int(w["position"]),
                    int(w["start_ms"]),
                    int(w["end_ms"]),
                ])

        verse_timings.append({
            "verse_key": verse_key,
            "timestamp_from": int(timestamp_from),
            "timestamp_to": int(timestamp_to),
            "duration": int(max(0, timestamp_to - timestamp_from)),
            "segments": segments,
        })

    return {
        "chapter_id": int(chapter_id),
        "duration": int(duration_ms),
        "format": "mp3",
        "verse_timings": verse_timings,
    }



def derive_audio_pattern_from_reciter_url(value):
    """
    Accept:
      https://site/reciter
      https://site/reciter/
      https://site/reciter/001.mp3
      https://site/reciter/{chapter}.mp3

    Return:
      https://site/reciter/{chapter}.mp3
    """
    value = str(value).strip()

    if "{chapter}" in value:
        return value

    value = value.rstrip("/")

    last = value.split("/")[-1]

    if last.lower().endswith(".mp3"):
        parent = "/".join(value.split("/")[:-1])
        return parent.rstrip("/") + "/{chapter}.mp3"

    return value + "/{chapter}.mp3"


def guess_reciter_name(value):
    value = str(value).strip().rstrip("/")

    if "{chapter}" in value:
        parts = value.split("/")
        if len(parts) >= 2:
            return parts[-2] or "reciter"
        return "reciter"

    last = value.split("/")[-1]

    if last.lower().endswith(".mp3"):
        parts = value.split("/")
        if len(parts) >= 2:
            return parts[-2] or "reciter"
        return "reciter"

    return last or "reciter"


def count_missing_output_segments(output):
    missing = 0
    for verse in output.get("verse_timings", []):
        for seg in verse.get("segments", []):
            if len(seg) != 3:
                missing += 1
    return missing


def count_estimated_mapped_words(mapped_words):
    return sum(1 for w in mapped_words if bool(w.get("estimated")))


def ensure_complete_output_or_fail(output, out_path):
    missing = count_missing_output_segments(output)

    if missing:
        raise RuntimeError(
            f"Output still has {missing} missing segments after completion: {out_path}"
        )

    return missing


# ---------------------------------------------------------------------------
# v21 quality/retry helpers
# ---------------------------------------------------------------------------

INTRO_PHRASES = [
    # Longest isti'adha variants first.
    ["اعوذ", "بالله", "السميع", "العليم", "من", "الشيطان", "الرجيم", "من", "همزه", "ونفخه", "ونفثه"],
    ["اعوذ", "بالله", "السميع", "العليم", "من", "الشيطان", "الرجيم"],
    ["اعوذ", "بالله", "العظيم", "من", "الشيطان", "الرجيم"],
    ["اعوذ", "بالله", "من", "الشيطان", "الرجيم"],
    ["اعوذ", "بالله", "من", "الشيطن", "الرجيم"],
    ["اعوذ", "بالله", "السميع", "العليم"],
    ["بسم", "الله", "الرحمن", "الرحيم"],
]

# Chapters that start with huruf muqatta'at.
# Opening-safe mode is stricter for these because Whisper often skips or
# misreads: الم، حم، طس، طسم، كهيعص، يس، ص، ق، ن، ...
MUQATTAAT_CHAPTERS = {
    2: "الم",
    3: "الم",
    7: "المص",
    10: "الر",
    11: "الر",
    12: "الر",
    13: "المر",
    14: "الر",
    15: "الر",
    19: "كهيعص",
    20: "طه",
    26: "طسم",
    27: "طس",
    28: "طسم",
    29: "الم",
    30: "الم",
    31: "الم",
    32: "الم",
    36: "يس",
    38: "ص",
    40: "حم",
    41: "حم",
    42: "حم عسق",
    43: "حم",
    44: "حم",
    45: "حم",
    46: "حم",
    50: "ق",
    68: "ن",
}


def is_muqattaat_chapter(chapter_id):
    try:
        return int(chapter_id) in MUQATTAAT_CHAPTERS
    except Exception:
        return False


def chapter_muqattaat_text(chapter_id):
    try:
        return MUQATTAAT_CHAPTERS.get(int(chapter_id))
    except Exception:
        return None



def is_opening_muqattaat_word(word_text, chapter_id=None, word_index=0):
    """
    v39 strict detection:
    Detect only actual opening huruf muqatta'at words, not any short word.
    v32-v38 had a broad fallback that incorrectly flagged words like
    إِنَّا in Surah 43 as muqatta'at.
    """
    if word_index is not None and int(word_index) > 3:
        return False

    norm = normalize_arabic(word_text or "")
    if not norm:
        return False

    exact_forms = {
        "الم", "المص", "الر", "المر", "كهيعص", "طه", "طسم", "طس",
        "يس", "ص", "حم", "عسق", "ق", "ن",
    }

    if norm not in exact_forms:
        return False

    if chapter_id is None:
        return True

    try:
        cid = int(chapter_id)
    except Exception:
        return True

    expected_raw = chapter_muqattaat_text(cid) or ""
    if not expected_raw:
        return False

    # Split before normalize, because normalize_arabic may remove spaces.
    expected_parts = [
        normalize_arabic(p)
        for p in re.split(r"\s+", expected_raw.strip())
        if normalize_arabic(p)
    ]

    expected_joined = normalize_arabic(expected_raw)

    if len(expected_parts) <= 1:
        return norm == expected_joined

    if int(word_index) < len(expected_parts):
        return norm == expected_parts[int(word_index)]

    return False


def muqattaat_madd_weight(word_text, chapter_id=None, word_index=0, base_weight=1, multiplier=7):
    """
    Huruf muqatta'at may be pronounced by letter names with madd:
    حم is closer to "حا ميم", not a two-letter normal word.
    When enabled, it may need more time weight during opening-gap distribution.
    """
    norm = normalize_arabic(word_text or "")
    if not is_opening_muqattaat_word(word_text, chapter_id=chapter_id, word_index=word_index):
        return max(1, int(base_weight))

    # Approximate pronounced length by known letter-name expansion.
    pronounced_units = {
        "حم": 6,       # حا + ميم
        "الم": 9,      # الف + لام + ميم
        "المص": 12,
        "الر": 8,
        "المر": 11,
        "كهيعص": 15,
        "طه": 5,
        "طسم": 9,
        "طس": 6,
        "يس": 6,
        "ص": 4,
        "ق": 4,
        "ن": 4,
        "عسق": 9,
    }.get(norm)

    if pronounced_units is None:
        pronounced_units = max(4, len(norm) * 3)

    return max(int(base_weight), int(pronounced_units * max(1, int(multiplier))))


def describe_opening_muqattaat(quran_words, chapter_id=None, max_words=4):
    out = []
    for i, w in enumerate((quran_words or [])[:max_words]):
        txt = w.get("text", "")
        if is_opening_muqattaat_word(txt, chapter_id=chapter_id, word_index=i):
            out.append({
                "index": i,
                "text": txt,
                "norm": normalize_arabic(txt),
                "chapter_muqattaat": chapter_muqattaat_text(chapter_id),
            })
    return out



def quran_opening_has_muqattaat(quran_words, chapter_id=None):
    if chapter_id is not None and is_muqattaat_chapter(chapter_id):
        return True

    if not quran_words:
        return False

    first_norm = normalize_arabic(quran_words[0].get("text", ""))
    known = {normalize_arabic(x) for x in MUQATTAAT_CHAPTERS.values()}
    return first_norm in known or len(first_norm) <= 5




# ---------------------------------------------------------------------------
# v44 opening fix: fold pronounced huruf muqatta'at letter names back to the
# Uthmani token BEFORE alignment.
#
# Whisper transcribes huruf muqatta'at by their spoken letter names, e.g.
# حمٓ -> "حا ميم" (two tokens) or "حاميم" (one token). Their normalized form
# does not equal the Quran token "حم", so DP alignment drifts at the opening
# and a few following words become estimated (estimated_after_opening).
#
# This step rewrites only the leading muqatta'at letter-name tokens so the
# opening Quran word matches exactly. It never changes Quran text, never
# invents timings (the merged span uses real ASR start/end), and only runs
# on muqatta'at chapters within a small leading scan window.
# ---------------------------------------------------------------------------

MUQATTAAT_LETTER_NAMES = {
    "ا": ["الف"],
    "ل": ["لام"],
    "م": ["ميم"],
    "ر": ["راء", "را"],
    "ص": ["صاد", "صا"],
    "ك": ["كاف", "كا"],
    "ه": ["هاء", "ها"],
    "ي": ["ياء", "يا"],
    "ع": ["عين"],
    "ط": ["طاء", "طا"],
    "س": ["سين"],
    "ح": ["حاء", "حا"],
    "ق": ["قاف", "قا"],
    "ن": ["نون"],
}


def _strip_hamza(value):
    return (value or "").replace("ء", "")


_MADD_VOWEL_RUN_RE = re.compile(r"([\u0627\u0648\u064a])\1+")


def _collapse_madd(value):
    """Collapse elongated madd vowels (ا/و/ي) repeated runs to a single
    instance so a stretched letter name like 'حاااميييم' matches 'حاميم'.
    Only long-vowel runs are collapsed; consonants are left intact so joined
    names like 'لامميم' (lam+mim) keep their internal letters."""
    return _MADD_VOWEL_RUN_RE.sub(r"\1", value or "")


def _letter_name_variants(letter):
    out = set()

    # The muqatta'at letter can surface in ASR either as its spoken name
    # ("حا"/"حاء", "ميم", "صاد") or as the bare Uthmani letter itself ("ح",
    # "م", "ص") -- the latter happens often when the Quran initial prompt
    # biases Whisper toward the written letter (e.g. حمٓ -> "حميم" = ح + ميم).
    bare = normalize_arabic(letter)
    if bare:
        out.add(bare)

    for name in MUQATTAAT_LETTER_NAMES.get(letter, []):
        n = normalize_arabic(name)
        if n:
            out.add(n)
            stripped = _strip_hamza(n)
            if stripped:
                out.add(stripped)
    return out


def _parse_letter_names_from_text(norm_text, letters):
    """
    Greedily consume the spoken letter names for `letters` (in order) from the
    start of norm_text. Returns the number of chars consumed if the full
    sequence is covered, else None.
    """
    pos = 0
    s = _collapse_madd(norm_text or "")
    for letter in letters:
        matched = False
        for variant in sorted(_letter_name_variants(letter), key=len, reverse=True):
            if variant and s.startswith(variant, pos):
                pos += len(variant)
                matched = True
                break
        if not matched:
            return None
    return pos


# Madd / nasal letters that can survive as a short trailing tail when a reciter
# stretches the FINAL muqatta'at letter name and Whisper transcribes the
# elongation as extra characters (e.g. a drawn-out ميم -> "...مين", لَمِّينَ).
_LETTER_NAME_TAIL_CHARS = set("اوينهء")


def _is_letter_name_phonetic_tail(tail):
    """True if `tail` is a short run of madd/nasal characters left over after a
    stretched final letter name. Bounded to a few characters so it can only mop
    up an elongation tail, never absorb a second real word."""
    if not tail or len(tail) > 3:
        return False
    return all(ch in _LETTER_NAME_TAIL_CHARS for ch in tail)


def _match_letter_names_multitoken(asr_words, start, letters, max_tokens=8):
    """
    Consume consecutive ASR tokens whose spoken letter names exactly cover
    `letters` in order. Each token must be fully consumed and contribute at
    least one letter. Returns the index of the last consumed token, or None.
    """
    li = 0
    ti = start
    end = min(len(asr_words), start + max_tokens)
    while li < len(letters) and ti < end:
        tok_norm = asr_words[ti].get("norm") or normalize_arabic(asr_words[ti].get("word", ""))
        tok_norm = _collapse_madd(tok_norm)
        if not tok_norm:
            return None

        pos = 0
        consumed_letters = 0
        while li + consumed_letters < len(letters):
            letter = letters[li + consumed_letters]
            variant_hit = None
            for variant in sorted(_letter_name_variants(letter), key=len, reverse=True):
                if variant and tok_norm.startswith(variant, pos):
                    variant_hit = variant
                    break
            if variant_hit is None:
                break
            pos += len(variant_hit)
            consumed_letters += 1

        if consumed_letters == 0:
            return None

        li += consumed_letters

        # The FINAL token of a multi-token letter-name run may carry a short
        # phonetic tail: when the reciter stretches the last letter's name, the
        # madd is transcribed as extra characters (e.g. an elongated ميم ->
        # "مين" in لَمِّينَ). Allow that tail, but ONLY once the full target
        # letter sequence has been covered, so intermediate tokens still must be
        # consumed cleanly and no real word is folded by accident.
        if li == len(letters):
            tail = tok_norm[pos:]
            if pos == len(tok_norm) or _is_letter_name_phonetic_tail(tail):
                return ti
            return None

        # Intermediate tokens must be clean, fully-consumed letter-name tokens.
        if pos != len(tok_norm):
            return None

        ti += 1

    return None


def _try_fold_targets_at(asr_words, start, targets):
    """
    Try to match the muqatta'at target norms consecutively starting at
    asr_words[start]. The first target MUST match for `start` to be a valid
    anchor; subsequent targets are folded greedily and matching stops at the
    first one that does not match (partial folds are allowed, e.g. حم folded
    but عسق left intact).

    Returns (next_index, fold_ops) when at least one real fold is produced,
    otherwise (start, None).
    """
    fold_ops = []
    i = start
    first_target_handled = False

    for t_index, target_norm in enumerate(targets):
        if i >= len(asr_words):
            break

        letters = list(target_norm)
        tok_norm = asr_words[i].get("norm") or normalize_arabic(asr_words[i].get("word", ""))
        tok_norm = _collapse_madd(tok_norm)

        # Case 1: token already equals the target -> valid anchor, no fold.
        if tok_norm == target_norm:
            i += 1
            if t_index == 0:
                first_target_handled = True
            continue

        # Case 2: a single token holds the joined letter names ("حاميم").
        consumed = _parse_letter_names_from_text(tok_norm, letters)
        if consumed is not None and consumed == len(tok_norm) and len(letters) >= 2:
            fold_ops.append({
                "start_index": i,
                "end_index": i,
                "target_norm": target_norm,
                "mode": "single_token",
            })
            i += 1
            if t_index == 0:
                first_target_handled = True
            continue

        # Case 2b: single-letter muqatta'at read by name ("صاد" -> ص).
        if consumed is not None and consumed == len(tok_norm) and len(letters) == 1 and tok_norm != target_norm:
            fold_ops.append({
                "start_index": i,
                "end_index": i,
                "target_norm": target_norm,
                "mode": "single_letter_name",
            })
            i += 1
            if t_index == 0:
                first_target_handled = True
            continue

        # Case 3: multiple tokens spelling the letter names ("حا" + "ميم").
        matched_end = _match_letter_names_multitoken(asr_words, i, letters)
        if matched_end is not None:
            fold_ops.append({
                "start_index": i,
                "end_index": matched_end,
                "target_norm": target_norm,
                "mode": "multi_token",
            })
            i = matched_end + 1
            if t_index == 0:
                first_target_handled = True
            continue

        # The opening target must anchor here; otherwise this start is invalid.
        if t_index == 0:
            return start, None
        break

    if not first_target_handled or not fold_ops:
        return start, None

    return i, fold_ops


def fold_muqattaat_letter_names_in_asr(asr_words, quran_words, chapter_id, max_scan=25):
    """
    Fold pronounced huruf muqatta'at letter names in the leading ASR tokens
    back to the chapter's Uthmani muqatta'at token(s) so alignment does not
    drift at the opening. Returns (new_asr_words, report).
    """
    if not asr_words or not is_muqattaat_chapter(chapter_id):
        return list(asr_words), []

    muq_words = describe_opening_muqattaat(quran_words or [], chapter_id=chapter_id)
    targets = [m.get("norm") for m in muq_words if m.get("norm")]
    if not targets:
        return list(asr_words), []

    scan_end = min(len(asr_words), max(1, int(max_scan)))

    for start in range(scan_end):
        _, fold_ops = _try_fold_targets_at(asr_words, start, targets)
        if not fold_ops:
            continue

        result = list(asr_words)
        report = []
        for op in sorted(fold_ops, key=lambda o: o["start_index"], reverse=True):
            i0 = op["start_index"]
            i1 = op["end_index"]
            block = result[i0:i1 + 1]
            merged = {
                "word": " ".join(str(b.get("word", "")) for b in block).strip(),
                "norm": op["target_norm"],
                "start_ms": int(block[0].get("start_ms")),
                "end_ms": int(block[-1].get("end_ms")),
                "muqattaat_folded": True,
                "muqattaat_fold_mode": op["mode"],
                "muqattaat_fold_source": [b.get("word") for b in block],
            }
            result[i0:i1 + 1] = [merged]
            report.append({
                "target_norm": op["target_norm"],
                "mode": op["mode"],
                "source_words": [b.get("word") for b in block],
                "start_ms": merged["start_ms"],
                "end_ms": merged["end_ms"],
            })

        report.reverse()
        print(
            "Muqattaat ASR fold: "
            + ", ".join(f"{'+'.join(r['source_words'])}->{r['target_norm']}" for r in report)
        )
        return result, report

    return list(asr_words), []


BASMALA_PHRASE = ["بسم", "الله", "الرحمن", "الرحيم"]


def quran_starts_with_phrase(quran_words, phrase, threshold=0.72):
    if not quran_words or len(quran_words) < len(phrase):
        return False

    q_norm = [normalize_arabic(w.get("text", "")) for w in quran_words[:len(phrase)]]
    p_norm = [normalize_arabic(x) for x in phrase]

    return all(similarity(q_norm[i], p_norm[i]) >= threshold for i in range(len(phrase)))


def quran_starts_with_basmala(quran_words, chapter_id=None):
    try:
        if int(chapter_id or 0) == 1:
            return quran_starts_with_phrase(quran_words, BASMALA_PHRASE, threshold=0.70)
    except Exception:
        pass

    return quran_starts_with_phrase(quran_words, BASMALA_PHRASE, threshold=0.70)


def phrase_is_basmala(phrase):
    if len(phrase) != len(BASMALA_PHRASE):
        return False
    p_norm = [normalize_arabic(x) for x in phrase]
    b_norm = [normalize_arabic(x) for x in BASMALA_PHRASE]
    return all(similarity(p_norm[i], b_norm[i]) >= 0.85 for i in range(len(BASMALA_PHRASE)))



OUTRO_PHRASES = [
    ["صدق", "الله", "العظيم"],
    ["صدق", "الله"],
]


def asr_norm_list(asr_words):
    return [normalize_arabic(w.get("word", "")) for w in asr_words]


def starts_with_phrase(norms, phrase):
    if len(norms) < len(phrase):
        return False
    return all(similarity(norms[i], phrase[i]) >= 0.70 for i in range(len(phrase)))


def ends_with_phrase(norms, phrase):
    if len(norms) < len(phrase):
        return False
    offset = len(norms) - len(phrase)
    return all(similarity(norms[offset + i], phrase[i]) >= 0.72 for i in range(len(phrase)))


def strip_asr_recitation_extras(asr_words, quran_words=None, strip_start=True, strip_end=True):
    """
    Remove common recitation extras not present in quran_words:
    - isti'adha
    - basmala at start of non-Fatiha/non-Taubah surahs
    - trailing "صدق الله العظيم"

    This does not touch Quran text. It only removes extra ASR words before alignment.
    """
    result = list(asr_words)
    removed = []

    if strip_start:
        changed = True
        while changed and result:
            changed = False
            norms = asr_norm_list(result)
            for phrase in sorted(INTRO_PHRASES, key=len, reverse=True):
                phrase_norm = [normalize_arabic(x) for x in phrase]
                if starts_with_phrase(norms, phrase_norm):
                    # Al-Fatihah: basmala is part of the surah text, not recitation intro.
                    # Do not strip it when quran_words itself starts with basmala.
                    if phrase_is_basmala(phrase) and quran_starts_with_basmala(quran_words):
                        continue

                    removed_words = result[:len(phrase)]
                    removed.append({
                        "side": "start",
                        "phrase": phrase,
                        "removed_words": [w.get("word") for w in removed_words],
                        "start_ms": removed_words[0].get("start_ms"),
                        "end_ms": removed_words[-1].get("end_ms"),
                    })
                    result = result[len(phrase):]
                    changed = True
                    break

    if strip_end:
        changed = True
        while changed and result:
            changed = False
            norms = asr_norm_list(result)
            for phrase in OUTRO_PHRASES:
                phrase_norm = [normalize_arabic(x) for x in phrase]
                if ends_with_phrase(norms, phrase_norm):
                    removed_words = result[-len(phrase):]
                    removed.append({
                        "side": "end",
                        "phrase": phrase,
                        "removed_words": [w.get("word") for w in removed_words],
                        "start_ms": removed_words[0].get("start_ms"),
                        "end_ms": removed_words[-1].get("end_ms"),
                    })
                    result = result[:-len(phrase)]
                    changed = True
                    break

    if removed:
        print(f"ASR recitation extras stripped: {len(removed)} block(s), words {len(asr_words)} -> {len(result)}")

    return result, removed


def score_alignment_candidate(quran_words, mapped, matched):
    q_count = max(1, len(quran_words))
    estimated = sum(1 for w in mapped if w.get("estimated"))
    missing = sum(1 for w in mapped if w.get("start_ms") is None or w.get("end_ms") is None)
    match_rate = matched / q_count

    # Matched count is primary. Missing/estimated are penalties.
    score = matched * 1000 - estimated * 30 - missing * 80
    return {
        "matched": matched,
        "match_rate": match_rate,
        "estimated": estimated,
        "missing": missing,
        "score": score,
    }


def build_alignment_variants(args):
    variants = []

    # User requested settings first.
    variants.append({
        "name": "user_settings",
        "alignment": args.alignment,
        "min_score": float(args.min_score),
        "lookahead": int(args.lookahead),
    })

    if not getattr(args, "no_auto_retry_alignment", False):
        for alignment, min_score, lookahead in [
            ("dp", 0.55, 15),
            ("dp", 0.50, 25),
            ("dp", 0.45, 35),
            ("dp", 0.40, 45),
            ("sequence", 0.45, 15),
            ("sequence", 0.40, 15),
            ("sequence", 0.35, 15),
        ]:
            variant = {"name": f"{alignment}_score_{min_score}_lookahead_{lookahead}", "alignment": alignment, "min_score": min_score, "lookahead": lookahead}
            if variant not in variants:
                variants.append(variant)

    # Deduplicate by effective values.
    out = []
    seen = set()
    for v in variants:
        key = (v["alignment"], round(float(v["min_score"]), 3), int(v["lookahead"]))
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out




# ---------------------------------------------------------------------------
# v24 char projection repair for estimated words
# ---------------------------------------------------------------------------

def build_quran_char_stream_for_projection(quran_words):
    chars = []
    spans = []

    for i, w in enumerate(quran_words):
        norm = w.get("norm") or normalize_arabic(w.get("text", ""))
        start = len(chars)
        chars.extend(list(norm))
        end = len(chars)
        spans.append({
            "word_index": i,
            "start_char": start,
            "end_char": end,
            "norm": norm,
        })

    return "".join(chars), spans


def build_asr_char_stream_for_projection(asr_words):
    chars = []
    char_times = []
    spans = []

    for i, w in enumerate(asr_words):
        norm = w.get("norm") or normalize_arabic(w.get("word", ""))
        if not norm:
            continue

        start_ms = int(w.get("start_ms", 0))
        end_ms = int(w.get("end_ms", start_ms))

        if end_ms <= start_ms:
            continue

        start_char = len(chars)
        n = max(1, len(norm))

        for k, ch in enumerate(norm):
            chars.append(ch)
            cs = start_ms + round((end_ms - start_ms) * k / n)
            ce = start_ms + round((end_ms - start_ms) * (k + 1) / n)
            char_times.append({
                "asr_word_index": i,
                "char_index_in_word": k,
                "start_ms": int(cs),
                "end_ms": int(ce),
                "char": ch,
            })

        spans.append({
            "asr_word_index": i,
            "start_char": start_char,
            "end_char": len(chars),
            "norm": norm,
            "word": w.get("word"),
            "start_ms": start_ms,
            "end_ms": end_ms,
        })

    return "".join(chars), char_times, spans


def build_char_alignment_map(q_stream, a_stream):
    import difflib

    matcher = difflib.SequenceMatcher(None, q_stream, a_stream, autojunk=False)
    q_to_a = {}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue

        length = min(i2 - i1, j2 - j1)
        for k in range(length):
            q_to_a[i1 + k] = j1 + k

    return q_to_a


def previous_timed_word(result, idx):
    for j in range(idx - 1, -1, -1):
        if result[j].get("start_ms") is not None and result[j].get("end_ms") is not None:
            return j
    return None


def next_timed_word(result, idx):
    for j in range(idx + 1, len(result)):
        if result[j].get("start_ms") is not None and result[j].get("end_ms") is not None:
            return j
    return None


def repair_estimated_words_by_char_projection(
    mapped_words,
    asr_words,
    min_char_ratio=0.55,
    min_chars=2,
    min_word_ms=70,
    max_projection_shift_ms=2500,
    intro_end_ms=None,
):
    """
    Repair only words that were completed by interpolation/splitting.

    It aligns normalized Quran character stream to normalized ASR character stream.
    If an estimated Quran word has enough exact character evidence in the ASR stream,
    we assign its timing from those ASR character timestamps.

    Hard rules:
    - Does not remap the whole surah.
    - Does not change Quran text.
    - Targets estimated/missing words only.
    - Respects previous/next timed word boundaries.
    - Never projects a word onto audio inside the detected leading recitation
      intro (isti'adha / basmala). When ``intro_end_ms`` is given, any candidate
      whose projected start falls before it is rejected, because those ASR
      characters belong to the excluded intro, not to the Quran text. This stops
      an opening muqatta'at word (e.g. الٓر) from being char-matched onto the
      isti'adha (e.g. ٱلرَّجِيم) and pulled back onto intro audio, which would
      erase its real, post-intro timing. Generic: no chapter/reciter hardcoding.
    """
    if not asr_words:
        return [dict(w) for w in mapped_words], []

    result = [dict(w) for w in mapped_words]
    q_stream, q_spans = build_quran_char_stream_for_projection(result)
    a_stream, a_char_times, _ = build_asr_char_stream_for_projection(asr_words)

    if not q_stream or not a_stream:
        return result, []

    q_to_a = build_char_alignment_map(q_stream, a_stream)
    report = []

    for idx, w in enumerate(result):
        has_timing = w.get("start_ms") is not None and w.get("end_ms") is not None
        needs = bool(w.get("estimated")) or (not has_timing)

        if not needs:
            continue

        span = q_spans[idx]
        q_start = span["start_char"]
        q_end = span["end_char"]
        norm = span["norm"]

        if not norm:
            continue

        q_chars = list(range(q_start, q_end))
        mapped_asr_chars = [q_to_a[c] for c in q_chars if c in q_to_a and q_to_a[c] < len(a_char_times)]

        mapped_unique = sorted(set(mapped_asr_chars))
        mapped_count = len(mapped_unique)

        if mapped_count < min_chars:
            continue

        ratio = mapped_count / max(1, len(norm))
        if ratio < min_char_ratio:
            continue

        projected_start = min(a_char_times[c]["start_ms"] for c in mapped_unique)
        projected_end = max(a_char_times[c]["end_ms"] for c in mapped_unique)

        if projected_end <= projected_start:
            continue

        # Never plant a Quran word on audio inside the detected leading
        # recitation intro (isti'adha / basmala). Those ASR characters are not
        # part of the Quran text, so a char match there (e.g. a muqatta'at word
        # "الر" hitting the isti'adha "الرجيم") is spurious; honoring it would
        # pull the word back onto excluded intro audio and erase its real,
        # post-intro timing. The first word has no previous timed neighbour, so
        # the max_projection_shift_ms guard below cannot catch this on its own.
        if intro_end_ms is not None and projected_start < int(intro_end_ms):
            continue

        prev_idx = previous_timed_word(result, idx)
        next_idx = next_timed_word(result, idx)

        bounded_start = int(projected_start)
        bounded_end = int(projected_end)

        if prev_idx is not None:
            prev_end = int(result[prev_idx]["end_ms"])
            # Avoid pulling a word too far away from its interpolated area.
            if has_timing and abs(bounded_start - int(w["start_ms"])) > max_projection_shift_ms:
                continue
            bounded_start = max(bounded_start, prev_end)

        if next_idx is not None:
            next_start = int(result[next_idx]["start_ms"])
            bounded_end = min(bounded_end, next_start)

        if bounded_end - bounded_start < min_word_ms:
            # If char span is too short but between good neighbors, use a tiny safe span.
            if next_idx is not None:
                bounded_end = min(int(result[next_idx]["start_ms"]), bounded_start + min_word_ms)
            else:
                bounded_end = bounded_start + min_word_ms

        if bounded_end <= bounded_start:
            continue

        old_start = w.get("start_ms")
        old_end = w.get("end_ms")

        result[idx]["start_ms"] = int(bounded_start)
        result[idx]["end_ms"] = int(bounded_end)
        result[idx]["estimated"] = False
        result[idx]["char_projected"] = True
        result[idx]["char_projection_ratio"] = round(ratio, 3)
        result[idx]["char_projection_chars"] = mapped_count
        result[idx]["score"] = max(float(result[idx].get("score") or 0), round(ratio, 3))
        result[idx]["method"] = str(result[idx].get("method", "")) + "+char_projection_repair"

        report.append({
            "index": idx,
            "verse_key": w.get("verse_key"),
            "position": w.get("position"),
            "text": w.get("text"),
            "norm": norm,
            "old_start_ms": old_start,
            "old_end_ms": old_end,
            "new_start_ms": int(bounded_start),
            "new_end_ms": int(bounded_end),
            "char_ratio": round(ratio, 3),
            "matched_chars": mapped_count,
            "word_chars": len(norm),
            "method": "char_projection_repair",
        })

    print(f"Char projection repairs: {len(report)}")
    return result, report


def recover_estimated_words_asr(
    mapped_words,
    args,
    wav_path,
    tmpdir,
    chapter_id=None,
    duration_ms=None,
):
    """Recover interior words the aligner had to ESTIMATE by re-transcribing
    just the audio between their real-timed neighbours.

    This is the interior counterpart of recover_dropped_opening_asr. Whisper
    routinely MERGES or SKIPS short, coarticulated function words mid-surah
    (e.g. أَمۡ, هُوَ, إِلَّا); the aligner then fills them by static
    interpolation between their real neighbours, which shows up as
    ``estimated`` words. Here we cut just the gap
    ``[prev_real_end .. next_real_start]`` (with a little outward context
    padding), run ASR on that clip alone WITHOUT the Quran initial_prompt, and
    if the dropped word surfaces we replace the interpolated timing with its
    REAL measured Whisper timestamp.

    Generic by construction: it is driven ONLY by the ``estimated`` flag plus
    neighbour timestamps. There is NO hardcoded chapter, reciter, or word
    list, so it applies to every surah and every reciter. When no confident
    match is found the safe interpolation is left untouched (never made
    worse). Opening-region estimates are skipped here on purpose; they are
    owned by recover_dropped_opening_asr.
    """
    report = {"attempted": False, "recovered": 0, "runs": [], "words": []}

    if getattr(args, "no_estimated_word_recover", False):
        return [dict(w) for w in mapped_words], report

    result = [dict(w) for w in mapped_words]
    n = len(result)

    def is_recoverable_estimate(w):
        return (
            bool(w.get("estimated"))
            and not bool(w.get("opening_region_estimated"))
            and not bool(w.get("intro_gap_estimated"))
        )

    # Group consecutive recoverable estimated words into runs.
    runs = []
    i = 0
    while i < n:
        if not is_recoverable_estimate(result[i]):
            i += 1
            continue
        j = i
        while j < n and is_recoverable_estimate(result[j]):
            j += 1
        runs.append((i, j - 1))
        i = j

    if not runs:
        return result, report

    report["attempted"] = True

    pad_ms = int(getattr(args, "estimated_word_recover_pad_ms", 600))
    max_window_ms = int(getattr(args, "estimated_word_recover_max_window_ms", 12000))
    min_score = float(getattr(args, "estimated_word_recover_min_score", 0.62))
    min_word_ms = int(getattr(args, "estimated_word_recover_min_word_ms", 80))
    max_runs = int(getattr(args, "estimated_word_recover_max_runs", 60))

    dur_cap = int(duration_ms) if duration_ms else None
    attempted_runs = 0
    total_recovered = 0

    for (a, b) in runs:
        if attempted_runs >= max_runs:
            break

        prev_idx = previous_timed_word(result, a)
        next_idx = next_timed_word(result, b)

        # Need at least one real anchor to bound the gap in absolute time.
        if prev_idx is None and next_idx is None:
            continue

        # Hard bounds: the missing word(s) live strictly between the real
        # neighbours, so recovered times are clamped to this window. This keeps
        # global order monotonic and prevents overlap with real words.
        if prev_idx is not None:
            win_start = int(result[prev_idx]["end_ms"])
        else:
            win_start = max(0, int(result[a].get("start_ms") or 0))

        if next_idx is not None:
            win_end = int(result[next_idx]["start_ms"])
        else:
            win_end = int(result[b].get("end_ms") or (win_start + max_window_ms))

        if win_end - win_start < min_word_ms:
            # No real audio room between neighbours; nothing to recover.
            continue
        if win_end - win_start > max_window_ms:
            # Window too wide to be a single dropped-word region; the miss is
            # ambiguous, so keep the safe interpolation.
            report["runs"].append({
                "start_index": a, "end_index": b,
                "win_start_ms": win_start, "win_end_ms": win_end,
                "skipped": "window_too_wide", "recovered": 0,
            })
            continue

        # Re-transcribe the gap with a little outward context padding so
        # Whisper has acoustic context, then keep only words inside the gap.
        clip_start_ms = max(0, win_start - pad_ms)
        clip_end_ms = win_end + pad_ms
        if dur_cap:
            clip_end_ms = min(clip_end_ms, dur_cap)
        if clip_end_ms <= clip_start_ms:
            continue

        attempted_runs += 1

        clip_path = str(Path(tmpdir) / f"est_recover_{win_start}_{win_end}.wav")
        try:
            extract_wav_segment(wav_path, clip_path, clip_start_ms, clip_end_ms)
        except Exception as e:
            print(f"  Estimated-word recovery: failed to extract clip {clip_start_ms}..{clip_end_ms}ms: {e}")
            continue

        try:
            clip_result = run_asr_backend(
                wav_path=clip_path,
                args=args,
                initial_prompt=None,
                duration_ms=clip_end_ms - clip_start_ms,
            )
        except Exception as e:
            print(f"  Estimated-word recovery: ASR failed on clip {clip_start_ms}..{clip_end_ms}ms: {e}")
            continue

        clip_words = extract_asr_words(clip_result)

        # Offset to absolute time and keep only candidates whose centre sits
        # inside the true gap [win_start, win_end].
        cand = []
        for w in clip_words:
            s = clip_start_ms + int(w["start_ms"])
            e = clip_start_ms + int(w["end_ms"])
            mid = (s + e) // 2
            if mid < win_start or mid > win_end:
                continue
            cand.append({
                "word": w["word"], "norm": w["norm"],
                "start_ms": max(win_start, s), "end_ms": min(win_end, e),
            })

        if not cand:
            report["runs"].append({
                "start_index": a, "end_index": b,
                "win_start_ms": win_start, "win_end_ms": win_end,
                "skipped": "no_asr_in_gap", "recovered": 0,
            })
            continue

        # Monotonic fuzzy map the missing Quran run onto the gap ASR words.
        q_region = [
            {"norm": result[k].get("norm") or normalize_arabic(result[k].get("text") or "")}
            for k in range(a, b + 1)
        ]
        pairs = map_region_fuzzy(q_region, cand, min_score)

        run_recovered = 0
        last_end = win_start
        for qi in range(len(q_region)):
            ai = pairs.get(qi)
            if ai is None:
                continue
            aw = cand[ai]
            s = max(int(aw["start_ms"]), last_end)
            e = int(aw["end_ms"])
            if e - s < min_word_ms:
                e = min(win_end, s + min_word_ms)
            if e <= s:
                continue
            idx = a + qi
            score = round(float(similarity(q_region[qi]["norm"], aw["norm"])), 3)
            old_s = result[idx].get("start_ms")
            old_e = result[idx].get("end_ms")
            result[idx]["start_ms"] = int(s)
            result[idx]["end_ms"] = int(e)
            result[idx]["estimated"] = False
            result[idx]["estimated_recovered"] = True
            result[idx]["score"] = max(float(result[idx].get("score") or 0), score)
            result[idx]["method"] = str(result[idx].get("method", "")) + "+estimated_word_recover"
            report["words"].append({
                "index": idx,
                "verse_key": result[idx].get("verse_key"),
                "position": result[idx].get("position"),
                "text": result[idx].get("text"),
                "norm": q_region[qi]["norm"],
                "asr_word": aw["word"],
                "old_start_ms": old_s, "old_end_ms": old_e,
                "new_start_ms": int(s), "new_end_ms": int(e),
                "score": score,
            })
            last_end = e
            run_recovered += 1

        total_recovered += run_recovered
        report["runs"].append({
            "start_index": a, "end_index": b,
            "win_start_ms": win_start, "win_end_ms": win_end,
            "asr_in_gap": len(cand), "recovered": run_recovered,
        })

    report["recovered"] = total_recovered
    if report["attempted"]:
        print(
            f"Estimated-word recovery: recovered {total_recovered} real ASR word(s) "
            f"across {attempted_runs} interior gap(s)."
        )
    return result, report


# ---------------------------------------------------------------------------
# CTC forced-alignment interior recovery (v59)
#
# Whisper is a GENERATIVE recogniser: it routinely merges/skips short
# coarticulated function words mid-surah (أَمۡ, هُوَ, إِلَّا, ...). When that
# happens the aligner has to ESTIMATE their timing by interpolation, and the
# Whisper-based interior recovery (recover_estimated_words_asr) often re-drops
# the very same word, so it can never fill the gap.
#
# CTC forced alignment fixes this structurally: we already KNOW the text (it is
# the Quran). We give a char-level CTC acoustic model (wav2vec2 with a native
# Arabic vocab) the EXACT known words of the gap and constrain-decode them
# against the gap audio. Forced alignment NEVER drops a token — every word gets
# a real, frame-measured span. This is generic across all surahs and reciters:
# it is driven only by the ``estimated`` flag plus neighbour timestamps, with no
# hardcoded chapter, reciter, or word list.
#
# Opening-region estimates are intentionally excluded here (they are owned by the
# opening gap/leading recovery + the deterministic muqatta'at bracket); this pass
# only touches interior words. It degrades gracefully: if torchaudio/transformers
# are not installed, it prints a clear one-time hint and leaves the safe
# interpolation untouched (never fabricates, never crashes the pipeline).
# ---------------------------------------------------------------------------

_CTC_MODEL_CACHE = {}


def _load_ctc_model(model_name, device):
    """Load (and cache) a wav2vec2 CTC processor+model for forced alignment."""
    key = (str(model_name), str(device))
    cached = _CTC_MODEL_CACHE.get(key)
    if cached is not None:
        print(f"Using cached CTC model: {model_name} on {device}")
        return cached
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    print(f"Loading CTC model: {model_name} on {device}")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
    _CTC_MODEL_CACHE[key] = (processor, model)
    return processor, model


def ctc_align_window(
    wav_path,
    transcript_texts,
    device,
    model_name,
    window_start_ms=0,
    window_sec=None,
    sample_rate=16000,
):
    """Force-align ``transcript_texts`` (a list of normalized strings) against a
    window of ``wav_path`` using a CTC acoustic model.

    Returns a list 1:1 with ``transcript_texts`` of either None (no alignable
    characters) or ``{"start_ms", "end_ms", "score"}`` with REAL measured
    timestamps (absolute, i.e. offset by ``window_start_ms``).

    This is the single source of truth for CTC forced alignment in the project;
    the standalone ``quran_opening_ctc.py`` opening tool delegates to it.
    """
    import torch
    import torchaudio
    import numpy as np
    import soundfile as sf

    if not hasattr(torchaudio.functional, "forced_align"):
        raise RuntimeError(
            "torchaudio.functional.forced_align is missing. Upgrade torchaudio "
            "to >=2.1 (pip install -U torchaudio)."
        )

    # Read audio with soundfile, NOT torchaudio.load: torchaudio>=2.8 routes
    # load() through load_with_torchcodec, which raises "TorchCodec is required"
    # unless the separate torchcodec package is installed. soundfile (already a
    # dependency) decodes the WAV directly into numpy; we wrap it as a [C, N]
    # tensor so the channel-mean / resample / windowing logic below is unchanged.
    audio_np, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)  # [N, C]
    waveform = torch.from_numpy(np.ascontiguousarray(audio_np.T))  # [C, N]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        sr = sample_rate

    start_sample = int(round(window_start_ms / 1000.0 * sr))
    start_sample = max(0, min(start_sample, waveform.shape[1]))
    if window_sec:
        end_sample = start_sample + int(round(float(window_sec) * sr))
        end_sample = min(end_sample, waveform.shape[1])
    else:
        end_sample = waveform.shape[1]
    window = waveform[:, start_sample:end_sample]
    if window.shape[1] == 0:
        raise RuntimeError("CTC alignment window is empty.")

    processor, model = _load_ctc_model(model_name, device)

    inputs = processor(
        window.squeeze(0).numpy(), sampling_rate=sr, return_tensors="pt"
    ).input_values.to(device)

    with torch.inference_mode():
        logits = model(inputs).logits
    emission = torch.log_softmax(logits, dim=-1).cpu()  # [1, T, V]
    num_frames = emission.shape[1]

    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id
    if blank_id is None:
        blank_id = vocab.get("<pad>", 0)
    word_delim = vocab.get("|")

    # Flatten transcript to one target token stream, recording each entry's
    # [start, end) token range within that stream.
    targets = []
    ranges = []  # one per transcript entry; None if no vocab chars
    for idx, text in enumerate(transcript_texts):
        ids = [vocab[c] for c in (text or "") if c in vocab]
        if not ids:
            ranges.append(None)
            continue
        start_idx = len(targets)
        targets.extend(ids)
        ranges.append((start_idx, len(targets)))
        if word_delim is not None and idx != len(transcript_texts) - 1:
            targets.append(word_delim)

    if not targets:
        return [None for _ in transcript_texts]
    if len(targets) > num_frames:
        raise RuntimeError(
            f"CTC transcript ({len(targets)} tokens) longer than audio frames "
            f"({num_frames}); window too short."
        )

    targets_t = torch.tensor([targets], dtype=torch.int32)
    aligned, scores = torchaudio.functional.forced_align(
        emission, targets_t, blank=blank_id
    )
    token_spans = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp())

    ms_per_frame = (window.shape[1] / num_frames) / sr * 1000.0

    results = []
    for rng in ranges:
        if rng is None:
            results.append(None)
            continue
        s_idx, e_idx = rng
        span_slice = token_spans[s_idx:e_idx]
        if not span_slice:
            results.append(None)
            continue
        start_frame = span_slice[0].start
        end_frame = span_slice[-1].end
        sc = [float(sp.score) for sp in span_slice]
        results.append({
            "start_ms": int(round(window_start_ms + start_frame * ms_per_frame)),
            "end_ms": int(round(window_start_ms + end_frame * ms_per_frame)),
            "score": round(sum(sc) / len(sc), 4) if sc else None,
        })
    return results


def recover_estimated_words_ctc(
    mapped_words,
    args,
    wav_path,
    tmpdir=None,
    chapter_id=None,
    duration_ms=None,
):
    """Recover interior ESTIMATED words via CTC forced alignment of their KNOWN
    Quran text against the audio between their real-timed neighbours.

    Runs AFTER the Whisper-based interior recovery as a stronger fallback: where
    Whisper keeps re-dropping a merged/short word, CTC forced alignment of the
    exact known text recovers a real measured span. Generic, no hardcoding;
    opening-region estimates are skipped; unmappable words keep their safe
    interpolation (never fabricated). Degrades gracefully when the CTC deps are
    missing.
    """
    report = {"attempted": False, "recovered": 0, "runs": [], "words": [],
              "model": getattr(args, "ctc_model", None)}

    if getattr(args, "no_ctc_word_recover", False):
        return [dict(w) for w in mapped_words], report

    result = [dict(w) for w in mapped_words]
    n = len(result)

    def is_recoverable_estimate(w):
        return (
            bool(w.get("estimated"))
            and not bool(w.get("opening_region_estimated"))
            and not bool(w.get("intro_gap_estimated"))
        )

    runs = []
    i = 0
    while i < n:
        if not is_recoverable_estimate(result[i]):
            i += 1
            continue
        j = i
        while j < n and is_recoverable_estimate(result[j]):
            j += 1
        runs.append((i, j - 1))
        i = j

    if not runs:
        return result, report

    model_name = getattr(args, "ctc_model", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
    min_score = float(getattr(args, "ctc_recover_min_score", 0.0))
    pad_ms = int(getattr(args, "ctc_recover_pad_ms", 150))
    max_window_ms = int(getattr(args, "ctc_recover_max_window_ms", 15000))
    chunk_ms = max(2000, int(getattr(args, "ctc_recover_chunk_ms", 45000)))
    min_word_ms = int(getattr(args, "ctc_recover_min_word_ms", 80))
    max_runs = int(getattr(args, "ctc_recover_max_runs", 200))
    device = getattr(args, "device_resolved", None) or pick_device(
        getattr(args, "device", "auto")
    )

    # Probe the CTC deps once; degrade gracefully with a clear hint if missing.
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # noqa: F401
    except Exception as exc:
        print(
            "CTC interior recovery: skipped (optional deps missing). "
            "To enable real timestamps for interior words Whisper drops, run:\n"
            "    pip install \"torchaudio>=2.1\" transformers\n"
            f"  ({exc})"
        )
        report["skipped"] = "deps_missing"
        return result, report

    report["attempted"] = True
    dur_cap = int(duration_ms) if duration_ms else None
    attempted_runs = 0
    total_recovered = 0

    for (a, b) in runs:
        if attempted_runs >= max_runs:
            break

        prev_idx = previous_timed_word(result, a)
        next_idx = next_timed_word(result, b)
        # Strict interior-only contract: require BOTH real anchors. A leading
        # run (prev_idx is None) is the surah OPENING — owned by the opening
        # recovery + deterministic muqatta'at bracket, and must NEVER be touched
        # here even if its estimated flags are absent. A trailing run (next_idx
        # is None) has no real upper bound. In both cases there is no pair of
        # REAL neighbour timestamps to bracket the gap, so we skip — recovered
        # spans are always clamped strictly between two real words.
        if prev_idx is None or next_idx is None:
            report["runs"].append({
                "start_index": a, "end_index": b,
                "skipped": "no_two_real_anchors", "recovered": 0,
            })
            continue

        win_start = int(result[prev_idx]["end_ms"])
        win_end = int(result[next_idx]["start_ms"])

        if win_end - win_start < min_word_ms:
            continue

        # Build one or more CTC sub-windows spanning this anchor-bounded gap.
        # NARROW gaps align in a single pass (unchanged behaviour). WIDE gaps -
        # a large stretch Whisper under-transcribed - are no longer skipped: the
        # gap is bracketed by two REAL anchor timestamps, so it is split into
        # sequential sub-windows using the interpolated word positions as cut
        # points and each chunk is force-aligned. This keeps every CTC forward
        # pass short enough to stay memory-safe on long surahs while still
        # recovering the dropped words with REAL measured spans - generic, no
        # per-surah tuning. Words a chunk cannot confidently place keep their
        # safe interpolation (the min_score floor below), so nothing is fabricated.
        subwins = []  # (sa, sb, aw_start, aw_end); token idxs sa..sb inclusive
        if win_end - win_start <= max_window_ms:
            aw_start = max(0, win_start - pad_ms)
            aw_end = win_end + pad_ms
            if dur_cap:
                aw_end = min(aw_end, dur_cap)
            if aw_end > aw_start:
                subwins.append((a, b, aw_start, aw_end))
        else:
            span_ms = win_end - win_start

            def _center(k):
                s0 = result[k].get("start_ms")
                e0 = result[k].get("end_ms")
                if s0 is not None and e0 is not None:
                    return (int(s0) + int(e0)) / 2.0
                # even-spread fallback when an estimate lacks a timing
                return win_start + (k - a + 0.5) * span_ms / (b - a + 1)

            n_chunks = max(1, (span_ms + chunk_ms - 1) // chunk_ms)
            step = span_ms / float(n_chunks)
            for c in range(n_chunks):
                cs = win_start + c * step
                ce = win_end if c == n_chunks - 1 else win_start + (c + 1) * step
                idxs = [
                    k for k in range(a, b + 1)
                    if (cs <= _center(k) < ce)
                    or (c == n_chunks - 1 and _center(k) >= cs)
                ]
                if not idxs:
                    continue
                sa, sb = idxs[0], idxs[-1]
                aw_start = max(0, int(round(cs)) - pad_ms)
                aw_end = int(round(ce)) + pad_ms
                if dur_cap:
                    aw_end = min(aw_end, dur_cap)
                if aw_end > aw_start:
                    subwins.append((sa, sb, aw_start, aw_end))

        if not subwins:
            continue

        attempted_runs += 1
        run_recovered = 0
        last_end = win_start
        run_failed = False
        # v81: proposals are buffered per run and committed AFTER a degenerate
        # guard, instead of mutating result[] span-by-span. The interior floor
        # default dropped from 0.10 to 0.0 (matching the underrun full-surah CTC
        # path, which has always accepted all non-None spans): interior gaps are
        # bracketed by TWO real anchors and force-align KNOWN correct text, so a
        # low emission score on a slow mujawwad madd (the conversational-Arabic
        # wav2vec2 model scores long vowels poorly) does not mean the placement
        # is wrong — field evidence: reciter-C/109 recovered 7/7 interior words at
        # floor 0.0 that floor 0.10 rejected. Safety: if the run COLLAPSES
        # (>=3 proposals and more than half squeezed to the minimum word width,
        # the signature of CTC failing to localize), low-score words keep their
        # safe interpolation — only spans clearing the legacy 0.10 floor commit.
        proposals = []  # (idx, s, e, sc, norm_text)
        for (sa, sb, aw_start, aw_end) in subwins:
            texts = [
                (result[k].get("norm") or normalize_arabic(result[k].get("text") or ""))
                for k in range(sa, sb + 1)
            ]
            try:
                spans = ctc_align_window(
                    wav_path=wav_path,
                    transcript_texts=texts,
                    device=device,
                    model_name=model_name,
                    window_start_ms=aw_start,
                    window_sec=(aw_end - aw_start) / 1000.0,
                )
            except Exception as e:
                print(
                    f"  CTC interior recovery: alignment failed on gap "
                    f"{win_start}..{win_end}ms window {aw_start}..{aw_end}ms: {e}"
                )
                run_failed = True
                continue

            for qi, span in enumerate(spans):
                if not span:
                    continue
                sc = span.get("score")
                if sc is not None and float(sc) < min_score:
                    continue
                s = max(int(span["start_ms"]), last_end, win_start)
                e = min(int(span["end_ms"]), win_end)
                if e - s < min_word_ms:
                    e = min(win_end, s + min_word_ms)
                if e <= s:
                    continue
                proposals.append((sa + qi, int(s), int(e), sc, texts[qi]))
                last_end = e

        # v81 degenerate-run guard, mirroring the underrun full-surah CTC guard
        # signals: a healthy force-alignment of known text between two real
        # anchors yields spans with real varied widths spread across the gap.
        # Two degenerate signatures (either one trips the guard):
        #   1. width collapse — most proposals squeezed to the minimum word
        #      width (CTC failed to localize);
        #   2. implausible packing — the proposals imply a recitation rate far
        #      above any humanly possible pace (piled into a tiny extent, i.e.
        #      "wide window but wrong placement").
        # On a degenerate run only words clearing the legacy 0.10 confidence
        # floor commit; the rest keep their safe interpolation (never fabricate
        # real-flagged timestamps).
        legacy_floor = 0.10
        floor_like = [p for p in proposals if (p[2] - p[1]) <= min_word_ms]
        width_collapsed = len(proposals) >= 3 and (2 * len(floor_like) > len(proposals))
        rate_bad = False
        if len(proposals) >= 3:
            extent_ms = max(p[2] for p in proposals) - min(p[1] for p in proposals)
            wps = (
                len(proposals) / (extent_ms / 1000.0)
                if extent_ms > 0 else float("inf")
            )
            max_wps = float(getattr(args, "underrun_ctc_degenerate_max_wps", 7.0))
            rate_bad = wps > max_wps
        degenerate_run = width_collapsed or rate_bad
        for (idx, s, e, sc, ntext) in proposals:
            if degenerate_run and (sc is None or float(sc) < legacy_floor):
                continue
            old_s = result[idx].get("start_ms")
            old_e = result[idx].get("end_ms")
            result[idx]["start_ms"] = int(s)
            result[idx]["end_ms"] = int(e)
            result[idx]["estimated"] = False
            result[idx]["ctc_recovered"] = True
            if sc is not None:
                result[idx]["score"] = max(float(result[idx].get("score") or 0), float(sc))
            result[idx]["method"] = str(result[idx].get("method", "")) + "+ctc_word_recover"
            report["words"].append({
                "index": idx,
                "verse_key": result[idx].get("verse_key"),
                "position": result[idx].get("position"),
                "text": result[idx].get("text"),
                "norm": ntext,
                "old_start_ms": old_s, "old_end_ms": old_e,
                "new_start_ms": int(s), "new_end_ms": int(e),
                "score": sc,
            })
            run_recovered += 1
        if degenerate_run and run_recovered < len(proposals):
            run_failed = True

        total_recovered += run_recovered
        run_report = {
            "start_index": a, "end_index": b,
            "win_start_ms": win_start, "win_end_ms": win_end,
            "recovered": run_recovered,
        }
        if len(subwins) > 1:
            run_report["subwindows"] = len(subwins)
        if run_failed and run_recovered == 0:
            run_report["skipped"] = "ctc_failed"
        report["runs"].append(run_report)

    report["recovered"] = total_recovered
    if report["attempted"]:
        print(
            f"CTC interior recovery: recovered {total_recovered} real word(s) "
            f"across {attempted_runs} interior gap(s)."
        )
    return result, report


def _dup_strip_leading_conjunction(norm):
    """Strip a single leading conjunction (فَ/وَ) from a normalized token so
    فان -> ان and وان -> ان. Used ONLY to treat two consecutive ayat that differ
    solely by a leading conjunction (e.g. Ash-Sharh 94:5 فَإِنَّ vs 94:6 إِنَّ) as
    the same repeated recitation. Comparison-only; never mutates a stored word."""
    s = str(norm or "")
    if len(s) > 1 and s[0] in ("ف", "و"):
        return s[1:]
    return s


def ayat_near_identical(norms_a, norms_b, min_len=2):
    """True when two ayat are the same recited text up to a leading conjunction:
    equal length (>= ``min_len`` tokens) and every token equal EXCEPT the first,
    which may differ only by a leading فَ/وَ. Generic detector for Quran-text
    consecutive duplicate ayat; no per-surah hardcoding."""
    if len(norms_a) != len(norms_b) or len(norms_a) < int(min_len):
        return False
    for k in range(1, len(norms_a)):
        if norms_a[k] != norms_b[k]:
            return False
    return _dup_strip_leading_conjunction(norms_a[0]) == _dup_strip_leading_conjunction(norms_b[0])


def adjacent_duplicate_ayah_regions(mapped_words):
    """Detect maximal runs of >= 2 consecutive ayat whose recited text is
    identical up to a leading conjunction (e.g. Ash-Sharh 94:5 فَإِنَّ مَعَ
    ٱلۡعُسۡرِ يُسۡرًا / 94:6 إِنَّ مَعَ ٱلۡعُسۡرِ يُسۡرًا).

    Returns a list of ``(start_index, end_index, [verse_keys])`` covering the full
    word span of each run. Detection is from the KNOWN Quran text only (verse_key
    + norms); requires real ``chapter:ayah`` verse keys and strictly consecutive
    ayah numbers, so a merely-repeated-but-separated phrase (e.g. Ar-Rahman's
    refrain, which has other ayat between occurrences) is NOT treated as a region.

    Whisper collapses such near-identical adjacent ayat into a single recitation
    and the DP tie-break spreads that one copy across BOTH ayat, leaving one whole
    copy ESTIMATED on a fake window whose timestamps slide onto the next ayah.
    Callers re-align these regions directly from the audio."""
    ayat = []  # [verse_key, chapter, ayah_num, [indices], [norms]]
    for idx, w in enumerate(mapped_words):
        vk = w.get("verse_key")
        norm = str(w.get("norm") or normalize_arabic(w.get("text") or ""))
        if ayat and ayat[-1][0] == vk and vk is not None:
            ayat[-1][3].append(idx)
            ayat[-1][4].append(norm)
            continue
        chapter = ayah_num = None
        if isinstance(vk, str) and ":" in vk:
            parts = vk.split(":")
            try:
                chapter, ayah_num = int(parts[0]), int(parts[1])
            except Exception:
                chapter = ayah_num = None
        ayat.append([vk, chapter, ayah_num, [idx], [norm]])

    regions = []
    n = len(ayat)
    i = 0
    while i < n - 1:
        j = i
        while (
            j + 1 < n
            and ayat[j][1] is not None
            and ayat[j][1] == ayat[j + 1][1]              # same chapter
            and ayat[j][2] is not None
            and ayat[j + 1][2] is not None
            and ayat[j + 1][2] == ayat[j][2] + 1          # strictly consecutive ayat
            and ayat_near_identical(ayat[j][4], ayat[j + 1][4])
        ):
            j += 1
        if j > i:
            start_idx = ayat[i][3][0]
            end_idx = ayat[j][3][-1]
            vks = [ayat[k][0] for k in range(i, j + 1)]
            regions.append((start_idx, end_idx, vks))
            i = j + 1
        else:
            i += 1
    return regions


def nonadjacent_duplicate_ayah_regions(
    mapped_words, max_intervening_ayat=1, max_span_words=40
):
    """Detect an ``A [X ...] A`` pattern: two ayat with identical recited text
    (up to a leading conjunction) separated by 1..``max_intervening_ayat``
    DIFFERENT intervening ayat, e.g. Al-Kafirun 109:3 وَلَآ أَنتُمۡ عَٰبِدُونَ مَآ
    أَعۡبُدُ == 109:5, with the different 109:4 between them.

    Whisper collapses the two identical copies (dropping one) exactly like the
    adjacent case, but ``adjacent_duplicate_ayah_regions`` never sees them because
    a DIFFERENT ayah sits between the copies. Returns a list of
    ``(start_index, end_index, [verse_keys])`` spanning the FIRST copy through the
    LAST copy inclusive (so the intervening ayah is inside the span), capped at
    ``max_span_words`` words. Detection is from the KNOWN Quran text only and
    requires real, strictly-consecutive ``chapter:ayah`` verse keys, so a refrain
    that recurs across the whole surah with many ayat between occurrences (e.g.
    Ar-Rahman) never forms a giant span. Each ayah is used by at most one region.
    Callers force-align each region from the audio between its two OUTER real
    anchors and adopt the result only when every word is placed; no per-surah
    hardcoding."""
    ayat = []  # [verse_key, chapter, ayah_num, [indices], [norms]]
    for idx, w in enumerate(mapped_words):
        vk = w.get("verse_key")
        norm = str(w.get("norm") or normalize_arabic(w.get("text") or ""))
        if ayat and ayat[-1][0] == vk and vk is not None:
            ayat[-1][3].append(idx)
            ayat[-1][4].append(norm)
            continue
        chapter = ayah_num = None
        if isinstance(vk, str) and ":" in vk:
            parts = vk.split(":")
            try:
                chapter, ayah_num = int(parts[0]), int(parts[1])
            except Exception:
                chapter = ayah_num = None
        ayat.append([vk, chapter, ayah_num, [idx], [norm]])

    max_intervening = max(1, int(max_intervening_ayat))
    regions = []
    n = len(ayat)
    used = [False] * n
    for i in range(n):
        if used[i]:
            continue
        if ayat[i][1] is None or ayat[i][2] is None:
            continue
        for j in range(i + 2, min(n, i + 2 + max_intervening)):
            if used[j] or ayat[j][1] != ayat[i][1] or ayat[j][2] is None:
                continue
            # every ayah in i..j must have a real same-chapter verse key and step
            # by exactly +1, so a numbering gap or an unparseable intermediate
            # verse key is rejected (endpoint arithmetic alone would admit it)
            consecutive = True
            for k in range(i + 1, j + 1):
                if (ayat[k][1] != ayat[i][1] or ayat[k][2] is None
                        or ayat[k][2] != ayat[k - 1][2] + 1):
                    consecutive = False
                    break
            if not consecutive:
                continue
            if not ayat_near_identical(ayat[i][4], ayat[j][4]):
                continue
            start_idx = ayat[i][3][0]
            end_idx = ayat[j][3][-1]
            if end_idx - start_idx + 1 > int(max_span_words):
                continue
            vks = [ayat[k][0] for k in range(i, j + 1)]
            regions.append((start_idx, end_idx, vks))
            for k in range(i, j + 1):
                used[k] = True
            break
    return regions


def _merge_dup_ayah_regions(
    result, adj_regions, na_regions, is_recoverable_estimate, max_span_words
):
    """Combine adjacent (priority) and non-adjacent duplicate-ayah regions into a
    non-overlapping list to force-align.

    Adjacent regions are the more specific pass and always win: they are kept as
    given (they never overlap each other by construction). Each NON-adjacent
    region is first GROWN outward across contiguous recoverable interior
    estimates (bounded by ``max_span_words`` and the array ends), THEN — after
    growth — dropped if it overlaps ANY adjacent region, and finally de-overlapped
    against the other non-adjacent regions (first-start-wins). Checking the
    adjacent overlap AFTER growth is essential: a grown span can reach into an
    adjacent region it did not originally touch, and dropping it there keeps the
    adjacent region from being silently displaced. Pure list logic; no CTC."""
    regions = list(adj_regions)
    adj_spans = [(a, b) for (a, b, _vks) in adj_regions]
    n_words = len(result)
    grown = []
    for (a, b, vks) in na_regions:
        while a - 1 >= 0 and is_recoverable_estimate(result[a - 1]) and (b - (a - 1) + 1) <= max_span_words:
            a -= 1
        while b + 1 < n_words and is_recoverable_estimate(result[b + 1]) and ((b + 1) - a + 1) <= max_span_words:
            b += 1
        if any(not (b < aa or a > bb) for (aa, bb) in adj_spans):
            continue
        grown.append((a, b, vks))
    grown.sort(key=lambda r: (r[0], r[1]))
    last_end = -1
    for (a, b, vks) in grown:
        if a <= last_end:
            continue
        regions.append((a, b, vks))
        last_end = b
    return regions


def recover_duplicate_adjacent_ayat_ctc(
    mapped_words,
    args,
    wav_path,
    chapter_id=None,
    duration_ms=None,
):
    """Re-align consecutive near-identical ayat as a whole region via CTC.

    Root cause (e.g. Ash-Sharh 94:5-6 فَإِنَّ/إِنَّ مَعَ ٱلۡعُسۡرِ يُسۡرًا):
    Whisper collapses the two near-identical ayat into a SINGLE recitation, and
    the DP alignment then faces a genuine tie — the one transcribed copy matches
    both ayat equally well — which it breaks by spreading that copy across BOTH
    ayat. That leaves one full copy ESTIMATED on a fake sub-second window while
    the audio of the dropped copy sits unused in a large gap before the next
    ayah, so the estimated timestamps "slide onto the ayah below". The per-gap
    interior CTC pass cannot fix this: its window is the (wrong) collapsed
    sub-window between the mis-assigned neighbours, which holds no matching audio.

    This pass detects such regions from the KNOWN Quran text, IGNORES the
    untrustworthy internal timings, and force-aligns the FULL region text against
    the audio between the region's two OUTER real anchors — where the audio does
    contain every copy. It overwrites even currently-"real" words in the region
    (they are the mis-assigned copy), so it adopts the re-alignment ONLY when
    forced alignment confidently placed EVERY word of the region; otherwise the
    region is left untouched. Generic across all surahs/reciters; fires only when
    a duplicate-ayah region still holds an estimated word; needs two real anchors
    bracketing the region; degrades gracefully when the CTC deps are missing."""
    report = {"attempted": False, "recovered": 0, "regions": [], "words": [],
              "model": getattr(args, "ctc_model", None)}
    result = [dict(w) for w in mapped_words]

    if getattr(args, "no_dup_ayat_ctc", False):
        return result, report

    def is_recoverable_estimate(w):
        return (
            bool(w.get("estimated"))
            and not bool(w.get("opening_region_estimated"))
            and not bool(w.get("intro_gap_estimated"))
        )

    regions = adjacent_duplicate_ayah_regions(result)

    # v74: also recover NON-adjacent identical ayat (e.g. Al-Kafirun 109:3 ==
    # 109:5 with the different 109:4 between them). Whisper collapses the two
    # identical copies just like the adjacent case, but the copies are separated
    # by one or more DIFFERENT ayat so adjacent_duplicate_ayah_regions never sees
    # them. _merge_dup_ayah_regions grows each such span outward across contiguous
    # interior estimates hugging it (so a dropped word in a neighbouring ayah —
    # e.g. 109:2 تَعۡبُدُونَ — is repaired in the same force-align), then drops any
    # that overlap an adjacent region (adjacent wins) and de-overlaps the rest.
    # The merged spans go through the SAME OUTER-anchor force-align +
    # adopt-only-if-complete loop below.
    if not getattr(args, "no_nonadjacent_dup_ayat_ctc", False):
        max_span = int(getattr(args, "nonadjacent_dup_max_span_words", 40))
        na_regions = nonadjacent_duplicate_ayah_regions(
            result,
            max_intervening_ayat=int(getattr(args, "nonadjacent_dup_max_intervening_ayat", 1)),
            max_span_words=max_span,
        )
        regions = _merge_dup_ayah_regions(
            result, regions, na_regions, is_recoverable_estimate, max_span
        )

    if not regions:
        return result, report

    regions = [
        (a, b, vks) for (a, b, vks) in regions
        if any(is_recoverable_estimate(result[k]) for k in range(a, b + 1))
    ]
    if not regions:
        return result, report

    model_name = getattr(args, "ctc_model", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
    min_score = float(getattr(args, "dup_ayat_ctc_min_score", 0.05))
    min_word_ms = int(getattr(args, "dup_ayat_ctc_min_word_ms", 80))
    pad_ms = int(getattr(args, "dup_ayat_ctc_pad_ms", 120))
    device = getattr(args, "device_resolved", None) or pick_device(
        getattr(args, "device", "auto")
    )

    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        import soundfile  # noqa: F401
    except Exception as exc:
        report["skipped"] = "deps_missing"
        print(
            "Duplicate-ayah CTC recovery: skipped (CTC deps missing: "
            f"{exc}). Install with: pip install -U torchaudio transformers soundfile"
        )
        return result, report

    report["attempted"] = True
    dur_cap = int(duration_ms) if duration_ms else None
    total_recovered = 0

    for (a, b, vks) in regions:
        prev_idx = previous_timed_word(result, a)
        next_idx = next_timed_word(result, b)
        # Need BOTH outer real anchors to bracket the region audio. A region at
        # the very surah opening/end is owned by the opening / underrun passes.
        if prev_idx is None or next_idx is None:
            report["regions"].append({
                "start_index": a, "end_index": b, "verse_keys": vks,
                "skipped": "no_two_real_anchors", "recovered": 0,
            })
            continue

        win_start = int(result[prev_idx]["end_ms"])
        win_end = int(result[next_idx]["start_ms"])
        span_count = b - a + 1
        if win_end - win_start < min_word_ms * span_count:
            report["regions"].append({
                "start_index": a, "end_index": b, "verse_keys": vks,
                "win_start_ms": win_start, "win_end_ms": win_end,
                "skipped": "window_too_small", "recovered": 0,
            })
            continue

        aw_start = max(0, win_start - pad_ms)
        aw_end = win_end + pad_ms
        if dur_cap:
            aw_end = min(aw_end, dur_cap)

        texts = [
            (result[k].get("norm") or normalize_arabic(result[k].get("text") or ""))
            for k in range(a, b + 1)
        ]
        try:
            spans = ctc_align_window(
                wav_path=wav_path,
                transcript_texts=texts,
                device=device,
                model_name=model_name,
                window_start_ms=aw_start,
                window_sec=(aw_end - aw_start) / 1000.0,
            )
        except Exception as e:
            print(
                f"  Duplicate-ayah CTC recovery: alignment failed on region "
                f"{a}..{b} ({'/'.join(str(v) for v in vks)}): {e}"
            )
            report["regions"].append({
                "start_index": a, "end_index": b, "verse_keys": vks,
                "skipped": "ctc_failed", "recovered": 0,
            })
            continue

        # This pass overwrites even currently-"real" region words (they are the
        # mis-assigned copy), so a partial result could corrupt good timings.
        # Adopt ONLY when forced alignment placed EVERY word of the region above
        # the floor; otherwise leave the region untouched.
        placed = []
        complete = len(spans) == span_count
        if complete:
            for span in spans:
                if (not span or span.get("start_ms") is None
                        or span.get("end_ms") is None):
                    complete = False
                    break
                sc = span.get("score")
                if sc is not None and float(sc) < min_score:
                    complete = False
                    break
                placed.append((int(span["start_ms"]), int(span["end_ms"]),
                               None if sc is None else float(sc)))
        if not complete:
            report["regions"].append({
                "start_index": a, "end_index": b, "verse_keys": vks,
                "win_start_ms": win_start, "win_end_ms": win_end,
                "skipped": "incomplete_alignment", "recovered": 0,
            })
            continue

        pre_estimated = sum(
            1 for k in range(a, b + 1) if bool(result[k].get("estimated"))
        )
        last_end = win_start
        for qi in range(span_count):
            idx = a + qi
            s0, e0, sc = placed[qi]
            s = max(s0, last_end, win_start)
            e = min(e0, win_end)
            if e < s + min_word_ms:
                e = min(win_end, s + min_word_ms)
            if e <= s:
                s = last_end
                e = min(win_end, s + min_word_ms)
            old_s = result[idx].get("start_ms")
            old_e = result[idx].get("end_ms")
            result[idx]["start_ms"] = int(s)
            result[idx]["end_ms"] = int(e)
            result[idx]["estimated"] = False
            result[idx]["ctc_dup_ayat_recovered"] = True
            if sc is not None:
                result[idx]["score"] = max(float(result[idx].get("score") or 0), sc)
            result[idx]["method"] = str(result[idx].get("method", "")) + "+dup_ayat_ctc"
            for stale in ("opening_region_estimated", "intro_gap_estimated"):
                result[idx].pop(stale, None)
            report["words"].append({
                "index": idx,
                "verse_key": result[idx].get("verse_key"),
                "position": result[idx].get("position"),
                "text": result[idx].get("text"),
                "norm": texts[qi],
                "old_start_ms": old_s, "old_end_ms": old_e,
                "new_start_ms": int(s), "new_end_ms": int(e),
                "score": sc,
            })
            last_end = e

        total_recovered += pre_estimated
        report["regions"].append({
            "start_index": a, "end_index": b, "verse_keys": vks,
            "win_start_ms": win_start, "win_end_ms": win_end,
            "recovered": pre_estimated, "adopted": True,
        })

    report["recovered"] = total_recovered
    if report["attempted"]:
        adopted = len([r for r in report["regions"] if r.get("adopted")])
        print(
            f"Duplicate-ayah CTC recovery: recovered {total_recovered} real "
            f"word span(s) across {adopted} region(s)."
        )
    return result, report


def underrun_intro_prefix_norms(quran_words, chapter_id=None):
    """Normalized token list for the standard recitation intro (isti'adha +
    basmala) that precedes a surah in most recordings.

    Used ONLY to absorb the leading intro audio when force-aligning a whole
    surah with CTC: the prefix tokens occupy the isti'adha/basmala region so the
    surah words anchor on the actual surah audio. These are the fixed liturgical
    formulas already referenced by ``build_quran_initial_prompt`` (NOT per-surah
    hardcoding). Forced alignment is monotonic and tolerant: if the reciter did
    not actually recite the isti'adha, those few tokens collapse into a tiny
    low-score span at the very start and the surah words still align correctly.
    """
    parts = []
    for w in "أعوذ بالله من الشيطان الرجيم".split():
        n = normalize_arabic(w)
        if n:
            parts.append(n)
    ch = chapter_id
    if ch != 9 and not quran_starts_with_basmala(quran_words, chapter_id=ch):
        for w in "بسم الله الرحمن الرحيم".split():
            n = normalize_arabic(w)
            if n:
                parts.append(n)
    return parts


def recover_underrun_surah_ctc(
    mapped_words,
    quran_words,
    args,
    wav_path,
    chapter_id=None,
    duration_ms=None,
    force=False,
    force_reason=None,
):
    """Last-resort full-surah CTC forced alignment for SEVERELY under-transcribed
    recordings.

    When Whisper structurally drops most of a surah (e.g. a 14-word surah
    transcribed as 5 words), no ASR-side recovery can reach the quality gate:
    the best alignment still leaves the majority of words MISSING or ESTIMATED.
    This pass force-aligns the KNOWN Quran text of the whole surah (with a
    standard isti'adha/basmala prefix to absorb the leading intro) against the
    audio with the char-level wav2vec2 CTC model. Forced alignment never drops a
    token, so every surah word gets a REAL measured span — including muqatta'at
    openings and madd (the acoustic model spans the elongated audio).

    Generic across all surahs/reciters; no per-surah hardcoding. Triggers only
    when a large fraction of words still lack real timestamps after every other
    repair, and only for clips short enough to align in one CTC pass, so healthy
    chapters are never touched. Degrades gracefully when the CTC deps are
    missing. Recovered spans replace ONLY words that lacked a real timestamp;
    words already aligned by ASR are kept.
    """
    report = {
        "attempted": False,
        "recovered": 0,
        "bad_fraction": 0.0,
        "model": getattr(args, "ctc_model", None),
    }
    result = [dict(w) for w in mapped_words]

    if getattr(args, "no_underrun_ctc", False):
        return result, report

    total = len(quran_words)
    if total == 0 or len(result) != total:
        return result, report

    def lacks_real(w):
        return (
            w.get("start_ms") is None
            or w.get("end_ms") is None
            or bool(w.get("estimated"))
        )

    bad = sum(1 for w in result if lacks_real(w))
    bad_fraction = bad / total
    report["bad_fraction"] = round(bad_fraction, 4)

    min_bad = float(getattr(args, "underrun_ctc_min_bad_fraction", 0.5))
    if bad_fraction < min_bad and not force:
        return result, report
    if force:
        # v84: escalation path — the opening-scoped CTC pass rejected its
        # alignment with a window-end squeeze (the first-real anchor bounding
        # the opening window is itself mis-timed too early, so NO window-scoped
        # attempt can succeed). The full-surah forced alignment is the only
        # pass that can both place the opening words AND move the bogus anchor
        # (misanchored-interior guard). Because the bad-fraction gate is
        # bypassed, adoption additionally requires a confident whole-surah
        # alignment (mean-score floor below); on failure everything is kept.
        report["forced"] = force_reason or True

    max_dur = int(getattr(args, "underrun_ctc_max_duration_ms", 120000))
    if duration_ms and int(duration_ms) > max_dur:
        report["skipped"] = "duration_exceeds_max"
        print(
            f"Underrun CTC fallback: skipped (clip {int(duration_ms)}ms > "
            f"max {max_dur}ms); raise --underrun-ctc-max-duration-ms to enable."
        )
        return result, report

    model_name = getattr(args, "ctc_model", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
    min_score = float(getattr(args, "underrun_ctc_min_score", 0.0))
    min_word_ms = int(getattr(args, "underrun_ctc_min_word_ms", 80))
    device = getattr(args, "device_resolved", None) or pick_device(
        getattr(args, "device", "auto")
    )

    # Probe CTC deps once; degrade gracefully with a clear hint if missing.
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        import soundfile  # noqa: F401
    except Exception as exc:
        report["error"] = f"ctc-deps-missing: {exc}"
        print(
            "Underrun CTC fallback: skipped (CTC deps missing: "
            f"{exc}). Install with: pip install -U torchaudio transformers soundfile"
        )
        return result, report

    full_prefix = underrun_intro_prefix_norms(quran_words, chapter_id)
    surah_norms = [str(w.get("norm") or "") for w in quran_words]

    # v84: like the opening-scoped pass (v83), the recited intro varies —
    # forcing intro tokens that are NOT in the audio does not merely collapse
    # at the start: with an absent isti'adha the 9-token prefix stretches over
    # the real basmala AND the first surah words, pushing the measured intro
    # end (and the whole opening) seconds too late. Try the standard intro
    # hypotheses and keep the one whose SURAH words score best; a shorter
    # prefix must win by a clear margin (longest-first order).
    istiadha_len = 5
    prefix_hypotheses = [list(full_prefix)]
    if len(full_prefix) > istiadha_len:
        prefix_hypotheses.append(list(full_prefix[istiadha_len:]))
    prefix_hypotheses.append([])

    report["attempted"] = True
    print(
        f"Underrun CTC fallback: {bad}/{total} word(s) lack real timestamps "
        f"({bad_fraction:.0%} bad) -> full-surah forced alignment "
        f"({len(prefix_hypotheses)} intro prefix hypotheses + {total} surah tokens)."
    )

    best = None  # (mean_surah_score, prefix, spans)
    align_error = None
    hyp_margin = float(getattr(args, "opening_ctc_prefix_hypothesis_margin", 0.02))
    for prefix_try in prefix_hypotheses:
        transcript = list(prefix_try) + surah_norms
        try:
            spans_try = ctc_align_window(
                wav_path,
                transcript,
                device,
                model_name,
                window_start_ms=0,
                window_sec=None,
            )
        except Exception as exc:
            align_error = str(exc)
            continue
        surah_try = spans_try[len(prefix_try):]
        if len(surah_try) != total:
            align_error = "span-count-mismatch"
            continue
        scores_try = [
            float(s.get("score") or 0.0)
            if s and s.get("start_ms") is not None and s.get("end_ms") is not None
            else 0.0
            for s in surah_try
        ]
        mean_try = sum(scores_try) / max(1, len(scores_try))
        if best is None or mean_try > best[0] + hyp_margin:
            best = (mean_try, list(prefix_try), spans_try)

    if best is None:
        report["error"] = align_error or "alignment failed"
        print(
            f"Underrun CTC fallback: alignment failed ({report['error']}); "
            "keeping prior alignment."
        )
        return result, report

    surah_mean_score, prefix, spans = best
    report["prefix_len"] = len(prefix)
    report["surah_mean_score"] = round(surah_mean_score, 4)
    print(
        f"Underrun CTC fallback: best intro hypothesis = {len(prefix)} intro "
        f"token(s) (mean surah score {surah_mean_score:.3f})."
    )

    prefix_spans = spans[: len(prefix)]
    surah_spans = spans[len(prefix):]

    if force:
        # v84 escalation confidence gate: the catastrophic-underrun trigger is
        # bypassed, so require positive acoustic evidence that the whole-surah
        # forced alignment actually matches the audio before adopting anything.
        force_floor = float(
            getattr(args, "underrun_ctc_force_min_mean_score", 0.15)
        )
        placed_scores = [
            float(s.get("score") or 0.0)
            for s in surah_spans
            if s and s.get("start_ms") is not None and s.get("end_ms") is not None
        ]
        force_mean = (
            sum(placed_scores) / len(placed_scores) if placed_scores else 0.0
        )
        report["force_mean_score"] = round(force_mean, 4)
        if len(placed_scores) != total or force_mean < force_floor:
            report["error"] = "force_low_confidence"
            print(
                f"Underrun CTC fallback (escalation): whole-surah alignment not "
                f"confident enough (placed {len(placed_scores)}/{total}, mean "
                f"score {force_mean:.3f} < {force_floor}); keeping prior alignment."
            )
            return result, report

    # The boundary between the forced-aligned intro tokens (isti'adha/basmala) and
    # the first surah token is the REAL measured intro end — far more trustworthy
    # than a strip detection derived from the broken under-transcription. Capture
    # it so the opening-intro verification can override the bogus stripped span.
    intro_end_candidates = [
        int(s["end_ms"]) for s in prefix_spans
        if s and s.get("end_ms") is not None
    ]
    intro_end_ms = max(intro_end_candidates) if intro_end_candidates else None

    # Was the intro (isti'adha/basmala) actually recited? When it was, the prefix
    # tokens force-align onto a real, non-degenerate audio region, so intro_end_ms
    # is a trustworthy boundary and any PRESERVED surah word timestamped before it
    # is mis-anchored onto the intro audio (a catastrophic-underrun DP artifact,
    # e.g. surah word #2 pinned onto the basmala at ~0ms) and must yield to its
    # full-surah CTC span. When the intro was NOT recited, the prefix collapses
    # into tiny low-score spans and intro_end_ms is meaningless, so we must NOT
    # override — a correctly-early opening word would be destroyed. Keyed only on
    # measurable prefix coverage/score: generic, no per-surah/per-reciter logic.
    intro_min_ms = int(getattr(args, "underrun_ctc_intro_min_ms", 700))
    intro_min_score = float(getattr(args, "underrun_ctc_intro_min_score", 0.15))
    intro_tol = int(getattr(args, "underrun_ctc_intro_tol_ms", 120))
    prefix_scores = [
        float(s["score"]) for s in prefix_spans
        if s and s.get("score") is not None
    ]
    prefix_mean_score = (
        sum(prefix_scores) / len(prefix_scores) if prefix_scores else 0.0
    )
    prefix_covered = bool(prefix_spans) and all(
        s and s.get("start_ms") is not None and s.get("end_ms") is not None
        for s in prefix_spans
    )
    intro_confident = bool(
        prefix_covered
        and len(prefix_scores) == len(prefix_spans)
        and intro_end_ms is not None
        and int(intro_end_ms) >= intro_min_ms
        and prefix_mean_score >= intro_min_score
    )
    report["intro_confident"] = intro_confident
    report["prefix_mean_score"] = round(prefix_mean_score, 4)

    def misanchored_into_intro(w):
        """A PRESERVED (ASR-'real') surah word whose start falls inside the
        confidently-measured intro audio is definitionally mis-anchored: no surah
        word legitimately overlaps a recited basmala. Only fires when the intro
        was actually recited (intro_confident), so a correctly-early opening word
        in a basmala-skipping recording is never clobbered."""
        if not intro_confident or intro_end_ms is None:
            return False
        s = w.get("start_ms")
        if s is None:
            return False
        return int(s) < int(intro_end_ms) - intro_tol

    # A PRESERVED word can also be mis-anchored far EARLIER than the full-surah
    # forced aligner places it -- the interior analogue of the intro artifact. When
    # Whisper collapses a long stretch into ONE hallucinated timestamp (e.g. a 27s
    # "الرحيم" spanning the whole surah, reciter-B Al-Masad 111), the sparse interior
    # recovery drops a LATE word into that huge ambiguous window and pins it early
    # (e.g. verse-5 "في" at ~5.3s). Left as a "real" anchor, the backward monotonic
    # clamp below then crushes every recovered word before it into a degenerate
    # cluster (18 words in ~1.7s). Forced alignment uses the FULL known text and is
    # strictly monotonic, so its span is the trustworthy POSITION: when the
    # forced-align start is MORE THAN interior_gap_ms LATER than the preserved
    # start, the preserved anchor is bogus and must yield to it. Purely
    # positional (madd only shifts a word LATER, never earlier, so it can never
    # trip this), generic, no per-surah/per-reciter logic. The gap is large enough
    # that ordinary ASR-vs-CTC jitter or a correctly-early basmala-skipping opening
    # word (forced-align only ~1-2s off) is never clobbered.
    interior_gap_ms = int(
        getattr(args, "underrun_ctc_interior_misanchor_gap_ms", 3000)
    )

    def misanchored_interior(idx):
        w = result[idx]
        s = w.get("start_ms")
        if s is None:
            return False
        fa = surah_spans[idx]
        if not fa or fa.get("start_ms") is None:
            return False
        return int(fa["start_ms"]) - int(s) > interior_gap_ms

    # Phase 1: assign forced-align spans to words that currently lack a real
    # timestamp. This pass fires ONLY on catastrophic under-transcription, where
    # the text is KNOWN and forced alignment never drops a token, so every word
    # must be placed: accept any non-None span (min_score is a tunable floor, 0.0
    # by default) and rely on the two-direction monotonic clamp + min_word_ms for
    # sanity. Words already aligned by ASR are preserved (both are real spans).
    recovered = 0
    overridden_intro = 0
    overridden_interior = 0
    for idx, span in enumerate(surah_spans):
        needs = lacks_real(result[idx])
        intro_bad = (not needs) and misanchored_into_intro(result[idx])
        interior_bad = (not needs) and (not intro_bad) and misanchored_interior(idx)
        if not needs and not intro_bad and not interior_bad:
            continue
        if not span or span.get("start_ms") is None or span.get("end_ms") is None:
            continue
        score = span.get("score")
        if score is not None and float(score) < min_score:
            continue
        start = int(span["start_ms"])
        end = int(span["end_ms"])
        # Raw forced-align width BEFORE the min_word_ms floor: this is the honest
        # signal of whether the CTC model actually localised the token. A raw width
        # below the floor means the aligner gave the token ~no real duration; the
        # compressed-cluster guard (below) uses this, NOT the post-floor/post-clamp
        # width, so a legit recovery that merely gets clamped near a preserved word
        # is not mistaken for a degenerate one.
        raw_width = end - start
        if end < start + min_word_ms:
            end = start + min_word_ms
        if duration_ms:
            end = min(end, int(duration_ms))
            start = min(start, max(0, end - min_word_ms))
        if end <= start:
            continue
        result[idx]["start_ms"] = start
        result[idx]["end_ms"] = end
        result[idx]["estimated"] = False
        result[idx]["score"] = float(score) if score is not None else 0.0
        result[idx]["ctc_underrun_recovered"] = True
        result[idx]["_ctc_raw_width_ms"] = raw_width
        if intro_bad:
            result[idx]["intro_misanchor_overridden"] = True
            overridden_intro += 1
        if interior_bad:
            result[idx]["interior_misanchor_overridden"] = True
            overridden_interior += 1
        for stale in ("opening_region_estimated", "intro_gap_estimated"):
            if stale in result[idx]:
                result[idx].pop(stale, None)
        recovered += 1
    report["intro_misanchor_overridden"] = overridden_intro
    report["interior_misanchor_overridden"] = overridden_interior
    if overridden_intro:
        print(
            f"Underrun CTC fallback: overrode {overridden_intro} intro-misanchored "
            f"preserved word(s) with forced-align spans (intro confidently recited)."
        )
    if overridden_interior:
        print(
            f"Underrun CTC fallback: overrode {overridden_interior} interior-misanchored "
            f"preserved word(s) with forced-align spans (preserved anchor pinned "
            f">{interior_gap_ms}ms before where forced alignment places the word)."
        )

    # Phase 2: monotonic clamp across the WHOLE surah timeline. CTC spans and
    # Whisper spans share one absolute timeline, but they can disagree slightly,
    # so a recovered word may overlap a preserved real-timed neighbour. Move ONLY
    # the CTC-recovered words (never the preserved ASR words): forward pass keeps
    # each recovered start at/after the previous real end; backward pass keeps
    # each recovered end at/before the next real start.
    if recovered:
        last_end = None
        for idx in range(total):
            w = result[idx]
            if w.get("start_ms") is None or w.get("end_ms") is None:
                continue
            if w.get("ctc_underrun_recovered"):
                if last_end is not None and w["start_ms"] < last_end:
                    w["start_ms"] = last_end
                if w["end_ms"] < w["start_ms"] + min_word_ms:
                    w["end_ms"] = w["start_ms"] + min_word_ms
                if duration_ms:
                    w["end_ms"] = min(w["end_ms"], int(duration_ms))
                    w["start_ms"] = min(w["start_ms"], max(0, w["end_ms"] - min_word_ms))
            last_end = w["end_ms"] if last_end is None else max(last_end, w["end_ms"])

        next_start = None
        for idx in range(total - 1, -1, -1):
            w = result[idx]
            if w.get("start_ms") is None or w.get("end_ms") is None:
                continue
            if w.get("ctc_underrun_recovered"):
                if next_start is not None and w["end_ms"] > next_start:
                    w["end_ms"] = next_start
                if w["end_ms"] < w["start_ms"] + min_word_ms:
                    w["start_ms"] = max(0, w["end_ms"] - min_word_ms)
            next_start = w["start_ms"] if next_start is None else min(next_start, w["start_ms"])

    # Compressed-cluster guard: on audio that BOTH Whisper and the CTC model fail
    # on (upstream audio/ASR failure, not a recoverable under-transcription), the
    # forced aligner cannot localise the known tokens and crams the whole opening
    # into a tiny window (e.g. 18 words in ~1.7s ~= 10 words/sec). Those spans are
    # non-zero and monotonic, so the degenerate-span guard (end<=start) misses them,
    # and marking them real would fake a 100%/0-estimated green for garbage timings.
    # Detect the implausible packing with TWO complementary signals and, if either
    # fires, REVERT the whole CTC recovery to estimated so the quality gate flags the
    # surah for manual review instead of false-greening it:
    #   1) RATE CEILING -- recovered words per second across the recovered window
    #      exceeds a plausible recitation ceiling. This is the primary signal: it
    #      catches the real catastrophe (reciter-B Al-Masad 111, ~10.4 w/s) where the
    #      aligner returns wide-but-overlapping spans that clamp down to a floor wall
    #      yet each RAW width is >= the floor (so the width signal alone misses it).
    #   2) RAW-WIDTH FLOOR FRACTION -- most recovered words had a RAW forced-align
    #      width (captured in Phase 1, before floor/clamp) below the floor, i.e. the
    #      aligner gave them ~no real duration. Catches zero-width degeneracy even
    #      when the spans happen to be spread out.
    # Both are rate CEILINGS (too many words too fast / no duration); madd only makes
    # words LONGER (slower, wider window -> LOWER rate), so neither can ever conflict
    # with the madd rule. Fully generic: no per-surah/per-reciter logic. A real
    # recovery spreads its words across the clip (well under the ceiling) and keeps
    # real, varied widths, so it is never mistaken for a degenerate one.
    recovered_idx = [
        i for i in range(total)
        if result[i].get("ctc_underrun_recovered")
        and result[i].get("start_ms") is not None
        and result[i].get("end_ms") is not None
    ]
    if recovered_idx:
        # Signal 1: recovered words-per-second across the recovered cluster window
        # (post-clamp final spans -- what actually gets delivered). A tiny/zero
        # window means everything piled up -> treat as infinitely fast.
        starts = [result[i]["start_ms"] for i in recovered_idx]
        ends = [result[i]["end_ms"] for i in recovered_idx]
        window_ms = max(ends) - min(starts)
        words_per_sec = (
            len(recovered_idx) / (window_ms / 1000.0)
            if window_ms > 0 else float("inf")
        )
        # Signal 2: raw-width floor fraction (Phase-1 raw widths, before floor/clamp).
        floor_like = [
            i for i in recovered_idx
            if result[i].get("_ctc_raw_width_ms") is not None
            and int(result[i]["_ctc_raw_width_ms"]) < min_word_ms
        ]
        floor_frac = len(floor_like) / len(recovered_idx)
        min_cluster = int(getattr(args, "underrun_ctc_degenerate_min_words", 4))
        max_floor_frac = float(getattr(args, "underrun_ctc_degenerate_floor_frac", 0.85))
        max_wps = float(getattr(args, "underrun_ctc_degenerate_max_wps", 7.0))
        rate_bad = words_per_sec > max_wps
        floor_bad = floor_frac >= max_floor_frac
        degenerate = len(recovered_idx) >= min_cluster and (rate_bad or floor_bad)
        report["ctc_recovery_floor_words"] = len(floor_like)
        report["ctc_recovery_floor_frac"] = round(floor_frac, 4)
        report["ctc_recovery_window_ms"] = window_ms
        report["ctc_recovery_words_per_sec"] = (
            round(words_per_sec, 3) if window_ms > 0 else None
        )
        report["ctc_recovery_degenerate"] = degenerate
        if degenerate:
            for i in recovered_idx:
                result[i]["estimated"] = True
                result[i]["ctc_recovery_degenerate"] = True
            reasons = []
            if rate_bad:
                rate_txt = (
                    f"{words_per_sec:.2f}" if window_ms > 0 else "inf"
                )
                reasons.append(
                    f"{len(recovered_idx)} words in {window_ms}ms "
                    f"= {rate_txt} words/sec > {max_wps:g} ceiling"
                )
            if floor_bad:
                reasons.append(
                    f"{len(floor_like)}/{len(recovered_idx)} recovered words had a "
                    f"raw forced-align width below the {min_word_ms}ms floor"
                )
            print(
                "Underrun CTC fallback: recovery is DEGENERATE ("
                + "; ".join(reasons)
                + ") -> reverting to estimated; audio is not auto-alignable, "
                "surah flagged for review."
            )
    # Drop the internal raw-width bookkeeping key so it never leaks into output.
    for i in range(total):
        result[i].pop("_ctc_raw_width_ms", None)

    # Cap the measured intro boundary against the earliest CTC-RECOVERED surah word
    # only — NOT against all real words. A preserved pre-existing "real" word can
    # itself be intro-misaligned in a catastrophic run; capping against it would pull
    # the boundary too early and silently hide a genuine opening overlap. The
    # recovered surah spans come from the SAME monotonic alignment as the intro
    # prefix, so they are the trustworthy anchor (max prefix end <= first recovered
    # surah start by construction). Leaving the boundary at the true prefix end means
    # a misaligned preserved opening word still surfaces as a review warning.
    recovered_starts = [
        w["start_ms"] for w in result
        if w.get("ctc_underrun_recovered") and w.get("start_ms") is not None
    ]
    earliest_recovered = min(recovered_starts) if recovered_starts else None
    if intro_end_ms is None:
        intro_end_ms = earliest_recovered
    elif earliest_recovered is not None:
        intro_end_ms = min(intro_end_ms, earliest_recovered)
    report["intro_end_ms"] = intro_end_ms

    report["recovered"] = recovered
    print(f"Underrun CTC fallback: recovered {recovered}/{total} real word span(s).")
    return result, report


def opening_ctc_leading_estimated_run(mapped_words):
    """Length of the leading run of words (from index 0) forming the broken
    opening island the opening-scoped CTC recovery targets.

    A word belongs to the island when it lacks a REAL timestamp (missing span
    or estimated) OR is still flagged ``opening_region_estimated`` — its timing
    came from the opening-safe interpolation repair, even if a later heuristic
    (e.g. a char-projection hit on the first word) flipped it to real. v82
    (reciter-C Al-Kafirun regression): char projection planted قل at the intro
    boundary as real, making the leading run 0, so the opening CTC silently
    skipped while يا أيها الكافرون لا stayed squeezed interpolations that the
    interior recovery also refuses to touch (it excludes opening-region words).
    Trusted recoveries pop the flag, so a genuinely repaired opening still
    terminates the island. Pure/deterministic; unit-testable without CTC deps."""
    n = 0
    for w in mapped_words:
        lacks_real = (
            w.get("start_ms") is None
            or w.get("end_ms") is None
            or bool(w.get("estimated"))
        )
        if lacks_real or bool(w.get("opening_region_estimated")):
            n += 1
        else:
            break
    return n


def opening_ctc_intro_budget(quran_words, args, chapter_id=None):
    """Plausible intro-sized lead-in for a surah opening: recited isti'adha
    (~5) + basmala (when the recited basmala is an intro, i.e. the surah text
    does not itself begin with the basmala and it is not surah 9) + any huruf
    muqatta'at rendered by letter name + a small buffer. Mirrors the quality
    intro budget used for opening-estimate forgiveness. Pure/unit-testable."""
    istiadha_budget = 5
    basmala_expected = (
        chapter_id != 9
        and not quran_starts_with_basmala(quran_words, chapter_id=chapter_id)
    )
    basmala_budget = len(BASMALA_PHRASE) if basmala_expected else 0
    muqattaat_budget = len(describe_opening_muqattaat(quran_words, chapter_id=chapter_id))
    forgive_buffer = max(0, int(getattr(args, "opening_estimate_forgive_buffer", 4)))
    return istiadha_budget + basmala_budget + muqattaat_budget + forgive_buffer


def opening_ctc_intro_inflation_plan(
    mapped_words,
    intro_audio_end_ms,
    inflated_intro_ms=12000,
    overlap_tol_ms=200,
    min_anchor_run=3,
    scan_limit=120,
):
    """Detect the Whisper 30s-decode-chunk intro-inflation artifact and describe
    the opening region that must be re-aligned.

    When a reciter leaves a long silence/pause after the basmala, Whisper (which
    decodes in 30s windows) stretches the last basmala word(s) across the WHOLE
    chunk, so the stripped intro block ends at ~30000ms (e.g. 29980ms). The real
    opening recitation, poorly transcribed in that mangled first chunk, then gets
    scattered: a few surah words grab tiny timestamps INSIDE the intro span
    (e.g. 0..760ms) while the true opening stays estimated. Those "false real"
    words at ~0ms defeat the opening-CTC recovery two ways: they cut the
    leading-estimated run short so the intro-budget gate never fires, and they
    make ``first_real_start`` = 0 so the CTC window collapses.

    This returns a plan dict with the trustworthy post-intro ANCHOR (a run of
    consecutive real, monotonic words whose starts are all at/after the inflated
    intro end) and the leading indices to DEMOTE to estimated, so the existing
    opening-CTC machinery re-aligns the entire opening window against the real
    recitation audio. Returns None when the artifact is absent: normal-length
    intro, no solid post-intro anchor, or no surah word mis-aligned inside the
    intro. Pure/deterministic — unit-testable without the CTC deps. Generic: no
    per-surah or per-reciter hardcoding; keyed only on the implausible intro span
    plus a mis-aligned opening word plus a solid resume anchor.

    v75: the same artifact occurs at a SHORTER span than the ~30s decode-chunk
    case when Whisper merges the basmala and the ENTIRE first ayah into one
    "basmala" block (e.g. Al-Insan 076: basmala + the 11-word verse 1 collapsed
    into a single ~14s intro block, so every verse-1 word was parked inside the
    intro in front of verse 2). The default threshold is therefore below the
    30s figure; a real isti'adha+basmala still never approaches it, and the
    anchor + demote guards (not the raw duration) are what prevent false fires.
    """
    if intro_audio_end_ms is None:
        return None
    try:
        intro_end = int(intro_audio_end_ms)
    except (TypeError, ValueError):
        return None
    if intro_end < int(inflated_intro_ms):
        return None
    total = len(mapped_words)
    if total == 0:
        return None

    def _lacks_real(w):
        return (
            w.get("start_ms") is None
            or w.get("end_ms") is None
            or bool(w.get("estimated"))
        )

    cutoff = intro_end - max(0, int(overlap_tol_ms))
    run = max(1, int(min_anchor_run))
    hi = min(total, max(1, int(scan_limit)))

    anchor_idx = None
    for i in range(hi):
        if i + run > total:
            break
        ok = True
        prev = None
        for j in range(i, i + run):
            w = mapped_words[j]
            s = w.get("start_ms")
            if _lacks_real(w) or s is None or int(s) < cutoff:
                ok = False
                break
            if prev is not None and int(s) < prev:
                ok = False
                break
            prev = int(s)
        if ok:
            anchor_idx = i
            break

    # anchor_idx == 0 means genuine recitation already resumes at index 0, so
    # there is no mis-aligned opening in front of it to re-align.
    if not anchor_idx:
        return None

    anchor_start = int(mapped_words[anchor_idx]["start_ms"])
    # Demote every real word positioned BEFORE the genuine resume anchor. The
    # mis-aligned opening word may be scattered at ~0ms deep inside the intro
    # span, OR parked right at the inflated intro end (start >= cutoff but still
    # < anchor_start) — e.g. reciter-B At-Taghabun, where the 30s basmala block
    # swallowed the whole first verse and its last real word sat exactly on the
    # boundary, leaving the leading run one short of the intro budget so the
    # opening-CTC gate never fired. Keying on the anchor start (not the intro
    # cutoff) catches both shapes with no per-surah tuning; anchor_start >= cutoff
    # always, so this strictly extends the previous cutoff-based rule.
    demote = [
        k
        for k in range(anchor_idx)
        if (not _lacks_real(mapped_words[k]))
        and mapped_words[k].get("start_ms") is not None
        and int(mapped_words[k]["start_ms"]) < anchor_start
    ]
    if not demote:
        return None

    return {
        "intro_audio_end_ms": intro_end,
        "cutoff_ms": cutoff,
        "anchor_index": anchor_idx,
        "anchor_start_ms": anchor_start,
        "demote_indices": demote,
    }


def recover_underrun_opening_ctc(
    mapped_words,
    quran_words,
    args,
    wav_path,
    chapter_id=None,
    duration_ms=None,
    intro_audio_end_ms=None,
):
    """Opening-scoped CTC forced alignment for a LONG, otherwise-healthy surah
    whose OPENING (first verse or two) Whisper dropped or mis-timed.

    The full-surah underrun CTC (``recover_underrun_surah_ctc``) only fires on
    catastrophic clips (>=50% of words lack a real timestamp) that are also short
    enough to align in one pass. A long surah that aligns fine everywhere except
    a broken ~30s opening never meets either bar, so its opening words keep fake
    interpolated timestamps behind an over-detected intro (e.g. Al-Furqan
    reciter-B: ayah 1 + the start of ayah 2 interpolated behind a bogus 30s intro,
    effective_match_rate ~98.5% -> needs review).

    This pass force-aligns ONLY the leading region: the isti'adha/basmala prefix
    plus the run of surah words from index 0 up to (but excluding) the first word
    that already has a REAL Whisper timestamp, against the audio window from 0ms
    to that first-real word's start (plus a small pad). The window is just the
    opening, so it always fits the CTC duration budget even for long surahs, and
    the healthy interior is never touched. Fires only when that leading estimated
    run exceeds the plausible intro budget (isti'adha + basmala + muqatta'at +
    buffer) — i.e. genuine dropped opening recitation, not a normal intro.
    Generic across all surahs/reciters; no per-surah hardcoding. Degrades
    gracefully when the CTC deps are missing. Replaces only opening words that
    lack a real timestamp; words already aligned by ASR are kept.
    """
    report = {"attempted": False, "recovered": 0, "leading_estimated": 0}
    result = [dict(w) for w in mapped_words]

    if getattr(args, "no_underrun_ctc", False) or getattr(args, "no_opening_ctc", False):
        return result, report

    total = len(quran_words)
    if total == 0 or len(result) != total:
        return result, report

    def lacks_real(w):
        return (
            w.get("start_ms") is None
            or w.get("end_ms") is None
            or bool(w.get("estimated"))
        )

    # v68: Whisper 30s-decode-chunk intro-inflation guard. When the stripped
    # intro block spans an implausibly long region (~30000ms) because Whisper
    # stretched the basmala across a long post-basmala pause, the dropped opening
    # gets scattered — a few surah words grab tiny timestamps inside the intro
    # while the true opening stays estimated. Demote those mis-aligned in-intro
    # words to estimated so the leading run extends to the first genuine
    # post-intro anchor and the opening-CTC window re-aligns the whole opening.
    inflation_plan = opening_ctc_intro_inflation_plan(
        result,
        intro_audio_end_ms,
        inflated_intro_ms=int(getattr(args, "opening_ctc_inflated_intro_ms", 12000)),
        overlap_tol_ms=int(getattr(args, "opening_ctc_intro_overlap_tol_ms", 200)),
        min_anchor_run=int(getattr(args, "opening_ctc_anchor_run", 3)),
        scan_limit=int(getattr(args, "opening_ctc_intro_scan_limit", 120)),
    )
    if inflation_plan:
        for k in inflation_plan["demote_indices"]:
            result[k]["estimated"] = True
            result[k]["intro_inflation_demoted"] = True
        report["intro_inflation"] = {
            "intro_audio_end_ms": inflation_plan["intro_audio_end_ms"],
            "anchor_index": inflation_plan["anchor_index"],
            "anchor_start_ms": inflation_plan["anchor_start_ms"],
            "demoted": len(inflation_plan["demote_indices"]),
        }
        print(
            "Opening CTC recovery: inflated-intro artifact detected "
            f"(intro end ~{inflation_plan['intro_audio_end_ms']}ms) — demoted "
            f"{len(inflation_plan['demote_indices'])} mis-aligned opening word(s) "
            f"before anchor index {inflation_plan['anchor_index']} "
            f"@ {inflation_plan['anchor_start_ms']}ms for re-alignment."
        )

    lead_end = opening_ctc_leading_estimated_run(result)
    report["leading_estimated"] = lead_end
    if lead_end == 0 or lead_end >= total:
        return result, report  # opening already real, or nothing else to anchor to

    intro_budget = opening_ctc_intro_budget(quran_words, args, chapter_id=chapter_id)
    report["intro_budget"] = intro_budget
    # The intro-budget gate guards the AMBIGUOUS case: a short leading run of
    # words that lack a real timestamp and might simply be an untimed intro. When
    # the inflated-intro plan fired we have POSITIVE evidence instead — real surah
    # words were parked inside the intro span in front of a solid post-intro
    # anchor — so the leading run is dropped recitation, not intro, no matter how
    # it compares to the budget. Without this, Al-Insan 076 (basmala + the whole
    # 11-word verse 1 merged into one ~14s block) leaves a leading run of 11 that
    # the 13-word intro budget would forgive, so the opening never gets re-aligned.
    if lead_end <= intro_budget and not inflation_plan:
        # v81: a budget-sized lead was previously ALWAYS presumed to be an
        # unrecited intro and skipped — which left genuinely dropped opening
        # recitation (e.g. reciter-C Al-Kafirun: Whisper drops قل يا أيها
        # الكافرون لا, 5 words <= budget 13) with interpolated timestamps that
        # v64 then forgave into a fake-clean effective rate. Reuse the v71
        # vocabulary rule: only isti'adha/basmala/muqatta'at tokens can
        # legitimately be an unrecited lead-in. If ANY leading word lacking a
        # real timestamp is arbitrary surah text, it is dropped RECITATION —
        # fall through and force-align the opening window so those words get
        # REAL measured spans instead of forgiven interpolations. An all-intro
        # lead (e.g. an Al-Fatiha take that starts on ٱلۡحَمۡدُ with 1:1 unrecited)
        # keeps the conservative skip so absent words are never given fake
        # timestamps. Generic: vocabulary-based, no per-surah/reciter logic.
        intro_norms = _opening_intro_prefix_norms()
        all_intro_vocab = all(
            (str(quran_words[i].get("norm") or "") or normalize_arabic(str(quran_words[i].get("text") or ""))) in intro_norms
            for i in range(lead_end)
        )
        if all_intro_vocab:
            return result, report  # plausible intro-sized lead-in, not dropped recitation
        # Falling through on a budget-sized lead: unlike the >budget case there
        # is no POSITIVE evidence the words exist acoustically before the first
        # real anchor (the recording might genuinely start late). Mark the run
        # so acceptance below is STRICT — adopt only an all-placed,
        # non-degenerate, confidence-cleared alignment; otherwise every word
        # keeps its safe interpolation (never fabricate real-flagged spans).
        small_lead_strict = True
    else:
        small_lead_strict = False

    # The first word with a REAL timestamp bounds the broken opening; align the
    # opening words in the audio BEFORE it.
    first_real_start = None
    for idx in range(lead_end, total):
        w = result[idx]
        if not lacks_real(w) and w.get("start_ms") is not None:
            first_real_start = int(w["start_ms"])
            break
    if first_real_start is None or first_real_start <= 0:
        return result, report

    pad_ms = max(0, int(getattr(args, "opening_ctc_pad_ms", 1500)))
    window_end_ms = first_real_start + pad_ms
    if duration_ms:
        window_end_ms = min(window_end_ms, int(duration_ms))
    window_sec = window_end_ms / 1000.0

    max_dur = int(getattr(args, "underrun_ctc_max_duration_ms", 120000))
    if window_end_ms > max_dur:
        report["skipped"] = "opening_window_exceeds_max"
        print(
            f"Opening CTC recovery: skipped (opening window {window_end_ms}ms > "
            f"max {max_dur}ms); raise --underrun-ctc-max-duration-ms to enable."
        )
        return result, report

    model_name = getattr(args, "ctc_model", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
    min_score = float(getattr(args, "underrun_ctc_min_score", 0.0))
    min_word_ms = int(getattr(args, "underrun_ctc_min_word_ms", 80))
    device = getattr(args, "device_resolved", None) or pick_device(
        getattr(args, "device", "auto")
    )

    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        import soundfile  # noqa: F401
    except Exception as exc:
        report["error"] = f"ctc-deps-missing: {exc}"
        print(
            "Opening CTC recovery: skipped (CTC deps missing: "
            f"{exc}). Install with: pip install -U torchaudio transformers soundfile"
        )
        return result, report

    # v83: the recited intro varies — some recordings have isti'adha + basmala,
    # some basmala only, some neither. Forcing an intro token that is NOT in the
    # audio does not harmlessly "collapse at the start": in a short opening
    # window it drags the whole monotonic alignment and destroys the surah-word
    # scores (109 reciter-C: istiadha=no in the audio, but 5 isti'adha tokens were
    # forced -> every opening word scored below the strict floor -> the genuine
    # dropped opening was rejected as low_confidence). Try the standard intro
    # prefix HYPOTHESES and keep the one whose surah words score best. Generic:
    # fixed liturgical formulas only, no per-surah/reciter logic.
    full_prefix = underrun_intro_prefix_norms(quran_words, chapter_id)
    istiadha_len = 5  # underrun_intro_prefix_norms always begins with isti'adha
    prefix_hypotheses = [list(full_prefix)]
    if len(full_prefix) > istiadha_len:
        prefix_hypotheses.append(list(full_prefix[istiadha_len:]))  # basmala only
    prefix_hypotheses.append([])  # no recited intro at all
    opening_norms = [str(quran_words[i].get("norm") or "") for i in range(lead_end)]

    report["attempted"] = True
    report["window_end_ms"] = window_end_ms
    print(
        f"Opening CTC recovery: {lead_end} leading word(s) lack real timestamps "
        f"(intro budget {intro_budget}) -> forced-align opening window "
        f"0..{window_end_ms}ms ({len(prefix_hypotheses)} intro prefix "
        f"hypothes{'es' if len(prefix_hypotheses) != 1 else 'is'}, "
        f"{lead_end} surah tokens)."
    )

    best = None  # (mean_opening_score, prefix, spans, squeezed)
    align_error = None
    # v84: window-end squeeze evidence. Counted only from a CREDIBLE source:
    # either the hypothesis finally selected as best, or a hypothesis whose
    # intro PREFIX force-aligned with high confidence (positive evidence that
    # prefix matches the actual recited intro, so its opening placement is
    # trustworthy even if it loses the mean-score selection). A random losing
    # hypothesis touching the boundary is NOT evidence.
    credible_squeeze = False
    boundary_tol_ms = int(getattr(args, "opening_ctc_boundary_tol_ms", 200))
    credible_prefix_floor = float(
        getattr(args, "opening_ctc_credible_prefix_score", 0.5)
    )
    for prefix_try in prefix_hypotheses:
        transcript = list(prefix_try) + opening_norms
        try:
            spans_try = ctc_align_window(
                wav_path,
                transcript,
                device,
                model_name,
                window_start_ms=0,
                window_sec=window_sec,
            )
        except Exception as exc:
            align_error = str(exc)
            continue
        opening_try = spans_try[len(prefix_try):]
        if len(opening_try) != lead_end:
            align_error = "span-count-mismatch"
            continue
        scores = [
            float(s.get("score") or 0.0)
            if s and s.get("start_ms") is not None and s.get("end_ms") is not None
            else 0.0
            for s in opening_try
        ]
        placed_ends = [
            int(s["end_ms"]) for s in opening_try
            if s and s.get("end_ms") is not None
        ]
        hyp_squeezed = bool(
            placed_ends and max(placed_ends) >= int(window_end_ms) - boundary_tol_ms
        )
        if hyp_squeezed and prefix_try:
            prefix_try_spans = spans_try[: len(prefix_try)]
            prefix_try_scores = [
                float(s.get("score") or 0.0)
                for s in prefix_try_spans
                if s and s.get("start_ms") is not None and s.get("end_ms") is not None
            ]
            prefix_try_mean = (
                sum(prefix_try_scores) / len(prefix_try_scores)
                if len(prefix_try_scores) == len(prefix_try)
                else 0.0
            )
            if prefix_try_mean >= credible_prefix_floor:
                credible_squeeze = True
        mean_score = sum(scores) / max(1, len(scores))
        # Hypotheses are ordered longest prefix first. A SHORTER prefix only
        # wins by a clear margin: on a near-tie the longer (more conservative)
        # prefix is kept, so the empty hypothesis cannot "steal" real intro
        # audio for the surah words off a marginal score difference.
        margin = float(getattr(args, "opening_ctc_prefix_hypothesis_margin", 0.02))
        if best is None or mean_score > best[0] + margin:
            best = (mean_score, list(prefix_try), spans_try, hyp_squeezed)

    if best is None:
        report["error"] = align_error or "alignment failed"
        print(
            f"Opening CTC recovery: alignment failed ({report['error']}); "
            "keeping prior alignment."
        )
        return result, report

    mean_opening_score, prefix, spans, best_squeezed = best
    boundary_hit = bool(best_squeezed or credible_squeeze)
    report["prefix_len"] = len(prefix)
    report["prefix_mean_opening_score"] = round(mean_opening_score, 4)
    print(
        f"Opening CTC recovery: best intro hypothesis = {len(prefix)} intro "
        f"token(s) (mean opening score {mean_opening_score:.3f})."
    )

    prefix_spans = spans[: len(prefix)]
    opening_spans = spans[len(prefix):]

    intro_end_candidates = [
        int(s["end_ms"]) for s in prefix_spans
        if s and s.get("end_ms") is not None
    ]
    intro_end_ms = max(intro_end_candidates) if intro_end_candidates else None

    if small_lead_strict:
        # v81 strict acceptance for the budget-sized non-intro-vocab lead: the
        # words are only PRESUMED dropped recitation — there is no positive
        # evidence they exist in the audio before the first real anchor. Adopt
        # the alignment only when it is unambiguous:
        #   1. all-placed — every leading word lacking a real timestamp got a
        #      usable span (v67 adopt-if-all-placed pattern);
        #   2. confidence — every span clears a small positive score floor
        #      (a truly absent word aligns near zero);
        #   3. non-degenerate — spans are not squeezed to minimum width or
        #      packed at an impossible recitation rate.
        # Otherwise every word keeps its safe interpolation and the v64
        # opening forgiveness handles quality exactly as before.
        strict_floor = float(getattr(args, "opening_ctc_small_lead_min_score", 0.05))
        cand = []
        reject = None
        for idx in range(lead_end):
            if not lacks_real(result[idx]):
                continue
            span = opening_spans[idx]
            if not span or span.get("start_ms") is None or span.get("end_ms") is None:
                reject = "unplaced_word"
                break
            sc = span.get("score")
            if sc is None or float(sc) < strict_floor:
                reject = "low_confidence"
                break
            s0 = int(span["start_ms"])
            e0 = min(int(span["end_ms"]), first_real_start)
            cand.append((s0, e0))
        if reject is None and len(cand) >= 3:
            floor_like = [1 for (s0, e0) in cand if (e0 - s0) <= min_word_ms]
            if 2 * len(floor_like) > len(cand):
                reject = "width_collapse"
            else:
                extent_ms = max(e for _, e in cand) - min(s for s, _ in cand)
                wps = len(cand) / (extent_ms / 1000.0) if extent_ms > 0 else float("inf")
                if wps > float(getattr(args, "underrun_ctc_degenerate_max_wps", 7.0)):
                    reject = "rate_ceiling"
        if reject is not None:
            report["small_lead_rejected"] = reject
            # v84: when the failed alignment was SQUEEZED against the window end,
            # the rejection is evidence the window ITSELF is too short — i.e. the
            # first "real" anchor bounding it is mis-timed too early (Whisper
            # timing compression, 109 reciter-C: anchor ما @6120ms vs ~12900ms
            # actual). Flag it so the caller can escalate to the full-surah CTC
            # pass, whose misanchored-interior guard can also move the bogus
            # anchor. Purely positional evidence; generic.
            if boundary_hit:
                report["anchor_suspect"] = True
            print(
                f"Opening CTC recovery: strict small-lead acceptance failed ({reject}); "
                + ("window-end squeeze detected (first-real anchor suspect); "
                   if boundary_hit else "")
                + "keeping safe interpolations (opening words may not be present in the audio)."
            )
            return result, report

    recovered = 0
    for idx in range(lead_end):
        span = opening_spans[idx]
        if not lacks_real(result[idx]):
            continue
        if not span or span.get("start_ms") is None or span.get("end_ms") is None:
            continue
        score = span.get("score")
        if score is not None and float(score) < min_score:
            continue
        start = int(span["start_ms"])
        end = int(span["end_ms"])
        if end < start + min_word_ms:
            end = start + min_word_ms
        # Never let a recovered opening word cross the first real anchor.
        end = min(end, first_real_start)
        start = min(start, max(0, end - min_word_ms))
        if end <= start:
            continue
        result[idx]["start_ms"] = start
        result[idx]["end_ms"] = end
        result[idx]["estimated"] = False
        result[idx]["score"] = float(score) if score is not None else 0.0
        result[idx]["ctc_opening_recovered"] = True
        for stale in ("opening_region_estimated", "intro_gap_estimated"):
            if stale in result[idx]:
                result[idx].pop(stale, None)
        recovered += 1

    # Monotonic clamp within the opening block (forward then backward), moving
    # ONLY recovered words and bounding them by the first real anchor.
    if recovered:
        last_end = None
        for idx in range(lead_end + 1):
            w = result[idx]
            if w.get("start_ms") is None or w.get("end_ms") is None:
                continue
            if w.get("ctc_opening_recovered"):
                if last_end is not None and w["start_ms"] < last_end:
                    w["start_ms"] = last_end
                if w["end_ms"] < w["start_ms"] + min_word_ms:
                    w["end_ms"] = w["start_ms"] + min_word_ms
                w["end_ms"] = min(w["end_ms"], first_real_start)
                w["start_ms"] = min(w["start_ms"], max(0, w["end_ms"] - min_word_ms))
            last_end = w["end_ms"] if last_end is None else max(last_end, w["end_ms"])

        next_start = None
        for idx in range(lead_end, -1, -1):
            w = result[idx]
            if w.get("start_ms") is None or w.get("end_ms") is None:
                continue
            if w.get("ctc_opening_recovered"):
                if next_start is not None and w["end_ms"] > next_start:
                    w["end_ms"] = next_start
                if w["end_ms"] < w["start_ms"] + min_word_ms:
                    w["start_ms"] = max(0, w["end_ms"] - min_word_ms)
            next_start = w["start_ms"] if next_start is None else min(next_start, w["start_ms"])

    recovered_starts = [
        w["start_ms"] for w in result
        if w.get("ctc_opening_recovered") and w.get("start_ms") is not None
    ]
    earliest_recovered = min(recovered_starts) if recovered_starts else None
    if intro_end_ms is None:
        intro_end_ms = earliest_recovered
    elif earliest_recovered is not None:
        intro_end_ms = min(intro_end_ms, earliest_recovered)
    report["intro_end_ms"] = intro_end_ms

    report["recovered"] = recovered
    print(f"Opening CTC recovery: recovered {recovered}/{lead_end} real word span(s).")
    return result, report


def enforce_opening_real_timestamps(
    mapped_words,
    asr_words,
    opening_gap_recover_report,
    quran_words=None,
    chapter_id=None,
    min_score=0.6,
    max_scan_words=25,
    min_word_ms=80,
):
    """v47: guarantee REAL Whisper timestamps for the surah-opening words.

    recover_dropped_opening_asr already splices REAL opening ASR words into the
    stream, but the downstream opening-safe leading repair can STILL statically
    redistribute the opening island when the first stable 5-word anchor lands a
    few words in. Example (An-Naml / 027): طسٓ is folded and recovered, yet the
    first stable run only begins at ٱلۡقُرۡءَانِ (index 3), so the leading repair
    flags طسٓ / تِلۡكَ / ءَايَٰتُ as ``opening_region_estimated`` and spreads them
    evenly from 0ms. That directly violates the hard rule that the muqatta'at
    opening AND the opening words must carry REAL Whisper timestamps.

    This pass fixes exactly that, WITHOUT re-transcribing: it takes the leading
    contiguous ESTIMATED island ``[0..k)`` and maps those Quran words onto the
    REAL recovered ASR words that already sit in the recovery window
    ``[gap_start .. first real anchor)``, assigning their measured Whisper
    timestamps and clearing the estimated flags. It runs ONLY when opening gap
    recovery actually recovered words, is monotonic, never overwrites a word that
    already has a real timestamp, and leaves any opening word it cannot confidently
    map untouched (still estimated) so nothing is silently fabricated. Generic by
    construction: no hardcoded surah, reciter, or word list.
    """
    report = {"attempted": False, "enforced": 0, "words": [], "reason": None}
    rec = opening_gap_recover_report if isinstance(opening_gap_recover_report, dict) else {}
    if not rec.get("attempted") or not rec.get("recovered"):
        report["reason"] = "no_opening_recovery"
        return [dict(w) for w in mapped_words], report

    result = [dict(w) for w in mapped_words]
    n = len(result)
    if n == 0:
        report["reason"] = "empty"
        return result, report

    # Leading contiguous estimated island starting at index 0 (exactly what the
    # opening-safe leading repair would have statically filled).
    island = []
    for i in range(0, min(n, int(max_scan_words or 25))):
        if bool(result[i].get("estimated")):
            island.append(i)
        else:
            break
    if not island:
        report["reason"] = "no_leading_estimated_island"
        return result, report

    anchor_idx = island[-1] + 1
    if anchor_idx >= n:
        report["reason"] = "no_real_anchor_after_island"
        return result, report
    anchor = result[anchor_idx]
    if anchor.get("start_ms") is None:
        report["reason"] = "anchor_not_timed"
        return result, report
    anchor_start = int(anchor["start_ms"])

    gap_start = int(rec.get("gap_start_ms") or 0)

    # Candidate REAL ASR words whose centre sits in [gap_start, anchor_start).
    # These are the recovered opening words (already onset-snapped + folded).
    cand = []
    for a in (asr_words or []):
        try:
            s = int(a.get("start_ms"))
            e = int(a.get("end_ms"))
        except Exception:
            continue
        mid = (s + e) // 2
        if mid < gap_start or mid >= anchor_start:
            continue
        cand.append({
            "word": a.get("word") or a.get("text") or "",
            "norm": a.get("norm") or normalize_arabic(a.get("word") or a.get("text") or ""),
            "start_ms": s,
            "end_ms": e,
        })
    cand.sort(key=lambda x: (x["start_ms"], x["end_ms"]))
    if not cand:
        report["reason"] = "no_real_asr_in_opening_window"
        return result, report

    report["attempted"] = True

    q_region = [
        {"norm": result[k].get("norm") or normalize_arabic(result[k].get("text") or "")}
        for k in island
    ]
    pairs = map_region_fuzzy(q_region, cand, min_score)

    enforced = 0
    last_end = gap_start
    for qi, idx in enumerate(island):
        ai = pairs.get(qi)
        if ai is None:
            continue
        aw = cand[ai]
        s = max(int(aw["start_ms"]), last_end)
        # Clamp strictly inside [.., anchor_start) so a long recovered token that
        # bleeds across the first real anchor can never overlap it (keeps global
        # order monotonic, mirroring recover_estimated_words_asr).
        e = min(int(aw["end_ms"]), anchor_start)
        if e - s < min_word_ms:
            e = min(anchor_start, s + min_word_ms)
        if e <= s or s >= anchor_start:
            continue
        score = round(float(similarity(q_region[qi]["norm"], aw["norm"])), 3)
        old_s = result[idx].get("start_ms")
        old_e = result[idx].get("end_ms")
        result[idx]["start_ms"] = int(s)
        result[idx]["end_ms"] = int(e)
        result[idx]["estimated"] = False
        # Clear the opening/intro estimate flags so the word is no longer counted
        # as a statically-estimated opening word by opening_estimate_summary.
        result[idx]["opening_region_estimated"] = False
        result[idx]["intro_gap_estimated"] = False
        result[idx]["opening_real_recovered"] = True
        result[idx]["score"] = max(float(result[idx].get("score") or 0), score)
        result[idx]["method"] = str(result[idx].get("method", "")) + "+opening_real_enforce"
        report["words"].append({
            "index": idx,
            "verse_key": result[idx].get("verse_key"),
            "position": result[idx].get("position"),
            "text": result[idx].get("text"),
            "norm": q_region[qi]["norm"],
            "asr_word": aw["word"],
            "old_start_ms": old_s, "old_end_ms": old_e,
            "new_start_ms": int(s), "new_end_ms": int(e),
            "score": score,
        })
        last_end = e
        enforced += 1

    report["enforced"] = enforced
    report["island_size"] = len(island)
    report["gap_start_ms"] = gap_start
    report["anchor_start_ms"] = anchor_start
    if enforced:
        first = report["words"][0]
        last = report["words"][-1]
        print(
            f"Opening real-timestamp enforcement: assigned REAL Whisper timestamps to "
            f"{enforced}/{len(island)} opening word(s) from recovered ASR "
            f"(first: [{first['text']}] @ {first['new_start_ms']}ms, "
            f"last: [{last['text']}] @ {last['new_end_ms']}ms)."
        )
    else:
        print("Opening real-timestamp enforcement: no opening estimate could be mapped to recovered ASR (left unchanged).")
    return result, report


def verify_opening_intro(mapped_words, stripped_report, quran_words=None, chapter_id=None, overlap_tolerance_ms=200, intro_audio_end_override=None):
    """v49: verify the leading isti'adha + basmala at the surah opening.

    The audio almost always opens with a recitation intro that is NOT part of the
    surah word list: the isti'adha (أعوذ بالله من الشيطان الرجيم) and/or the basmala
    (بسم الله الرحمن الرحيم — except At-Tawbah, and except Al-Fatihah where the
    basmala is itself the first ayah). ``strip_asr_recitation_extras`` removes those
    intro tokens from the ASR stream before alignment so the first REAL surah word
    (often a muqatta'at letter) anchors to the audio AFTER the intro.

    This pass NEVER mutates timings. It confirms and reports, generically (no
    hardcoded surah/reciter):
      - which intro phrases were detected and their measured Whisper audio span,
      - whether the basmala is expected for this surah (skipped for 9 / Al-Fatihah),
      - that the first real (non-estimated) surah word starts AFTER the intro span,
        i.e. that no surah opening word was wrongly aligned onto basmala/isti'adha
        audio.
    Returns ``(report, problems)`` where ``problems`` is a list of human-readable
    issues fed into the quality gate so ``--fail-on-low-quality`` can flag them.
    """
    report = {
        "istiadha_detected": False,
        "basmala_detected": False,
        "basmala_expected": False,
        "intro_blocks": [],
        "intro_audio_end_ms": None,
        "first_real_word": None,
        "overlap_words": [],
    }
    problems = []

    try:
        ch = int(chapter_id) if chapter_id is not None else None
    except Exception:
        ch = None
    # Basmala is recited before every surah except At-Tawbah (9), and for
    # Al-Fatihah it is the first ayah (so it is part of the text, not an intro).
    report["basmala_expected"] = bool(
        ch != 9 and not quran_starts_with_basmala(quran_words or [], chapter_id=ch)
    )

    starts = [x for x in (stripped_report or []) if x.get("side") == "start"]
    intro_end = None
    for blk in starts:
        phrase = blk.get("phrase") or []
        is_bas = phrase_is_basmala(phrase)
        if is_bas:
            report["basmala_detected"] = True
        else:
            report["istiadha_detected"] = True
        e = blk.get("end_ms")
        report["intro_blocks"].append({
            "kind": "basmala" if is_bas else "istiadha",
            "phrase": " ".join(phrase),
            "start_ms": blk.get("start_ms"),
            "end_ms": e,
        })
        try:
            e = int(e)
            intro_end = e if intro_end is None else max(intro_end, e)
        except Exception:
            pass

    report["intro_audio_end_ms"] = intro_end

    # When the full-surah underrun CTC pass recovered the surah, it provides a
    # MEASURED intro boundary (end of the forced-aligned isti'adha/basmala tokens,
    # capped at the first surah word's start). That is authoritative: the
    # stripped-intro end can be wildly wrong in exactly this case, because the
    # severe under-transcription that triggered the fallback also corrupts the
    # strip detection (e.g. a single "intro" block spanning nearly the whole clip).
    if intro_audio_end_override is not None:
        intro_end = int(intro_audio_end_override)
        report["intro_audio_end_ms"] = intro_end
        report["intro_audio_end_source"] = "underrun_ctc"

    # First real (timed, non-estimated) surah word.
    for i, w in enumerate(mapped_words or []):
        if bool(w.get("estimated")):
            continue
        s = w.get("start_ms")
        if s is None:
            continue
        report["first_real_word"] = {
            "index": i,
            "text": w.get("text"),
            "verse_key": w.get("verse_key"),
            "start_ms": int(s),
        }
        break

    # Surah opening words whose REAL timestamp falls well inside the stripped
    # intro audio span => a Quran word was aligned onto basmala/isti'adha audio.
    # A small tolerance absorbs boundary jitter (onset snapping / madd tails)
    # near the intro cutoff so legitimate alignments are not flagged across a
    # full 114-surah x 57-reciter batch.
    if intro_end is not None:
        cutoff = int(intro_end) - max(0, int(overlap_tolerance_ms or 0))
        report["overlap_cutoff_ms"] = cutoff
        for i, w in enumerate(mapped_words or []):
            if i > 40:
                break
            if bool(w.get("estimated")):
                continue
            s = w.get("start_ms")
            if s is None:
                continue
            if int(s) < cutoff:
                report["overlap_words"].append({
                    "index": i,
                    "text": w.get("text"),
                    "verse_key": w.get("verse_key"),
                    "start_ms": int(s),
                })
        if report["overlap_words"]:
            ow = report["overlap_words"][0]
            problems.append(
                f"{len(report['overlap_words'])} surah opening word(s) start before the "
                f"stripped intro ends (~{intro_end}ms, tol {int(overlap_tolerance_ms or 0)}ms) — "
                f"a Quran word may be aligned onto basmala/isti'adha audio "
                f"(first: [{ow['text']}] @ {ow['start_ms']}ms)."
            )

    return report, problems


def recount_matched_after_repairs(mapped_words):
    return sum(
        1 for w in mapped_words
        if w.get("start_ms") is not None
        and w.get("end_ms") is not None
        and not bool(w.get("estimated"))
    )





# ---------------------------------------------------------------------------
# v25 basmala/intro guard and leading-gap repair
# ---------------------------------------------------------------------------

def start_extra_info(stripped_report):
    starts = [x for x in (stripped_report or []) if x.get("side") == "start"]
    if not starts:
        return None

    removed_words = []
    start_ms = None
    end_ms = None

    for item in starts:
        removed_words.extend(item.get("removed_words") or [])
        if item.get("start_ms") is not None:
            start_ms = item.get("start_ms") if start_ms is None else min(start_ms, item.get("start_ms"))
        if item.get("end_ms") is not None:
            end_ms = item.get("end_ms") if end_ms is None else max(end_ms, item.get("end_ms"))

    return {
        "removed_count": len(removed_words),
        "removed_words": removed_words,
        "start_ms": start_ms,
        "end_ms": end_ms,
    }


def choose_intro_verify_report(selected_stripped_report, global_stripped_report):
    """Pick the intro strip report used to verify/exclude the recitation intro.

    The candidate selected for alignment is the un-stripped ``original_asr`` for
    every muqatta'at chapter (so the opening fold/repair runs on the real audio),
    and its strip report therefore carries no leading intro block. The intro was
    still detected globally (``strip_asr_recitation_extras`` on the raw ASR), and
    that same global span already drives the timing exclusion fed into
    ``apply_leading_intro_gap_repair``. Use it for the verification/quality check
    too, so the isti'adha/basmala are recognized and the opening-overlap guard
    runs against the real intro end. Generic across all surahs/reciters; no
    hardcoding, no timestamp mutation.
    """
    def _has_start(rep):
        return any((blk or {}).get("side") == "start" for blk in (rep or []))

    if _has_start(selected_stripped_report):
        return selected_stripped_report
    if _has_start(global_stripped_report):
        return global_stripped_report
    return selected_stripped_report


def detected_intro_end_ms(stripped_report, detected_intro_info=None):
    """End (ms) of the detected leading recitation intro (isti'adha and/or
    basmala), or None when no intro was detected.

    Mirrors how ``verify_opening_intro`` and ``apply_leading_intro_gap_repair``
    derive the intro span, so every downstream stage excludes the SAME audio
    region. Prefers the strip report's leading ``start`` blocks and falls back
    to ``detected_intro_info`` (the globally-detected intro) when the selected
    candidate carried no strip block (e.g. un-stripped original_asr on a
    muqatta'at chapter). Generic: no chapter/reciter hardcoding, no mutation.
    """
    intro_end = None
    for blk in (stripped_report or []):
        if (blk or {}).get("side") != "start":
            continue
        try:
            e = int(blk.get("end_ms"))
        except (TypeError, ValueError):
            continue
        intro_end = e if intro_end is None else max(intro_end, e)
    if intro_end is None and detected_intro_info:
        try:
            intro_end = int(detected_intro_info.get("end_ms"))
        except (TypeError, ValueError):
            intro_end = None
    return intro_end


def count_intro_false_hits(mapped_words, removed_count):
    """
    Detect if original_asr wrongly matched Quran opening words to intro words
    such as bismala/isti'adha.
    """
    if not removed_count:
        return 0

    hits = 0
    for w in mapped_words[:80]:
        if w.get("estimated"):
            continue

        ai = w.get("asr_index")
        try:
            ai = int(ai)
        except Exception:
            continue

        if ai < int(removed_count):
            hits += 1

    return hits




def detect_silence_regions_ms(samples, sr, start_ms, end_ms, min_silence_ms=140):
    import numpy as _np

    start_ms = int(max(0, start_ms))
    end_ms = int(max(start_ms, end_ms))

    if end_ms - start_ms < min_silence_ms:
        return []

    s = int(sr * start_ms / 1000)
    e = int(sr * end_ms / 1000)
    region = samples[s:e]

    if len(region) < int(sr * min_silence_ms / 1000):
        return []

    frame_ms = 25
    hop_ms = 10
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))

    rms = []
    starts = []

    for i in range(0, max(1, len(region) - frame + 1), hop):
        chunk = region[i:i + frame]
        if len(chunk) == 0:
            continue
        rms.append(float(_np.sqrt(_np.mean(chunk * chunk))))
        starts.append(i)

    if not rms:
        return []

    rms = _np.array(rms, dtype="float32")

    p10 = float(_np.percentile(rms, 10))
    p20 = float(_np.percentile(rms, 20))
    p35 = float(_np.percentile(rms, 35))
    med = float(_np.percentile(rms, 50))

    threshold = max(10.0, min(p20 * 1.20, p35 * 0.75, med * 0.32), p10 * 1.60)
    low = rms <= threshold
    need = max(1, int(round(min_silence_ms / hop_ms)))

    regions = []
    run_start = None

    for idx, is_low in enumerate(low):
        if is_low and run_start is None:
            run_start = idx
        elif (not is_low) and run_start is not None:
            if idx - run_start >= need:
                st = int(start_ms + starts[run_start] * 1000 / sr)
                en = int(start_ms + (starts[idx - 1] + frame) * 1000 / sr)
                regions.append({
                    "start_ms": st,
                    "end_ms": en,
                    "mid_ms": int((st + en) / 2),
                    "duration_ms": int(en - st),
                    "threshold": round(float(threshold), 3),
                })
            run_start = None

    if run_start is not None and len(low) - run_start >= need:
        st = int(start_ms + starts[run_start] * 1000 / sr)
        en = int(start_ms + (starts[-1] + frame) * 1000 / sr)
        regions.append({
            "start_ms": st,
            "end_ms": en,
            "mid_ms": int((st + en) / 2),
            "duration_ms": int(en - st),
            "threshold": round(float(threshold), 3),
        })

    merged = []
    for r in regions:
        if not merged or r["start_ms"] - merged[-1]["end_ms"] > 120:
            merged.append(dict(r))
        else:
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], r["end_ms"])
            merged[-1]["mid_ms"] = int((merged[-1]["start_ms"] + merged[-1]["end_ms"]) / 2)
            merged[-1]["duration_ms"] = merged[-1]["end_ms"] - merged[-1]["start_ms"]

    return merged


def grouped_leading_indices_by_verse(result, leading_indices):
    groups = []
    by_key = {}

    for idx in leading_indices:
        key = result[idx].get("verse_key")
        if key not in by_key:
            by_key[key] = {
                "verse_key": key,
                "indices": [],
            }
            groups.append(by_key[key])
        by_key[key]["indices"].append(idx)

    return groups


def assign_words_to_span_weighted(result, indices, start_ms, end_ms, min_word_ms=90, method_suffix="+intro_gap_after_basmala"):
    if not indices:
        return []

    start_ms = int(start_ms)
    end_ms = int(max(start_ms + len(indices) * min_word_ms, end_ms))

    weights = []
    for idx in indices:
        norm = result[idx].get("norm") or normalize_arabic(result[idx].get("text", ""))
        weights.append(max(1, len(norm)))

    total_weight = max(1, sum(weights))
    cursor = start_ms
    report = []

    for pos, idx in enumerate(indices):
        old_start = result[idx].get("start_ms")
        old_end = result[idx].get("end_ms")

        if pos == len(indices) - 1:
            e = end_ms
        else:
            e = start_ms + round((end_ms - start_ms) * sum(weights[:pos + 1]) / total_weight)

        if e - cursor < min_word_ms and pos != len(indices) - 1:
            e = min(end_ms, cursor + min_word_ms)

        result[idx]["start_ms"] = int(cursor)
        result[idx]["end_ms"] = int(e)
        result[idx]["estimated"] = True
        result[idx]["intro_gap_estimated"] = True
        result[idx]["method"] = str(result[idx].get("method", "")) + method_suffix

        report.append({
            "index": idx,
            "verse_key": result[idx].get("verse_key"),
            "position": result[idx].get("position"),
            "text": result[idx].get("text"),
            "old_start_ms": old_start,
            "old_end_ms": old_end,
            "new_start_ms": int(cursor),
            "new_end_ms": int(e),
        })

        cursor = e

    return report


def choose_silence_boundaries_for_verse_groups(groups, gap_start, gap_end, silence_regions):
    need = max(0, len(groups) - 1)
    if need <= 0 or not silence_regions:
        return [], []

    total_weight = 0
    group_weights = []

    for group in groups:
        w = sum(int(x) for x in group.get("weights_by_index", {}).values())
        group_weights.append(max(1, w))
        total_weight += max(1, w)

    desired = []
    accum = 0
    for w in group_weights[:-1]:
        accum += w
        desired.append(int(gap_start + (gap_end - gap_start) * accum / max(1, total_weight)))

    margin = 350
    candidates = [
        r for r in silence_regions
        if r["mid_ms"] > gap_start + margin and r["mid_ms"] < gap_end - margin
    ]

    chosen = []
    used = set()

    for target in desired:
        best_idx = None
        best_score = None

        for ci, r in enumerate(candidates):
            if ci in used:
                continue

            mid = r["mid_ms"]
            if chosen and mid <= chosen[-1]["mid_ms"] + 500:
                continue

            dist = abs(mid - target)
            dur_bonus = min(500, int(r.get("duration_ms", 0)))
            score = dist - dur_bonus * 0.35

            if best_score is None or score < best_score:
                best_score = score
                best_idx = ci

        if best_idx is not None:
            used.add(best_idx)
            chosen.append(candidates[best_idx])

    chosen = sorted(chosen, key=lambda r: r["mid_ms"])
    return [r["mid_ms"] for r in chosen], chosen





def opening_first_reliable_index(mapped_words, intro_end=0, min_score=0.88):
    """
    First safe anchor after the introduction/opening confusion.
    A word is reliable when it is not estimated, has timing after intro_end,
    and has a good score. This avoids anchoring on weak matches to basmala/huruf.
    """
    for i, w in enumerate(mapped_words):
        if bool(w.get("estimated")):
            continue
        if w.get("start_ms") is None or w.get("end_ms") is None:
            continue
        try:
            st = int(w.get("start_ms"))
        except Exception:
            continue
        if st < int(intro_end):
            continue
        try:
            score = float(w.get("score", 0))
        except Exception:
            score = 0.0
        if score >= min_score:
            return i

    # Fallback: any non-estimated timed word after intro_end.
    for i, w in enumerate(mapped_words):
        if not bool(w.get("estimated")) and w.get("start_ms") is not None and w.get("end_ms") is not None:
            try:
                if int(w.get("start_ms")) >= int(intro_end):
                    return i
            except Exception:
                pass

    return None



def is_reliable_timed_word(w, intro_end=0, min_score=0.88):
    if bool(w.get("estimated")):
        return False
    if w.get("start_ms") is None or w.get("end_ms") is None:
        return False
    try:
        st = int(w.get("start_ms"))
    except Exception:
        return False
    if st < int(intro_end):
        return False
    try:
        score = float(w.get("score", 0))
    except Exception:
        score = 0.0
    return score >= min_score


def opening_stable_anchor_index(
    mapped_words,
    intro_end=0,
    min_score=0.88,
    consecutive=5,
    max_scan_words=80,
):
    """
    v31: Do not close the opening region at the first reliable word.
    Close it at the first stable run of N reliable words.
    This handles basmala + muqatta'at openings where one early word aligns
    but the next few words are still part of the opening confusion.
    """
    consecutive = max(1, int(consecutive or 1))
    max_scan_words = max(consecutive, int(max_scan_words or 80))
    scan_end = min(len(mapped_words), max_scan_words)

    for i in range(0, scan_end):
        if i + consecutive > len(mapped_words):
            break
        ok = True
        for j in range(i, min(len(mapped_words), i + consecutive)):
            if not is_reliable_timed_word(mapped_words[j], intro_end=intro_end, min_score=min_score):
                ok = False
                break
        if ok:
            return i

    return opening_first_reliable_index(mapped_words, intro_end=intro_end, min_score=min_score)



def opening_absorb_tail_anchor_index(
    mapped_words,
    base_anchor_idx,
    intro_end=0,
    min_score=0.88,
    consecutive=5,
    max_scan_words=80,
):
    """
    v40 opening island:
    If the first stable anchor closes the opening too early, a few estimated
    words may remain immediately after it and get counted as main-region errors.
    For opening-safe chapters, absorb those early estimated words into the
    opening island and close the island only at the next stable reliable run.

    This is intentionally limited to the beginning scan window so it does not
    hide genuine main-surah ASR misses.
    """
    info = {
        "changed": False,
        "base_anchor_index": base_anchor_idx,
        "new_anchor_index": base_anchor_idx,
        "absorbed_estimated_indices": [],
        "last_absorbed_estimated_index": None,
        "max_scan_words": int(max_scan_words or 80),
    }

    if base_anchor_idx is None:
        return base_anchor_idx, info

    try:
        base_anchor_idx = int(base_anchor_idx)
    except Exception:
        return base_anchor_idx, info

    if base_anchor_idx < 0 or base_anchor_idx >= len(mapped_words):
        return base_anchor_idx, info

    consecutive = max(1, int(consecutive or 1))
    scan_end = min(len(mapped_words), max(base_anchor_idx + 1, int(max_scan_words or 80)))

    early_estimated = []
    for i in range(base_anchor_idx, scan_end):
        if bool(mapped_words[i].get("estimated")):
            early_estimated.append(i)

    if not early_estimated:
        return base_anchor_idx, info

    last_est = max(early_estimated)
    search_start = min(len(mapped_words) - 1, max(base_anchor_idx + 1, last_est + 1))

    # Prefer the next stable reliable run after the last early estimate.
    new_anchor = None
    for i in range(search_start, scan_end):
        if i + consecutive > len(mapped_words):
            break
        ok = True
        for j in range(i, min(len(mapped_words), i + consecutive)):
            if not is_reliable_timed_word(mapped_words[j], intro_end=intro_end, min_score=min_score):
                ok = False
                break
        if ok:
            new_anchor = i
            break

    # Fallback: first reliable word after the last early estimate.
    if new_anchor is None:
        for i in range(search_start, scan_end):
            if is_reliable_timed_word(mapped_words[i], intro_end=intro_end, min_score=min_score):
                new_anchor = i
                break

    if new_anchor is None or new_anchor <= base_anchor_idx:
        return base_anchor_idx, info

    info.update({
        "changed": True,
        "new_anchor_index": int(new_anchor),
        "absorbed_estimated_indices": early_estimated[:50],
        "last_absorbed_estimated_index": int(last_est),
    })
    return int(new_anchor), info


def opening_has_leading_problem(mapped_words, quran_words=None, chapter_id=None, max_scan_words=40):
    """
    Detect opening confusion:
    - chapter with huruf muqatta'at
    - or leading estimated words
    - or weak early scores
    """
    scan = mapped_words[:max_scan_words]
    if not scan:
        return False

    if quran_opening_has_muqattaat(quran_words or [], chapter_id=chapter_id):
        return True

    leading_estimated = 0
    for w in scan:
        if bool(w.get("estimated")):
            leading_estimated += 1
        else:
            break

    if leading_estimated >= 2:
        return True

    weak = 0
    for w in scan[:10]:
        try:
            score = float(w.get("score", 0))
        except Exception:
            score = 0.0
        if score < 0.75:
            weak += 1

    return weak >= 3


_OPENING_INTRO_PREFIX_NORMS_CACHE = None


def _opening_intro_prefix_norms():
    # Normalized vocabulary of the only tokens that may legitimately precede the
    # first recited ayah: the isti'adha, the basmala, and every huruf muqatta'at
    # form. Used to gate the v71 leading-estimated-run classification so that a
    # genuinely dropped first ayah (arbitrary Quran text) is never mistaken for
    # an unrecited opening. Chapter-agnostic on purpose: a non-muqatta'at surah
    # cannot legitimately open with a muqatta'at token, so the global set is safe
    # and needs no chapter_id.
    global _OPENING_INTRO_PREFIX_NORMS_CACHE
    if _OPENING_INTRO_PREFIX_NORMS_CACHE is not None:
        return _OPENING_INTRO_PREFIX_NORMS_CACHE
    tokens = list(BASMALA_PHRASE)
    tokens += [
        "اعوذ", "أعوذ", "بالله", "من", "الشيطان", "الشيطن", "الرجيم",
        "السميع", "العليم", "العظيم", "همزه", "ونفخه", "ونفثه",
    ]
    tokens += [
        "الم", "المص", "الر", "المر", "كهيعص", "طه", "طسم", "طس",
        "يس", "ص", "حم", "عسق", "ق", "ن",
    ]
    norms = set()
    for t in tokens:
        n = normalize_arabic(t)
        if n:
            norms.add(n)
    _OPENING_INTRO_PREFIX_NORMS_CACHE = norms
    return norms


def opening_estimate_summary(mapped_words):
    estimated = [i for i, w in enumerate(mapped_words) if bool(w.get("estimated"))]
    flagged = [i for i, w in enumerate(mapped_words) if bool(w.get("opening_region_estimated") or w.get("intro_gap_estimated"))]
    # v70: a contiguous run of estimated words at the very start of the surah IS
    # the opening (unrecited isti'adha / basmala / huruf muqatta'at that never
    # entered the audio), even when no intro-gap repair flagged it. Example: an
    # Al-Fatiha recording that begins directly on ٱلۡحَمۡدُ with the basmala (1:1)
    # absent -- the first real word then sits at ~0ms so apply_leading_intro_gap_
    # repair bails (no audio span before it to redistribute into) and the leading
    # basmala estimates carry no opening_region flag. Without this they were
    # mis-filed as estimated_after_opening and never reached the v64 intro-budget
    # forgiveness, so a fully-aligned surah reported a fake sub-100% match_rate.
    #
    # v71: the leading run is gated by VOCABULARY, not position alone. Only
    # estimated words whose normalized form is an expected opening token -- the
    # isti'adha, the basmala, or the surah's huruf muqatta'at -- may join the
    # leading run. This closes the false-forgiveness hole in the purely
    # positional v70 rule: a reciter who genuinely DROPS the first (non-intro)
    # ayah would otherwise have that short drop (<= the intro budget of
    # isti'adha 5 + buffer 4) silently forgiven as if it were the opening. A
    # dropped ayah word is arbitrary Quran text, not an intro token, so it breaks
    # the run and is correctly counted against quality (estimated_after_opening).
    # Real-timed opening words (e.g. a measured muqatta'at) are estimated=False
    # and also break the run so their timestamps are preserved, and the v64
    # budget cap still limits how many leading estimates are forgiven.
    intro_prefix_norms = _opening_intro_prefix_norms()
    leading_run = []
    for i, w in enumerate(mapped_words):
        if not bool(w.get("estimated")):
            break
        wn = w.get("norm") or normalize_arabic(w.get("text") or w.get("word") or "")
        if wn in intro_prefix_norms:
            leading_run.append(i)
        else:
            break
    opening_set = set(flagged) | set(leading_run)
    after_opening = [i for i in estimated if i not in opening_set]
    return {
        "estimated_total": len(estimated),
        "opening_estimated": len(opening_set),
        "estimated_after_opening": len(after_opening),
        "estimated_after_opening_indices": after_opening[:50],
    }


def demote_degenerate_real_words(mapped_words):
    # A word flagged real (estimated=False) must carry a physically valid,
    # positive-duration span. A missing bound, or end_ms <= start_ms (zero /
    # negative duration), means the word was never actually placed on the
    # timeline -- typically the monotonic clamp collapsing a full-surah-CTC
    # recovered word against a mis-anchored neighbour (e.g. a catastrophic
    # under-transcription that wrongly matches a surah word to the basmala at
    # 0ms squeezes the true opening words down to 0-0). Such a span is not a
    # real timestamp, so demote it to estimated. That way the opening-safe
    # quality gate counts it against the effective match rate instead of
    # reporting a false 100% for a surah whose opening never actually aligned.
    #
    # This NEVER judges duration LENGTH -- a long madd word keeps its real span
    # untouched. Only an impossible zero/negative-duration (or missing) span is
    # demoted, so the "never flag by duration alone" contract is preserved.
    # Generic; no per-surah or per-reciter logic.
    demoted = []
    for i, w in enumerate(mapped_words):
        if bool(w.get("estimated")):
            continue
        s = w.get("start_ms")
        e = w.get("end_ms")
        invalid = False
        if s is None or e is None:
            invalid = True
        else:
            try:
                invalid = float(e) <= float(s)
            except (TypeError, ValueError):
                invalid = True
        if invalid:
            w["estimated"] = True
            w["degenerate_span_demoted"] = True
            demoted.append(i)
    return mapped_words, {"demoted": len(demoted), "indices": demoted[:50]}



def apply_leading_intro_gap_repair(
    mapped_words,
    stripped_report,
    wav_path=None,
    min_word_ms=90,
    use_audio_pauses=True,
    pause_min_ms=140,
    quran_words=None,
    chapter_id=None,
    opening_safe_mode=True,
    opening_anchor_min_score=0.88,
    opening_stable_anchor_words=5,
    opening_stable_anchor_max_scan_words=80,
    opening_muqattaat_madd_weight=7,
    opening_muqattaat_min_ms=900,
    opening_muqattaat_mode="auto",
    opening_absorb_early_estimates=True,
    opening_absorb_max_scan_words=80,
    detected_intro_info=None,
    keep_opening_real_timed=True,
    deterministic_muqattaat_bracket=True,
):
    info = start_extra_info(stripped_report)
    has_intro = bool(info and info.get("end_ms") is not None)

    # v55: the leading isti'adha/basmala is the only variable lead-in; the first
    # surah word (muqatta'at included) is fixed text, so once an intro is
    # confidently detected its end time is exactly where the surah starts -- that
    # is the anchor we must count from. When the un-stripped original_asr
    # candidate wins selection (always the case for muqatta'at chapters, which
    # stay on the original_asr + fold path), this function would otherwise get an
    # empty strip report, fall back to intro_end=0, and spread the opening words
    # statically from 0ms straight over the recited isti'adha/basmala/الٓر audio.
    # Honor the externally-detected intro boundary so the opening always begins
    # AFTER the intro, generically for every surah.
    if not has_intro and detected_intro_info and detected_intro_info.get("end_ms") is not None:
        info = dict(detected_intro_info)
        has_intro = True

    # Normal intro/basmala/isti'adha path.
    if has_intro:
        intro_end = int(info["end_ms"])
    else:
        # Extra opening-safe path for muqatta'at chapters even if Whisper failed
        # to explicitly transcribe basmala/isti'adha.
        if not opening_safe_mode:
            return [dict(w) for w in mapped_words], []

        if not opening_has_leading_problem(
            mapped_words,
            quran_words=quran_words,
            chapter_id=chapter_id,
        ):
            return [dict(w) for w in mapped_words], []

        intro_end = 0
        info = {
            "removed_count": 0,
            "removed_words": [],
            "start_ms": 0,
            "end_ms": 0,
            "synthetic_opening": True,
            "reason": "muqattaat_or_leading_estimated_without_detected_intro",
        }

    result = [dict(w) for w in mapped_words]

    first_reliable_idx = opening_stable_anchor_index(
        result,
        intro_end=intro_end,
        min_score=opening_anchor_min_score,
        consecutive=opening_stable_anchor_words,
        max_scan_words=opening_stable_anchor_max_scan_words,
    )

    base_first_reliable_idx = first_reliable_idx
    opening_absorb_info = {
        "changed": False,
        "base_anchor_index": first_reliable_idx,
        "new_anchor_index": first_reliable_idx,
        "absorbed_estimated_indices": [],
        "last_absorbed_estimated_index": None,
        "max_scan_words": int(opening_absorb_max_scan_words or 80),
    }

    # v45: never let absorb-tail promote the anchor forward when the opening is
    # already clean from index 0. The first word (e.g. the muqatta'at token حم,
    # now matched with its REAL ASR span thanks to the bare-letter fold) is then
    # reliable at index 0, so the repair must bail and preserve its real
    # timestamps. Promoting the anchor past 0 here would statically redistribute
    # those real-timed opening words -- exactly what must never happen.
    if opening_absorb_early_estimates and base_first_reliable_idx not in (None, 0):
        first_reliable_idx, opening_absorb_info = opening_absorb_tail_anchor_index(
            result,
            base_anchor_idx=first_reliable_idx,
            intro_end=intro_end,
            min_score=opening_anchor_min_score,
            consecutive=opening_stable_anchor_words,
            max_scan_words=opening_absorb_max_scan_words,
        )

    # v56: never statically redistribute opening words that ALREADY carry a
    # trustworthy REAL (Whisper-measured) timestamp. The score-based stable
    # anchor needs a run of high-confidence matches, so when only the very
    # first word is broken (e.g. an unfolded muqatta'at that got a false hit on
    # the basmala) the anchor lands deep in the verse and the old code smeared
    # every real-timed word before it (تلك/ءايات/الكتاب... lose their measured
    # times). Instead, scan the leading island for the first word with a real
    # (non-estimated), post-intro, monotonic timestamp and clamp the repair to
    # it: only the genuinely-unmatched leading words (the muqatta'at) are then
    # redistributed across [intro_end, first_real_word_start] -- a real,
    # audio-bounded span -- while every measured opening word is kept intact.
    # Generic across surahs/reciters; it can only SHRINK the repaired island,
    # never fabricate timings.
    opening_real_timed_clamp = {
        "applied": False,
        "real_anchor_index": None,
        "base_first_reliable_index": first_reliable_idx,
        "deterministic_muqattaat": False,
        "muqattaat_word_count": None,
    }

    # v57 deterministic muqatta'at bracket. For a chapter we KNOW opens with
    # huruf muqatta'at, the muqatta'at words occupy the first M fixed Quran
    # positions (M = number of muqatta'at words in the surah's text, e.g.
    # "الر" -> 1, "حم عسق" -> 2). We do NOT verify the muqatta'at acoustically
    # (Whisper routinely drops/garbles it) -- we trust the surah definition.
    # The opening's first normally-recited word is the first REAL
    # (Whisper-measured) word at index >= M. We bracket the muqatta'at across
    # [intro_end, that word's start] and keep every measured word from there
    # on. This is purely position-based and reciter-agnostic: as soon as the
    # reciter speaks after the intro it is marked as the muqatta'at, and the
    # moment a real next word begins, that is where the muqatta'at ends.
    muq_count = 0
    if deterministic_muqattaat_bracket and is_muqattaat_chapter(chapter_id):
        _muq_txt = chapter_muqattaat_text(chapter_id) or ""
        muq_count = len([p for p in re.split(r"\s+", _muq_txt.strip()) if p])
    opening_real_timed_clamp["muqattaat_word_count"] = muq_count or None

    if keep_opening_real_timed and first_reliable_idx not in (None, 0):
        real_anchor_idx = None
        deterministic = False

        # Deterministic path: search at/after the known muqatta'at words for the
        # first real-timed word. Everything in [M, that word) is unmatched
        # opening audio (the muqatta'at + any dropped words), safe to bracket;
        # we stop AT the first real word so measured words are never smeared.
        if muq_count > 0:
            scan_limit = min(
                len(result),
                muq_count + int(opening_stable_anchor_max_scan_words or 80),
            )
            prev_real_start = intro_end
            for i in range(muq_count, scan_limit):
                w = result[i]
                st = w.get("start_ms")
                if (
                    not w.get("estimated", False)
                    and not w.get("opening_region_estimated", False)
                    and st is not None
                    and int(st) > intro_end + min_word_ms
                    and int(st) >= prev_real_start
                ):
                    real_anchor_idx = i
                    deterministic = True
                    break

        # Fallback (v56): non-muqatta'at chapters, or muqatta'at boundary not
        # found -- scan the leading island for the first real-timed word.
        if real_anchor_idx is None:
            prev_real_start = intro_end
            for i in range(0, first_reliable_idx):
                w = result[i]
                st = w.get("start_ms")
                if (
                    not w.get("estimated", False)
                    and not w.get("opening_region_estimated", False)
                    and st is not None
                    and int(st) > intro_end + min_word_ms
                    and int(st) >= prev_real_start
                ):
                    real_anchor_idx = i
                    break

        if real_anchor_idx is not None and real_anchor_idx != first_reliable_idx:
            opening_real_timed_clamp["applied"] = True
            opening_real_timed_clamp["real_anchor_index"] = real_anchor_idx
            opening_real_timed_clamp["deterministic_muqattaat"] = deterministic
            first_reliable_idx = real_anchor_idx

    if first_reliable_idx is None or first_reliable_idx == 0:
        return result, []

    # Only repair the opening island before the final stable anchor.
    next_start = int(result[first_reliable_idx]["start_ms"])
    if next_start <= intro_end + min_word_ms:
        return result, []

    leading_indices = list(range(0, first_reliable_idx))
    groups = grouped_leading_indices_by_verse(result, leading_indices)

    for group in groups:
        wb = {}
        for idx in group["indices"]:
            norm = result[idx].get("norm") or normalize_arabic(result[idx].get("text", ""))
            base_w = max(1, len(norm))
            is_muq = is_opening_muqattaat_word(
                result[idx].get("text", ""),
                chapter_id=chapter_id,
                word_index=idx,
            )

            # v33: do not assume every reciter reads huruf muqatta'at with
            # long madd. In auto mode, first trust audio silence boundaries.
            # Extra weighting is only forced in "weighted" mode. In "auto",
            # weighting is applied later only if no useful silence boundary exists.
            if is_muq and opening_muqattaat_mode == "weighted":
                wb[idx] = muqattaat_madd_weight(
                    result[idx].get("text", ""),
                    chapter_id=chapter_id,
                    word_index=idx,
                    base_weight=base_w,
                    multiplier=opening_muqattaat_madd_weight,
                )
                result[idx]["opening_muqattaat_madd"] = True
                result[idx]["opening_muqattaat_mode"] = "weighted"
                result[idx]["opening_muqattaat_weight"] = wb[idx]
            else:
                wb[idx] = base_w
                if is_muq:
                    result[idx]["opening_muqattaat_madd"] = False
                    result[idx]["opening_muqattaat_mode"] = opening_muqattaat_mode
                    result[idx]["opening_muqattaat_weight"] = wb[idx]
        group["weights_by_index"] = wb

    silence_regions = []
    selected_silences = []
    boundaries = []

    if wav_path and use_audio_pauses and len(groups) > 1:
        try:
            samples, sr = load_wav_mono_float(wav_path)
            silence_regions = detect_silence_regions_ms(
                samples=samples,
                sr=sr,
                start_ms=intro_end,
                end_ms=next_start,
                min_silence_ms=pause_min_ms,
            )
            boundaries, selected_silences = choose_silence_boundaries_for_verse_groups(
                groups=groups,
                gap_start=intro_end,
                gap_end=next_start,
                silence_regions=silence_regions,
            )
        except Exception as e:
            print(f"WARNING: Opening gap silence repair disabled: {e}")
            silence_regions = []
            selected_silences = []
            boundaries = []

    # v33 adaptive muqatta'at handling:
    # - If audio silences gave enough boundaries, do not force madd duration.
    # - If no useful boundary exists, and mode is auto, give opening muqatta'at
    #   moderate extra weight as a fallback only.
    has_enough_silence_boundaries = len(boundaries) >= len(groups) - 1 and len(groups) > 1

    if (
        opening_muqattaat_mode == "auto"
        and not has_enough_silence_boundaries
        and groups
    ):
        for group in groups[:1]:
            for idx in group.get("indices") or []:
                if is_opening_muqattaat_word(result[idx].get("text", ""), chapter_id=chapter_id, word_index=idx):
                    norm = result[idx].get("norm") or normalize_arabic(result[idx].get("text", ""))
                    base_w = max(1, len(norm))
                    auto_multiplier = max(2, min(4, int(opening_muqattaat_madd_weight or 3)))
                    group["weights_by_index"][idx] = muqattaat_madd_weight(
                        result[idx].get("text", ""),
                        chapter_id=chapter_id,
                        word_index=idx,
                        base_weight=base_w,
                        multiplier=auto_multiplier,
                    )
                    result[idx]["opening_muqattaat_madd"] = True
                    result[idx]["opening_muqattaat_mode"] = "auto_fallback_no_silence_boundary"
                    result[idx]["opening_muqattaat_weight"] = group["weights_by_index"][idx]

    if len(boundaries) < len(groups) - 1:
        total_weight = 0
        weights = []
        for group in groups:
            w = sum(group.get("weights_by_index", {}).values())
            weights.append(max(1, w))
            total_weight += max(1, w)

        fallback = []
        accum = 0
        for w in weights[:-1]:
            accum += w
            fallback.append(intro_end + round((next_start - intro_end) * accum / max(1, total_weight)))

        merged = []
        for i in range(len(groups) - 1):
            if i < len(boundaries):
                merged.append(boundaries[i])
            else:
                merged.append(fallback[i])

        clean = []
        last = intro_end
        for b in merged:
            b = int(max(last + min_word_ms, min(next_start - min_word_ms, b)))
            clean.append(b)
            last = b
        boundaries = clean

    # Ensure opening huruf muqatta'at with madd, especially حم, is not
    # compressed as a short word if no good silence boundary was found.
    if groups and boundaries:
        first_indices = groups[0].get("indices") or []
        first_has_muqattaat = any(
            is_opening_muqattaat_word(result[idx].get("text", ""), chapter_id=chapter_id, word_index=idx)
            for idx in first_indices
        )
        if first_has_muqattaat:
            force_muq_min = (
                opening_muqattaat_mode == "weighted"
                or (
                    opening_muqattaat_mode == "auto"
                    and not has_enough_silence_boundaries
                    and int(opening_muqattaat_min_ms or 0) > 0
                )
            )
            if force_muq_min:
                min_end = intro_end + max(0, int(opening_muqattaat_min_ms or 0))
                max_end = next_start - min_word_ms * max(1, len(groups) - 1)
                if max_end > intro_end:
                    boundaries[0] = int(max(boundaries[0], min(min_end, max_end)))

    spans = []
    cursor = intro_end
    for i, group in enumerate(groups):
        end = boundaries[i] if i < len(boundaries) else next_start
        spans.append((group, int(cursor), int(end)))
        cursor = int(end)

    report = []

    method_name = "opening_safe_silence_repair" if selected_silences else "opening_safe_gap_repair"
    if has_intro:
        method_name += "_after_intro"

    for group, s, e in spans:
        group_report = assign_words_to_span_weighted(
            result,
            group["indices"],
            s,
            e,
            min_word_ms=min_word_ms,
            method_suffix="+" + method_name,
        )

        for item in group_report:
            idx = item["index"]
            result[idx]["opening_region_estimated"] = True
            result[idx]["opening_safe_method"] = method_name

            item.update({
                "intro_end_ms": intro_end,
                "has_detected_intro": has_intro,
                "intro_info": info,
                "chapter_has_muqattaat": quran_opening_has_muqattaat(quran_words or [], chapter_id=chapter_id),
                "muqattaat": chapter_muqattaat_text(chapter_id),
                "first_reliable_index": first_reliable_idx,
                "first_reliable_word": result[first_reliable_idx].get("text"),
                "first_reliable_start_ms": next_start,
                "first_reliable_score": result[first_reliable_idx].get("score"),
                "base_first_reliable_index_before_absorb": base_first_reliable_idx,
                "opening_absorb_early_estimates": bool(opening_absorb_early_estimates),
                "opening_absorb_info": opening_absorb_info,
                "keep_opening_real_timed": bool(keep_opening_real_timed),
                "deterministic_muqattaat_bracket": bool(deterministic_muqattaat_bracket),
                "opening_real_timed_clamp": opening_real_timed_clamp,
                "stable_anchor_words": opening_stable_anchor_words,
                "stable_anchor_max_scan_words": opening_stable_anchor_max_scan_words,
                "opening_muqattaat": describe_opening_muqattaat(quran_words or [], chapter_id=chapter_id),
                "opening_muqattaat_mode": opening_muqattaat_mode,
                "opening_muqattaat_madd_weight": opening_muqattaat_madd_weight,
                "opening_muqattaat_min_ms": opening_muqattaat_min_ms,
                "opening_muqattaat_used_audio_boundaries": bool(has_enough_silence_boundaries),
                "word_is_muqattaat_madd": bool(result[idx].get("opening_muqattaat_madd")),
                "word_muqattaat_weight": result[idx].get("opening_muqattaat_weight"),
                "verse_span_start_ms": s,
                "verse_span_end_ms": e,
                "used_audio_silences": bool(selected_silences),
                "all_silence_regions_count": len(silence_regions),
                "selected_silences": selected_silences,
                "method": method_name,
            })
            report.append(item)

    print(f"Opening-safe leading repairs: {len(report)}")
    if report:
        print(
            f"Opening stable anchor: index={first_reliable_idx}, "
            f"word={result[first_reliable_idx].get('text')}, "
            f"score={result[first_reliable_idx].get('score')}, "
            f"required_run={opening_stable_anchor_words}"
        )
        if opening_absorb_info.get("changed"):
            print(
                "Opening island absorbed early estimates: "
                f"count={len(opening_absorb_info.get('absorbed_estimated_indices') or [])}, "
                f"old_anchor={opening_absorb_info.get('base_anchor_index')}, "
                f"new_anchor={opening_absorb_info.get('new_anchor_index')}, "
                f"last_estimated={opening_absorb_info.get('last_absorbed_estimated_index')}"
            )
    opening_muq = describe_opening_muqattaat(quran_words or [], chapter_id=chapter_id)
    if opening_muq:
        print(
            "Opening muqattaat detected: "
            + ", ".join(x.get("text", "") for x in opening_muq)
            + f" | mode={opening_muqattaat_mode}"
        )

    if selected_silences:
        dprint("Opening selected silence boundaries: " + ", ".join(str(x.get("mid_ms")) for x in selected_silences))

    return result, report


def select_best_alignment_candidate(candidates, has_leading_intro, chapter_has_muqattaat):
    """v54: pick the best alignment candidate.

    When a leading isti'adha/basmala intro was confidently detected at the very
    start AND this is NOT a muqatta'at chapter, that intro is pure non-Quran
    audio that must be excluded. The un-stripped ``original_asr`` candidate can
    still tie on score (the DP simply leaves the first surah words *estimated*
    rather than visibly false-matching the intro tokens, so the intro-false-hit
    penalty never fires). If ``original_asr`` then wins, the opening repair runs
    with no intro boundary and spreads the first surah words statically from 0ms
    straight over the isti'adha/basmala audio -- exactly the "you didn't remove
    the intro" symptom, and it drags the match rate below the quality gate.

    For these chapters we restrict the choice to the stripped candidates so the
    first real surah word anchors on real audio AFTER the intro. Muqatta'at
    chapters are left untouched: their openings rely on the original_asr +
    opening-safe fold path, so they must keep competing normally.
    """
    if not candidates:
        return None

    def _key(c):
        return (c.get("quality_score", c.get("score", 0)), c.get("matched", 0), c.get("match_rate", 0))

    def _coverage_key(c):
        # Rank purely by how many real Quran words a candidate aligns, ignoring
        # quality_score (which the intro penalty can drive arbitrarily negative).
        return (c.get("matched", 0), c.get("match_rate", 0), c.get("quality_score", c.get("score", 0)))

    selectable = candidates
    if has_leading_intro and not chapter_has_muqattaat:
        stripped_only = [c for c in candidates if c.get("asr_set") == "stripped_recitation_extras"]
        if stripped_only:
            # v60 safety: only force the stripped subset when it actually anchors
            # real audio. Under severe under-transcription the stripper can remove
            # almost every ASR word (e.g. a 14-word surah transcribed as 5 words,
            # with 4 stripped as a mis-detected isti'adha/basmala), leaving a
            # DEGENERATE stripped candidate that matches NOTHING. Only force the
            # stripped subset when its best member aligns at least one real word.
            best_stripped = max(stripped_only, key=_key)
            if int(best_stripped.get("matched", 0)) > 0:
                selectable = stripped_only

    best = max(selectable, key=_key)

    # v61: never ship an all-missing alignment when a real one exists. The intro
    # false-match penalty (default 1e6) can push a matched>0 ``original_asr``
    # candidate's quality_score FAR below a degenerate stripped candidate that
    # matched NOTHING (severe under-transcription where the stripper removed
    # nearly every ASR word). In that case quality_score is meaningless, so the
    # plain ``max(..., _key)`` above would still pick the all-missing candidate
    # -> guaranteed FAILED chapter. Fall back to whichever candidate actually
    # aligns the most real Quran words; the full-surah CTC fallback then lifts
    # that partial alignment to real per-word timestamps.
    if int(best.get("matched", 0)) <= 0:
        real = [c for c in candidates if int(c.get("matched", 0)) > 0]
        if real:
            best = max(real, key=_coverage_key)

    return best


def quick_alignment_match_count(quran_words, asr_words, args):
    """v60: cheap single-pass count of how many Quran words an ASR set can place.

    Used only to DECIDE between two transcriptions of the same audio (the normal
    prompted pass vs a no-prompt re-transcription). It runs one DP alignment and
    returns the matched count. It deliberately measures real alignment coverage
    rather than raw ASR word count, so a longer-but-hallucinated transcription
    cannot win: hallucinated tokens do not land on Quran word positions and so do
    not raise the matched count. The threshold mirrors map_quran_to_audio's own
    default (0.55) so spurious low-similarity fuzzy hits do not inflate the count;
    both transcriptions are measured with the SAME threshold, so the comparison
    is a fair relative one.
    """
    if not quran_words or not asr_words:
        return 0
    try:
        _, matched = map_quran_to_audio(
            quran_words=quran_words,
            asr_words=asr_words,
            min_score=0.55,
            lookahead=45,
            alignment="dp",
            complete_output=False,
        )
        return int(matched)
    except Exception:
        return -1


def underrun_recover_should_attempt(matched_count, quran_count, min_coverage):
    """v60/v65: decide whether a second (no-prompt) transcription pass is worth it.

    True only when coverage is poor, so the common, well-transcribed case keeps
    its single-pass cost and can never be regressed by this feature. v65: the
    first argument is the number of Quran words the ASR actually ALIGNS (matched
    count), not the raw ASR token count. Raw token count over-counts intro tokens
    (basmala/isti'adha) and can mask dropped ayah words, so gating on matched
    count is the true coverage signal.
    """
    if quran_count <= 0:
        return False
    return (matched_count / quran_count) < float(min_coverage)


def underrun_recover_should_adopt(primary_match, alt_match):
    """v60: adopt the no-prompt transcription only if it aligns strictly more
    real Quran words than the prompted one.

    The matched count is the complete cheap signal at this pre-repair stage:
    nothing is interpolated yet, so estimated == 0 for both candidates and
    missing == quran_count - matched. "More matched" is therefore identically
    "fewer missing", and a strictly-greater match count is a real coverage win,
    not a noisier transcription.
    """
    return int(alt_match) > int(primary_match)


def align_with_quality_retries(quran_words, asr_words, args):
    """
    Try multiple safe alignment variants and keep the best result.
    This avoids saving a weak one-pass alignment such as 98.09% match when
    another configuration can do better.
    """
    asr_sets = [{
        "name": "original_asr",
        "asr_words": asr_words,
        "stripped": [],
    }]

    global_stripped_report = []
    global_start_extra = None

    if not getattr(args, "no_strip_recitation_extras", False):
        stripped_asr, stripped_report = strip_asr_recitation_extras(asr_words, quran_words=quran_words)
        global_stripped_report = stripped_report or []
        global_start_extra = start_extra_info(global_stripped_report)

        if len(stripped_asr) != len(asr_words):
            asr_sets.append({
                "name": "stripped_recitation_extras",
                "asr_words": stripped_asr,
                "stripped": stripped_report,
            })

    variants = build_alignment_variants(args)
    candidates = []

    print("Alignment quality candidates:")
    for asr_set in asr_sets:
        for variant in variants:
            mapped, matched = map_quran_to_audio(
                quran_words=quran_words,
                asr_words=asr_set["asr_words"],
                min_score=variant["min_score"],
                lookahead=variant["lookahead"],
                alignment=variant["alignment"],
                complete_output=args.complete_output,
            )
            metrics = score_alignment_candidate(quran_words, mapped, matched)

            intro_false_hits = 0
            if (
                asr_set["name"] == "original_asr"
                and global_start_extra
                and getattr(args, "penalize_intro_false_matches", True)
            ):
                intro_false_hits = count_intro_false_hits(mapped, global_start_extra.get("removed_count", 0))

            opening_safe_candidate = (
                not getattr(args, "no_opening_safe_mode", False)
                and not getattr(args, "strict_intro_false_match_penalty", False)
                and quran_opening_has_muqattaat(
                    quran_words,
                    chapter_id=getattr(args, "_current_chapter_id", None),
                )
            )

            # In muqatta'at/opening-safe chapters, the first ASR words may be
            # basmala/isti'adha while Quran starts with حم/طس/الم/etc.
            # v25-v29 penalized original_asr so heavily that it preferred a
            # weaker stripped candidate. But opening-safe repair will replace
            # only the opening region, so false intro hits inside that region
            # should not destroy candidate selection.
            effective_intro_penalty = 0 if opening_safe_candidate else args.intro_false_match_penalty
            quality_score = metrics["score"] - (intro_false_hits * effective_intro_penalty)

            cand = {
                **variant,
                "asr_set": asr_set["name"],
                "asr_words_count": len(asr_set["asr_words"]),
                "asr_words_for_repair": asr_set["asr_words"],
                "stripped": asr_set["stripped"],
                "mapped": mapped,
                "matched": matched,
                "intro_false_hits": intro_false_hits,
                "opening_safe_candidate": bool(opening_safe_candidate),
                "effective_intro_false_match_penalty": effective_intro_penalty,
                "quality_score": quality_score,
                **metrics,
            }
            candidates.append(cand)

            intro_note = f" intro_false_hits={intro_false_hits}" if intro_false_hits else ""
            if opening_safe_candidate and intro_false_hits:
                intro_note += " opening_safe_penalty=0"

            print(
                f"  - {asr_set['name']} | {variant['alignment']} "
                f"min={variant['min_score']} lookahead={variant['lookahead']} "
                f"matched={matched}/{len(quran_words)} rate={metrics['match_rate']:.2%} "
                f"estimated={metrics['estimated']} missing={metrics['missing']}{intro_note}"
            )

    chapter_has_muqattaat = bool(
        quran_opening_has_muqattaat(
            quran_words, chapter_id=getattr(args, "_current_chapter_id", None)
        )
    )
    has_leading_intro = bool(global_start_extra)
    if has_leading_intro and not chapter_has_muqattaat:
        print(
            "Leading intro detected (non-muqattaat): restricting selection to the "
            f"stripped candidate so the isti'adha/basmala "
            f"({global_start_extra.get('removed_count')} word(s)) is excluded from the opening."
        )

    best = select_best_alignment_candidate(
        candidates,
        has_leading_intro=has_leading_intro,
        chapter_has_muqattaat=chapter_has_muqattaat,
    )

    print("Selected alignment candidate:")
    print(
        f"  {best['asr_set']} | {best['alignment']} min={best['min_score']} "
        f"lookahead={best['lookahead']} matched={best['matched']}/{len(quran_words)} "
        f"rate={best['match_rate']:.2%} estimated={best['estimated']} missing={best['missing']}"
    )

    report = []
    for cand in candidates:
        item = {k: v for k, v in cand.items() if k not in ("mapped", "asr_words_for_repair")}
        item["selected"] = cand is best
        report.append(item)

    return best["mapped"], best["matched"], best["asr_words_count"], best["stripped"], report, best["asr_words_for_repair"], global_start_extra, global_stripped_report


def output_quality_status(match_rate, estimated_count, final_missing, args):
    min_rate = float(getattr(args, "quality_min_match", 0.995))
    max_estimated = int(getattr(args, "quality_max_estimated", 0))

    ok = (match_rate >= min_rate) and (estimated_count <= max_estimated) and (final_missing == 0)
    issues = []

    if match_rate < min_rate:
        issues.append(f"match_rate {match_rate:.2%} < {min_rate:.2%}")
    if estimated_count > max_estimated:
        issues.append(f"estimated_timings {estimated_count} > {max_estimated}")
    if final_missing:
        issues.append(f"final_missing_segments {final_missing} > 0")

    return ok, issues


def detect_recitation_anomalies(
    mapped_words,
    asr_words,
    repeat_similarity=0.80,
    phrase_similarity=0.80,
    phrase_min_words=2,
    max_examples=50,
    asr_index_trusted=True,
):
    """
    Verification-only pass. It NEVER changes timestamps and NEVER corrects the
    text. The aligner already gives every Quran word its real position + timing,
    and the leading isti'adha / basmala are detected and stripped elsewhere
    (verify_opening_intro + recitation-extras stripping), so this pass does NOT
    judge whether a word was pronounced "correctly": ASR spelling noise (e.g.
    ٱلْأَرْض transcribed as "لوض") is intentionally ignored — we only take each
    word's position and timing.

    The ONE thing it reports is when the reciter REPEATED something, because a
    repeat is the only event that injects extra audio the timing extractor must
    account for:

    - repeated word: a single EXTRA (unmatched) ASR token that duplicates a
      neighbouring matched word.
    - repeated phrase / sentence: a CONSECUTIVE run of extra (unmatched) ASR
      tokens that duplicates a consecutive run of matched Quran words (the
      reciter went back and re-recited a phrase or a whole ayah).

    Reads only fields the aligner already produced (norm, asr_index, start/end).
    It is fast: a single linear scan over the unmatched ASR tokens (O(n log n)).
    Repeat detection needs the ASR stream that asr_index refers to, so when the
    index space is not trusted it is skipped (see asr_index_trusted).
    """
    mapped_words = mapped_words or []
    asr_words = asr_words or []

    def _norm_of(w):
        if not isinstance(w, dict):
            return ""
        return w.get("norm") or normalize_arabic(w.get("word", "") or w.get("text", "") or "")

    # Quran words matched to REAL audio (not estimated / interpolated), indexed by
    # their ASR position so we can look up what sits before/after a repeat.
    matched = []
    used_indices = set()
    index_to_word = {}
    for i, w in enumerate(mapped_words):
        if not isinstance(w, dict) or bool(w.get("estimated")):
            continue
        ai = w.get("asr_index")
        if ai is None or w.get("start_ms") is None or w.get("end_ms") is None:
            continue
        matched.append((i, w))
        try:
            ai = int(ai)
        except (TypeError, ValueError):
            continue
        used_indices.add(ai)
        index_to_word[ai] = (i, w)
    used_sorted = sorted(used_indices)

    repeated_words = []
    repeated_phrases = []

    def _flag_word(k):
        """A single extra token -> repeated word (nearest matched neighbour)."""
        tnorm = _norm_of(asr_words[k])
        if not tnorm:
            return
        p = bisect.bisect_left(used_sorted, k)
        neighbours = []
        if p < len(used_sorted):
            neighbours.append(used_sorted[p])
        if p - 1 >= 0:
            neighbours.append(used_sorted[p - 1])
        best = None
        for ni in neighbours:
            wi, w = index_to_word[ni]
            sim = similarity(tnorm, _norm_of(w))
            if sim >= repeat_similarity and (best is None or sim > best[0]):
                best = (sim, ni, wi, w)
        if best is not None:
            sim, ni, wi, w = best
            repeated_words.append({
                "word_index": wi,
                "verse_key": w.get("verse_key"),
                "position": w.get("position"),
                "text": w.get("text"),
                "repeated_norm": tnorm,
                "similarity": round(sim, 3),
                "matched_asr_index": ni,
                "matched_span_ms": [w.get("start_ms"), w.get("end_ms")],
                "extra_asr_index": k,
                "extra_span_ms": [asr_words[k].get("start_ms"), asr_words[k].get("end_ms")],
            })

    if asr_index_trusted and asr_words and used_sorted:
        # Group EXTRA (unmatched) ASR tokens into runs of consecutive indices.
        extra_runs = []
        cur = []
        for k, tok in enumerate(asr_words):
            if k not in used_indices and isinstance(tok, dict) and _norm_of(tok):
                if cur and k == cur[-1] + 1:
                    cur.append(k)
                else:
                    if cur:
                        extra_runs.append(cur)
                    cur = [k]
            elif cur:
                extra_runs.append(cur)
                cur = []
        if cur:
            extra_runs.append(cur)

        for run in extra_runs:
            L = len(run)
            run_norms = [_norm_of(asr_words[k]) for k in run]
            a, b = run[0], run[-1]
            pos_a = bisect.bisect_left(used_sorted, a)
            pos_b = bisect.bisect_right(used_sorted, b)

            flagged_phrase = False
            if L >= phrase_min_words:
                # The matched Quran words just before / after the run are the
                # candidate phrase the reciter repeated. One occurrence stays
                # matched; the other is this unmatched run.
                before_idx = used_sorted[max(0, pos_a - L):pos_a]
                after_idx = used_sorted[pos_b:pos_b + L]

                def _seq_sim(cand_idx):
                    if len(cand_idx) != L:
                        return 0.0
                    s = 0.0
                    for tnorm, ci in zip(run_norms, cand_idx):
                        s += similarity(tnorm, _norm_of(index_to_word[ci][1]))
                    return s / L

                before_sim = _seq_sim(before_idx)
                after_sim = _seq_sim(after_idx)
                if before_sim >= after_sim:
                    best_sim, cand, where = before_sim, before_idx, "before"
                else:
                    best_sim, cand, where = after_sim, after_idx, "after"

                if best_sim >= phrase_similarity and len(cand) == L:
                    dup = [index_to_word[ci][1] for ci in cand]
                    repeated_phrases.append({
                        "num_words": L,
                        "repeated_text": " ".join((w.get("text") or "") for w in dup),
                        "repeated_norm": " ".join(run_norms),
                        "similarity": round(best_sim, 3),
                        "duplicated_word_indices": [index_to_word[ci][0] for ci in cand],
                        "verse_keys": sorted({w.get("verse_key") for w in dup if w.get("verse_key")}),
                        "duplicated_span_ms": [dup[0].get("start_ms"), dup[-1].get("end_ms")],
                        "extra_asr_indices": [a, b],
                        "extra_span_ms": [asr_words[a].get("start_ms"), asr_words[b].get("end_ms")],
                        "duplicated_position": where,
                    })
                    flagged_phrase = True

            if not flagged_phrase:
                # Not a phrase repeat: test each token as an isolated word repeat.
                for k in run:
                    _flag_word(k)

    repeated_total = len(repeated_words) + len(repeated_phrases)
    return {
        "params": {
            "repeat_similarity": repeat_similarity,
            "phrase_similarity": phrase_similarity,
            "phrase_min_words": phrase_min_words,
            "repeats_checked": bool(asr_index_trusted),
        },
        "matched_words": len(matched),
        "repeated_word_count": len(repeated_words),
        "repeated_phrase_count": len(repeated_phrases),
        "repeated_count": repeated_total,
        "anomaly_count": repeated_total,
        "repeated_words": repeated_words[:max_examples],
        "repeated_phrases": repeated_phrases[:max_examples],
    }


def _is_valid_existing_timing(out_path):
    """Resume guard: an existing chapter output counts as 'done' only if it parses
    as JSON and carries actual word/segment timing data. Truncated files left by a
    hard interruption (the old non-atomic write path, or a manual kill) are
    rejected so --resume rebuilds them instead of skipping them forever."""
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return False

    if isinstance(data, dict):
        # build_output() always emits a non-empty verse_timings list for a real
        # chapter; treat that as the completion marker.
        vt = data.get("verse_timings")
        if isinstance(vt, list) and len(vt) > 0:
            return True
        for key in ("segments", "words", "ayahs", "verses", "result"):
            val = data.get(key)
            if isinstance(val, (list, dict)) and len(val) > 0:
                return True
        return False
    if isinstance(data, list):
        return len(data) > 0
    return False


def process_chapter(args, chapter_id):
    chapter_padded = pad3(chapter_id)

    if args.audio and not args.all:
        audio_source = args.audio
    else:
        audio_source = args.audio_pattern.replace("{chapter}", chapter_padded)

    out_path = args.out

    if args.all:
        out_path = str(Path(args.out_dir) / f"{chapter_padded}.json")

    if getattr(args, "resume", False) and Path(out_path).exists():
        # Only skip on resume if the existing file is a STRUCTURALLY VALID,
        # non-empty timing JSON. A leftover truncated file from a hard kill would
        # otherwise be skipped forever; here it is treated as not-done and rebuilt.
        if _is_valid_existing_timing(out_path):
            print("")
            print("=" * 72)
            print(f"Chapter: {chapter_id}")
            print(f"SKIPPED existing: {out_path}")
            return {
                "chapter_id": chapter_id,
                "output": out_path,
                "skipped": True,
                "status": "skipped_existing",
            }
        print("")
        print("=" * 72)
        print(f"Chapter: {chapter_id}")
        print(f"RESUME: existing output is invalid/incomplete, rebuilding: {out_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        audio_path = tmpdir / f"{chapter_padded}.mp3"
        wav_path = tmpdir / f"{chapter_padded}.wav"

        print("")
        print("=" * 72)
        print(f"Chapter: {chapter_id}")
        print(f"Audio: {audio_source}")
        print(f"Output: {out_path}")

        print("Preparing audio...")
        copy_or_download_audio(audio_source, audio_path, insecure_ssl=args.insecure_ssl)

        duration_ms = ffprobe_duration_ms(audio_path)

        print("Converting to 16k mono WAV...")
        convert_to_wav(audio_path, wav_path)

        print("Loading local Quran words...")
        quran_words = load_quran_words(args.quran_words, chapter_id)
        print(f"Quran words: {len(quran_words)}")

        initial_prompt = None
        if not args.no_quran_initial_prompt:
            initial_prompt = build_quran_initial_prompt(
                quran_words=quran_words,
                chapter_id=chapter_id,
                prompt_words=args.initial_prompt_words,
                include_basmala=not args.no_initial_prompt_basmala,
                include_istiatha=args.initial_prompt_istiatha,
            )

        # v43: external chunking is OFF by default.
        # v41/v42 proved faster, but it can hurt alignment by cutting Quran context.
        # Use --asr-external-chunking only for explicit experiments.
        use_external_chunking = bool(getattr(args, "asr_external_chunking", False)) and not bool(getattr(args, "no_asr_external_chunking", False))

        if use_external_chunking:
            whisper_result = run_asr_backend_chunked(
                wav_path=wav_path,
                args=args,
                initial_prompt=initial_prompt,
                duration_ms=duration_ms,
                tmpdir=tmpdir,
            )
        else:
            whisper_result = run_asr_backend(
                wav_path=wav_path,
                args=args,
                initial_prompt=initial_prompt,
                duration_ms=duration_ms,
            )

        asr_chunk_report = whisper_result.get("external_chunk_report", []) if isinstance(whisper_result, dict) else []
        asr_words = extract_asr_words(whisper_result)
        print(f"ASR words: {len(asr_words)}")

        # v60: severe under-transcription recovery. Some reciters/surahs make
        # Whisper DROP a large fraction of the words (e.g. a 14-word surah
        # transcribed as 5 words, or a long surah missing a third of its words).
        # The Quran initial_prompt plus the preceding basmala can bias the
        # decoder to skip ahead. When coverage is poor we re-transcribe the WHOLE
        # clip with NO prompt and keep whichever transcription actually ALIGNS
        # more real Quran words (so a longer-but-hallucinated transcript can never
        # win). The second pass only runs on the poor-coverage minority, so the
        # common case keeps its single-pass cost. Generic across surahs/reciters.
        underrun_recover_report = {"attempted": False, "adopted": False}
        if (
            not getattr(args, "no_underrun_recover", False)
            and initial_prompt
            and quran_words
        ):
            # v65: coverage MUST be measured by how many Quran words the ASR can
            # actually PLACE, not by the raw ASR token count. A reciter who recites
            # the basmala/isti'adha adds intro tokens that pad the raw count, so an
            # ASR pass that DROPS real ayah words but transcribes the intro can still
            # reach len(asr) == len(quran) and look "complete" while aligning far
            # fewer words. Real example (An-Nasr / reciter-B): 19 raw tokens for 19
            # Quran words (basmala's 4 tokens padding 4 dropped interior words of
            # ayah 2), so token-coverage = 1.0 and recovery was WRONGLY skipped,
            # even though only 15/19 words actually aligned (78.9%). Gating on the
            # matched count is the true, generic coverage signal.
            primary_match = quick_alignment_match_count(quran_words, asr_words, args)
            if primary_match < 0:
                primary_match = len(asr_words)  # alignment failed: fall back to raw token count
            min_coverage = float(getattr(args, "underrun_recover_min_coverage", 0.85))
            coverage = primary_match / max(1, len(quran_words))
            underrun_recover_report["coverage"] = round(coverage, 4)
            underrun_recover_report["primary_asr_words"] = len(asr_words)
            underrun_recover_report["primary_match"] = primary_match
            underrun_recover_report["quran_words"] = len(quran_words)
            if underrun_recover_should_attempt(primary_match, len(quran_words), min_coverage):
                underrun_recover_report["attempted"] = True
                print(
                    f"Severe under-transcription suspected: ASR aligned only "
                    f"{primary_match}/{len(quran_words)} Quran word(s) "
                    f"(coverage {coverage:.1%} < {min_coverage:.0%}; "
                    f"{len(asr_words)} raw ASR token(s)); re-transcribing the whole "
                    f"clip WITHOUT the Quran initial_prompt to recover the dropped words."
                )
                try:
                    if use_external_chunking:
                        alt_result = run_asr_backend_chunked(
                            wav_path=wav_path,
                            args=args,
                            initial_prompt=None,
                            duration_ms=duration_ms,
                            tmpdir=tmpdir,
                        )
                    else:
                        alt_result = run_asr_backend(
                            wav_path=wav_path,
                            args=args,
                            initial_prompt=None,
                            duration_ms=duration_ms,
                        )
                    alt_words = extract_asr_words(alt_result)
                    alt_match = quick_alignment_match_count(quran_words, alt_words, args)
                    underrun_recover_report["alt_asr_words"] = len(alt_words)
                    underrun_recover_report["alt_match"] = alt_match
                    if underrun_recover_should_adopt(primary_match, alt_match):
                        print(
                            f"  Under-transcription recovery: no-prompt pass aligns "
                            f"{alt_match} Quran word(s) vs {primary_match} (prompted) and "
                            f"transcribed {len(alt_words)} word(s); adopting the "
                            f"no-prompt transcription."
                        )
                        asr_words = alt_words
                        underrun_recover_report["adopted"] = True
                    else:
                        print(
                            f"  Under-transcription recovery: no-prompt pass did not "
                            f"improve alignment ({alt_match} <= {primary_match}); keeping "
                            f"the prompted transcription."
                        )
                except Exception as e:
                    print(
                        f"  Under-transcription recovery failed: {e}; keeping the "
                        f"prompted transcription."
                    )

        opening_gap_recover_report = {}
        if not getattr(args, "no_opening_gap_recover", False):
            asr_words, opening_gap_recover_report = recover_dropped_opening_asr(
                asr_words,
                args=args,
                wav_path=wav_path,
                tmpdir=tmpdir,
                chapter_id=chapter_id,
                duration_ms=duration_ms,
            )
            if opening_gap_recover_report.get("recovered"):
                print(f"ASR words after opening gap recovery: {len(asr_words)}")

        muqattaat_fold_report = []
        if not getattr(args, "no_muqattaat_asr_fold", False):
            asr_words, muqattaat_fold_report = fold_muqattaat_letter_names_in_asr(
                asr_words,
                quran_words=quran_words,
                chapter_id=chapter_id,
                max_scan=getattr(args, "muqattaat_asr_fold_max_scan", 25),
            )
            if muqattaat_fold_report:
                print(f"ASR words after muqattaat fold: {len(asr_words)}")

        alignment_quality_report = []
        stripped_recitation_report = []
        selected_asr_words_count = len(asr_words)
        args._current_chapter_id = chapter_id

        mapped, matched, selected_asr_words_count, stripped_recitation_report, alignment_quality_report, selected_asr_words_for_repair, detected_intro_info, global_intro_report = align_with_quality_retries(
            quran_words=quran_words,
            asr_words=asr_words,
            args=args,
        )

        intro_gap_report = []
        if not args.no_intro_gap_repair:
            mapped, intro_gap_report = apply_leading_intro_gap_repair(
                mapped,
                stripped_recitation_report,
                wav_path=wav_path,
                min_word_ms=args.intro_gap_min_word_ms,
                use_audio_pauses=not args.no_intro_gap_audio_pauses,
                pause_min_ms=args.intro_gap_pause_min_ms,
                quran_words=quran_words,
                chapter_id=chapter_id,
                opening_safe_mode=not args.no_opening_safe_mode,
                opening_anchor_min_score=args.opening_anchor_min_score,
                opening_stable_anchor_words=args.opening_stable_anchor_words,
                opening_stable_anchor_max_scan_words=args.opening_stable_anchor_max_scan_words,
                opening_muqattaat_madd_weight=args.opening_muqattaat_madd_weight,
                opening_muqattaat_min_ms=args.opening_muqattaat_min_ms,
                opening_muqattaat_mode=args.opening_muqattaat_mode,
                opening_absorb_early_estimates=not args.no_opening_absorb_early_estimates,
                opening_absorb_max_scan_words=args.opening_absorb_max_scan_words,
                detected_intro_info=detected_intro_info,
                keep_opening_real_timed=not args.no_opening_keep_real_timed,
                deterministic_muqattaat_bracket=not args.no_deterministic_muqattaat_bracket,
            )

        char_projection_report = []
        if not args.no_char_projection_repair:
            char_projection_intro_end_ms = detected_intro_end_ms(
                choose_intro_verify_report(stripped_recitation_report, global_intro_report),
                detected_intro_info,
            )
            mapped, char_projection_report = repair_estimated_words_by_char_projection(
                mapped_words=mapped,
                asr_words=selected_asr_words_for_repair,
                min_char_ratio=args.char_projection_min_ratio,
                min_chars=args.char_projection_min_chars,
                min_word_ms=args.char_projection_min_word_ms,
                max_projection_shift_ms=args.char_projection_max_shift_ms,
                intro_end_ms=char_projection_intro_end_ms,
            )
            if char_projection_report:
                matched = recount_matched_after_repairs(mapped)

        # v47: after the opening-safe leading repair, the surah-opening island can
        # still be STATICALLY estimated when the stable anchor lands a few words in
        # (e.g. An-Naml طسٓ/تِلۡكَ/ءَايَٰتُ before ٱلۡقُرۡءَانِ). If opening gap
        # recovery already pulled REAL ASR words into that window, assign their
        # measured Whisper timestamps to the opening words instead of static ones.
        opening_real_enforce_report = {"attempted": False, "enforced": 0}
        if not getattr(args, "no_opening_real_timestamp_enforce", False):
            mapped, opening_real_enforce_report = enforce_opening_real_timestamps(
                mapped_words=mapped,
                asr_words=selected_asr_words_for_repair,
                opening_gap_recover_report=opening_gap_recover_report,
                quran_words=quran_words,
                chapter_id=chapter_id,
                min_score=getattr(args, "opening_real_timestamp_enforce_min_score", 0.6),
                max_scan_words=getattr(args, "opening_real_timestamp_enforce_max_scan", 25),
            )
            if opening_real_enforce_report.get("enforced"):
                matched = recount_matched_after_repairs(mapped)

        estimated_word_recover_report = {"attempted": False, "recovered": 0}
        if not getattr(args, "no_estimated_word_recover", False):
            mapped, estimated_word_recover_report = recover_estimated_words_asr(
                mapped_words=mapped,
                args=args,
                wav_path=wav_path,
                tmpdir=tmpdir,
                chapter_id=chapter_id,
                duration_ms=duration_ms,
            )
            if estimated_word_recover_report.get("recovered"):
                matched = recount_matched_after_repairs(mapped)

        # v59: CTC forced-alignment fallback for interior words Whisper keeps
        # dropping. Aligns the KNOWN Quran text of each remaining estimated gap
        # against its audio, so merged/short function words get REAL spans.
        ctc_word_recover_report = {"attempted": False, "recovered": 0}
        if not getattr(args, "no_ctc_word_recover", False):
            mapped, ctc_word_recover_report = recover_estimated_words_ctc(
                mapped_words=mapped,
                args=args,
                wav_path=wav_path,
                tmpdir=tmpdir,
                chapter_id=chapter_id,
                duration_ms=duration_ms,
            )
            if ctc_word_recover_report.get("recovered"):
                matched = recount_matched_after_repairs(mapped)

        # v67: duplicate consecutive-ayah CTC realignment. When Whisper collapses
        # two near-identical adjacent ayat (e.g. Ash-Sharh 94:5-6 فَإِنَّ/إِنَّ مَعَ
        # ٱلۡعُسۡرِ يُسۡرًا) into a single recitation, the DP tie spreads that copy
        # across BOTH ayat, leaving one whole copy ESTIMATED on a fake window while
        # the dropped copy's audio sits unused before the next ayah — the interior
        # CTC above cannot reach it (its window is the wrong collapsed sub-window).
        # Re-align the whole duplicate region between its outer real anchors.
        dup_ayat_ctc_report = {"attempted": False, "recovered": 0}
        if not getattr(args, "no_dup_ayat_ctc", False):
            mapped, dup_ayat_ctc_report = recover_duplicate_adjacent_ayat_ctc(
                mapped_words=mapped,
                args=args,
                wav_path=wav_path,
                chapter_id=chapter_id,
                duration_ms=duration_ms,
            )
            if dup_ayat_ctc_report.get("recovered"):
                matched = recount_matched_after_repairs(mapped)

        # v61: full-surah CTC fallback for SEVERELY under-transcribed clips.
        # When Whisper structurally drops most of a surah, the passes above
        # cannot reach the quality gate (the opening island has no anchors for
        # interior recovery). Force-align the KNOWN whole-surah text against the
        # audio so every word — openings, madd, function words — gets a REAL
        # measured span. Triggers only on catastrophic under-transcription;
        # healthy chapters are untouched.
        underrun_ctc_report = {"attempted": False, "recovered": 0}
        if not getattr(args, "no_underrun_ctc", False):
            mapped, underrun_ctc_report = recover_underrun_surah_ctc(
                mapped_words=mapped,
                quran_words=quran_words,
                args=args,
                wav_path=wav_path,
                chapter_id=chapter_id,
                duration_ms=duration_ms,
            )
            if underrun_ctc_report.get("recovered"):
                matched = recount_matched_after_repairs(mapped)

        # v66: opening-scoped CTC recovery for a LONG, otherwise-healthy surah
        # whose OPENING Whisper dropped/mis-timed. The full-surah CTC above only
        # fires on catastrophic (>=50% bad) short clips, so a long surah with a
        # broken ~30s opening (e.g. Al-Furqan reciter-B) is never recovered — its
        # opening words keep fake interpolated timestamps. Force-align just the
        # opening window; the healthy interior is untouched. Skip when the
        # full-surah pass already ran (they address the same words).
        opening_ctc_report = {"attempted": False, "recovered": 0}
        if not underrun_ctc_report.get("recovered") and not getattr(args, "no_opening_ctc", False):
            _opening_ctc_intro_end = detected_intro_end_ms(
                choose_intro_verify_report(stripped_recitation_report, global_intro_report),
                detected_intro_info,
            )
            mapped, opening_ctc_report = recover_underrun_opening_ctc(
                mapped_words=mapped,
                quran_words=quran_words,
                args=args,
                wav_path=wav_path,
                chapter_id=chapter_id,
                duration_ms=duration_ms,
                intro_audio_end_ms=_opening_ctc_intro_end,
            )
            if opening_ctc_report.get("recovered") or opening_ctc_report.get("intro_inflation"):
                matched = recount_matched_after_repairs(mapped)

            # v84: escalation — the opening pass rejected its alignment AND
            # every hypothesis was squeezed against the window end. That means
            # the first-real anchor bounding the window is itself mis-timed too
            # early (Whisper timing compression), so no window-scoped attempt
            # can ever succeed. Run the full-surah CTC pass with the
            # bad-fraction gate bypassed: forced alignment of the KNOWN text is
            # strictly monotonic over the whole clip, places the opening words
            # AND moves the bogus anchor via the misanchored-interior guard.
            # Adoption is protected by the escalation mean-score floor.
            if (
                opening_ctc_report.get("anchor_suspect")
                and not underrun_ctc_report.get("recovered")
                and not getattr(args, "no_underrun_ctc", False)
            ):
                print(
                    "Opening CTC escalation: window-end squeeze suggests the "
                    "first-real anchor is mis-timed; escalating to full-surah "
                    "forced alignment."
                )
                mapped, opening_escalation_report = recover_underrun_surah_ctc(
                    mapped_words=mapped,
                    quran_words=quran_words,
                    args=args,
                    wav_path=wav_path,
                    chapter_id=chapter_id,
                    duration_ms=duration_ms,
                    force=True,
                    force_reason="opening_window_squeeze",
                )
                opening_ctc_report["escalation"] = opening_escalation_report
                if opening_escalation_report.get("recovered"):
                    underrun_ctc_report = opening_escalation_report
                    matched = recount_matched_after_repairs(mapped)

        # Degenerate-span guard: after every recovery pass, demote any word still
        # flagged real (estimated=False) but carrying an impossible zero/negative-
        # duration (or missing) span down to estimated. Such a word was never
        # genuinely placed (e.g. the monotonic clamp collapsing a CTC-recovered
        # opening to 0-0 behind a surah word mis-anchored onto the basmala), so
        # counting it as matched produced a false 100% for a broken opening.
        # Demoting it lets the opening-safe quality gate flag the surah honestly.
        mapped, degenerate_span_report = demote_degenerate_real_words(mapped)
        if degenerate_span_report.get("demoted"):
            matched = recount_matched_after_repairs(mapped)
            print(
                f"Degenerate-span guard: demoted "
                f"{degenerate_span_report['demoted']} real word(s) with impossible "
                f"spans to estimated."
            )

        match_rate = matched / max(1, len(quran_words))

        print(f"Matched after repairs: {matched}/{len(quran_words)}")
        print(f"Match rate after repairs: {match_rate:.2%}")

        madd_changes = []
        if not args.no_madd_fix:
            mapped, madd_changes = apply_madd_only_extension(
                mapped,
                wav_path=wav_path,
                mode=args.madd_mode,
                max_extend_ms=args.madd_max_extend_ms,
                verse_end_max_extend_ms=args.madd_verse_end_max_extend_ms,
                min_gap_ms=args.madd_min_gap_ms,
                pause_min_ms=args.madd_pause_min_ms,
                use_audio_pauses=not args.no_madd_audio_pauses,
            )

        output = build_output(
            chapter_id=chapter_id,
            duration_ms=duration_ms,
            mapped_words=mapped,
            interpolate=args.interpolate_missing,
        )

        final_missing = count_missing_output_segments(output)
        estimated_count = count_estimated_mapped_words(mapped)
        opening_summary = opening_estimate_summary(mapped)

        # Verification-only pass (per reciter, never edits timestamps and never
        # corrects text): the ONLY thing it reports is whether the reciter
        # repeated a word or a whole phrase/ayah.
        recitation_anomalies = {}
        if not getattr(args, "no_recitation_anomaly_check", False):
            recitation_anomalies = detect_recitation_anomalies(
                mapped,
                selected_asr_words_for_repair,
                repeat_similarity=getattr(args, "recitation_repeat_similarity", 0.80),
                phrase_similarity=getattr(args, "recitation_phrase_similarity", 0.80),
            )
            if recitation_anomalies.get("anomaly_count"):
                print(
                    "Recitation verification: "
                    f"repeated_words={recitation_anomalies['repeated_word_count']}, "
                    f"repeated_phrases={recitation_anomalies['repeated_phrase_count']} "
                    "(verification only, timestamps unchanged)"
                )

        # v49: verify the leading isti'adha + basmala were detected/stripped and
        # that no surah opening word was aligned onto that intro audio.
        # The selected candidate is the un-stripped original_asr for every
        # muqatta'at chapter, so its strip report has no leading intro block.
        # Fall back to the globally-detected intro (the same span the timing
        # repair uses) so istiadha/basmala are still recognized and the
        # opening-overlap quality check runs against the real intro end.
        intro_report_for_verify = choose_intro_verify_report(
            stripped_recitation_report, global_intro_report
        )
        underrun_intro_end_override = (
            underrun_ctc_report.get("intro_end_ms")
            if underrun_ctc_report.get("recovered")
            else None
        )
        if underrun_intro_end_override is None and opening_ctc_report.get("recovered"):
            underrun_intro_end_override = opening_ctc_report.get("intro_end_ms")
        opening_intro_verify, opening_intro_problems = verify_opening_intro(
            mapped, intro_report_for_verify, quran_words=quran_words,
            chapter_id=chapter_id, intro_audio_end_override=underrun_intro_end_override,
        )
        _ivfw = opening_intro_verify.get("first_real_word") or {}
        print(
            "Opening intro verify: "
            f"istiadha={'yes' if opening_intro_verify['istiadha_detected'] else 'no'}, "
            f"basmala={'yes' if opening_intro_verify['basmala_detected'] else 'no'} "
            f"(expected={'yes' if opening_intro_verify['basmala_expected'] else 'no'}), "
            f"intro_audio_end={opening_intro_verify['intro_audio_end_ms']}ms, "
            f"first_real_word=[{_ivfw.get('text')}] @ {_ivfw.get('start_ms')}ms, "
            f"opening_overlap={len(opening_intro_verify['overlap_words'])}"
        )
        if opening_intro_verify["basmala_expected"] and not opening_intro_verify["basmala_detected"]:
            print(
                "  Note: basmala expected for this surah but none was detected/stripped at the "
                "start (reciter may omit it, or Whisper dropped it)."
            )
        for _p in opening_intro_problems:
            print(f"  Opening intro WARNING: {_p}")

        quality_match_rate = match_rate
        quality_estimated_count = estimated_count
        opening_accepted_for_quality = False

        if not getattr(args, "no_allow_opening_estimates", False):
            opening_estimated = int(opening_summary.get("opening_estimated", 0) or 0)
            estimated_after_opening = int(opening_summary.get("estimated_after_opening", 0) or 0)

            if opening_estimated > 0:
                # v64: cap how many opening words may be "forgiven" as intro
                # artifacts. The only legitimate variable lead-in is the recited
                # isti'adha (أعوذ بالله... ~5 words) + the basmala (~4 words, when
                # the surah expects one) + any huruf muqatta'at Whisper renders by
                # letter name, plus a small buffer for madd/anchor lag. Forgiving
                # ALL opening estimates unconditionally meant a whole dropped
                # opening verse (e.g. Whisper mis-recognising the first ~30s of an
                # Idris-Abkr-style long opening) was reported as a perfect
                # effective_match_rate=100% / status ok, hiding fake interpolated
                # timestamps on real recited words. Beyond the intro budget the
                # excess is genuine under-transcription: credit only the plausible
                # intro portion and count the rest against quality so it surfaces
                # (QUALITY WARNING -> needs_review) instead of masquerading as
                # perfect. Generic for every surah/reciter; no per-surah
                # hardcoding. --no-opening-estimate-cap restores the old
                # forgive-everything behaviour; --opening-estimate-forgive-buffer
                # tunes the slack.
                istiadha_budget = 5
                basmala_budget = len(BASMALA_PHRASE) if opening_intro_verify.get("basmala_expected") else 0
                muqattaat_budget = len(describe_opening_muqattaat(quran_words, chapter_id=chapter_id))
                forgive_buffer = max(0, int(getattr(args, "opening_estimate_forgive_buffer", 4)))
                intro_budget = istiadha_budget + basmala_budget + muqattaat_budget + forgive_buffer

                cap_enabled = not getattr(args, "no_opening_estimate_cap", False)
                if cap_enabled and opening_estimated > intro_budget:
                    forgiven = intro_budget
                    excess = opening_estimated - forgiven
                else:
                    forgiven = opening_estimated
                    excess = 0

                opening_accepted_for_quality = True
                quality_estimated_count = estimated_after_opening + excess
                effective_matched = min(len(quran_words), int(matched) + forgiven)
                quality_match_rate = effective_matched / max(1, len(quran_words))

                print(
                    "Opening-safe quality: "
                    f"opening_estimated={opening_estimated}, "
                    f"estimated_after_opening={estimated_after_opening}, "
                    f"effective_match_rate={quality_match_rate:.2%}"
                )

                if excess > 0:
                    print(
                        "Opening under-transcription: "
                        f"opening_estimated={opening_estimated} exceeds intro budget={intro_budget} "
                        f"(istiadha={istiadha_budget}, basmala={basmala_budget}, "
                        f"muqattaat={muqattaat_budget}, buffer={forgive_buffer}); "
                        f"{excess} opening word(s) are dropped recitation with interpolated "
                        f"(non-real) timestamps, NOT intro — counted against quality."
                    )

                if isinstance(opening_gap_recover_report, dict) and opening_gap_recover_report.get("attempted"):
                    print("")
                    print(
                        "WARNING: opening words are still STATICALLY ESTIMATED "
                        f"({opening_estimated} word(s)) even after opening gap recovery "
                        f"recovered {int(opening_gap_recover_report.get('recovered', 0) or 0)} real ASR word(s) "
                        f"in {opening_gap_recover_report.get('gap_start_ms')}ms.."
                        f"{opening_gap_recover_report.get('gap_end_ms')}ms."
                    )
                    print(
                        "  These opening timings are NOT real Whisper timestamps. "
                        "Inspect the .opening_gap_recover.json / .opening_summary.json, and consider "
                        "lowering --opening-gap-recover-min-ms or widening --opening-gap-recover-scan-ms. "
                        "Use --fail-on-low-quality to make this a hard error instead of a warning."
                    )

        if getattr(args, "complete_output", False):
            ensure_complete_output_or_fail(output, out_path)
            final_missing = 0

        quality_ok, quality_issues = output_quality_status(quality_match_rate, quality_estimated_count, final_missing, args)
        if opening_intro_problems:
            quality_ok = False
            quality_issues = list(quality_issues) + opening_intro_problems
        if (
            recitation_anomalies.get("anomaly_count")
            and getattr(args, "fail_on_recitation_anomalies", False)
        ):
            quality_ok = False
            quality_issues = list(quality_issues) + [
                "recitation_anomalies: "
                f"repeated_words={recitation_anomalies['repeated_word_count']}, "
                f"repeated_phrases={recitation_anomalies['repeated_phrase_count']}"
            ]
        if not quality_ok:
            print("")
            print("QUALITY WARNING: output needs review.")
            for issue in quality_issues:
                print(f"  - {issue}")
            if getattr(args, "fail_on_low_quality", False):
                raise RuntimeError("Low quality output: " + "; ".join(quality_issues))

        save_json(out_path, output, minify=args.minify)
        print(f"Saved: {out_path}")
        print(f"Final missing segments: {final_missing}")
        print(f"Estimated timings inside main script: {estimated_count}")

        if args.debug:
            # Keep the per-reciter output folder clean (only NNN.json). All debug
            # side-files go to a SEPARATE debug folder when --debug-dir is set
            # (the batch runner points this at debug_logs/<reciter>/), so a failed
            # chapter can be located and re-run without touching the timings.
            _debug_dir = Path(getattr(args, "debug_dir", None) or Path(out_path).parent)
            _debug_dir.mkdir(parents=True, exist_ok=True)
            _debug_base = _debug_dir / chapter_padded

            debug_path = str(_debug_base.with_suffix(".debug.json"))
            save_json(debug_path, mapped, minify=False)
            print(f"Debug saved: {debug_path}")

            asr_path = str(_debug_base.with_suffix(".asr_words.json"))
            save_json(asr_path, asr_words, minify=False)
            print(f"ASR words saved: {asr_path}")

            # The selected/stripped ASR stream that mapped.asr_index references.
            # Persist it so the offline validator can recompute repeat detection
            # in the SAME index space (the raw .asr_words.json may differ).
            selasr_path = str(_debug_base.with_suffix(".selected_asr_words.json"))
            save_json(selasr_path, selected_asr_words_for_repair, minify=False)
            print(f"Selected ASR words saved: {selasr_path}")

            if asr_chunk_report:
                asr_chunks_path = str(_debug_base.with_suffix(".asr_chunks.json"))
                save_json(asr_chunks_path, asr_chunk_report, minify=False)
                print(f"ASR chunks saved: {asr_chunks_path}")

            madd_path = str(_debug_base.with_suffix(".madd_fix.json"))
            save_json(madd_path, madd_changes, minify=False)
            print(f"Madd fix saved: {madd_path}")

            alignq_path = str(_debug_base.with_suffix(".alignment_quality.json"))
            save_json(alignq_path, alignment_quality_report, minify=False)
            print(f"Alignment quality saved: {alignq_path}")

            stripped_path = str(_debug_base.with_suffix(".stripped_recitation.json"))
            save_json(stripped_path, stripped_recitation_report, minify=False)
            print(f"Stripped recitation extras saved: {stripped_path}")

            charproj_path = str(_debug_base.with_suffix(".char_projection_repair.json"))
            save_json(charproj_path, char_projection_report, minify=False)
            print(f"Char projection repair saved: {charproj_path}")

            introgap_path = str(_debug_base.with_suffix(".intro_gap_repair.json"))
            save_json(introgap_path, intro_gap_report, minify=False)
            print(f"Intro gap repair saved: {introgap_path}")

            muqfold_path = str(_debug_base.with_suffix(".muqattaat_fold.json"))
            save_json(muqfold_path, muqattaat_fold_report, minify=False)
            print(f"Muqattaat fold saved: {muqfold_path}")

            opengap_path = str(_debug_base.with_suffix(".opening_gap_recover.json"))
            save_json(opengap_path, opening_gap_recover_report, minify=False)
            print(f"Opening gap recovery saved: {opengap_path}")

            underrun_path = str(_debug_base.with_suffix(".underrun_recover.json"))
            save_json(underrun_path, underrun_recover_report, minify=False)
            print(f"Under-transcription recovery saved: {underrun_path}")

            estrecover_path = str(_debug_base.with_suffix(".estimated_word_recover.json"))
            save_json(estrecover_path, estimated_word_recover_report, minify=False)
            print(f"Estimated-word recovery saved: {estrecover_path}")

            ctcrecover_path = str(_debug_base.with_suffix(".ctc_word_recover.json"))
            save_json(ctcrecover_path, ctc_word_recover_report, minify=False)
            print(f"CTC interior recovery saved: {ctcrecover_path}")

            dupayat_path = str(_debug_base.with_suffix(".dup_ayat_ctc_recover.json"))
            save_json(dupayat_path, dup_ayat_ctc_report, minify=False)
            print(f"Duplicate-ayah CTC recovery saved: {dupayat_path}")

            openreal_path = str(_debug_base.with_suffix(".opening_real_enforce.json"))
            save_json(openreal_path, opening_real_enforce_report, minify=False)
            print(f"Opening real-timestamp enforcement saved: {openreal_path}")

            openctc_path = str(_debug_base.with_suffix(".opening_ctc_recover.json"))
            save_json(openctc_path, opening_ctc_report, minify=False)
            print(f"Opening CTC recovery saved: {openctc_path}")

            degenspan_path = str(_debug_base.with_suffix(".degenerate_span_demote.json"))
            save_json(degenspan_path, degenerate_span_report, minify=False)
            print(f"Degenerate-span demote saved: {degenspan_path}")

            prompt_path = str(_debug_base.with_suffix(".initial_prompt.txt"))
            Path(prompt_path).write_text(initial_prompt or "", encoding="utf-8")
            print(f"Initial prompt saved: {prompt_path}")

            openingsum_path = str(_debug_base.with_suffix(".opening_summary.json"))
            save_json(openingsum_path, opening_estimate_summary(mapped), minify=False)
            print(f"Opening summary saved: {openingsum_path}")

            openingintro_path = str(_debug_base.with_suffix(".opening_intro_verify.json"))
            save_json(openingintro_path, opening_intro_verify, minify=False)
            print(f"Opening intro verify saved: {openingintro_path}")

            recanom_path = str(_debug_base.with_suffix(".recitation_anomalies.json"))
            save_json(recanom_path, recitation_anomalies, minify=False)
            print(f"Recitation anomalies saved: {recanom_path}")

        if match_rate < args.warn_below:
            print("")
            print("WARNING: Low match rate.")
            print("Try:")
            print("  --model large-v3")
            print("  --min-score 0.35")
            print("  --alignment dp")

        return {
            "chapter_id": chapter_id,
            "audio": audio_source,
            "output": out_path,
            "duration_ms": duration_ms,
            "quran_words": len(quran_words),
            "asr_words": len(asr_words),
            "selected_asr_words": selected_asr_words_count,
            "asr_chunks": len(asr_chunk_report) if asr_chunk_report else 0,
            "matched": matched,
            "match_rate": round(match_rate, 6),
            "quality_match_rate": round(quality_match_rate, 6),
            "estimated_timings": estimated_count,
            "quality_estimated_timings": quality_estimated_count,
            "opening_accepted_for_quality": opening_accepted_for_quality,
            "opening_estimate_summary": opening_summary,
            "opening_intro_verify": opening_intro_verify,
            "recitation_anomalies": {
                "repeated_words": int(recitation_anomalies.get("repeated_word_count", 0) or 0),
                "repeated_phrases": int(recitation_anomalies.get("repeated_phrase_count", 0) or 0),
                "repeated": int(recitation_anomalies.get("repeated_count", 0) or 0),
                "total": int(recitation_anomalies.get("anomaly_count", 0) or 0),
            },
            "final_missing_segments": final_missing,
            "quality_ok": quality_ok,
            "quality_issues": quality_issues,
            "status": "ok" if quality_ok else "needs_review",
        }




# ---------------------------------------------------------------------------
# v22 existing timings validator and fixer
# ---------------------------------------------------------------------------

def expected_words_by_chapter(quran_words_path):
    """
    Return {chapter_id: expected_word_count} from quran_words.json.
    """
    with open(quran_words_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected = {}
    for ch in range(1, 115):
        words = data.get(str(ch), [])
        expected[ch] = len(words)

    return expected


def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


def is_chapter_timing_json(path):
    return bool(re.fullmatch(r"\d{3}\.json", Path(path).name))


def count_segments_in_output(output):
    count = 0
    for vt in output.get("verse_timings", []):
        count += len(vt.get("segments", []) or [])
    return count




# ---------------------------------------------------------------------------
# v23 log-based validation
# ---------------------------------------------------------------------------

def normalize_rel_path_for_log(path):
    try:
        return str(Path(path).as_posix()).lstrip("./")
    except Exception:
        return str(path).replace("\\", "/").lstrip("./")


def parse_alignment_logs(log_root="logs"):
    """
    Parse batch/single-run logs and return quality data by output JSON path.

    Supports log lines like:
      Chapter: 27
      Output: timings/abo/027.json
      Quran words: 1151
      ASR words: 1192
      Completed missing inside main script: 22
      Matched: 1129/1151
      Match rate: 98.09%
      Final missing segments: 0
      Estimated timings inside main script: 22
    """
    log_root = Path(log_root)
    by_output = {}
    by_reciter_chapter = {}

    if not log_root.exists():
        return {
            "log_root": str(log_root),
            "logs_found": 0,
            "by_output": by_output,
            "by_reciter_chapter": by_reciter_chapter,
        }

    log_files = sorted(log_root.glob("*.log"))

    chapter_re = re.compile(r"^Chapter:\s*(\d+)\s*$")
    output_re = re.compile(r"^Output:\s*(.+?)\s*$")
    audio_re = re.compile(r"^Audio:\s*(.+?)\s*$")
    quran_words_re = re.compile(r"^Quran words:\s*(\d+)")
    asr_words_re = re.compile(r"^ASR words:\s*(\d+)")
    completed_missing_re = re.compile(r"^Completed missing inside main script:\s*(\d+)")
    matched_re = re.compile(r"^Matched:\s*(\d+)\s*/\s*(\d+)")
    match_rate_re = re.compile(r"^Match rate:\s*([0-9.]+)%")
    final_missing_re = re.compile(r"^Final missing segments:\s*(\d+)")
    estimated_re = re.compile(r"^Estimated timings inside main script:\s*(\d+)")
    saved_re = re.compile(r"^Saved:\s*(.+?)\s*$")
    failed_re = re.compile(r"^FAILED chapter\s+(\d+):\s*(.+?)\s*$")

    def finish_run(run, log_file):
        if not run:
            return

        out = run.get("output") or run.get("saved")
        chapter = run.get("chapter")

        if out:
            key = normalize_rel_path_for_log(out)
            run["output"] = key
            run["log_file"] = str(log_file)
            run["reciter"] = Path(key).parent.name
            by_output[key] = dict(run)

        if chapter is not None:
            reciter = run.get("reciter")
            if not reciter and out:
                reciter = Path(normalize_rel_path_for_log(out)).parent.name
            if reciter:
                by_reciter_chapter[f"{reciter}:{int(chapter):03d}"] = dict(run)

    for log_file in log_files:
        current = None

        try:
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        for raw in lines:
            line = raw.strip()

            m = chapter_re.match(line)
            if m:
                finish_run(current, log_file)
                current = {
                    "chapter": int(m.group(1)),
                    "status": "started",
                }
                continue

            if current is None:
                # Some logs can start with failure lines. Ignore until Chapter.
                continue

            m = output_re.match(line)
            if m:
                current["output"] = normalize_rel_path_for_log(m.group(1))
                current["reciter"] = Path(current["output"]).parent.name
                continue

            m = saved_re.match(line)
            if m:
                current["saved"] = normalize_rel_path_for_log(m.group(1))
                current["status"] = "saved"
                continue

            m = audio_re.match(line)
            if m:
                current["audio"] = m.group(1)
                continue

            m = quran_words_re.match(line)
            if m:
                current["quran_words"] = int(m.group(1))
                continue

            m = asr_words_re.match(line)
            if m:
                current["asr_words"] = int(m.group(1))
                continue

            m = completed_missing_re.match(line)
            if m:
                current["completed_missing"] = int(m.group(1))
                continue

            m = matched_re.match(line)
            if m:
                current["matched"] = int(m.group(1))
                current["match_total"] = int(m.group(2))
                continue

            m = match_rate_re.match(line)
            if m:
                current["match_rate"] = float(m.group(1)) / 100.0
                continue

            m = final_missing_re.match(line)
            if m:
                current["final_missing_segments"] = int(m.group(1))
                continue

            m = estimated_re.match(line)
            if m:
                current["estimated_timings"] = int(m.group(1))
                continue

            m = failed_re.match(line)
            if m:
                current["status"] = "failed"
                current["error"] = m.group(2)
                continue

        finish_run(current, log_file)

    return {
        "log_root": str(log_root),
        "logs_found": len(log_files),
        "by_output": by_output,
        "by_reciter_chapter": by_reciter_chapter,
    }


def find_log_quality_for_timing(timing_path, parsed_logs):
    if not parsed_logs:
        return None

    timing_path = Path(timing_path)
    rel = normalize_rel_path_for_log(timing_path)

    by_output = parsed_logs.get("by_output", {})
    if rel in by_output:
        return by_output[rel]

    # Try common path variants
    variants = [
        normalize_rel_path_for_log(timing_path.as_posix()),
        normalize_rel_path_for_log("./" + timing_path.as_posix()),
        normalize_rel_path_for_log(str(timing_path)),
    ]

    for v in variants:
        if v in by_output:
            return by_output[v]

    reciter = timing_path.parent.name
    chapter = timing_path.stem

    by_rc = parsed_logs.get("by_reciter_chapter", {})
    return by_rc.get(f"{reciter}:{chapter}")



class _IntroBudgetArgs:
    """Minimal stand-in so the validator can reuse opening_ctc_intro_budget()
    (which reads args.opening_estimate_forgive_buffer) without threading the full
    argparse namespace. Keeps the validator's intro budget byte-identical to the
    generator's, guaranteeing they agree on the opening-safe effective rate."""

    __slots__ = ("opening_estimate_forgive_buffer",)

    def __init__(self, forgive_buffer):
        self.opening_estimate_forgive_buffer = forgive_buffer


def validate_one_timing_file(
    timing_path,
    expected_counts=None,
    min_match=0.9955,
    max_estimated=0,
    min_word_ms=20,
    max_word_ms=12000,
    require_debug=False,
    parsed_logs=None,
    use_logs=False,
    use_debug_quality=True,
    debug_root=None,
    respect_manual_lock=True,
    severe_match=0.90,
    opening_estimate_forgive_buffer=4,
):
    timing_path = Path(timing_path)
    item = {
        "file": str(timing_path),
        "reciter": timing_path.parent.name,
        "chapter": None,
        "status": "ok",
        "needs_fix": False,
        "needs_review": False,
        "manual_locked": False,
        "issues": [],
        "review_issues": [],
        "warnings": [],
        "metrics": {},
        "related_files": {},
    }

    # Manual-lock: the dashboard writes a <chapter>.manual_lock.json marker next
    # to the timing file the moment a human saves manual edits. A hand-corrected
    # surah is FINAL — re-aligning it with the same audio+model would only
    # reproduce the same best-effort result and clobber the human's work. So a
    # locked file is reported ok/locked and skipped by the fix loop. Pass
    # --fix-override-manual (respect_manual_lock=False) to re-align it anyway.
    if respect_manual_lock:
        try:
            lock_path = timing_path.with_suffix(".manual_lock.json")
            if lock_path.exists():
                item["manual_locked"] = True
                item["status"] = "ok"
                item["warnings"].append("manual_locked_skipping_checks")
                try:
                    item["chapter"] = int(timing_path.stem)
                except (TypeError, ValueError):
                    pass
                return item
        except OSError:
            pass

    output, err = safe_load_json(timing_path)
    if err:
        item["status"] = "needs_fix"
        item["needs_fix"] = True
        item["issues"].append(f"cannot_read_json: {err}")
        return item

    if not isinstance(output, dict):
        item["status"] = "needs_fix"
        item["needs_fix"] = True
        item["issues"].append("json_root_is_not_object")
        return item

    chapter_id = output.get("chapter_id")
    try:
        chapter_id = int(chapter_id)
    except Exception:
        chapter_id = None

    item["chapter"] = chapter_id

    verse_timings = output.get("verse_timings")
    if not isinstance(verse_timings, list) or not verse_timings:
        item["issues"].append("missing_or_empty_verse_timings")

    duration_ms = output.get("duration")
    try:
        duration_ms = int(duration_ms)
    except Exception:
        duration_ms = None

    expected_words = None
    if expected_counts and chapter_id:
        expected_words = int(expected_counts.get(chapter_id, 0) or 0)

    total_segments = count_segments_in_output(output)
    item["metrics"]["expected_words"] = expected_words
    item["metrics"]["total_segments"] = total_segments
    item["metrics"]["duration_ms"] = duration_ms

    if expected_words and total_segments != expected_words:
        item["issues"].append(f"segment_count_mismatch: got {total_segments}, expected {expected_words}")

    if chapter_id == 1:
        # Fatihah must include the opening basmala from quran_words.json.
        # This is a structural guard; if quran_words expects 29 words and output has 25,
        # the mismatch above will fail the file.
        item["metrics"]["fatiha_basmala_required"] = True
        if expected_words:
            item["metrics"]["fatiha_expected_words_including_basmala"] = expected_words

    last_global_start = -1
    last_global_end = -1
    zero_or_negative = 0
    too_short = 0
    too_long = 0
    backward = 0
    bad_shapes = 0
    outside_duration = 0

    for vt in verse_timings or []:
        key = vt.get("verse_key")
        ts_from = vt.get("timestamp_from")
        ts_to = vt.get("timestamp_to")

        try:
            ts_from_i = int(ts_from)
            ts_to_i = int(ts_to)

            if ts_to_i < ts_from_i:
                item["issues"].append(f"{key}: verse_to_before_from")
        except Exception:
            item["issues"].append(f"{key}: bad_verse_timestamp")

        prev_start = -1
        prev_end = -1

        for seg in vt.get("segments", []) or []:
            if not isinstance(seg, list) or len(seg) != 3:
                bad_shapes += 1
                continue

            pos, start_ms, end_ms = seg

            try:
                start_ms = int(start_ms)
                end_ms = int(end_ms)
            except Exception:
                bad_shapes += 1
                continue

            dur = end_ms - start_ms

            if dur <= 0:
                zero_or_negative += 1

            if dur > 0 and dur < min_word_ms:
                too_short += 1

            if dur > max_word_ms:
                too_long += 1

            if start_ms < prev_start:
                backward += 1

            if start_ms < last_global_start:
                backward += 1

            if duration_ms is not None and (start_ms < 0 or end_ms > duration_ms + 2500):
                outside_duration += 1

            prev_start = max(prev_start, start_ms)
            prev_end = max(prev_end, end_ms)
            last_global_start = max(last_global_start, start_ms)
            last_global_end = max(last_global_end, end_ms)

    item["metrics"]["zero_or_negative_segments"] = zero_or_negative
    item["metrics"]["too_short_segments"] = too_short
    item["metrics"]["too_long_segments"] = too_long
    item["metrics"]["backward_segments"] = backward
    item["metrics"]["bad_segment_shapes"] = bad_shapes
    item["metrics"]["outside_duration_segments"] = outside_duration

    if bad_shapes:
        item["issues"].append(f"bad_segment_shapes: {bad_shapes}")
    if zero_or_negative:
        item["issues"].append(f"zero_or_negative_segments: {zero_or_negative}")
    if backward:
        item["issues"].append(f"backward_segments: {backward}")
    if outside_duration:
        item["issues"].append(f"outside_duration_segments: {outside_duration}")
    if too_long:
        item["warnings"].append(f"suspicious_long_words: {too_long}")
    if too_short:
        item["warnings"].append(f"suspicious_short_words: {too_short}")

    def _sidecar(suffix):
        # Prefer the sidecar next to the timing JSON. In batch mode the debug/
        # diagnostic side-files (*.debug.json, *.alignment_quality.json, ...) are
        # written under <debug-root>/<reciter>/ to keep timing folders clean, so
        # fall back there when the local one is absent. Without this, quality
        # checks silently no-op on batch outputs and broken-but-matched files
        # (low match_rate / estimated words) are never flagged.
        local = timing_path.with_suffix(suffix)
        if local.exists():
            return local
        if debug_root:
            candidate = Path(debug_root) / timing_path.parent.name / (timing_path.stem + suffix)
            if candidate.exists():
                return candidate
        return local

    debug_path = _sidecar(".debug.json")
    asr_path = _sidecar(".asr_words.json")
    alignq_path = _sidecar(".alignment_quality.json")
    stripped_path = _sidecar(".stripped_recitation.json")
    madd_path = _sidecar(".madd_fix.json")

    item["related_files"] = {
        "debug": str(debug_path) if debug_path.exists() else None,
        "asr_words": str(asr_path) if asr_path.exists() else None,
        "alignment_quality": str(alignq_path) if alignq_path.exists() else None,
        "stripped_recitation": str(stripped_path) if stripped_path.exists() else None,
        "madd_fix": str(madd_path) if madd_path.exists() else None,
    }

    log_quality = None
    if use_logs:
        log_quality = find_log_quality_for_timing(timing_path, parsed_logs)
        if log_quality:
            item["related_files"]["log_file"] = log_quality.get("log_file")
            item["metrics"]["log_status"] = log_quality.get("status")
            item["metrics"]["log_quran_words"] = log_quality.get("quran_words")
            item["metrics"]["log_asr_words"] = log_quality.get("asr_words")
            item["metrics"]["log_matched"] = log_quality.get("matched")
            item["metrics"]["log_match_total"] = log_quality.get("match_total")
            item["metrics"]["log_match_rate"] = round(float(log_quality.get("match_rate")), 6) if log_quality.get("match_rate") is not None else None
            item["metrics"]["log_completed_missing"] = log_quality.get("completed_missing")
            item["metrics"]["log_estimated_timings"] = log_quality.get("estimated_timings")
            item["metrics"]["log_final_missing_segments"] = log_quality.get("final_missing_segments")

            if log_quality.get("status") == "failed":
                item["issues"].append(f"log_says_failed: {log_quality.get('error', '')}")

            # Option 1 (effective-rate gate): interpolated/completed word COUNTS
            # are no longer a standalone review trigger. They already depress the
            # match rate checked below, which is the authoritative bar. A surah
            # that still meets the match-rate threshold is OK even if a few
            # opening/within-tolerance words were interpolated.

            if log_quality.get("final_missing_segments") is not None and int(log_quality.get("final_missing_segments")) > 0:
                item["issues"].append(f"log_final_missing_segments: {int(log_quality.get('final_missing_segments'))}")

            # The log's "Match rate" is the RAW (pre-opening-forgiveness) rate,
            # so it can false-flag a surah whose only interpolated words are the
            # opening. When debug-quality is enabled (default) the opening-safe
            # EFFECTIVE rate computed from the debug + opening_summary sidecars is
            # authoritative, so don't let the raw log rate override it. Only fall
            # back to the raw log rate when debug-quality validation is off.
            if (not use_debug_quality) and log_quality.get("match_rate") is not None and float(log_quality.get("match_rate")) < min_match:
                item["review_issues"].append(f"log_match_rate_below_threshold: {float(log_quality.get('match_rate')):.4%} < {min_match:.4%}")
        else:
            item["warnings"].append("no_matching_log_quality_found")

    estimated_count = None
    debug_count = None
    matched_count = None
    match_rate = None
    low_score_count = None

    if use_debug_quality and debug_path.exists():
        debug_data, debug_err = safe_load_json(debug_path)

        if debug_err:
            item["warnings"].append(f"cannot_read_debug: {debug_err}")
        elif isinstance(debug_data, list):
            debug_count = len(debug_data)
            estimated_count = 0
            low_score_count = 0
            matched_count = 0

            for w in debug_data:
                if not isinstance(w, dict):
                    continue

                if bool(w.get("estimated")):
                    estimated_count += 1
                else:
                    matched_count += 1

                score = w.get("score")
                try:
                    if score is not None and float(score) < 0.45:
                        low_score_count += 1
                except Exception:
                    pass

            if expected_words:
                match_rate = matched_count / max(1, expected_words)

            item["metrics"]["debug_words"] = debug_count
            item["metrics"]["estimated_words"] = estimated_count
            item["metrics"]["matched_words_from_debug"] = matched_count
            item["metrics"]["match_rate_from_debug"] = round(match_rate, 6) if match_rate is not None else None
            item["metrics"]["low_score_words_lt_0_45"] = low_score_count

            # Option 1: judge by the OPENING-SAFE EFFECTIVE match rate, mirroring
            # exactly what the generator reports ("Opening-safe quality:
            # effective_match_rate=..."), instead of the raw matched/expected rate
            # plus a separate zero-estimated gate. A plausible intro's worth of
            # opening estimates (isti'adha + basmala + huruf muqatta'at + buffer)
            # is forgiven -- the SAME intro budget the generator uses via
            # opening_ctc_intro_budget() -- so a surah whose only interpolated
            # words are the opening (or few enough to still clear the bar) counts
            # as OK. Estimated words are NOT a standalone review trigger anymore:
            # they already lower the effective rate, which is the authoritative
            # >=min_match bar. Generic; no per-surah hardcoding.
            effective_estimated = estimated_count
            effective_match_rate = match_rate
            opening_summary_path = _sidecar(".opening_summary.json")
            if opening_summary_path.exists() and expected_words:
                osum, osum_err = safe_load_json(opening_summary_path)
                if not osum_err and isinstance(osum, dict):
                    # Harden against a present-but-corrupt sidecar (non-numeric or
                    # negative counts): on any coercion failure fall back to the
                    # raw match rate instead of aborting the whole validation.
                    try:
                        opening_estimated = max(0, int(osum.get("opening_estimated", 0) or 0))
                        estimated_after_opening = max(0, int(osum.get("estimated_after_opening", 0) or 0))
                    except (TypeError, ValueError):
                        opening_estimated = 0
                        estimated_after_opening = 0
                        item["warnings"].append("bad_opening_summary_counts_using_raw_rate")
                    if opening_estimated > 0:
                        chapter_id = item.get("chapter")
                        if chapter_id is None:
                            try:
                                chapter_id = int(timing_path.stem)
                            except (TypeError, ValueError):
                                chapter_id = None
                        quran_words_for_budget = [
                            {"norm": w.get("norm", ""), "text": w.get("text", "")}
                            for w in debug_data if isinstance(w, dict)
                        ]
                        try:
                            intro_budget = opening_ctc_intro_budget(
                                quran_words_for_budget,
                                _IntroBudgetArgs(opening_estimate_forgive_buffer),
                                chapter_id=chapter_id,
                            )
                        except Exception:
                            intro_budget = 0
                        excess = max(0, opening_estimated - intro_budget)
                        effective_estimated = estimated_after_opening + excess
                        effective_match_rate = (expected_words - effective_estimated) / max(1, expected_words)
                        item["metrics"]["opening_estimated"] = opening_estimated
                        item["metrics"]["estimated_after_opening"] = estimated_after_opening
                        item["metrics"]["effective_estimated_words"] = effective_estimated
                        item["metrics"]["effective_match_rate"] = round(effective_match_rate, 6)

            if effective_match_rate is not None and effective_match_rate < min_match:
                item["review_issues"].append(f"match_rate_below_threshold: {effective_match_rate:.4%} < {min_match:.4%}")

            # SEVERE quality shortfall is RED / needs_fix, not mild YELLOW.
            # A structurally-sound file whose FINAL effective match rate (real,
            # non-estimated words / expected) is far below the bar is not "a bit
            # below ideal" — most of the surah is interpolated, not really
            # aligned, so it is genuinely wrong and must surface as needs_fix
            # rather than hide as needs_review. With the v63 chunked-CTC gap
            # recovery a re-align can now actually fix these, so flagging them
            # RED is productive instead of an endless re-align loop. Set
            # --validate-severe-match 0 to disable and keep everything yellow.
            if (
                severe_match
                and effective_match_rate is not None
                and effective_match_rate < float(severe_match)
            ):
                item["issues"].append(
                    f"severe_under_transcription: effective match {effective_match_rate:.4%} "
                    f"< {float(severe_match):.4%} (most of the surah is interpolated, "
                    f"not really aligned)"
                )

            if low_score_count:
                item["warnings"].append(f"low_score_words_lt_0_45: {low_score_count}")

            # Recitation verification over existing output. Prefer the live
            # report (computed with the correct selected-ASR index space); else
            # recompute from the selected-ASR dump; else fall back to the raw
            # ASR dump WITHOUT repeat detection, because its index space may not
            # match the saved mapping and would create false repeat flags.
            recanom_path = _sidecar(".recitation_anomalies.json")
            selasr_path = _sidecar(".selected_asr_words.json")
            anomalies = None
            repeats_checked = True

            if recanom_path.exists():
                rep_data, rep_err = safe_load_json(recanom_path)
                if not rep_err and isinstance(rep_data, dict):
                    anomalies = {
                        "repeated_word_count": int(rep_data.get("repeated_word_count", 0) or 0),
                        "repeated_phrase_count": int(rep_data.get("repeated_phrase_count", 0) or 0),
                        "repeated_count": int(rep_data.get("repeated_count", 0) or 0),
                        "anomaly_count": int(rep_data.get("anomaly_count", 0) or 0),
                    }
                    repeats_checked = bool((rep_data.get("params") or {}).get("repeats_checked", True))

            if anomalies is None and selasr_path.exists():
                asr_data, asr_err = safe_load_json(selasr_path)
                if not asr_err and isinstance(asr_data, list):
                    anomalies = detect_recitation_anomalies(debug_data, asr_data, asr_index_trusted=True)

            if anomalies is None and asr_path.exists():
                asr_data, asr_err = safe_load_json(asr_path)
                if asr_err:
                    item["warnings"].append(f"cannot_read_asr_words: {asr_err}")
                elif isinstance(asr_data, list):
                    repeats_checked = False
                    anomalies = detect_recitation_anomalies(debug_data, asr_data, asr_index_trusted=False)

            if anomalies is not None:
                item["metrics"]["recitation_repeated_words"] = anomalies["repeated_word_count"]
                item["metrics"]["recitation_repeated_phrases"] = anomalies["repeated_phrase_count"]
                item["metrics"]["recitation_repeated"] = anomalies["repeated_count"]
                item["metrics"]["recitation_repeats_checked"] = repeats_checked
                if not repeats_checked:
                    item["warnings"].append(
                        "recitation_repeat_check_skipped: no selected-ASR/report dump "
                        "(raw ASR index space is unreliable for repeat detection)"
                    )
                if anomalies["anomaly_count"]:
                    item["warnings"].append(
                        "recitation_anomalies: "
                        f"repeated_words={anomalies['repeated_word_count']}, "
                        f"repeated_phrases={anomalies['repeated_phrase_count']}"
                    )
        else:
            item["warnings"].append("debug_file_not_list")
    else:
        if use_debug_quality:
            msg = "missing_debug_file_cannot_verify_alignment_quality"
            if require_debug:
                item["issues"].append(msg)
            else:
                # Missing debug alone is not a failure. It only means quality proof is unavailable.
                if log_quality:
                    item["warnings"].append("missing_debug_file_but_log_quality_used")
                else:
                    item["warnings"].append(msg)
        else:
            item["warnings"].append("debug_quality_check_disabled")

    if alignq_path.exists():
        alignq_data, alignq_err = safe_load_json(alignq_path)
        if alignq_err:
            item["warnings"].append(f"cannot_read_alignment_quality: {alignq_err}")
        else:
            item["metrics"]["alignment_quality_candidates"] = len(alignq_data) if isinstance(alignq_data, list) else None
    else:
        item["warnings"].append("missing_alignment_quality_file_not_a_failure")

    # Structural problems (item["issues"]) are the only RED / needs_fix state:
    # broken JSON, missing/backward/out-of-range segments, wrong word count, a
    # generation FAILURE, or leftover missing segments. Those are re-alignable.
    #
    # Quality shortfalls (item["review_issues"]) are YELLOW / needs_review: the
    # file is structurally sound but a bit below the ideal quality bar (match
    # rate under threshold, some estimated/interpolated words). Re-aligning the
    # SAME audio with the SAME model just reproduces the same best-effort result,
    # so treating these as needs_fix made them re-align on every run forever and
    # never turn green. As needs_review they leave the default fix loop (only
    # --fix-include-review retries them) and surface honestly for human review.
    if item["issues"]:
        item["status"] = "needs_fix"
        item["needs_fix"] = True
    elif item["needs_review"] or item["review_issues"]:
        item["status"] = "needs_review"
        item["needs_review"] = True
    else:
        item["status"] = "ok"

    return item


def discover_timing_files(root, reciters=None):
    root = Path(root)

    if reciters:
        files = []
        for reciter in reciters:
            d = root / reciter
            if d.exists():
                files.extend(sorted(p for p in d.glob("*.json") if is_chapter_timing_json(p)))
        return files

    return sorted(p for p in root.glob("*/*.json") if is_chapter_timing_json(p))



# ---------------------------------------------------------------------------
# v29 in-place repair for existing timing JSON files
# ---------------------------------------------------------------------------

def normalize_existing_output_structure(output):
    """
    Safe no-ASR repair for existing final JSON:
    - drops bad-shaped segments
    - sorts segments by start time inside each verse
    - recalculates timestamp_from/timestamp_to/duration
    - clamps negative starts to 0
    It never changes text because final timing JSON has no text.
    """
    if not isinstance(output, dict):
        raise ValueError("output is not object")

    changed = False
    out = dict(output)
    verse_timings = out.get("verse_timings")
    if not isinstance(verse_timings, list):
        raise ValueError("missing verse_timings list")

    new_verses = []

    for vt in verse_timings:
        if not isinstance(vt, dict):
            changed = True
            continue

        new_vt = dict(vt)
        segs = []

        for seg in vt.get("segments", []) or []:
            if not isinstance(seg, list) or len(seg) != 3:
                changed = True
                continue

            try:
                pos = int(seg[0])
                st = max(0, int(seg[1]))
                en = max(st + 1, int(seg[2]))
            except Exception:
                changed = True
                continue

            if [pos, st, en] != seg:
                changed = True

            segs.append([pos, st, en])

        segs.sort(key=lambda x: (x[1], x[0]))

        if segs:
            ts_from = min(x[1] for x in segs)
            ts_to = max(x[2] for x in segs)
        else:
            ts_from = 0
            ts_to = 0

        if new_vt.get("timestamp_from") != ts_from or new_vt.get("timestamp_to") != ts_to:
            changed = True

        new_vt["timestamp_from"] = int(ts_from)
        new_vt["timestamp_to"] = int(ts_to)
        new_vt["duration"] = int(max(0, ts_to - ts_from))
        new_vt["segments"] = segs
        new_verses.append(new_vt)

    out["verse_timings"] = new_verses
    return out, changed


def rebuild_output_from_debug_file(timing_path, expected_counts=None):
    timing_path = Path(timing_path)
    debug_path = timing_path.with_suffix(".debug.json")
    if not debug_path.exists():
        return None, "missing_debug"

    debug_data, err = safe_load_json(debug_path)
    if err:
        return None, f"cannot_read_debug: {err}"

    if not isinstance(debug_data, list) or not debug_data:
        return None, "debug_not_list_or_empty"

    out, err = safe_load_json(timing_path)
    if err or not isinstance(out, dict):
        out = {}

    chapter_id = out.get("chapter_id") or debug_data[0].get("verse_key", "0:0").split(":")[0]
    try:
        chapter_id = int(chapter_id)
    except Exception:
        return None, "cannot_detect_chapter"

    expected = None
    if expected_counts:
        expected = int(expected_counts.get(chapter_id, 0) or 0)

    if expected and len(debug_data) != expected:
        return None, f"debug_word_count_mismatch: got {len(debug_data)}, expected {expected}"

    for i, w in enumerate(debug_data):
        if not isinstance(w, dict):
            return None, f"debug_word_not_object_at_{i}"
        if w.get("start_ms") is None or w.get("end_ms") is None:
            return None, f"debug_word_missing_timing_at_{i}"

    duration_ms = out.get("duration")
    try:
        duration_ms = int(duration_ms)
    except Exception:
        max_end = max(int(w.get("end_ms") or 0) for w in debug_data)
        duration_ms = max_end

    rebuilt = build_output(
        chapter_id=chapter_id,
        duration_ms=duration_ms,
        mapped_words=debug_data,
        interpolate=False,
    )

    return rebuilt, "rebuilt_from_debug"


def repair_one_existing_timing_file(timing_path, expected_counts=None, use_debug=True, backup=True):
    timing_path = Path(timing_path)
    result = {
        "file": str(timing_path),
        "status": "unchanged",
        "method": None,
        "backup": None,
        "error": None,
    }

    original, err = safe_load_json(timing_path)
    if err:
        result["status"] = "failed"
        result["error"] = f"cannot_read_json: {err}"
        return result

    repaired = None
    method = None

    if use_debug:
        rebuilt, reason = rebuild_output_from_debug_file(timing_path, expected_counts=expected_counts)
        if rebuilt is not None:
            repaired = rebuilt
            method = reason
        else:
            result["debug_rebuild_skipped"] = reason

    if repaired is None:
        try:
            repaired, changed = normalize_existing_output_structure(original)
            method = "normalized_final_json" if changed else "already_structurally_normal"
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            return result

    if repaired == original:
        result["status"] = "unchanged"
        result["method"] = method
        return result

    if backup:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = timing_path.with_suffix(timing_path.suffix + f".repair_backup_{stamp}")
        shutil.copy2(timing_path, backup_path)
        result["backup"] = str(backup_path)

    save_json(timing_path, repaired, minify=False)
    result["status"] = "repaired"
    result["method"] = method
    return result


def repair_existing_timings_in_place(args):
    expected_counts = expected_words_by_chapter(args.quran_words)

    reciters = None
    if getattr(args, "validate_reciter_list_url", None):
        reciters = fetch_reciter_folders_from_url(
            args.validate_reciter_list_url,
            insecure_ssl=getattr(args, "insecure_ssl", False),
        )
    elif args.validate_reciter_folders:
        reciters = [x.rstrip("/") for x in args.validate_reciter_folders]

    files = discover_timing_files(args.validate_timings_root, reciters=reciters)

    results = []
    for path in files:
        results.append(repair_one_existing_timing_file(
            path,
            expected_counts=expected_counts,
            use_debug=not args.no_repair_use_debug,
            backup=not args.no_repair_backup_existing,
        ))

    report = {
        "script_version": SCRIPT_VERSION,
        "root": args.validate_timings_root,
        "total_files": len(files),
        "repaired": sum(1 for x in results if x.get("status") == "repaired"),
        "unchanged": sum(1 for x in results if x.get("status") == "unchanged"),
        "failed": sum(1 for x in results if x.get("status") == "failed"),
        "items": results,
    }

    report_path = Path(args.repair_existing_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report_path, report, minify=False)

    print("")
    print("=" * 72)
    print("Existing timing in-place repair finished")
    print(f"Root: {args.validate_timings_root}")
    print(f"Files: {report['total_files']}")
    print(f"Repaired: {report['repaired']}")
    print(f"Unchanged: {report['unchanged']}")
    print(f"Failed: {report['failed']}")
    print(f"Report: {report_path}")
    print("=" * 72)

    return report



def validation_report_summary(items):
    return {
        "total_files": len(items),
        "ok": sum(1 for x in items if x["status"] == "ok"),
        "needs_review": sum(1 for x in items if x["status"] == "needs_review"),
        "needs_fix": sum(1 for x in items if x["status"] == "needs_fix"),
        "reciters": len(sorted(set(x.get("reciter") for x in items))),
    }


def validate_existing_timings(args):
    expected_counts = expected_words_by_chapter(args.quran_words)

    use_logs_for_validation = bool(args.validate_use_logs) and not bool(args.validate_structural_only)
    use_debug_quality = (not bool(args.validate_structural_only)) and bool(args.validate_use_debug_quality)

    parsed_logs = parse_alignment_logs(args.validate_log_root) if use_logs_for_validation else None

    if parsed_logs:
        print(f"Parsed logs from {args.validate_log_root}: {parsed_logs.get('logs_found', 0)} log file(s), {len(parsed_logs.get('by_output', {}))} output entries")
    elif args.validate_use_logs and args.validate_structural_only:
        print("Log parsing skipped because --validate-structural-only is enabled.")
    else:
        print("Log parsing disabled. Use --validate-use-logs only when logs are fresh and match these outputs.")

    reciters = None
    if getattr(args, "validate_reciter_list_url", None):
        reciters = fetch_reciter_folders_from_url(
            args.validate_reciter_list_url,
            insecure_ssl=getattr(args, "insecure_ssl", False),
        )
    elif args.validate_reciter_folders:
        reciters = [x.rstrip("/") for x in args.validate_reciter_folders]

    files = discover_timing_files(args.validate_timings_root, reciters=reciters)

    items = []
    for path in files:
        items.append(validate_one_timing_file(
            path,
            expected_counts=expected_counts,
            min_match=args.validate_min_match,
            max_estimated=args.validate_max_estimated,
            severe_match=getattr(args, "validate_severe_match", 0.90),
            min_word_ms=args.validate_min_word_ms,
            max_word_ms=args.validate_max_word_ms,
            require_debug=args.validate_require_debug,
            parsed_logs=parsed_logs,
            use_logs=use_logs_for_validation,
            use_debug_quality=use_debug_quality,
            debug_root=getattr(args, "validate_debug_root", None),
            respect_manual_lock=not getattr(args, "fix_override_manual", False),
            opening_estimate_forgive_buffer=getattr(args, "opening_estimate_forgive_buffer", 4),
        ))

    summary = validation_report_summary(items)
    report = {
        "script_version": SCRIPT_VERSION,
        "root": args.validate_timings_root,
        "thresholds": {
            "min_match": args.validate_min_match,
            "max_estimated": args.validate_max_estimated,
            "severe_match": getattr(args, "validate_severe_match", 0.90),
            "min_word_ms": args.validate_min_word_ms,
            "max_word_ms": args.validate_max_word_ms,
            "require_debug": args.validate_require_debug,
            "use_logs": use_logs_for_validation,
            "use_debug_quality": use_debug_quality,
            "structural_only": args.validate_structural_only,
            "log_root": args.validate_log_root,
            "debug_root": getattr(args, "validate_debug_root", None),
        },
        "log_parse": {
            "logs_found": parsed_logs.get("logs_found", 0) if parsed_logs else 0,
            "outputs_found": len(parsed_logs.get("by_output", {})) if parsed_logs else 0,
        },
        "summary": summary,
        "items": items,
    }

    report_path = Path(args.validate_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report_path, report, minify=False)

    fail_list_path = Path(args.validate_fail_list)
    fail_list_path.parent.mkdir(parents=True, exist_ok=True)

    with open(fail_list_path, "w", encoding="utf-8") as f:
        for item in items:
            if item.get("needs_fix"):
                f.write(item["file"] + "\n")

    review_list_path = Path(str(fail_list_path).replace(".txt", "_review.txt"))
    with open(review_list_path, "w", encoding="utf-8") as f:
        for item in items:
            if item.get("status") == "needs_review":
                f.write(item["file"] + "\n")

    print("")
    print("=" * 72)
    print("Existing timings validation finished")
    print(f"Root: {args.validate_timings_root}")
    print(f"Files: {summary['total_files']}")
    print(f"OK: {summary['ok']}")
    print(f"Needs review: {summary['needs_review']}")
    print(f"Needs fix: {summary['needs_fix']}")
    print(f"Report: {report_path}")
    print(f"Needs fix list: {fail_list_path}")
    print(f"Needs review list: {review_list_path}")
    print("=" * 72)

    return report


def backup_existing_timing_family(timing_path):
    timing_path = Path(timing_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = timing_path.parent / f"_bad_backup_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    stem = timing_path.stem
    moved = []

    for p in timing_path.parent.glob(stem + ".*"):
        if p.is_file():
            dst = backup_dir / p.name
            shutil.move(str(p), str(dst))
            moved.append(str(dst))

    return str(backup_dir), moved


def _enable_quality_validation_for_review_fix(args):
    """--fix-include-review signals the user wants EVERY low-quality surah
    re-aligned, not just structurally broken ones. Quality lives in two places
    and the defaults hide both:

      1. .debug.json per-word matched/estimated data (next to each timing, or
         under --validate-debug-root/<reciter>/ for clean batch outputs). It is
         only read in quality mode, but the default is structural-only
         (JSON/shape checks only), so low-match surahs stay 'ok'.
      2. logs/*.log generation records ("QUALITY WARNING: output needs review",
         match_rate, estimated, failed). They are only read with
         --validate-use-logs, which is off by default. This is exactly the
         signal the dashboard's "Review" count comes from.

    Without turning BOTH on, --fix-include-review re-aligns only the .debug.json
    failures and silently ignores the (usually larger) set of log-flagged review
    surahs. This forces both on so every flagged surah surfaces as needs_fix and
    gets re-aligned. Log-based checks assume the logs match the current outputs;
    omit --fix-include-review to skip them."""
    if not getattr(args, "fix_include_review", False):
        return
    enabled = []
    if getattr(args, "validate_structural_only", True):
        args.validate_structural_only = False
        enabled.append(".debug.json quality checks")
    if not getattr(args, "validate_use_logs", False):
        args.validate_use_logs = True
        enabled.append("logs/*.log quality checks")
    if enabled:
        print("Note: --fix-include-review enabled " + " and ".join(enabled) +
              " so review surahs (from .debug.json AND logs) are detected and "
              "re-aligned. Log checks assume logs match current outputs.")


def fix_existing_bad_timings(args, report=None):
    _enable_quality_validation_for_review_fix(args)
    if report is None:
        report_path = Path(args.validate_report)
        if report_path.exists():
            report, err = safe_load_json(report_path)
            if err:
                raise RuntimeError(f"Cannot read validation report: {err}")
        else:
            report = validate_existing_timings(args)

    bad_items = [x for x in report.get("items", []) if x.get("needs_fix")]

    if args.fix_include_review:
        bad_items.extend([x for x in report.get("items", []) if x.get("status") == "needs_review"])

    # Deduplicate by file.
    dedup = []
    seen = set()
    for item in bad_items:
        f = item.get("file")
        if f and f not in seen:
            seen.add(f)
            dedup.append(item)

    print("")
    print("=" * 72)
    print(f"Fixing existing bad timing files: {len(dedup)}")
    print("=" * 72)

    fixed = []
    failed = []

    base_url = str(args.batch_base_url).rstrip("/")

    for item in dedup:
        timing_path = Path(item["file"])
        reciter = item.get("reciter") or timing_path.parent.name
        chapter = item.get("chapter")

        if not chapter:
            try:
                chapter = int(timing_path.stem)
            except Exception:
                failed.append({"file": str(timing_path), "error": "cannot_determine_chapter"})
                continue

        if args.fix_backup_existing and timing_path.exists():
            backup_dir, moved = backup_existing_timing_family(timing_path)
            print(f"Backed up {timing_path.name} family to: {backup_dir}")

        child = argparse.Namespace(**vars(args))
        child.reciter_url = f"{base_url}/{reciter}"
        child.audio_pattern = derive_audio_pattern_from_reciter_url(child.reciter_url)
        child.all = False
        child.chapter = int(chapter)
        child.audio = None
        child.out = str(timing_path)
        child.out_dir = str(timing_path.parent)
        child.resume = False
        child.debug = True
        child.summary_file = None

        try:
            print("")
            print("-" * 72)
            print(f"Rebuilding: {reciter} chapter {chapter} -> {timing_path}")
            print("-" * 72)
            result = process_chapter(child, int(chapter))
            fixed.append({
                "file": str(timing_path),
                "reciter": reciter,
                "chapter": int(chapter),
                "result": result,
            })
        except Exception as e:
            failed.append({
                "file": str(timing_path),
                "reciter": reciter,
                "chapter": int(chapter),
                "error": str(e),
            })

    fix_report = {
        "fixed_count": len(fixed),
        "failed_count": len(failed),
        "fixed": fixed,
        "failed": failed,
    }

    fix_report_path = Path(args.fix_report)
    fix_report_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(fix_report_path, fix_report, minify=False)

    print("")
    print("=" * 72)
    print("Fix finished")
    print(f"Fixed: {len(fixed)}")
    print(f"Failed: {len(failed)}")
    print(f"Fix report: {fix_report_path}")
    print("=" * 72)

    # Re-validate after fixing so the report, needs_fix list AND the dashboard
    # reflect the repaired files. The initial validation report was written
    # BEFORE the re-alignment, so without this pass every fixed surah would stay
    # red and the needs_fix count would never drop, even though the underlying
    # timing (and its fresh local .debug.json/.alignment_quality.json) improved.
    # Re-alignment writes fresh debug side-files next to each timing, and the
    # validator prefers those local files over the stale batch debug_logs copies,
    # so this recomputes each status honestly (still-bad surahs stay flagged).
    if getattr(args, "fix_revalidate", True) and fixed:
        print("")
        print("=" * 72)
        print("Re-validating timings after fix (refreshes report, needs_fix list, dashboard)...")
        print("=" * 72)
        fix_report["revalidation"] = validate_existing_timings(args).get("summary")
        save_json(fix_report_path, fix_report, minify=False)

    if failed:
        raise SystemExit(1)

    return fix_report




def main():
    parser = argparse.ArgumentParser(
        description="Quran Timing Helper (qurani.io) — generate word-level Quran timing JSON from recitation MP3s using Whisper + wav2vec2 forced alignment."
    )

    parser.add_argument("--quran-words", default="quran_words.json")
    parser.add_argument("--download-quran-words-only", action="store_true", help="Only generate quran_words.json then exit.")
    parser.add_argument("--force-quran-words", action="store_true", help="Regenerate quran_words.json even if it exists.")
    parser.add_argument("--raw-kmaslesa", default="raw_kmaslesa_pages.json")
    parser.add_argument("--quran-words-mismatches", default="quran_words_mismatches.json")
    parser.add_argument("--quran-words-sleep", type=float, default=0.1)


    parser.add_argument("--chapter", type=int)
    parser.add_argument("--audio")

    parser.add_argument("--all", action="store_true")
    parser.add_argument("--audio-pattern")
    parser.add_argument("--reciter-url", help="Base reciter URL, pattern URL, or one MP3 URL like .../001.mp3. With no --chapter, it generates all 114.")
    parser.add_argument("--out-dir")
    parser.add_argument("--resume", action="store_true", help="Skip chapters whose output JSON already exists.")
    parser.add_argument("--summary-file", help="Write a JSON summary for all generated chapters.")

    parser.add_argument("--out", default="timing.json")

    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--model-file")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])

    parser.add_argument("--asr-external-chunking", action="store_true", help="v43: enable experimental external ASR chunking. Default is OFF.")
    parser.add_argument("--no-asr-external-chunking", action="store_true", help="v43: force whole-WAV ASR in one call. This is the default.")
    parser.add_argument("--asr-opening-chunk-s", type=float, default=10.0, help="v43 optional chunking: first opening chunk duration in seconds. Default: 10 seconds.")
    parser.add_argument("--asr-chunk-s", type=float, default=30.0, help="v43 optional chunking: body core chunk duration in seconds after the opening chunk. Default: 30 seconds.")
    parser.add_argument("--asr-chunk-boundary-mode", default="silence", choices=["silence", "fixed"], help="v43 optional chunking: choose body chunk boundaries by nearby silence or fixed duration.")
    parser.add_argument("--asr-chunk-overlap-s", type=float, default=1.5, help="v43 optional chunking: add this much left/right audio context to chunks, then trim overlap words.")
    parser.add_argument("--asr-chunk-boundary-search-ms", type=int, default=2500, help="v43 optional chunking: search window around target body chunk boundary for a silence.")
    parser.add_argument("--asr-chunk-min-silence-ms", type=int, default=180, help="v43 optional chunking: minimum silence length used for silence-aware chunk boundaries.")
    parser.add_argument("--asr-prompt-every-chunk", action="store_true", help="v43 optional chunking: pass Quran initial_prompt to every ASR chunk, not only the opening chunk.")
    parser.add_argument("--openai-fp16", action="store_true", dest="openai_fp16", help="Force fp16 for OpenAI Whisper. Usually only for CUDA.")
    parser.add_argument("--no-openai-fp16", action="store_false", dest="openai_fp16", help="Disable fp16 for OpenAI Whisper.")
    parser.set_defaults(openai_fp16=None)

    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--lookahead", type=int, default=15)
    parser.add_argument("--alignment", choices=["dp", "sequence"], default="dp")
    parser.add_argument("--complete-output", action="store_true", help="Force complete JSON output. This is enabled by default unless --strict is used.")
    parser.add_argument("--strict", action="store_true", help="Do not complete missing timings. Leave missing segments visible and allow non-complete JSON.")
    parser.add_argument("--warn-below", type=float, default=0.85)

    parser.add_argument("--interpolate-missing", action="store_true")

    parser.add_argument("--no-madd-fix", action="store_true", help="Disable v18 madd-only end extension.")
    parser.add_argument("--madd-mode", choices=["strong", "broad"], default="broad", help="strong = clear Uthmani madd signs only; broad = long-vowel letters too.")
    parser.add_argument("--madd-max-extend-ms", type=int, default=1800, help="Max end extension inside the same verse.")
    parser.add_argument("--madd-verse-end-max-extend-ms", type=int, default=3500, help="Max end extension for final word before next verse.")
    parser.add_argument("--madd-min-gap-ms", type=int, default=80)
    parser.add_argument("--madd-pause-min-ms", type=int, default=90)
    parser.add_argument("--no-madd-audio-pauses", action="store_true", help="Do not use audio silence to stop verse-ending madd.")

    parser.add_argument("--no-auto-retry-alignment", action="store_true", help="Disable v21 automatic alignment retry variants.")
    parser.add_argument("--no-strip-recitation-extras", action="store_true", help="Do not strip leading isti'adha/basmala or trailing outro from ASR before alignment.")
    parser.add_argument("--quality-min-match", type=float, default=0.995, help="Mark output as needs_review below this match rate.")
    parser.add_argument("--quality-max-estimated", type=int, default=0, help="Mark output as needs_review if estimated timings exceed this count.")
    parser.add_argument("--fail-on-low-quality", action="store_true", help="Raise an error instead of saving when quality is below threshold.")
    parser.add_argument("--no-recitation-anomaly-check", action="store_true", help="Disable the per-reciter recitation verification pass that flags repeated (re-recited) words and phrases/ayat. Default: enabled for every reciter; it NEVER changes timestamps and NEVER corrects text, it only reports repeats.")
    parser.add_argument("--fail-on-recitation-anomalies", action="store_true", help="Treat any detected repeated word or repeated phrase/ayah as needs_review (and a hard error together with --fail-on-low-quality).")
    parser.add_argument("--recitation-repeat-similarity", type=float, default=0.80, help="A single extra (unmatched) ASR token at least this similar to an adjacent matched word is flagged as a repeated word.")
    parser.add_argument("--recitation-phrase-similarity", type=float, default=0.80, help="A consecutive run of extra (unmatched) ASR tokens whose average per-position similarity to an adjacent matched run is at least this is flagged as a repeated phrase/ayah.")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", help="Write all debug side-files (*.debug.json, *.opening_*.json, etc.) to this folder instead of next to the timing JSON. In batch mode this is set automatically to <batch-debug-root>/<reciter>/.")
    parser.add_argument("--minify", action="store_true")

    parser.add_argument("--insecure-ssl", action="store_true")
    parser.add_argument("--ca-file")


    # Integrated batch runner args
    parser.add_argument("--batch-reciter-folders", nargs="*", help="Run specific reciter folders, e.g. reciter-one reciter-two.")
    parser.add_argument("--batch-reciter-file", help="Text file with one reciter folder per line.")
    parser.add_argument("--batch-reciter-list-url", help="Direct URL to a plain-text reciter folder list (one folder name per line, # comments allowed). The tool downloads the list and runs every folder against --batch-base-url.")
    parser.add_argument("--batch-base-url", default="https://qurani.io/quransound")
    parser.add_argument("--batch-out-root", default="timings", help="Root folder for clean timing output: <out-root>/<reciter>/001.json..114.json (no debug files).")
    parser.add_argument("--batch-log-root", default="logs")
    parser.add_argument("--batch-debug-root", default="debug_logs", help="Root folder for ALL debug/diagnostic files in batch mode: per-reciter logs, summary.json, and *.debug.json/*.opening_*.json side-files go under <debug-root>/<reciter>/. Keeps the timing output folders clean. Default: debug_logs.")


    # v22 existing generated JSON validation/fix args
    parser.add_argument("--validate-existing-timings", action="store_true", help="Validate already generated timings under timings/<reciter>/*.json.")
    parser.add_argument("--validate-timings-root", default="timings")
    parser.add_argument("--validate-report", default="logs/timings_validation_report.json")
    parser.add_argument("--validate-fail-list", default="logs/timings_needs_fix.txt")
    parser.add_argument("--validate-reciter-folders", nargs="*", help="Validate only these reciter folders under --validate-timings-root.")
    parser.add_argument("--validate-reciter-list-url", help="Direct URL to a plain-text reciter folder list; only these folders are validated.")
    parser.add_argument("--validate-min-match", type=float, default=0.9955)
    parser.add_argument("--validate-max-estimated", type=int, default=0)
    parser.add_argument("--validate-severe-match", type=float, default=0.90, help="v63: effective match rate below which a structurally-sound file is reported RED / needs_fix (severe under-transcription: most of the surah is interpolated, not really aligned) instead of mild YELLOW / needs_review. Cleanly separates genuinely-wrong surahs from borderline ones just under the ideal bar. Set 0 to disable and keep all quality shortfalls yellow. Default: 0.90.")
    parser.add_argument("--validate-min-word-ms", type=int, default=20)
    parser.add_argument("--validate-max-word-ms", type=int, default=12000)
    parser.add_argument("--validate-require-debug", action="store_true", help="Mark files without .debug.json as needs_fix instead of needs_review.")
    parser.add_argument("--validate-debug-root", default="debug_logs", help="Folder where batch mode wrote *.debug.json / *.alignment_quality.json side-files (<debug-root>/<reciter>/<chapter>.debug.json). The validator falls back here when they are not next to the timing JSON, so quality checks work on clean batch outputs. Default: debug_logs.")

    parser.add_argument("--fix-existing-bad", action="store_true", help="After validation, rerun files marked needs_fix.")
    parser.add_argument("--fix-include-review", action="store_true", help="Also rerun files marked needs_review.")
    parser.add_argument("--fix-backup-existing", action="store_true", default=True, help="Backup old file family before rebuilding.")
    parser.add_argument("--no-fix-backup-existing", action="store_false", dest="fix_backup_existing")
    parser.add_argument("--fix-report", default="logs/timings_fix_report.json")
    parser.add_argument("--no-fix-revalidate", action="store_false", dest="fix_revalidate", help="Do not re-validate after fixing. By default the tool re-validates the timings once the fix pass finishes so the report, needs_fix list, and dashboard reflect the newly repaired files (reds clear and the count drops).")
    parser.set_defaults(fix_revalidate=True)
    parser.add_argument("--fix-override-manual", action="store_true", default=False, help="Re-align surahs even if they were hand-corrected in the dashboard (i.e. have a <chapter>.manual_lock.json marker). By default such files are treated as ok/locked and skipped so re-runs never reproduce the old result or overwrite manual edits. Use this only when you deliberately want to discard manual fixes and let the model re-align them.")



    parser.add_argument("--validate-use-logs", action="store_true", default=False, help="Use logs/*.log quality data. Enable only when logs are fresh and match these outputs.")
    parser.add_argument("--no-validate-use-logs", action="store_false", dest="validate_use_logs")
    parser.add_argument("--validate-log-root", default="logs", help="Folder containing old batch logs like reciter-B_v19_all.log.")



    parser.add_argument("--no-char-projection-repair", action="store_true", help="Disable v24 char-stream repair for estimated words.")
    parser.add_argument("--char-projection-min-ratio", type=float, default=0.55)
    parser.add_argument("--char-projection-min-chars", type=int, default=2)
    parser.add_argument("--char-projection-min-word-ms", type=int, default=70)
    parser.add_argument("--char-projection-max-shift-ms", type=int, default=2500)



    parser.add_argument("--no-quran-initial-prompt", action="store_true", help="Disable Quran text initial_prompt for Whisper.")
    parser.add_argument("--initial-prompt-words", type=int, default=100)
    parser.add_argument("--no-initial-prompt-basmala", action="store_true", help="Do not include bismillah in Whisper initial_prompt.")
    parser.add_argument("--intro-false-match-penalty", type=int, default=1000000)
    parser.add_argument("--no-penalize-intro-false-matches", action="store_false", dest="penalize_intro_false_matches")
    parser.set_defaults(penalize_intro_false_matches=True)
    parser.add_argument("--no-intro-gap-repair", action="store_true", help="Disable redistribution of leading estimated words after basmala/intro.")
    parser.add_argument("--intro-gap-min-word-ms", type=int, default=90)



    parser.add_argument("--no-intro-gap-audio-pauses", action="store_true", help="Disable audio silence detection inside leading intro gap.")
    parser.add_argument("--intro-gap-pause-min-ms", type=int, default=140)



    parser.add_argument("--validate-structural-only", action="store_true", default=True, help="Default: validate JSON structure/timing only, do not fail files because debug/log quality data is missing.")
    parser.add_argument("--validate-quality", action="store_false", dest="validate_structural_only", help="Enable strict quality validation using debug files and optionally fresh logs.")
    parser.add_argument("--validate-use-debug-quality", action="store_true", default=True, help="When --validate-quality is used, read .debug.json quality data if present.")
    parser.add_argument("--no-validate-use-debug-quality", action="store_false", dest="validate_use_debug_quality")



    parser.add_argument("--no-opening-safe-mode", action="store_true", help="Disable v28 opening-safe mode for basmala/isti'adha/muqatta'at beginnings.")
    parser.add_argument("--opening-anchor-min-score", type=float, default=0.88)
    parser.add_argument("--initial-prompt-istiatha", dest="initial_prompt_istiatha", action="store_true", default=True, help="Add isti'adha to Whisper initial prompt for reciters that start with it. Default: ON (most reciters open with أعوذ بالله).")
    parser.add_argument("--no-initial-prompt-istiatha", dest="initial_prompt_istiatha", action="store_false", help="Do not add isti'adha to the Whisper initial prompt.")



    parser.add_argument("--repair-existing-timings", action="store_true", help="Repair existing final JSON in-place without rerunning Whisper.")
    parser.add_argument("--no-repair-use-debug", action="store_true", help="Do not rebuild final JSON from .debug.json during --repair-existing-timings.")
    parser.add_argument("--no-repair-backup-existing", action="store_true", help="Do not create .repair_backup files before in-place repair.")
    parser.add_argument("--repair-existing-report", default="logs/timings_repair_existing_report.json")



    parser.add_argument("--strict-intro-false-match-penalty", action="store_true", help="Use old strict penalty for original ASR intro false matches even in muqatta'at/opening-safe chapters.")
    parser.add_argument("--no-allow-opening-estimates", action="store_true", help="Do not accept opening-region estimates for quality scoring.")
    parser.add_argument("--no-opening-estimate-cap", action="store_true", help="v64: disable the opening-estimate intro budget cap and restore the old behaviour of forgiving EVERY opening-region estimate for quality scoring. By default only a plausible intro's worth of opening estimates (isti'adha + basmala + huruf muqatta'at + buffer) is forgiven; beyond that the excess is treated as genuine dropped recitation and counted against quality (QUALITY WARNING -> needs_review) instead of being hidden behind a fake effective_match_rate=100%%. Default: cap enabled.")
    parser.add_argument("--opening-estimate-forgive-buffer", type=int, default=4, help="v64: extra opening-region estimated words allowed on top of the isti'adha + basmala + huruf-muqatta'at intro word budget before the excess is counted against quality. Absorbs madd/anchor lag at the opening. Default: 4.")



    parser.add_argument("--opening-stable-anchor-words", type=int, default=5, help="v31: close opening only after this many consecutive reliable words.")
    parser.add_argument("--opening-stable-anchor-max-scan-words", type=int, default=80, help="v31: search this many initial words for the stable opening anchor.")
    parser.add_argument("--no-opening-absorb-early-estimates", action="store_true", help="v40: disable absorbing early estimated words into the opening island.")
    parser.add_argument("--no-opening-keep-real-timed", action="store_true", help="v56: disable keeping real (Whisper-measured) opening word timestamps; redistribute the whole leading island statically as before.")
    parser.add_argument("--no-deterministic-muqattaat-bracket", action="store_true", help="v57: disable the deterministic muqatta'at bracket. By default, for chapters known to open with huruf muqatta'at, the opening is anchored on the known muqatta'at word count and the muqatta'at is bracketed over [intro_end, first real word] WITHOUT verifying it acoustically. Disabling falls back to the v56 score-anchor clamp.")
    parser.add_argument("--opening-absorb-max-scan-words", type=int, default=80, help="v40: absorb estimated words inside this initial word window into the opening island.")



    parser.add_argument("--opening-muqattaat-madd-weight", type=int, default=7, help="v32: extra timing weight for opening huruf muqatta'at such as حم.")
    parser.add_argument("--opening-muqattaat-min-ms", type=int, default=900, help="v32: minimum first-span duration for opening muqatta'at when possible.")



    parser.add_argument("--opening-muqattaat-mode", choices=["auto", "off", "weighted"], default="auto", help="v33: auto trusts audio silences first; weighted forces extra duration; off disables extra muqatta'at weighting.")

    parser.add_argument("--no-muqattaat-asr-fold", action="store_true", help="v44: disable folding pronounced muqatta'at letter names (e.g. 'حا ميم') back to the Uthmani token before alignment. Default: enabled, and only ever runs on muqatta'at chapters.")
    parser.add_argument("--muqattaat-asr-fold-max-scan", type=int, default=25, help="v44: scan this many leading ASR tokens to locate and fold the opening muqatta'at letter names. Default: 25.")
    parser.add_argument("--no-opening-gap-recover", action="store_true", help="v46: disable re-transcription recovery of a dropped surah opening. By default, if Whisper leaves a large leading gap with ZERO tokens (e.g. it skipped حمٓ + the first verses), the gap clip is re-transcribed without the Quran initial_prompt to recover REAL word timestamps. Default: enabled.")
    parser.add_argument("--opening-gap-recover-min-ms", type=int, default=4000, help="v46: minimum leading inter-word ASR gap (ms) that triggers opening re-transcription recovery. Default: 4000.")
    parser.add_argument("--opening-gap-recover-scan-ms", type=int, default=90000, help="v46: only look for the dropped-opening gap within this many ms from the start. Default: 90000.")
    parser.add_argument("--opening-gap-recover-window-ms", type=int, default=30000, help="v46: max sub-window (ms) per re-transcription clip; longer gaps are split into overlapping windows so each stays within one Whisper decode window. Default: 30000.")
    parser.add_argument("--no-opening-leading-gap-recover", action="store_true", help="v47: disable LEADING-gap opening recovery. By default, when Whisper drops the muqatta'at recited from the very start (so its first transcribed word lands well after the opening, e.g. طسٓ in An-Naml with no transcribed basmala before it), the region [0..first_asr_word] is re-transcribed as its own clip (no Quran initial_prompt) to recover REAL opening timestamps instead of statically estimating them from 0ms. Generic across all muqatta'at surahs/reciters. Default: enabled.")
    parser.add_argument("--no-opening-onset-snap", action="store_true", help="v46: disable snapping the first recovered opening word's start to the real audio onset. By default, after recovery the first word's start is pulled back to the end of the last silence region before it, because Whisper's DTW clips the soft onset of a long madd (e.g. حمٓ). This uses a measured silence boundary (real audio), not static estimation. Default: enabled.")
    parser.add_argument("--opening-onset-snap-min-ms", type=int, default=200, help="v46: only snap the recovered opening onset when Whisper's start is at least this many ms later than the measured audio onset. Default: 200.")
    parser.add_argument("--opening-onset-snap-max-ms", type=int, default=4000, help="v46: never pull the recovered opening onset back by more than this many ms (safety bound against over-extension). Default: 4000.")

    parser.add_argument("--no-opening-real-timestamp-enforce", action="store_true", help="v47: disable opening REAL-timestamp enforcement. By default, after the opening-safe leading repair, any surah-opening word still STATICALLY estimated (because the stable anchor landed a few words in, e.g. طسٓ/تِلۡكَ/ءَايَٰتُ before ٱلۡقُرۡءَانِ in An-Naml) is re-mapped onto the REAL recovered ASR words already present in the opening recovery window, taking their measured Whisper timestamps (no re-transcription). Generic across all muqatta'at surahs/reciters. Default: enabled.")
    parser.add_argument("--opening-real-timestamp-enforce-min-score", type=float, default=0.6, help="v47: minimum fuzzy similarity to map a still-estimated opening word onto a recovered real ASR word during opening REAL-timestamp enforcement. Default: 0.6.")
    parser.add_argument("--opening-real-timestamp-enforce-max-scan", type=int, default=25, help="v47: scan at most this many leading words for the estimated opening island during opening REAL-timestamp enforcement. Default: 25.")
    parser.add_argument("--no-estimated-word-recover", action="store_true", help="v47: disable interior estimated-word recovery. By default, any mid-surah word the aligner had to ESTIMATE (interpolate/split) gets a targeted re-transcription of just the gap between its real-timed neighbours (no Quran initial_prompt); if Whisper surfaces the dropped word there, its REAL timestamp replaces the interpolation. Generic across all surahs/reciters; opening-region estimates are excluded (handled by opening gap recovery). Default: enabled.")
    parser.add_argument("--no-underrun-recover", action="store_true", help="v60: disable severe under-transcription recovery. By default, when the prompted Whisper pass covers far fewer words than the surah has (e.g. a short surah transcribed as 5 of 14 words, or a long surah missing a large fraction), the WHOLE clip is re-transcribed WITHOUT the Quran initial_prompt and whichever transcription ALIGNS more real Quran words is kept (a longer-but-hallucinated transcript cannot win). Generic across all surahs/reciters; the second pass runs only on the poor-coverage minority. Default: enabled.")
    parser.add_argument("--underrun-recover-min-coverage", type=float, default=0.85, help="v60: coverage threshold (ASR words / Quran words) below which the no-prompt under-transcription recovery pass is triggered. Default: 0.85.")
    parser.add_argument("--estimated-word-recover-pad-ms", type=int, default=600, help="v47: outward acoustic-context padding (ms) added on each side of the interior gap clip before re-transcription; recovered words are still clamped to the gap between real neighbours. Default: 600.")
    parser.add_argument("--estimated-word-recover-max-window-ms", type=int, default=12000, help="v47: skip interior gaps wider than this (ms) — too wide to be a single dropped word, so the safe interpolation is kept. Default: 12000.")
    parser.add_argument("--estimated-word-recover-min-score", type=float, default=0.62, help="v47: minimum fuzzy similarity for a re-transcribed gap word to be accepted as the missing Quran word. Default: 0.62.")
    parser.add_argument("--estimated-word-recover-min-word-ms", type=int, default=80, help="v47: minimum duration (ms) assigned to a recovered interior word. Default: 80.")
    parser.add_argument("--estimated-word-recover-max-runs", type=int, default=60, help="v47: safety cap on how many interior estimated-word gaps to re-transcribe per chapter. Default: 60.")
    parser.add_argument("--no-ctc-word-recover", action="store_true", help="v59: disable CTC forced-alignment recovery of interior estimated words. By default, any mid-surah word still ESTIMATED after Whisper recovery is recovered by force-aligning its KNOWN Quran text against the gap audio with a char-level wav2vec2 CTC model — forced alignment never drops a word, so merged/short function words (أَمۡ, هُوَ, إِلَّا) get REAL measured spans. Generic across all surahs/reciters; opening-region estimates are excluded; unmappable words keep their safe interpolation. Needs torchaudio>=2.1 + transformers (degrades gracefully with a hint if missing). Default: enabled.")
    parser.add_argument("--ctc-model", default="jonatasgrosman/wav2vec2-large-xlsr-53-arabic", help="v59: Hugging Face wav2vec2 CTC model (native Arabic char vocab) used for interior forced-alignment recovery. Default: jonatasgrosman/wav2vec2-large-xlsr-53-arabic.")
    parser.add_argument("--ctc-recover-min-score", type=float, default=0.0, help="v81: minimum average CTC acoustic score for a force-aligned interior word to replace its interpolation. Default dropped from 0.10 to 0.0 (matching the underrun full-surah CTC path): interior gaps are bracketed by two REAL anchors and force-align KNOWN correct text, so slow-mujawwad madd words with low emission scores are still positionally sound; a per-gap degenerate guard reverts low-score words to safe interpolation when the run collapses to minimum-width spans. Default: 0.0.")
    parser.add_argument("--opening-ctc-prefix-hypothesis-margin", type=float, default=0.02, help="v83: a SHORTER intro-prefix hypothesis (basmala-only / no intro) must beat the longer one's mean opening score by at least this margin to be selected; on a near-tie the longer, more conservative prefix wins. Default: 0.02.")
    parser.add_argument("--opening-ctc-boundary-tol-ms", type=int, default=200, help="v84: tolerance (ms) for detecting a window-end squeeze in the opening CTC pass — if any hypothesis places its last opening word within this distance of the window end AND the strict gate rejects, the first-real anchor bounding the window is flagged suspect and the full-surah CTC escalation fires. Default: 200.")
    parser.add_argument("--opening-ctc-credible-prefix-score", type=float, default=0.5, help="v84: a window-end squeeze only counts as anchor-suspect evidence when it comes from a CREDIBLE hypothesis: the one finally selected as best, or one whose intro PREFIX force-aligned with at least this mean acoustic score (positive evidence that prefix matches the actual recited intro). A random losing hypothesis touching the boundary never triggers escalation. Default: 0.5.")
    parser.add_argument("--underrun-ctc-force-min-mean-score", type=float, default=0.15, help="v84: when the full-surah CTC pass runs as an ESCALATION (opening window squeeze) rather than via the catastrophic bad-fraction trigger, require the whole-surah forced alignment to place every word with at least this MEAN acoustic score before adopting anything. Default: 0.15.")
    parser.add_argument("--opening-ctc-small-lead-min-score", type=float, default=0.05, help="v81: strict per-word score floor for the opening CTC pass when the leading estimated run is within the intro budget (non-intro-vocab small lead). Unlike the >budget case there is no positive evidence the words exist in the audio, so the alignment is adopted only if ALL words place with at least this confidence and pass the degenerate-shape checks; otherwise safe interpolations are kept. Default: 0.05.")
    parser.add_argument("--ctc-recover-pad-ms", type=int, default=150, help="v59: small symmetric acoustic-context padding (ms) added around the interior gap before CTC alignment; recovered spans are still clamped to the gap between real neighbours. Default: 150.")
    parser.add_argument("--ctc-recover-max-window-ms", type=int, default=15000, help="v59: interior gaps wider than this (ms) are aligned by CHUNKED CTC (see --ctc-recover-chunk-ms) instead of a single pass; a gap at or below this width aligns in one pass. Default: 15000.")
    parser.add_argument("--ctc-recover-chunk-ms", type=int, default=45000, help="v63: when an interior estimated-word gap bracketed by two REAL anchor timestamps is wider than --ctc-recover-max-window-ms, it is no longer skipped: it is split into sequential sub-windows of at most this many ms (using the interpolated word positions as cut points) and each chunk is force-aligned. Recovers the large under-transcribed stretches Whisper drops on long surahs while keeping every CTC forward pass memory-safe. Generic; healthy chapters are untouched; words a chunk cannot confidently place keep their safe interpolation. Default: 45000.")
    parser.add_argument("--ctc-recover-min-word-ms", type=int, default=80, help="v59: minimum duration (ms) assigned to a CTC-recovered interior word. Default: 80.")
    parser.add_argument("--ctc-recover-max-runs", type=int, default=200, help="v59: safety cap on how many interior estimated-word gaps to CTC-align per chapter (raised in v63 so heavily fragmented under-transcribed surahs are fully covered). Default: 200.")
    parser.add_argument("--no-dup-ayat-ctc", action="store_true", help="v67: disable duplicate consecutive-ayah CTC realignment. By default, when two near-identical adjacent ayat (e.g. Ash-Sharh 94:5 فَإِنَّ مَعَ ٱلۡعُسۡرِ يُسۡرًا / 94:6 إِنَّ مَعَ ٱلۡعُسۡرِ يُسۡرًا) are collapsed by Whisper into a single recitation, the DP tie spreads that one copy across BOTH ayat and one whole copy is left ESTIMATED with fake timestamps that slide onto the next ayah. This pass detects such regions from the KNOWN Quran text (strictly consecutive ayat, identical up to a leading conjunction) and force-aligns the FULL region against the audio between its two OUTER real anchors, where every copy is present. It adopts the re-alignment only when forced alignment confidently placed every region word. Generic across all surahs/reciters; fires only when a duplicate-ayah region still holds an estimated word; degrades gracefully if CTC deps are missing. Default: enabled.")
    parser.add_argument("--no-nonadjacent-dup-ayat-ctc", action="store_true", help="v74: disable NON-adjacent duplicate-ayah CTC realignment. By default, two ayat with identical recited text separated by a small number of DIFFERENT intervening ayat (e.g. Al-Kafirun 109:3 وَلَآ أَنتُمۡ عَٰبِدُونَ مَآ أَعۡبُدُ == 109:5, with 109:4 between them) — which Whisper collapses and drops one copy of, exactly like the adjacent case but invisible to the adjacent detector — are force-aligned as one span (first copy..last copy, grown outward across any contiguous interior estimates hugging it) between the span's two OUTER real anchors, adopting the re-alignment only when forced alignment placed EVERY word. Detection is from the KNOWN Quran text (strictly consecutive real verse keys, identical up to a leading conjunction), so a whole-surah refrain (e.g. Ar-Rahman) never forms a giant span. Generic across all surahs/reciters; fires only when such a span still holds an estimated word; degrades gracefully if CTC deps are missing. Default: enabled.")
    parser.add_argument("--nonadjacent-dup-max-intervening-ayat", type=int, default=1, help="v74: maximum number of DIFFERENT ayat allowed between the two identical copies for the non-adjacent duplicate-ayah pass (Al-Kafirun 109:3/109:5 have exactly 1). Kept small so a whole-surah refrain never forms a giant span. Default: 1.")
    parser.add_argument("--nonadjacent-dup-max-span-words", type=int, default=40, help="v74: skip a non-adjacent duplicate-ayah span wider than this many words (safety cap so only compact repeated-ayah regions are force-aligned as a whole). Default: 40.")
    parser.add_argument("--no-underrun-ctc", action="store_true", help="v61: disable the full-surah CTC fallback for severely under-transcribed clips. By default, when Whisper structurally drops most of a surah (so the majority of words are still MISSING/ESTIMATED after every other repair), the KNOWN whole-surah text is force-aligned against the audio with the wav2vec2 CTC model — forced alignment never drops a token, so every word (openings, madd, function words) gets a REAL measured span. A standard isti'adha/basmala prefix absorbs the leading intro. Generic across all surahs/reciters; healthy chapters are untouched; degrades gracefully if CTC deps are missing. Default: enabled.")
    parser.add_argument("--no-opening-ctc", action="store_true", help="v66: disable the opening-scoped CTC recovery. By default, a LONG surah that aligns fine everywhere EXCEPT a dropped/mis-timed OPENING (first verse or two) — which the full-surah CTC never reaches because the clip is neither >=50%% bad nor short enough — has just its opening window force-aligned with the wav2vec2 CTC model (isti'adha/basmala prefix + the leading estimated surah words, against 0..first-real-word). Fires only when the leading estimated run exceeds a plausible intro budget (isti'adha + basmala + muqatta'at + buffer), i.e. genuine dropped opening recitation. Generic; healthy openings untouched; degrades gracefully if CTC deps are missing. Default: enabled.")
    parser.add_argument("--opening-ctc-pad-ms", type=int, default=1500, help="v66: pad (ms) added after the first real word's start when sizing the opening CTC audio window, so a word straddling that boundary is not clipped. Default: 1500.")
    parser.add_argument("--opening-ctc-inflated-intro-ms", type=int, default=12000, help="v68/v75: intro span (ms) at/above which the stripped basmala/isti'adha block is treated as INFLATED by Whisper (a long post-basmala pause stretches the intro toward ~30000ms; or, v75, the basmala and the entire first ayah get merged into one ~14s block). A real isti'adha+basmala never approaches this, and the anchor+demote guards (not the duration) prevent false fires, so above it any surah word mis-aligned inside the intro is re-aligned by the opening CTC. Default: 12000.")
    parser.add_argument("--opening-ctc-intro-overlap-tol-ms", type=int, default=200, help="v68: tolerance (ms) subtracted from the inflated intro end when deciding whether an opening surah word is mis-aligned inside the intro (start < intro_end - tol) and when locating the post-intro resume anchor. Default: 200.")
    parser.add_argument("--opening-ctc-anchor-run", type=int, default=3, help="v68: number of consecutive real, monotonic, at/after-intro-end words required to accept a post-intro resume anchor for the inflated-intro guard, so a lone fluke timestamp is not used as the re-alignment boundary. Default: 3.")
    parser.add_argument("--opening-ctc-intro-scan-limit", type=int, default=120, help="v68: how many leading words to scan for the post-intro resume anchor in the inflated-intro guard. Raised from 60 to 120 so a genuine anchor that appears late (very fast recitation or unusual tokenization of a long opening) is not missed, which would leave the artifact unrepaired. Default: 120.")
    parser.add_argument("--underrun-ctc-min-bad-fraction", type=float, default=0.5, help="v61: trigger the full-surah CTC fallback only when at least this fraction of surah words still lack a real timestamp (MISSING or ESTIMATED) after all other repairs. High by default so only catastrophic under-transcription triggers it. Default: 0.5.")
    parser.add_argument("--underrun-ctc-max-duration-ms", type=int, default=120000, help="v61: skip the full-surah CTC fallback for clips longer than this (ms) — too long to force-align in one CTC pass without excessive memory. Default: 120000 (2 min).")
    parser.add_argument("--underrun-ctc-min-score", type=float, default=0.0, help="v61: minimum average CTC acoustic score for a full-surah force-aligned word span to be accepted; below this the word keeps its prior (estimated/missing) state. This pass fires ONLY on catastrophic under-transcription where the text is KNOWN and forced alignment never drops a token, so the default is 0.0 (accept every forced-aligned span, bounded by the monotonic clamp + min-word-ms) to guarantee full coverage. Raise it to reject low-confidence spans. Default: 0.0.")
    parser.add_argument("--underrun-ctc-min-word-ms", type=int, default=80, help="v61: minimum duration (ms) assigned to a full-surah CTC-recovered word span. Default: 80.")



    parser.add_argument("--asr-backend", choices=["openai-whisper", "hf-transformers"], default="openai-whisper", help="ASR backend. Use hf-transformers for Quran-specific Hugging Face Whisper models.")
    parser.add_argument("--hf-model", default="IJyad/whisper-large-v3-Tarteel", help="Hugging Face ASR model id for --asr-backend hf-transformers.")
    parser.add_argument("--hf-chunk-length-s", type=int, default=30, help="HF ASR chunk length in seconds.")
    parser.add_argument("--hf-stride-length-s", type=int, default=5, help="HF ASR stride length in seconds.")
    parser.add_argument("--hf-torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--hf-trust-remote-code", action="store_true")



    parser.add_argument("--no-hf-fix-timestamp-config", action="store_true", help="Disable v36 auto patch for missing Whisper no_timestamps_token_id in Hugging Face checkpoints.")



    parser.add_argument("--hf-pass-language-kwargs", action="store_true", help="Pass language/task to HF generate(). Default off because some Quran checkpoints have outdated generation_config.")



    parser.add_argument("--no-hf-fix-alignment-heads", action="store_true", help="Disable v38 auto patch for missing Whisper alignment_heads in Hugging Face checkpoints.")
    parser.add_argument("--hf-alignment-heads-from", default="auto", help="Source model to copy alignment_heads from, e.g. openai/whisper-large-v3-turbo, or auto.")


    args = parser.parse_args()

    global DEBUG_SCREEN
    DEBUG_SCREEN = bool(getattr(args, "debug", False))

    print(f"{TOOL_NAME} — {TOOL_SITE}")
    print(f"Script version: {SCRIPT_VERSION}")

    if getattr(args, "asr_backend", "openai-whisper") == "hf-transformers":
        print(f"ASR backend: hf-transformers")
        print(f"HF model: {args.hf_model}")
        if args.device == "mps":
            print("Mac MPS note: HF transformers is supported, but first run downloads the model and may be slower than openai-whisper.")
    else:
        print("ASR backend: openai-whisper")

    # Complete JSON is the safe default for app output:
    # no [position] segments in final files unless user explicitly asks for --strict.
    if args.strict:
        args.complete_output = False
    else:
        args.complete_output = True

    if args.reciter_url:
        args.audio_pattern = derive_audio_pattern_from_reciter_url(args.reciter_url)

        # If user gives a reciter link without --chapter, generate all 114 automatically.
        if not args.chapter and not args.audio:
            args.all = True

        if not args.out_dir:
            args.out_dir = str(Path("timings") / guess_reciter_name(args.reciter_url))

    if args.all and not args.summary_file:
        args.summary_file = str(Path(args.out_dir) / "summary.json") if args.out_dir else "summary.json"

    configure_ssl(use_certifi=True, insecure_ssl=args.insecure_ssl, ca_file=args.ca_file)

    args.quran_words = ensure_quran_words_file(args)

    if args.download_quran_words_only:
        print("quran_words generation done.")
        return

    if args.repair_existing_timings:
        repair_existing_timings_in_place(args)
        return

    if args.validate_existing_timings and not args.fix_existing_bad:
        validate_existing_timings(args)
        return

    require_binary("ffmpeg")
    require_binary("ffprobe")

    args.device_resolved = pick_device(args.device)
    print(f"Using device: {args.device_resolved}")

    if args.validate_existing_timings and args.fix_existing_bad:
        _enable_quality_validation_for_review_fix(args)
        report = validate_existing_timings(args)
        fix_existing_bad_timings(args, report=report)
        return

    if args.fix_existing_bad and not args.validate_existing_timings:
        # Targeted fix: re-align ONLY the files already flagged in the saved
        # validation report (needs_fix, plus needs_review when
        # --fix-include-review) WITHOUT first re-validating the whole tree.
        # fix_existing_bad_timings(report=None) reads args.validate_report from
        # disk. Pair with --no-fix-revalidate to also skip the post-fix full
        # re-scan, so nothing outside the flagged worklist is ever touched.
        fix_existing_bad_timings(args)
        return

    if args.batch_reciter_list_url or args.batch_reciter_folders or args.batch_reciter_file:
        run_batch_reciters(args)
        return

    if args.all:
        if not args.audio_pattern or not args.out_dir:
            raise SystemExit("--all requires --audio-pattern or --reciter-url, and --out-dir")

        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

        summary = {
            "audio_pattern": args.audio_pattern,
            "out_dir": args.out_dir,
            "model": args.model,
            "asr_backend": args.asr_backend,
            "hf_model": getattr(args, "hf_model", None),
            "device": args.device_resolved,
            "alignment": args.alignment,
            "complete_output": args.complete_output,
            "chapters": [],
            "failed": [],
        }

        for chapter_id in range(1, 115):
            try:
                item = process_chapter(args, chapter_id)
                summary["chapters"].append(item)
            except Exception as e:
                print(f"FAILED chapter {chapter_id}: {e}")
                summary["failed"].append({
                    "chapter_id": chapter_id,
                    "error": str(e),
                    "status": "failed",
                })

        summary["ok_count"] = sum(1 for x in summary["chapters"] if x and x.get("status") == "ok")
        summary["skipped_count"] = sum(1 for x in summary["chapters"] if x and x.get("status") == "skipped_existing")
        summary["failed_count"] = len(summary["failed"])
        summary["total_estimated_timings"] = sum(int(x.get("estimated_timings", 0)) for x in summary["chapters"] if x)
        summary["total_final_missing_segments"] = sum(int(x.get("final_missing_segments", 0)) for x in summary["chapters"] if x)
        summary["total_recitation_repeated_words"] = sum(int((x.get("recitation_anomalies") or {}).get("repeated_words", 0)) for x in summary["chapters"] if x)
        summary["total_recitation_repeated_phrases"] = sum(int((x.get("recitation_anomalies") or {}).get("repeated_phrases", 0)) for x in summary["chapters"] if x)
        summary["total_recitation_repeated"] = sum(int((x.get("recitation_anomalies") or {}).get("repeated", 0)) for x in summary["chapters"] if x)
        summary["chapters_with_recitation_anomalies"] = [
            x.get("chapter_id") for x in summary["chapters"]
            if x and int((x.get("recitation_anomalies") or {}).get("total", 0))
        ]

        if args.summary_file:
            Path(args.summary_file).parent.mkdir(parents=True, exist_ok=True)
            save_json(args.summary_file, summary, minify=False)
            print("")
            print("=" * 72)
            print(f"Summary saved: {args.summary_file}")
            print(f"OK: {summary['ok_count']}")
            print(f"Skipped: {summary['skipped_count']}")
            print(f"Failed: {summary['failed_count']}")
            print(f"Total estimated timings: {summary['total_estimated_timings']}")
            print(f"Total final missing segments: {summary['total_final_missing_segments']}")

        if summary["failed_count"]:
            raise SystemExit(1)

    else:
        if args.reciter_url and args.chapter and not args.audio:
            args.audio = args.audio_pattern.replace("{chapter}", pad3(args.chapter))

        if not args.chapter or not args.audio:
            raise SystemExit("Single mode requires --chapter and --audio, or --chapter with --reciter-url")

        process_chapter(args, args.chapter)


if __name__ == "__main__":
    main()
