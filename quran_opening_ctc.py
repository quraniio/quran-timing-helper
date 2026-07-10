#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone CTC forced-alignment for the OPENING of any surah.

Why this exists
---------------
Whisper is an ASR model: it transcribes, then derives word timestamps by DTW.
For huruf muqatta'at (e.g. الٓر, كهيعص, الٓمٓ) Whisper tokenizes the fused written
form as ONE token, but the audio contains the spoken letter NAMES recited
separately ("ألف لام راء"). The single token cannot be mapped onto the several
spoken units, so the muqatta'at gets dropped / zero-duration and the opening
drifts.

Forced alignment is the right tool because we already KNOW the text (the Quran).
We feed a CTC acoustic model (wav2vec2 / native Arabic vocab) the EXACT text and
constrain-decode it against the audio. The single trick that makes muqatta'at
work: expand each muqatta'at word into its spoken letter names BEFORE aligning,
then merge the per-letter spans back into the one muqatta'at word.

This is GENERIC: it works for every surah and every reciter. The muqatta'at
table and letter-name map come from quran_forced_align_full.py (Quran facts,
not per-reciter hardcoding). Non-muqatta'at surahs simply align their opening
words directly.

This tool is intentionally standalone so the approach can be validated in
isolation (on real audio, on your Mac) BEFORE it is wired into the main twin
scripts.

Usage
-----
    python quran_opening_ctc.py \
        --chapter 10 \
        --audio "https://qurani.io/quransound/reciter-A/010.mp3" \
        --quran-words quran_words.json \
        --out opening_ctc/reciter-A_010.json

Optional:
    --intro-end-ms 5880     # skip recited isti'adha/basmala lead-in
    --window-sec 120        # how many seconds from window start to align
    --opening-words 14      # how many leading Quran words to align
    --model jonatasgrosman/wav2vec2-large-xlsr-53-arabic
    --device auto|mps|cuda|cpu

Requirements:
    Run this in the SAME environment as the main pipeline (quran_forced_align_full.py),
    which already provides torch + whisper. This tool additionally needs
    torchaudio (>=2.1) and transformers for the CTC step:

        pip install "torchaudio>=2.1" transformers
        # torch + whisper are already present if the main pipeline runs here

    ffmpeg available on PATH.

Note: this tool deliberately reuses quran_forced_align_full.py as the single
source of truth for Quran facts (muqatta'at table, letter names, normalize,
Quran-words/audio loaders) to avoid data drift. That module imports whisper at
load time, so whisper must be installed (it always is in the pipeline env).
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the single source of truth for Quran facts + audio/text helpers.
# This module imports torch + whisper at load time (always present in the main
# pipeline env); fail with a clear, actionable message if they are missing.
try:
    import quran_forced_align_full as q  # noqa: E402
except ImportError as _exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Could not import quran_forced_align_full (the shared Quran-facts module).\n"
        "Run this tool in the same environment as the main pipeline, where torch "
        "and whisper are installed:\n"
        "    pip install torch whisper torchaudio transformers\n"
        f"Import error: {_exc}"
    )


# ---------------------------------------------------------------------------
# Pure logic (no torch / no audio) -- unit-testable.
# ---------------------------------------------------------------------------

def expand_muqattaat_word_to_letter_names(word_text):
    """
    Expand a single muqatta'at Quran word into its spoken letter names.

    "الٓر"  -> ["الف", "لام", "راء"]
    "كهيعص" -> ["كاف", "هاء", "ياء", "عين", "صاد"]
    "حمٓ"   -> ["حاء", "ميم"]

    Uses the canonical (first) variant from MUQATTAAT_LETTER_NAMES. Falls back
    to the bare letter if a name is unknown (should not happen for the 14
    huruf muqatta'at).
    """
    norm = q.normalize_arabic(word_text or "")
    names = []
    for letter in list(norm):
        variants = q.MUQATTAAT_LETTER_NAMES.get(letter)
        name = variants[0] if variants else letter
        names.append(q.normalize_arabic(name))
    return names


def build_opening_transcript(quran_words, chapter_id, opening_words):
    """
    Build the alignment transcript for the opening region.

    Returns a list of entries, each:
        {
          "text":        normalized text to align (a letter name OR a word),
          "quran_index": index into quran_words this entry belongs to,
          "letter":      the muqatta'at letter (or None for normal words),
          "is_muqattaat": bool,
        }

    Muqatta'at words expand to one entry per spoken letter name; all those
    entries share the same quran_index so their spans can be merged back into
    the single Quran word afterwards.
    """
    transcript = []
    n = min(int(opening_words), len(quran_words))
    for i in range(n):
        w = quran_words[i]
        text = w.get("text", "")
        if q.is_opening_muqattaat_word(text, chapter_id=chapter_id, word_index=i):
            for letter, name in zip(
                list(q.normalize_arabic(text)),
                expand_muqattaat_word_to_letter_names(text),
            ):
                if not name:
                    continue
                transcript.append({
                    "text": name,
                    "quran_index": i,
                    "letter": letter,
                    "is_muqattaat": True,
                })
        else:
            norm = q.normalize_arabic(text)
            if norm:
                transcript.append({
                    "text": norm,
                    "quran_index": i,
                    "letter": None,
                    "is_muqattaat": False,
                })
    return transcript


def merge_spans_to_quran(transcript, spans, quran_words):
    """
    Merge per-transcript-entry spans back to one row per Quran word.

    `spans[i]` corresponds to `transcript[i]` and is either None (no alignment)
    or a dict with real {"start_ms", "end_ms", "score"}.

    For a muqatta'at word the merged span covers [min start, max end] across its
    letter spans -- i.e. the real audio span of the whole recited muqatta'at.
    Per-letter detail is kept under "letters" for inspection / debugging.
    """
    by_index = {}
    for entry, span in zip(transcript, spans):
        qi = entry["quran_index"]
        slot = by_index.setdefault(qi, {
            "quran_index": qi,
            "verse_key": quran_words[qi].get("verse_key"),
            "position": quran_words[qi].get("position"),
            "text": quran_words[qi].get("text"),
            "is_muqattaat": entry["is_muqattaat"],
            "start_ms": None,
            "end_ms": None,
            "scores": [],
            "letters": [],
            "aligned": False,
        })
        if entry["is_muqattaat"]:
            slot["letters"].append({
                "letter": entry["letter"],
                "name": entry["text"],
                "start_ms": (span or {}).get("start_ms"),
                "end_ms": (span or {}).get("end_ms"),
                "score": (span or {}).get("score"),
            })
        if not span:
            continue
        slot["aligned"] = True
        s, e = span.get("start_ms"), span.get("end_ms")
        if s is not None:
            slot["start_ms"] = s if slot["start_ms"] is None else min(slot["start_ms"], s)
        if e is not None:
            slot["end_ms"] = e if slot["end_ms"] is None else max(slot["end_ms"], e)
        if span.get("score") is not None:
            slot["scores"].append(span["score"])

    out = []
    for qi in sorted(by_index):
        slot = by_index[qi]
        scores = slot.pop("scores")
        slot["score"] = round(sum(scores) / len(scores), 4) if scores else None
        if not slot["letters"]:
            slot.pop("letters")
        out.append(slot)
    return out


# ---------------------------------------------------------------------------
# CTC runtime (torch / torchaudio / transformers) -- lazy imported.
# ---------------------------------------------------------------------------

def ctc_align(wav_path, transcript_texts, device, model_name,
              window_start_ms=0, window_sec=None, sample_rate=16000):
    """
    Force-align `transcript_texts` (a list of normalized strings) against a
    window of `wav_path`. Returns a list (1:1 with transcript_texts) of either
    None or {"start_ms", "end_ms", "score"} with REAL measured timestamps.

    Delegates to quran_forced_align_full.ctc_align_window, the single source of
    truth for CTC forced alignment, so the opening tool and the in-pipeline
    interior recovery can never drift apart.
    """
    try:
        return q.ctc_align_window(
            wav_path=wav_path,
            transcript_texts=transcript_texts,
            device=device,
            model_name=model_name,
            window_start_ms=window_start_ms,
            window_sec=window_sec,
            sample_rate=sample_rate,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "CTC alignment needs torch>=2.1, torchaudio>=2.1 and transformers.\n"
            "Install:  pip install \"torch>=2.1\" \"torchaudio>=2.1\" transformers\n"
            f"Import error: {exc}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(args):
    quran_words_path = Path(args.quran_words)
    if not quran_words_path.exists():
        raise SystemExit(
            f"Quran words file not found: {quran_words_path}\n"
            "Run the main pipeline once (it generates quran_words.json) or pass "
            "--quran-words pointing at an existing one."
        )

    quran_words = q.load_quran_words(str(quran_words_path), args.chapter)
    is_muq = q.is_muqattaat_chapter(args.chapter)
    transcript = build_opening_transcript(quran_words, args.chapter, args.opening_words)
    if not transcript:
        raise SystemExit("Empty opening transcript -- nothing to align.")

    print(f"Chapter {args.chapter} | muqatta'at chapter: {is_muq} "
          f"| opening words: {args.opening_words} | transcript tokens: {len(transcript)}")
    print("Transcript:", " ".join(e["text"] for e in transcript))

    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "audio.input"
        wav = Path(td) / "audio.wav"
        q.copy_or_download_audio(args.audio, str(raw), insecure_ssl=args.insecure_ssl)
        q.convert_to_wav(str(raw), str(wav))

        device = q.pick_device(args.device)
        print(f"Device: {device} | model: {args.model}")
        spans = ctc_align(
            str(wav),
            [e["text"] for e in transcript],
            device=device,
            model_name=args.model,
            window_start_ms=args.intro_end_ms,
            window_sec=args.window_sec,
        )

    merged = merge_spans_to_quran(transcript, spans, quran_words)

    print("\nOpening word timings (REAL, from CTC forced alignment):")
    for row in merged:
        flag = "MUQ" if row.get("is_muqattaat") else "   "
        print(f"  [{flag}] {row['verse_key']:>6}  {row['text']:<12} "
              f"start={row['start_ms']}  end={row['end_ms']}  score={row['score']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chapter_id": args.chapter,
            "model": args.model,
            "intro_end_ms": args.intro_end_ms,
            "is_muqattaat_chapter": is_muq,
            "opening_words": merged,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {out_path}")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Standalone CTC forced alignment for the opening of any surah."
    )
    parser.add_argument("--chapter", type=int, required=True)
    parser.add_argument("--audio", required=True, help="Audio file path or URL.")
    parser.add_argument("--quran-words", default="quran_words.json")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    parser.add_argument("--opening-words", type=int, default=14,
                        help="How many leading Quran words to align.")
    parser.add_argument("--intro-end-ms", type=int, default=0,
                        help="Skip recited isti'adha/basmala lead-in (window start).")
    parser.add_argument("--window-sec", type=float, default=120.0,
                        help="Seconds from window start to feed the aligner.")
    parser.add_argument("--model", default="jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
                        help="HF wav2vec2 CTC model with a native Arabic vocab.")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--insecure-ssl", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
