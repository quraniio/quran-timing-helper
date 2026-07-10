#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the pure logic of quran_opening_ctc (no torch / no audio)."""

import sys
import types
from pathlib import Path

# Stub heavy optional deps so the modules import in a plain Python 3.x env.
for _name in ("torch", "whisper", "requests", "numpy"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

sys.path.insert(0, str(Path(__file__).resolve().parent))

import quran_opening_ctc as oc  # noqa: E402


def _qw(verse_key, position, text):
    return {"verse_key": verse_key, "position": position, "text": text,
            "norm": __import__("quran_forced_align_full").normalize_arabic(text)}


def test_expand_alr():
    # الٓر -> alif laam raa
    assert oc.expand_muqattaat_word_to_letter_names("الٓر") == ["الف", "لام", "راء"]


def test_expand_kahayaaynsad():
    assert oc.expand_muqattaat_word_to_letter_names("كهيعص") == [
        "كاف", "هاء", "ياء", "عين", "صاد",
    ]


def test_expand_ham():
    assert oc.expand_muqattaat_word_to_letter_names("حمٓ") == ["حاء", "ميم"]


def test_transcript_muqattaat_chapter_expands_only_opening_word():
    # Surah 10 (Yunus): الٓر then تلك ءايات ...
    quran_words = [
        _qw("10:1", 1, "الٓر"),
        _qw("10:1", 2, "تِلْكَ"),
        _qw("10:1", 3, "ءَايَٰتُ"),
        _qw("10:1", 4, "ٱلْكِتَٰبِ"),
    ]
    t = oc.build_opening_transcript(quran_words, 10, opening_words=4)
    # 3 letter-name entries for الٓر + 3 normal words
    muq = [e for e in t if e["is_muqattaat"]]
    normal = [e for e in t if not e["is_muqattaat"]]
    assert [e["text"] for e in muq] == ["الف", "لام", "راء"]
    assert all(e["quran_index"] == 0 for e in muq)
    assert len(normal) == 3
    assert [e["quran_index"] for e in normal] == [1, 2, 3]


def test_transcript_non_muqattaat_chapter_no_expansion():
    # Surah 1 (Al-Fatihah) opening words -- no muqatta'at.
    quran_words = [
        _qw("1:2", 1, "ٱلْحَمْدُ"),
        _qw("1:2", 2, "لِلَّهِ"),
        _qw("1:2", 3, "رَبِّ"),
    ]
    t = oc.build_opening_transcript(quran_words, 1, opening_words=3)
    assert all(not e["is_muqattaat"] for e in t)
    assert [e["quran_index"] for e in t] == [0, 1, 2]


def test_merge_muqattaat_letters_into_one_word_real_span():
    quran_words = [
        _qw("10:1", 1, "الٓر"),
        _qw("10:1", 2, "تِلْكَ"),
    ]
    transcript = oc.build_opening_transcript(quran_words, 10, opening_words=2)
    # Fake CTC spans: 3 letter spans (real, monotonic) + 1 word span.
    spans = [
        {"start_ms": 5880, "end_ms": 6800, "score": 0.9},   # الف
        {"start_ms": 6800, "end_ms": 8200, "score": 0.8},   # لام (madd)
        {"start_ms": 8200, "end_ms": 10100, "score": 0.85},  # راء (madd)
        {"start_ms": 10100, "end_ms": 11540, "score": 0.95},  # تلك
    ]
    merged = oc.merge_spans_to_quran(transcript, spans, quran_words)
    alr = merged[0]
    tilka = merged[1]
    # الٓر span = full real audio span of the recited letters.
    assert alr["start_ms"] == 5880
    assert alr["end_ms"] == 10100
    assert alr["is_muqattaat"] is True
    assert len(alr["letters"]) == 3
    # تلك keeps its own real span (not smeared).
    assert tilka["start_ms"] == 10100
    assert tilka["end_ms"] == 11540


def test_merge_handles_unaligned_entry():
    quran_words = [_qw("2:1", 1, "الٓمٓ"), _qw("2:2", 2, "ذَٰلِكَ")]
    transcript = oc.build_opening_transcript(quran_words, 2, opening_words=2)
    # First letter fails to align (None), rest ok.
    spans = [None,
             {"start_ms": 2000, "end_ms": 3000, "score": 0.7},
             {"start_ms": 3000, "end_ms": 4500, "score": 0.7},
             {"start_ms": 4500, "end_ms": 5200, "score": 0.9}]
    merged = oc.merge_spans_to_quran(transcript, spans, quran_words)
    alm = merged[0]
    # Still aligned via the letters that DID align; never fabricates.
    assert alm["aligned"] is True
    assert alm["start_ms"] == 2000
    assert alm["end_ms"] == 4500


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
