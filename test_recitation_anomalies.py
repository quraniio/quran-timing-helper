#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for detect_recitation_anomalies (repeat-detection, verify-only)."""

import sys
import types
from pathlib import Path

# Stub heavy optional deps so the module imports in a plain Python 3.x env.
for _name in ("torch", "whisper", "requests", "numpy"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

# Stub the optional CTC deps so the dep-probe in recover_estimated_words_ctc
# succeeds in a plain Python env; the actual aligner is monkeypatched in tests.
_torchaudio_stub = sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
_torchaudio_stub.functional = types.SimpleNamespace(forced_align=lambda *a, **k: None)
_transformers_stub = sys.modules.setdefault("transformers", types.ModuleType("transformers"))
_transformers_stub.Wav2Vec2ForCTC = object
_transformers_stub.Wav2Vec2Processor = object

sys.path.insert(0, str(Path(__file__).resolve().parent))

import quran_forced_align_full as q  # noqa: E402


def _mw(text, norm, start, end, score, asr_index, estimated=False, verse_key="2:1", pos=1):
    return {
        "text": text, "norm": norm, "start_ms": start, "end_ms": end,
        "score": score, "asr_index": asr_index, "estimated": estimated,
        "verse_key": verse_key, "position": pos,
    }


def _asr(norm, start, end):
    return {"word": norm, "norm": norm, "start_ms": start, "end_ms": end}


def test_repeated_word_flagged():
    mapped = []
    asr = []
    t = 0
    used_asr_index = 0
    for i in range(10):
        norm = "الله"
        # leave a gap in asr indices to insert an extra repeat token after word 5
        mapped.append(_mw("ٱللَّه", norm, t, t + 300, 0.97, used_asr_index, pos=i + 1))
        asr.append(_asr(norm, t, t + 300))
        t += 350
        used_asr_index += 1
        if i == 5:
            # reciter repeats "الله" -> extra unmatched ASR token right after
            asr.append(_asr("الله", t, t + 300))
            t += 350
            used_asr_index += 1  # this index is NOT referenced by any mapped word

    rep = q.detect_recitation_anomalies(mapped, asr)
    assert rep["repeated_word_count"] == 1, rep
    assert rep["repeated_phrase_count"] == 0, rep
    assert rep["repeated_count"] >= 1, rep
    r = rep["repeated_words"][0]
    assert r["repeated_norm"] == "الله"
    assert r["similarity"] >= 0.8


def test_repeated_phrase_flagged():
    # The reciter says the whole sequence, then re-recites the phrase
    # "رب العالمين" (a consecutive run of two extra ASR tokens).
    norms = ["الحمد", "لله", "رب", "العالمين", "الرحمن", "الرحيم", "مالك", "يوم"]
    mapped = []
    asr = []
    t = 0
    ai = 0
    for i, n in enumerate(norms):
        mapped.append(_mw(n, n, t, t + 300, 0.97, ai, pos=i + 1))
        asr.append(_asr(n, t, t + 300))
        t += 350
        ai += 1
        if i == 3:
            for rn in ("رب", "العالمين"):
                asr.append(_asr(rn, t, t + 300))  # extra, unreferenced indices
                t += 350
                ai += 1

    rep = q.detect_recitation_anomalies(mapped, asr)
    assert rep["repeated_phrase_count"] == 1, rep
    assert rep["repeated_word_count"] == 0, rep
    ph = rep["repeated_phrases"][0]
    assert ph["num_words"] == 2, ph
    assert ph["repeated_norm"] == "رب العالمين", ph
    assert ph["similarity"] >= 0.8, ph
    assert ph["verse_keys"], ph


def test_no_false_positives_clean():
    mapped = []
    asr = []
    t = 0
    for i in range(40):
        norm = "والعصر"
        mapped.append(_mw("وَٱلْعَصْر", norm, t, t + 500, 1.0, i, pos=i + 1))
        asr.append(_asr(norm, t, t + 500))
        t += 550
    rep = q.detect_recitation_anomalies(mapped, asr)
    assert rep["anomaly_count"] == 0, rep
    assert rep["repeated_word_count"] == 0, rep
    assert rep["repeated_phrase_count"] == 0, rep


def test_estimated_words_not_flagged():
    # An estimated (non-real) opening word with no asr_index must be ignored.
    mapped = [
        _mw("الٓمٓ", "الم", 0, 0, 0.0, None, estimated=True, pos=1),
    ]
    for i in range(25):
        mapped.append(_mw("ذلك", "ذلك", 100 + i * 300, 100 + i * 300 + 250, 0.95, i, pos=i + 2))
    asr = [_asr("ذلك", 100 + i * 300, 100 + i * 300 + 250) for i in range(25)]
    rep = q.detect_recitation_anomalies(mapped, asr)
    assert rep["anomaly_count"] == 0, rep


def test_untrusted_asr_disables_repeats():
    # When the saved ASR stream does NOT share the mapping's asr_index space
    # (e.g. raw vs stripped), repeat detection must be turned OFF so that
    # unmatched intro/basmala tokens are never mis-flagged as repeats.
    mapped = []
    asr = []
    t = 0
    ai = 0
    for i in range(10):
        mapped.append(_mw("ٱللَّه", "الله", t, t + 300, 0.97, ai, pos=i + 1))
        asr.append(_asr("الله", t, t + 300))
        t += 350
        ai += 1
        if i == 5:
            asr.append(_asr("الله", t, t + 300))  # extra repeat token
            t += 350
            ai += 1

    trusted = q.detect_recitation_anomalies(mapped, asr, asr_index_trusted=True)
    untrusted = q.detect_recitation_anomalies(mapped, asr, asr_index_trusted=False)

    assert trusted["repeated_count"] >= 1, trusted
    assert untrusted["repeated_count"] == 0, untrusted
    assert untrusted["params"]["repeats_checked"] is False


def _cand(asr_set, quality_score, matched=100, match_rate=1.0):
    return {
        "asr_set": asr_set,
        "quality_score": quality_score,
        "score": quality_score,
        "matched": matched,
        "match_rate": match_rate,
    }


def test_select_prefers_stripped_when_intro_detected_non_muqattaat():
    # v54: Al-An'am style. original_asr ties (intro-false-hit penalty never
    # fired because the opening got estimated, not false-matched), so without
    # the fix original_asr (first in list) would win the tie and keep the
    # isti'adha/basmala intro -> opening spread over the intro audio.
    candidates = [
        _cand("original_asr", quality_score=3025),
        _cand("stripped_recitation_extras", quality_score=3025),
    ]
    best = q.select_best_alignment_candidate(
        candidates, has_leading_intro=True, chapter_has_muqattaat=False
    )
    assert best["asr_set"] == "stripped_recitation_extras", best


def test_select_keeps_original_for_muqattaat_chapter():
    # Muqatta'at openings rely on the original_asr + opening-safe fold path and
    # must never be forced onto the stripped candidate.
    candidates = [
        _cand("original_asr", quality_score=3025),
        _cand("stripped_recitation_extras", quality_score=3025),
    ]
    best = q.select_best_alignment_candidate(
        candidates, has_leading_intro=True, chapter_has_muqattaat=True
    )
    assert best["asr_set"] == "original_asr", best


def test_select_no_intro_uses_pure_quality():
    # No leading intro detected -> behave exactly like the old max() selection.
    candidates = [
        _cand("original_asr", quality_score=3000),
        _cand("stripped_recitation_extras", quality_score=2990),
    ]
    best = q.select_best_alignment_candidate(
        candidates, has_leading_intro=False, chapter_has_muqattaat=False
    )
    assert best["asr_set"] == "original_asr", best


def test_select_stripped_still_loses_to_clearly_better_original_when_no_intro():
    candidates = [
        _cand("original_asr", quality_score=3050),
        _cand("stripped_recitation_extras", quality_score=3010),
    ]
    best = q.select_best_alignment_candidate(
        candidates, has_leading_intro=False, chapter_has_muqattaat=False
    )
    assert best["asr_set"] == "original_asr", best


def test_select_intro_detected_picks_best_among_stripped_variants():
    # When restricted to stripped, still pick the highest-quality stripped one.
    candidates = [
        _cand("original_asr", quality_score=9999),
        _cand("stripped_recitation_extras", quality_score=3010, match_rate=0.99),
        _cand("stripped_recitation_extras", quality_score=3010, match_rate=1.0),
    ]
    best = q.select_best_alignment_candidate(
        candidates, has_leading_intro=True, chapter_has_muqattaat=False
    )
    assert best["asr_set"] == "stripped_recitation_extras", best
    assert best["match_rate"] == 1.0, best


def test_select_intro_detected_but_no_stripped_candidate_falls_back():
    # Defensive: if somehow no stripped candidate exists, do not crash.
    candidates = [_cand("original_asr", quality_score=3025)]
    best = q.select_best_alignment_candidate(
        candidates, has_leading_intro=True, chapter_has_muqattaat=False
    )
    assert best["asr_set"] == "original_asr", best


def _opening_mapped(intro_estimated=4, reliable=6, reliable_start=10000, verse_key="10:1"):
    words = []
    for i in range(intro_estimated):
        words.append({
            "text": "كلمة", "norm": "كلمة", "start_ms": 0, "end_ms": 0,
            "score": 0.0, "estimated": True, "verse_key": verse_key, "position": i + 1,
        })
    t = reliable_start
    for j in range(reliable):
        words.append({
            "text": "ثابتة", "norm": "ثابتة", "start_ms": t, "end_ms": t + 400,
            "score": 0.95, "estimated": False, "verse_key": verse_key, "position": intro_estimated + j + 1,
        })
        t += 500
    return words


def test_detected_intro_applied_when_strip_report_empty():
    # v55: original_asr won selection so stripped_report is empty, but the intro
    # WAS detected globally. The opening must be placed AFTER the detected intro
    # end (7000ms), never spread from 0 over the isti'adha/basmala audio.
    mapped = _opening_mapped()
    quran_words = [{"text": "الٓر"}] + [{"text": "كلمة"} for _ in range(9)]
    detected = {"removed_count": 9, "removed_words": ["x"] * 9, "start_ms": 0, "end_ms": 7000}
    result, report = q.apply_leading_intro_gap_repair(
        mapped,
        stripped_report=[],
        wav_path=None,
        use_audio_pauses=False,
        quran_words=quran_words,
        chapter_id=10,
        opening_safe_mode=True,
        detected_intro_info=detected,
    )
    assert report, "expected an opening repair to run"
    leading = [w for w in result[:4]]
    assert all(int(w["start_ms"]) >= 7000 for w in leading), [w["start_ms"] for w in leading]
    assert int(result[0]["start_ms"]) == 7000, result[0]["start_ms"]


def test_detected_intro_applied_for_muqattaat_chapter():
    # Same as above but emphasises the muqatta'at path: chapter 10 (الٓر) keeps
    # the original_asr+fold candidate, yet the detected intro must still push the
    # muqatta'at opening past the recited isti'adha/basmala.
    mapped = _opening_mapped(verse_key="10:1")
    quran_words = [{"text": "الٓر"}] + [{"text": "تلك"} for _ in range(9)]
    detected = {"removed_count": 9, "removed_words": ["x"] * 9, "start_ms": 0, "end_ms": 6500}
    result, report = q.apply_leading_intro_gap_repair(
        mapped,
        stripped_report=[],
        wav_path=None,
        use_audio_pauses=False,
        quran_words=quran_words,
        chapter_id=10,
        opening_safe_mode=True,
        detected_intro_info=detected,
    )
    assert report, "expected an opening repair to run"
    assert int(result[0]["start_ms"]) >= 6500, result[0]["start_ms"]


def _ch10_opening_mapped():
    # Reproduces the real reciter-A/ch10 opening: الٓر got a FALSE match on the last
    # basmala token (5160-5880, during the intro), while تلك/ءايات/الكتاب/الحكيم
    # and the rest of 10:2 carry REAL consecutive Whisper timestamps (10100ms+)
    # but with low fuzzy scores, so the score-based stable anchor only forms a
    # run of 5 deep at index 14 (أنذر). The old code smeared indices 0..13.
    words = []
    # idx0: الٓر -- false match inside the intro span.
    words.append({"text": "الٓرۚ", "norm": "الر", "start_ms": 5160, "end_ms": 5880,
                  "score": 0.5, "estimated": False, "verse_key": "10:1", "position": 1})
    # idx1..4: 10:1 tail with REAL low-score timestamps.
    real_open = [("تِلۡكَ", 10100, 11540), ("ءَايَٰتُ", 11540, 13700),
                 ("ٱلۡكِتَٰبِ", 13700, 14460), ("ٱلۡحَكِيمِ", 14460, 15560)]
    for k, (t, s, e) in enumerate(real_open):
        words.append({"text": t, "norm": t, "start_ms": s, "end_ms": e,
                      "score": 0.72, "estimated": False, "verse_key": "10:1", "position": k + 2})
    # idx5..13: start of 10:2 with REAL low-score timestamps (still < run-of-5).
    t = 16080
    for k in range(9):
        words.append({"text": "كلمة", "norm": "كلمة", "start_ms": t, "end_ms": t + 800,
                      "score": 0.74, "estimated": False, "verse_key": "10:2", "position": k + 1})
        t += 900
    # idx14..19: reliable run (score 0.95) -> stable anchor lands at index 14.
    t = 27520
    for k in range(6):
        words.append({"text": "ثابتة", "norm": "ثابتة", "start_ms": t, "end_ms": t + 400,
                      "score": 0.95, "estimated": False, "verse_key": "10:2", "position": k + 10})
        t += 500
    return words


def test_keep_real_timed_opening_words_after_unfolded_muqattaat():
    # v56: only the unfolded muqatta'at (الٓر, idx0) should be redistributed,
    # across [intro_end=5880, تلك_real_start=10100] -- a real audio-bounded span.
    # Every measured opening word (تلك/ءايات/...) MUST keep its Whisper time.
    mapped = _ch10_opening_mapped()
    quran_words = [{"text": "الٓر"}] + [{"text": "كلمة"} for _ in range(19)]
    detected = {"removed_count": 9, "removed_words": ["x"] * 9, "start_ms": 0, "end_ms": 5880}
    result, report = q.apply_leading_intro_gap_repair(
        mapped,
        stripped_report=[],
        wav_path=None,
        use_audio_pauses=False,
        quran_words=quran_words,
        chapter_id=10,
        opening_safe_mode=True,
        detected_intro_info=detected,
    )
    assert report, "expected an opening repair to run"
    # الٓر redistributed to the real madd span after the intro.
    assert int(result[0]["start_ms"]) == 5880, result[0]["start_ms"]
    assert int(result[0]["end_ms"]) == 10100, result[0]["end_ms"]
    # Real-timed opening words preserved EXACTLY (not smeared).
    assert int(result[1]["start_ms"]) == 10100, result[1]["start_ms"]
    assert int(result[2]["start_ms"]) == 11540, result[2]["start_ms"]
    assert int(result[3]["start_ms"]) == 13700, result[3]["start_ms"]
    assert int(result[4]["start_ms"]) == 14460, result[4]["start_ms"]
    assert int(result[5]["start_ms"]) == 16080, result[5]["start_ms"]
    clamp = report[0].get("opening_real_timed_clamp")
    assert clamp and clamp.get("applied") and clamp.get("real_anchor_index") == 1, clamp


def test_keep_real_timed_disabled_falls_back_to_static():
    # With the v56 guard disabled, the old behavior returns: the whole leading
    # island 0..13 is statically redistributed and تلك loses its real 10100ms.
    mapped = _ch10_opening_mapped()
    quran_words = [{"text": "الٓر"}] + [{"text": "كلمة"} for _ in range(19)]
    detected = {"removed_count": 9, "removed_words": ["x"] * 9, "start_ms": 0, "end_ms": 5880}
    result, report = q.apply_leading_intro_gap_repair(
        mapped,
        stripped_report=[],
        wav_path=None,
        use_audio_pauses=False,
        quran_words=quran_words,
        chapter_id=10,
        opening_safe_mode=True,
        detected_intro_info=detected,
        keep_opening_real_timed=False,
    )
    assert report, "expected an opening repair to run"
    # تلك no longer at its real time; the whole island was smeared.
    assert int(result[1]["start_ms"]) != 10100, result[1]["start_ms"]
    clamp = report[0].get("opening_real_timed_clamp")
    assert clamp and not clamp.get("applied"), clamp


def test_no_intro_signal_and_no_opening_safe_returns_unchanged():
    # Sanity: with no strip report, no detected intro, and opening-safe disabled,
    # the repair must be a no-op (nothing to anchor on).
    mapped = _opening_mapped()
    before = [int(w["start_ms"]) for w in mapped]
    result, report = q.apply_leading_intro_gap_repair(
        mapped,
        stripped_report=[],
        wav_path=None,
        use_audio_pauses=False,
        quran_words=[{"text": "كلمة"} for _ in range(10)],
        chapter_id=10,
        opening_safe_mode=False,
        detected_intro_info=None,
    )
    assert report == [], report
    assert [int(w["start_ms"]) for w in result] == before


def _ch19_opening_mapped():
    # Surah Maryam: كٓهيعٓصٓ (19:1) then ذِكۡرُ رَحۡمَتِ رَبِّكَ ... . Whisper drops
    # the muqatta'at (idx0 estimated), while ذكر and the following words carry
    # REAL low-score Whisper timestamps. A reliable run forms only at idx5.
    words = []
    words.append({"text": "كٓهيعٓصٓ", "norm": "كهيعص", "start_ms": 0, "end_ms": 0,
                  "score": 0.0, "estimated": True, "verse_key": "19:1", "position": 1})
    real_open = [("ذِكۡرُ", 10000, 10800), ("رَحۡمَتِ", 10800, 11600),
                 ("رَبِّكَ", 11600, 12400), ("عَبۡدَهُۥ", 12400, 13400)]
    for k, (t, s, e) in enumerate(real_open):
        words.append({"text": t, "norm": t, "start_ms": s, "end_ms": e,
                      "score": 0.72, "estimated": False, "verse_key": "19:2", "position": k + 1})
    t = 14000
    for k in range(6):
        words.append({"text": "ثابتة", "norm": "ثابتة", "start_ms": t, "end_ms": t + 400,
                      "score": 0.95, "estimated": False, "verse_key": "19:2", "position": k + 6})
        t += 500
    return words


def test_deterministic_bracket_kahyaaas_ch19():
    # v57: KNOW that surah 19 opens with one muqatta'at word (كهيعص). Without any
    # acoustic verification of the muqatta'at, bracket it across
    # [intro_end=6500, ذكر_real_start=10000]; ذكر and everything after keep their
    # measured times. This is the user's worked example.
    mapped = _ch19_opening_mapped()
    quran_words = [{"text": "كٓهيعٓصٓ"}] + [{"text": "كلمة"} for _ in range(10)]
    detected = {"removed_count": 7, "removed_words": ["x"] * 7, "start_ms": 0, "end_ms": 6500}
    result, report = q.apply_leading_intro_gap_repair(
        mapped,
        stripped_report=[],
        wav_path=None,
        use_audio_pauses=False,
        quran_words=quran_words,
        chapter_id=19,
        opening_safe_mode=True,
        detected_intro_info=detected,
    )
    assert report, "expected an opening repair to run"
    assert int(result[0]["start_ms"]) == 6500, result[0]["start_ms"]
    assert int(result[0]["end_ms"]) == 10000, result[0]["end_ms"]
    assert int(result[1]["start_ms"]) == 10000, result[1]["start_ms"]
    clamp = report[0].get("opening_real_timed_clamp")
    assert clamp and clamp.get("applied"), clamp
    assert clamp.get("deterministic_muqattaat") is True, clamp
    assert clamp.get("real_anchor_index") == 1, clamp
    assert clamp.get("muqattaat_word_count") == 1, clamp


def _ch42_opening_mapped_false_muqattaat():
    # Surah ash-Shura opens with TWO muqatta'at words across two verses:
    # حمٓ (42:1) and عٓسٓقٓ (42:2). Here BOTH get false low-score matches near the
    # intro (5200 / 5700), the first real recited word starts at 9000, and a
    # reliable run forms only at idx7. The v56 island scan from index 0 would
    # latch onto حم's false 5200 timestamp and bail (keeping the false times);
    # v57 anchors on the KNOWN muqatta'at word count (2) and brackets BOTH.
    words = []
    words.append({"text": "حمٓ", "norm": "حم", "start_ms": 5200, "end_ms": 5600,
                  "score": 0.5, "estimated": False, "verse_key": "42:1", "position": 1})
    words.append({"text": "عٓسٓقٓ", "norm": "عسق", "start_ms": 5700, "end_ms": 6000,
                  "score": 0.5, "estimated": False, "verse_key": "42:2", "position": 1})
    real_open = [("كلمة", 9000, 9800), ("كلمة", 9800, 10600),
                 ("كلمة", 10600, 11400), ("كلمة", 11400, 12200)]
    for k, (t, s, e) in enumerate(real_open):
        words.append({"text": t, "norm": t, "start_ms": s, "end_ms": e,
                      "score": 0.72, "estimated": False, "verse_key": "42:3", "position": k + 1})
    t = 13000
    for k in range(6):
        words.append({"text": "ثابتة", "norm": "ثابتة", "start_ms": t, "end_ms": t + 400,
                      "score": 0.95, "estimated": False, "verse_key": "42:3", "position": k + 6})
        t += 500
    return words


def test_deterministic_bracket_multiword_ch42_skips_false_muqattaat_match():
    mapped = _ch42_opening_mapped_false_muqattaat()
    quran_words = [{"text": "حمٓ"}, {"text": "عٓسٓقٓ"}] + [{"text": "كلمة"} for _ in range(14)]
    detected = {"removed_count": 5, "removed_words": ["x"] * 5, "start_ms": 0, "end_ms": 4800}
    result, report = q.apply_leading_intro_gap_repair(
        mapped,
        stripped_report=[],
        wav_path=None,
        use_audio_pauses=False,
        quran_words=quran_words,
        chapter_id=42,
        opening_safe_mode=True,
        detected_intro_info=detected,
    )
    assert report, "expected an opening repair to run"
    # Both muqatta'at words are bracketed across the real post-intro span; the
    # false 5700 timestamp on عسق is discarded.
    assert int(result[0]["start_ms"]) == 4800, result[0]["start_ms"]
    assert int(result[1]["end_ms"]) == 9000, result[1]["end_ms"]
    assert int(result[1]["start_ms"]) != 5700, result[1]["start_ms"]
    assert int(result[0]["end_ms"]) == int(result[1]["start_ms"]), (result[0], result[1])
    # First real recited word keeps its measured time.
    assert int(result[2]["start_ms"]) == 9000, result[2]["start_ms"]
    clamp = report[0].get("opening_real_timed_clamp")
    assert clamp and clamp.get("applied"), clamp
    assert clamp.get("deterministic_muqattaat") is True, clamp
    assert clamp.get("real_anchor_index") == 2, clamp
    assert clamp.get("muqattaat_word_count") == 2, clamp


def test_deterministic_bracket_disabled_falls_back_to_v56():
    # With the deterministic bracket disabled, the v56 island scan latches onto
    # حم's false 5200 timestamp, clamps the anchor to index 0, and bails -- so
    # عسق keeps its false 5700 time (the regression v57 fixes).
    mapped = _ch42_opening_mapped_false_muqattaat()
    quran_words = [{"text": "حمٓ"}, {"text": "عٓسٓقٓ"}] + [{"text": "كلمة"} for _ in range(14)]
    detected = {"removed_count": 5, "removed_words": ["x"] * 5, "start_ms": 0, "end_ms": 4800}
    result, report = q.apply_leading_intro_gap_repair(
        mapped,
        stripped_report=[],
        wav_path=None,
        use_audio_pauses=False,
        quran_words=quran_words,
        chapter_id=42,
        opening_safe_mode=True,
        detected_intro_info=detected,
        deterministic_muqattaat_bracket=False,
    )
    assert report == [], report
    assert int(result[1]["start_ms"]) == 5700, result[1]["start_ms"]


def test_deterministic_bracket_spans_dropped_word_after_muqattaat():
    # M=1 (كهيعص) but Whisper ALSO dropped the first normal word ذكر (idx1
    # estimated). The deterministic scan starts at M and walks PAST the dropped
    # word to the first REAL recited word (idx2), bracketing both the muqatta'at
    # and the dropped word across [intro_end, idx2_start]. No acoustic
    # verification of the muqatta'at; the first measured word is preserved.
    words = []
    words.append({"text": "كٓهيعٓصٓ", "norm": "كهيعص", "start_ms": 0, "end_ms": 0,
                  "score": 0.0, "estimated": True, "verse_key": "19:1", "position": 1})
    words.append({"text": "ذِكۡرُ", "norm": "ذكر", "start_ms": 0, "end_ms": 0,
                  "score": 0.0, "estimated": True, "verse_key": "19:2", "position": 1})
    real_open = [("رَحۡمَتِ", 11000, 11800), ("رَبِّكَ", 11800, 12600),
                 ("عَبۡدَهُۥ", 12600, 13600)]
    for k, (t, s, e) in enumerate(real_open):
        words.append({"text": t, "norm": t, "start_ms": s, "end_ms": e,
                      "score": 0.72, "estimated": False, "verse_key": "19:2", "position": k + 2})
    t = 14200
    for k in range(6):
        words.append({"text": "ثابتة", "norm": "ثابتة", "start_ms": t, "end_ms": t + 400,
                      "score": 0.95, "estimated": False, "verse_key": "19:2", "position": k + 6})
        t += 500
    quran_words = [{"text": "كٓهيعٓصٓ"}, {"text": "ذِكۡرُ"}] + [{"text": "كلمة"} for _ in range(9)]
    detected = {"removed_count": 7, "removed_words": ["x"] * 7, "start_ms": 0, "end_ms": 6500}
    result, report = q.apply_leading_intro_gap_repair(
        words,
        stripped_report=[],
        wav_path=None,
        use_audio_pauses=False,
        quran_words=quran_words,
        chapter_id=19,
        opening_safe_mode=True,
        detected_intro_info=detected,
    )
    assert report, "expected an opening repair to run"
    # Muqatta'at + dropped word bracketed across the real post-intro span.
    assert int(result[0]["start_ms"]) == 6500, result[0]["start_ms"]
    # First REAL recited word keeps its measured time and bounds the bracket.
    assert int(result[2]["start_ms"]) == 11000, result[2]["start_ms"]
    # The dropped word now carries an estimated time INSIDE the bracket.
    assert 6500 <= int(result[1]["start_ms"]) <= 11000, result[1]["start_ms"]
    assert int(result[1]["end_ms"]) <= 11000, result[1]["end_ms"]
    clamp = report[0].get("opening_real_timed_clamp")
    assert clamp and clamp.get("applied"), clamp
    assert clamp.get("deterministic_muqattaat") is True, clamp
    assert clamp.get("real_anchor_index") == 2, clamp
    assert clamp.get("muqattaat_word_count") == 1, clamp


def test_choose_intro_verify_report_falls_back_to_global():
    # original_asr won selection (muqatta'at chapter) -> selected report empty,
    # but the intro was detected globally -> verification must use the global one.
    global_rep = [
        {"side": "start", "phrase": ["اعوذ", "بالله", "من", "الشيطان", "الرجيم"],
         "start_ms": 500, "end_ms": 4000, "removed_words": ["اعوذ"]},
        {"side": "start", "phrase": ["بسم", "الله", "الرحمن", "الرحيم"],
         "start_ms": 4000, "end_ms": 7000, "removed_words": ["بسم"]},
    ]
    assert q.choose_intro_verify_report([], global_rep) is global_rep
    # When the selected candidate has its own intro block (non-muqatta'at path)
    # it is preferred unchanged.
    selected = [{"side": "start", "phrase": ["بسم", "الله", "الرحمن", "الرحيم"],
                 "start_ms": 0, "end_ms": 3000}]
    assert q.choose_intro_verify_report(selected, global_rep) is selected
    # Nothing detected anywhere -> behavior unchanged (returns selected).
    assert q.choose_intro_verify_report([], []) == []


def test_verify_opening_intro_recognizes_istiadha_and_basmala_from_global():
    # Surah 10 (الٓر): reciter said BOTH isti'adha and basmala. original_asr won
    # so the selected strip report is empty; the global report carries both.
    global_rep = [
        {"side": "start", "phrase": ["اعوذ", "بالله", "من", "الشيطان", "الرجيم"],
         "start_ms": 500, "end_ms": 4000},
        {"side": "start", "phrase": ["بسم", "الله", "الرحمن", "الرحيم"],
         "start_ms": 4000, "end_ms": 7000},
    ]
    rep = q.choose_intro_verify_report([], global_rep)
    quran_words = [{"text": "الٓر"}, {"text": "تِلۡكَ"}]

    # Correct timing: الٓر begins AFTER the intro -> both detected, no overlap.
    mapped_ok = [
        {"text": "الٓرۚ", "norm": "الر", "start_ms": 7200, "end_ms": 8000,
         "estimated": False, "verse_key": "10:1"},
        {"text": "تِلۡكَ", "norm": "تلك", "start_ms": 8000, "end_ms": 8600,
         "estimated": False, "verse_key": "10:1"},
    ]
    report_ok, problems_ok = q.verify_opening_intro(
        mapped_ok, rep, quran_words=quran_words, chapter_id=10)
    assert report_ok["istiadha_detected"] is True, report_ok
    assert report_ok["basmala_detected"] is True, report_ok
    assert report_ok["intro_audio_end_ms"] == 7000, report_ok
    assert problems_ok == [], problems_ok

    # Bug timing (the surah-10 symptom): الٓر at 2000ms lands ON the intro audio
    # -> now flagged, because verification sees the real intro end (7000ms).
    mapped_bad = [
        {"text": "الٓرۚ", "norm": "الر", "start_ms": 2000, "end_ms": 2600,
         "estimated": False, "verse_key": "10:1"},
        {"text": "تِلۡكَ", "norm": "تلك", "start_ms": 8000, "end_ms": 8600,
         "estimated": False, "verse_key": "10:1"},
    ]
    report_bad, problems_bad = q.verify_opening_intro(
        mapped_bad, rep, quran_words=quran_words, chapter_id=10)
    assert report_bad["intro_audio_end_ms"] == 7000, report_bad
    assert len(report_bad["overlap_words"]) == 1, report_bad
    assert problems_bad, "a surah word starting inside the intro must be flagged"


def _surah10_opening_charproj_fixture():
    # Reproduces the Yunus (surah 10) opening: the muqatta'at الٓر was correctly
    # bracketed onto its REAL post-intro audio (5880..10100) by the intro-gap
    # repair. The selected original_asr still carries the isti'adha word الرجيم
    # at 2000..2580 (inside the intro), whose first 3 letters char-match الٓر.
    mapped = [
        {"text": "الٓرۚ", "norm": "الر", "start_ms": 5880, "end_ms": 10100,
         "score": 1.0, "asr_index": None, "estimated": True,
         "verse_key": "10:1", "position": 1, "intro_gap_estimated": True,
         "opening_region_estimated": True},
        {"text": "تِلۡكَ", "norm": "تلك", "start_ms": 10100, "end_ms": 11540,
         "score": 1.0, "asr_index": 11, "estimated": False,
         "verse_key": "10:1", "position": 2},
        {"text": "ءَايَٰتُ", "norm": "ايات", "start_ms": 11540, "end_ms": 13700,
         "score": 1.0, "asr_index": 12, "estimated": False,
         "verse_key": "10:1", "position": 3},
    ]
    asr = [
        _asr("الرجيم", 2000, 2580),   # last isti'adha word (intro audio)
        _asr("تلك", 10100, 11540),
        _asr("ايات", 11540, 13700),
    ]
    return mapped, asr


def test_char_projection_keeps_muqattaat_off_intro_audio():
    # With the detected intro end passed in, the guard must REJECT projecting
    # الٓر onto the isti'adha الرجيم; its real bracketed timing must be kept.
    mapped, asr = _surah10_opening_charproj_fixture()
    repaired, report = q.repair_estimated_words_by_char_projection(
        mapped_words=mapped, asr_words=asr, intro_end_ms=5880,
    )
    assert repaired[0]["start_ms"] == 5880
    assert repaired[0]["end_ms"] == 10100
    # No char-projection entry may move the opening word into the intro.
    for r in report:
        if r["index"] == 0:
            assert r["new_start_ms"] >= 5880


def test_char_projection_without_intro_guard_reproduces_regression():
    # Control: WITHOUT the intro end, the first word (no previous timed
    # neighbour, so the shift cap cannot fire) is wrongly pulled back onto the
    # isti'adha audio. This is exactly the bug the intro guard fixes.
    mapped, asr = _surah10_opening_charproj_fixture()
    repaired, _ = q.repair_estimated_words_by_char_projection(
        mapped_words=mapped, asr_words=asr,
    )
    assert repaired[0]["start_ms"] < 5880


def test_char_projection_no_intro_chapter_still_repairs():
    # No leading intro (e.g. a surah where the reciter omits it / At-Tawbah):
    # intro_end_ms=None must leave normal char-projection behavior intact, so a
    # genuinely estimated interior word still gets its real timing from ASR.
    mapped = [
        {"text": "قُلۡ", "norm": "قل", "start_ms": 0, "end_ms": 400,
         "score": 1.0, "asr_index": 0, "estimated": False,
         "verse_key": "9:1", "position": 1},
        {"text": "هُوَ", "norm": "هو", "start_ms": 400, "end_ms": 700,
         "score": 0.0, "asr_index": None, "estimated": True,
         "verse_key": "9:1", "position": 2},
        {"text": "ٱللَّهُ", "norm": "الله", "start_ms": 1200, "end_ms": 1600,
         "score": 1.0, "asr_index": 2, "estimated": False,
         "verse_key": "9:1", "position": 3},
    ]
    asr = [
        _asr("قل", 0, 400),
        _asr("هو", 800, 1100),     # real timing for the estimated middle word
        _asr("الله", 1200, 1600),
    ]
    repaired, report = q.repair_estimated_words_by_char_projection(
        mapped_words=mapped, asr_words=asr, intro_end_ms=None,
    )
    assert repaired[1]["estimated"] is False
    assert any(r["index"] == 1 for r in report)


def _ctc_args(**over):
    base = dict(
        no_ctc_word_recover=False,
        ctc_model="dummy-model",
        ctc_recover_min_score=0.10,
        ctc_recover_pad_ms=150,
        ctc_recover_max_window_ms=15000,
        ctc_recover_min_word_ms=80,
        ctc_recover_max_runs=60,
        device_resolved="cpu",
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _ctc_mapped():
    # real | estimated interior | real
    return [
        {"text": "قُلۡ", "norm": "قل", "start_ms": 600, "end_ms": 1000,
         "score": 1.0, "estimated": False, "verse_key": "112:1", "position": 1},
        {"text": "هُوَ", "norm": "هو", "start_ms": 1200, "end_ms": 1500,
         "score": 0.0, "estimated": True, "verse_key": "112:1", "position": 2},
        {"text": "ٱللَّهُ", "norm": "الله", "start_ms": 2000, "end_ms": 2400,
         "score": 1.0, "estimated": False, "verse_key": "112:1", "position": 3},
    ]


def test_ctc_interior_recovery_fills_estimated_word(monkeypatched=None):
    saved = q.ctc_align_window
    try:
        q.ctc_align_window = lambda **kw: [{"start_ms": 1300, "end_ms": 1700, "score": 0.9}]
        out, report = q.recover_estimated_words_ctc(_ctc_mapped(), _ctc_args(), wav_path="x.wav")
    finally:
        q.ctc_align_window = saved
    assert report["attempted"] is True
    assert report["recovered"] == 1
    w = out[1]
    assert w["estimated"] is False
    assert w["ctc_recovered"] is True
    # clamped strictly inside the [prev_end=1000, next_start=2000] gap
    assert 1000 <= w["start_ms"] < w["end_ms"] <= 2000
    assert "+ctc_word_recover" in w["method"]


def test_ctc_interior_recovery_rejects_low_score():
    saved = q.ctc_align_window
    try:
        q.ctc_align_window = lambda **kw: [{"start_ms": 1300, "end_ms": 1700, "score": 0.02}]
        out, report = q.recover_estimated_words_ctc(_ctc_mapped(), _ctc_args(), wav_path="x.wav")
    finally:
        q.ctc_align_window = saved
    assert report["recovered"] == 0
    assert out[1]["estimated"] is True  # safe interpolation kept, never fabricated


def test_ctc_interior_recovery_skips_opening_region():
    mapped = _ctc_mapped()
    mapped[1]["opening_region_estimated"] = True
    saved = q.ctc_align_window
    called = {"n": 0}
    def _fake(**kw):
        called["n"] += 1
        return [{"start_ms": 1300, "end_ms": 1700, "score": 0.9}]
    try:
        q.ctc_align_window = _fake
        out, report = q.recover_estimated_words_ctc(mapped, _ctc_args(), wav_path="x.wav")
    finally:
        q.ctc_align_window = saved
    assert called["n"] == 0          # opening estimates are owned elsewhere
    assert report["recovered"] == 0
    assert out[1]["estimated"] is True


def test_ctc_interior_recovery_skips_leading_run_protects_opening():
    # Leading estimated run with NO real word before it = the surah opening.
    # It must be skipped structurally even WITHOUT opening_region_estimated flags.
    mapped = [
        {"text": "الٓر", "norm": "الر", "start_ms": 0, "end_ms": 400,
         "score": 0.0, "estimated": True, "verse_key": "10:1", "position": 1},
        {"text": "تِلۡكَ", "norm": "تلك", "start_ms": 400, "end_ms": 800,
         "score": 0.0, "estimated": True, "verse_key": "10:1", "position": 2},
        {"text": "ءَايَٰتُ", "norm": "ايات", "start_ms": 2000, "end_ms": 2400,
         "score": 1.0, "estimated": False, "verse_key": "10:1", "position": 3},
    ]
    saved = q.ctc_align_window
    called = {"n": 0}
    def _fake(**kw):
        called["n"] += 1
        return [{"start_ms": 100, "end_ms": 300, "score": 0.9}]
    try:
        q.ctc_align_window = _fake
        out, report = q.recover_estimated_words_ctc(mapped, _ctc_args(), wav_path="x.wav")
    finally:
        q.ctc_align_window = saved
    assert called["n"] == 0
    assert report["recovered"] == 0
    assert out[0]["estimated"] is True and out[1]["estimated"] is True


def test_ctc_interior_recovery_skips_trailing_run():
    # Trailing estimated run with NO real word after it has no upper anchor.
    mapped = [
        {"text": "قُلۡ", "norm": "قل", "start_ms": 600, "end_ms": 1000,
         "score": 1.0, "estimated": False, "verse_key": "112:1", "position": 1},
        {"text": "هُوَ", "norm": "هو", "start_ms": 1200, "end_ms": 1500,
         "score": 0.0, "estimated": True, "verse_key": "112:1", "position": 2},
    ]
    saved = q.ctc_align_window
    called = {"n": 0}
    def _fake(**kw):
        called["n"] += 1
        return [{"start_ms": 1300, "end_ms": 1700, "score": 0.9}]
    try:
        q.ctc_align_window = _fake
        out, report = q.recover_estimated_words_ctc(mapped, _ctc_args(), wav_path="x.wav")
    finally:
        q.ctc_align_window = saved
    assert called["n"] == 0
    assert report["recovered"] == 0
    assert out[1]["estimated"] is True


def test_ctc_interior_recovery_disabled_flag():
    saved = q.ctc_align_window
    try:
        q.ctc_align_window = lambda **kw: [{"start_ms": 1300, "end_ms": 1700, "score": 0.9}]
        out, report = q.recover_estimated_words_ctc(
            _ctc_mapped(), _ctc_args(no_ctc_word_recover=True), wav_path="x.wav")
    finally:
        q.ctc_align_window = saved
    assert report["attempted"] is False
    assert out[1]["estimated"] is True


def test_ctc_interior_recovery_degrades_when_deps_missing():
    # Temporarily remove the transformers stub so the dep-probe fails.
    saved_mod = sys.modules.pop("transformers", None)
    try:
        out, report = q.recover_estimated_words_ctc(_ctc_mapped(), _ctc_args(), wav_path="x.wav")
    finally:
        if saved_mod is not None:
            sys.modules["transformers"] = saved_mod
    assert report.get("skipped") == "deps_missing"
    assert report["recovered"] == 0
    assert out[1]["estimated"] is True  # never crashes, never fabricates


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
