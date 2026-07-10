#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v60 unit tests: under-transcription recovery decision + degenerate-stripped
candidate-selection safety guard.

These cover the Al-Asr (103) / Mansour Al-Salemi hard-FAILED case, where Whisper
under-transcribed a 14-word surah as 5 words, the stripper removed 4 as a
mis-detected isti'adha/basmala, and the v54 "restrict to stripped candidate"
rule then forced an all-missing (matched=0/14) candidate over a real one.
"""

import sys
import types
from pathlib import Path

# numpy must be the REAL module: quick_alignment_match_count runs the DP aligner.
for _name in ("torch", "whisper", "requests"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
try:
    import numpy  # noqa: F401
except Exception:
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))

_torchaudio_stub = sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
_torchaudio_stub.functional = types.SimpleNamespace(forced_align=lambda *a, **k: None)
_transformers_stub = sys.modules.setdefault("transformers", types.ModuleType("transformers"))
_transformers_stub.Wav2Vec2ForCTC = object
_transformers_stub.Wav2Vec2Processor = object
sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))

sys.path.insert(0, str(Path(__file__).resolve().parent))

import quran_forced_align_full as q  # noqa: E402


def _cand(asr_set, matched, missing, score):
    return {
        "asr_set": asr_set,
        "matched": matched,
        "missing": missing,
        "match_rate": matched / 14.0,
        "quality_score": score,
        "score": score,
    }


# ---- degenerate-stripped safety guard -------------------------------------

def test_degenerate_stripped_does_not_force_failure():
    """The Al-Asr case: stripped candidate matched 0/14, original matched 3/14.
    The guard must NOT force the all-missing stripped candidate."""
    original = _cand("original_asr", matched=3, missing=0, score=3000)
    stripped = _cand("stripped_recitation_extras", matched=0, missing=14, score=-1120)
    best = q.select_best_alignment_candidate(
        [original, stripped],
        has_leading_intro=True,
        chapter_has_muqattaat=False,
    )
    assert best is original, "must not pick the all-missing stripped candidate"
    assert best["matched"] == 3


def test_healthy_stripped_is_still_forced():
    """Normal non-muqattaat leading-intro case: when the stripped candidate
    actually anchors real audio, the v54 restriction is preserved so the
    isti'adha/basmala stays excluded from the opening."""
    original = _cand("original_asr", matched=10, missing=0, score=9000)
    stripped = _cand("stripped_recitation_extras", matched=8, missing=0, score=8000)
    best = q.select_best_alignment_candidate(
        [original, stripped],
        has_leading_intro=True,
        chapter_has_muqattaat=False,
    )
    assert best is stripped, "healthy stripped candidate must still be forced"


def test_muqattaat_never_restricts():
    """Muqatta'at chapters keep competing normally (opening-safe fold path)."""
    original = _cand("original_asr", matched=12, missing=0, score=12000)
    stripped = _cand("stripped_recitation_extras", matched=0, missing=14, score=-1120)
    best = q.select_best_alignment_candidate(
        [original, stripped],
        has_leading_intro=True,
        chapter_has_muqattaat=True,
    )
    assert best is original


def test_no_leading_intro_picks_best_score():
    original = _cand("original_asr", matched=14, missing=0, score=14000)
    stripped = _cand("stripped_recitation_extras", matched=9, missing=0, score=9000)
    best = q.select_best_alignment_candidate(
        [original, stripped],
        has_leading_intro=False,
        chapter_has_muqattaat=False,
    )
    assert best is original


# ---- quick_alignment_match_count ------------------------------------------

def _qw(norm, key, pos):
    return {"norm": norm, "word": norm, "verse_key": key, "position": pos}


def _aw(norm, start, end):
    return {"word": norm, "norm": norm, "start_ms": start, "end_ms": end}


def test_quick_match_prefers_better_coverage():
    quran = [
        _qw("والعصر", "103:1", 1),
        _qw("ان", "103:2", 1),
        _qw("الانسان", "103:2", 2),
        _qw("لفي", "103:2", 3),
        _qw("خسر", "103:2", 4),
    ]
    poor = [_aw("والعصر", 0, 500)]
    full = [
        _aw("والعصر", 0, 500),
        _aw("ان", 600, 800),
        _aw("الانسان", 900, 1400),
        _aw("لفي", 1500, 1700),
        _aw("خسر", 1800, 2200),
    ]
    args = types.SimpleNamespace()
    poor_n = q.quick_alignment_match_count(quran, poor, args)
    full_n = q.quick_alignment_match_count(quran, full, args)
    assert full_n > poor_n, f"full coverage ({full_n}) should beat poor ({poor_n})"
    assert full_n >= 4


def test_quick_match_empty_inputs():
    args = types.SimpleNamespace()
    assert q.quick_alignment_match_count([], [_aw("x", 0, 1)], args) == 0
    assert q.quick_alignment_match_count([_qw("x", "1:1", 1)], [], args) == 0


# ---- trigger + adoption decision helpers ----------------------------------

def test_should_attempt_triggers_on_severe_underrun():
    # Al-Asr: 5 ASR words for 14 Quran words -> coverage 0.357 < 0.85.
    assert q.underrun_recover_should_attempt(5, 14, 0.85) is True


def test_should_attempt_skips_healthy_coverage():
    # A well-transcribed surah must NOT pay the second pass.
    assert q.underrun_recover_should_attempt(13, 14, 0.85) is False
    assert q.underrun_recover_should_attempt(20, 14, 0.85) is False


def test_should_attempt_boundary_is_strict():
    # Exactly at the threshold does not trigger (strict <).
    assert q.underrun_recover_should_attempt(85, 100, 0.85) is False
    assert q.underrun_recover_should_attempt(84, 100, 0.85) is True


def test_should_attempt_guards_zero_quran():
    assert q.underrun_recover_should_attempt(0, 0, 0.85) is False


def test_intro_token_padding_masks_dropped_words():
    """v65 regression lock: the An-Nasr / reciter-B failure mode. The reciter says
    the basmala (4 intro tokens) while Whisper DROPS interior ayah words, so the
    RAW token count equals the Quran word count and looks 'complete', while far
    fewer words actually align. The gate must fire on matched-count coverage, not
    on raw token count."""
    quran = [
        _qw("اذا", "110:1", 1),
        _qw("جاء", "110:1", 2),
        _qw("نصر", "110:1", 3),
        _qw("الله", "110:1", 4),
        _qw("والفتح", "110:1", 5),
        _qw("ورايت", "110:2", 1),
        _qw("الناس", "110:2", 2),
        _qw("يدخلون", "110:2", 3),
        _qw("في", "110:2", 4),
        _qw("دين", "110:2", 5),
    ]
    # 10 raw ASR tokens for 10 Quran words: 4 basmala intro tokens pad the count
    # while 4 interior ayah-2 words (الناس/يدخلون/في/دين) are dropped.
    asr = [
        _aw("بسم", 0, 300),
        _aw("الله", 300, 600),
        _aw("الرحمن", 600, 900),
        _aw("الرحيم", 900, 1200),
        _aw("اذا", 3000, 3300),
        _aw("جاء", 3300, 3600),
        _aw("نصر", 3600, 3900),
        _aw("الله", 3900, 4200),
        _aw("والفتح", 4200, 4600),
        _aw("ورايت", 4600, 5000),
    ]
    args = types.SimpleNamespace()
    matched = q.quick_alignment_match_count(quran, asr, args)
    # Raw-token gate (the OLD bug) would compute 10/10 = 1.0 and skip recovery.
    assert q.underrun_recover_should_attempt(len(asr), len(quran), 0.85) is False
    # Matched-count gate (v65) sees only the real ayah words align and DOES fire.
    assert matched < len(asr), f"expected dropped words, matched={matched}"
    assert q.underrun_recover_should_attempt(matched, len(quran), 0.85) is True


def test_should_adopt_only_on_strict_improvement():
    assert q.underrun_recover_should_adopt(0, 12) is True    # Al-Asr recovery
    assert q.underrun_recover_should_adopt(3, 3) is False    # no improvement -> keep prompted
    assert q.underrun_recover_should_adopt(10, 4) is False   # alt worse -> keep prompted


# ---- v61: poisoned quality_score selection rescue -------------------------

def test_poisoned_quality_score_still_picks_real_alignment():
    """The REAL Al-Asr failure: the intro false-match penalty (1e6) drives the
    matched=3 original_asr candidate's quality_score far below the matched=0
    stripped candidate. quality_score is meaningless here; the coverage rescue
    must still pick the candidate that aligns real words."""
    original = _cand("original_asr", matched=3, missing=0, score=2670 - 3 * 1000000)
    stripped = _cand("stripped_recitation_extras", matched=0, missing=14, score=-1120)
    best = q.select_best_alignment_candidate(
        [original, stripped],
        has_leading_intro=True,
        chapter_has_muqattaat=False,
    )
    assert best is original, "must rescue the real alignment despite poisoned score"
    assert best["matched"] == 3


def test_coverage_rescue_prefers_most_matched():
    """Among only-real candidates the rescue must maximise matched count, not the
    intro-penalised quality_score (which would favour fewer intro false hits)."""
    few = _cand("original_asr", matched=1, missing=0, score=1000 - 1 * 1000000)
    many = _cand("original_asr", matched=3, missing=0, score=3000 - 3 * 1000000)
    degenerate = _cand("stripped_recitation_extras", matched=0, missing=14, score=-1120)
    best = q.select_best_alignment_candidate(
        [few, many, degenerate],
        has_leading_intro=True,
        chapter_has_muqattaat=False,
    )
    assert best is many and best["matched"] == 3


def test_all_degenerate_returns_something():
    """If every candidate matched 0, selection must not crash; returns a candidate."""
    a = _cand("original_asr", matched=0, missing=14, score=-5)
    b = _cand("stripped_recitation_extras", matched=0, missing=14, score=-1120)
    best = q.select_best_alignment_candidate(
        [a, b], has_leading_intro=True, chapter_has_muqattaat=False
    )
    assert best in (a, b)


# ---- v61: full-surah CTC fallback -----------------------------------------

def _surah14():
    texts = [
        "والعصر", "ان", "الانسان", "لفي", "خسر", "الا", "الذين", "امنوا",
        "وعملوا", "الصالحات", "وتواصوا", "بالحق", "وتواصوا", "بالصبر",
    ]
    return [
        {"norm": q.normalize_arabic(t), "word": t, "verse_key": f"103:{i}", "position": i}
        for i, t in enumerate(texts, 1)
    ]


def _underrun_mapped(quran_words):
    """3 real words at the front, the remaining 11 estimated (no real time)."""
    out = []
    for i, w in enumerate(quran_words):
        m = {"norm": w["norm"], "word": w["word"]}
        if i < 3:
            m["start_ms"] = 1000 + i * 500
            m["end_ms"] = 1400 + i * 500
            m["estimated"] = False
        else:
            m["start_ms"] = None
            m["end_ms"] = None
            m["estimated"] = True
        out.append(m)
    return out


def _ctc_args():
    return types.SimpleNamespace(device_resolved="cpu")


def test_underrun_ctc_recovers_all_words(monkeypatch=None):
    quran = _surah14()
    mapped = _underrun_mapped(quran)
    prefix_len = len(q.underrun_intro_prefix_norms(quran, 103))
    assert prefix_len > 0, "intro prefix should be non-empty for ch103"

    def fake_align(wav_path, transcript, device, model_name, window_start_ms=0, window_sec=None):
        assert len(transcript) == prefix_len + len(quran)
        # Tight intro tokens (end well before the preserved words at 1000ms), then a
        # realistic surah layout spread across the clip at a plausible rate -- well
        # under the degenerate rate ceiling, not crammed into a tiny window.
        spans = []
        for k in range(len(transcript)):
            if k < prefix_len:
                start, end = 100 * k, 100 * k + 90
            else:
                start = 3000 + (k - prefix_len) * 600
                end = start + 400
            spans.append({"start_ms": start, "end_ms": end, "score": 0.5})
        return spans

    q.ctc_align_window = fake_align
    try:
        result, report = q.recover_underrun_surah_ctc(
            mapped_words=mapped,
            quran_words=quran,
            args=_ctc_args(),
            wav_path="dummy.wav",
            chapter_id=103,
            duration_ms=30000,
        )
    finally:
        q.ctc_align_window = q.ctc_align_window  # keep reference; module reload not needed

    assert report["attempted"] is True
    # 3 front words already had REAL Whisper spans -> preserved, not re-counted;
    # only the 11 lacking words are recovered.
    assert report["recovered"] == 11, report
    assert all(w.get("start_ms") is not None and not w.get("estimated") for w in result)
    # the 3 preserved words keep their original Whisper timestamps
    assert result[0]["start_ms"] == 1000 and not result[0].get("ctc_underrun_recovered")
    # monotonic non-overlapping across the whole (preserved + recovered) timeline
    for a, b in zip(result, result[1:]):
        assert a["end_ms"] <= b["start_ms"] + 1
    assert q.recount_matched_after_repairs(result) == 14


def test_underrun_ctc_degenerate_recovery_reverts_to_estimated():
    """Audio that BOTH Whisper and the CTC model fail on: forced alignment cannot
    localise the tokens and returns near-zero RAW widths for (almost) every word
    (they then get pinned at the floor, non-zero & monotonic, so the degenerate-
    SPAN guard misses them). The compressed-cluster guard must detect the raw-width
    degeneracy and REVERT the whole recovery to estimated so the quality gate flags
    the surah instead of false-greening garbage timings."""
    quran = _surah14()
    mapped = _underrun_mapped(quran)  # 3 preserved real, 11 lacking

    def fake_align(wav_path, transcript, device, model_name, window_start_ms=0, window_sec=None):
        # RAW zero-width spans: the aligner gave every token no real duration.
        return [{"start_ms": 10 * k, "end_ms": 10 * k, "score": 0.5}
                for k in range(len(transcript))]

    q.ctc_align_window = fake_align
    result, report = q.recover_underrun_surah_ctc(
        mapped_words=mapped, quran_words=quran, args=_ctc_args(),
        wav_path="dummy.wav", chapter_id=103, duration_ms=30000,
    )

    assert report["attempted"] is True
    assert report.get("ctc_recovery_degenerate") is True, report
    assert report.get("ctc_recovery_floor_frac", 0) >= 0.85, report
    # every CTC-recovered word reverted to estimated (so the quality gate flags it)
    recovered = [w for w in result if w.get("ctc_underrun_recovered")]
    assert recovered, "expected some recovered words"
    assert all(w.get("estimated") is True for w in recovered), recovered
    # preserved ASR words are untouched (still real)
    assert result[0].get("estimated") is False and result[0]["start_ms"] == 1000
    # matched collapses to just the 3 preserved real words -> RED, not a fake green
    assert q.recount_matched_after_repairs(result) == 3
    # internal bookkeeping key must not leak into output
    assert all("_ctc_raw_width_ms" not in w for w in result)


def test_underrun_ctc_minority_floor_stays_real():
    """A healthy recovery where the CTC gave real, varied widths to most words and
    only a minority landed below the floor must NOT be flagged degenerate: the
    guard keys on the RAW-width floor fraction, and a minority never crosses the
    threshold, so the words stay real."""
    quran = _surah14()
    mapped = _underrun_mapped(quran)
    prefix_len = len(q.underrun_intro_prefix_norms(quran, 103))

    def fake_align(wav_path, transcript, device, model_name, window_start_ms=0, window_sec=None):
        spans = []
        for k in range(len(transcript)):
            surah_idx = k - prefix_len
            # one lacking surah word gets a zero-width raw span; the rest are wide
            if surah_idx == 5:
                spans.append({"start_ms": 300 * k, "end_ms": 300 * k, "score": 0.5})
            else:
                spans.append({"start_ms": 300 * k, "end_ms": 300 * k + 250, "score": 0.5})
        return spans

    q.ctc_align_window = fake_align
    result, report = q.recover_underrun_surah_ctc(
        mapped_words=mapped, quran_words=quran, args=_ctc_args(),
        wav_path="dummy.wav", chapter_id=103, duration_ms=30000,
    )

    assert report["attempted"] is True
    assert report.get("ctc_recovery_degenerate") is False, report
    assert 0 < report.get("ctc_recovery_floor_frac", 0) < 0.85, report
    assert all(w.get("start_ms") is not None and not w.get("estimated") for w in result)


def test_underrun_ctc_rate_ceiling_catches_wide_but_packed_spans():
    """The exact v78 miss: the aligner returns WIDE raw spans (each >= the min-word
    floor, so the raw-width floor-fraction signal stays near zero and never trips),
    but they are all piled on top of each other, so the monotonic clamp packs the
    recovered words into a tiny window at an impossible recitation rate. The v79
    RATE-CEILING signal must catch this even though the floor-fraction signal does
    not, and revert the recovered words to estimated -> RED, not a fake green."""
    quran = _surah14()
    mapped = _underrun_mapped(quran)  # 3 preserved real, 11 lacking
    prefix_len = len(q.underrun_intro_prefix_norms(quran, 103))

    def fake_align(wav_path, transcript, device, model_name, window_start_ms=0, window_sec=None):
        spans = []
        for k in range(len(transcript)):
            if k < prefix_len:
                start, end = 100 * k, 100 * k + 90  # tight intro, ends before 1000ms
            else:
                # every surah token has a WIDE raw width (400ms >= floor) but is
                # piled at the same start -> heavy overlap -> clamp collapses them.
                start, end = 3000, 3400
            spans.append({"start_ms": start, "end_ms": end, "score": 0.5})
        return spans

    q.ctc_align_window = fake_align
    result, report = q.recover_underrun_surah_ctc(
        mapped_words=mapped, quran_words=quran, args=_ctc_args(),
        wav_path="dummy.wav", chapter_id=103, duration_ms=30000,
    )

    assert report["attempted"] is True
    # RATE signal fires: recovered words packed far above real recitation rate
    assert report.get("ctc_recovery_words_per_sec", 0) > 7.0, report
    # FLOOR signal did NOT catch it (raw widths are wide, not floored)
    assert report.get("ctc_recovery_floor_frac", 1.0) < 0.85, report
    assert report.get("ctc_recovery_degenerate") is True, report
    # every CTC-recovered word reverted to estimated so the quality gate flags RED
    recovered = [w for w in result if w.get("ctc_underrun_recovered")]
    assert recovered, "expected some recovered words"
    assert all(w.get("estimated") is True for w in recovered), recovered
    # preserved ASR words untouched, matched collapses to the 3 real words -> RED
    assert result[0].get("estimated") is False and result[0]["start_ms"] == 1000
    assert q.recount_matched_after_repairs(result) == 3
    # internal bookkeeping key must not leak into output
    assert all("_ctc_raw_width_ms" not in w for w in result)


def test_underrun_ctc_interior_misanchor_overridden_spreads_cluster():
    """v80 root fix for reciter-B Al-Masad 111: Whisper collapses ~27s into ONE
    hallucinated timestamp, so the sparse interior recovery drops a LATE verse word
    ('في', 111:5:1) into that huge window and pins it EARLY (~5.3s). Left as a
    'real' anchor, the backward monotonic clamp crushes every recovered word before
    it into a degenerate ~1.7s cluster -> the whole surah reverts to estimated (RED).
    The full-surah forced aligner (full known text, monotonic) actually places that
    word ~20s later; overriding the bogus early anchor with the forced-align span
    lets the recovered opening keep its spread spans, so the surah aligns for real
    instead of being flagged. Purely positional (madd-safe), generic."""
    quran = _surah14()
    prefix_len = len(q.underrun_intro_prefix_norms(quran, 103))
    assert prefix_len > 0
    # The LAST word is a preserved 'real' anchor pinned FAR too early (3000ms),
    # exactly like 'في' at 5.3s; every earlier word lacks a real timestamp.
    last = len(quran) - 1
    mapped = []
    for i, w in enumerate(quran):
        m = {"norm": w["norm"], "word": w["word"]}
        if i == last:
            m["start_ms"] = 3000
            m["end_ms"] = 3400
            m["estimated"] = False
        else:
            m["start_ms"] = None
            m["end_ms"] = None
            m["estimated"] = True
        mapped.append(m)

    def fake_align(wav_path, transcript, device, model_name, window_start_ms=0, window_sec=None):
        # Forced alignment SPREADS the whole surah across the clip (the raw spans
        # are wide and monotonic) and correctly puts the last word ~13.4s in --
        # ~10s LATER than the bogus 3000ms preserved anchor.
        spans = []
        for k in range(len(transcript)):
            if k < prefix_len:
                start = 100 * k                    # tight intro, ends before 3000ms
            else:
                start = 3000 + (k - prefix_len) * 800
            spans.append({"start_ms": start, "end_ms": start + 400, "score": 0.5})
        return spans

    q.ctc_align_window = fake_align
    result, report = q.recover_underrun_surah_ctc(
        mapped_words=mapped, quran_words=quran, args=_ctc_args(),
        wav_path="dummy.wav", chapter_id=103, duration_ms=30000,
    )

    # the bogus early anchor was recognised as interior-misanchored and overridden
    assert report.get("interior_misanchor_overridden") == 1, report
    assert result[last].get("interior_misanchor_overridden") is True
    assert result[last].get("ctc_underrun_recovered") is True
    # it now sits ~13s in (its forced-align position), not at the bogus 3000ms
    assert result[last]["start_ms"] >= 12000, result[last]
    # with the bad anchor gone, the recovered opening keeps its SPREAD spans instead
    # of being crushed -> NOT degenerate, everything stays REAL (a true recovery)
    assert report.get("ctc_recovery_degenerate") is False, report
    assert all(w.get("start_ms") is not None and not w.get("estimated") for w in result)
    # the opening cluster really is spread across seconds, not packed at a floor wall
    assert result[last - 1]["start_ms"] - result[0]["start_ms"] > 5000, result
    # monotonic, non-overlapping, all 14 words matched -> real >= gate, no fake green
    for a, b in zip(result, result[1:]):
        assert a["end_ms"] <= b["start_ms"], (a, b)
    assert q.recount_matched_after_repairs(result) == 14
    assert all("_ctc_raw_width_ms" not in w for w in result)


def test_underrun_ctc_small_drift_preserved_word_not_overridden():
    """Anti-clobber guard for the v80 interior-misanchor override: a CORRECTLY
    placed preserved word whose forced-align span drifts only slightly LATER (well
    within interior_gap_ms, i.e. ordinary ASR-vs-CTC jitter) must be PRESERVED
    untouched -- never mistaken for a bogus early anchor."""
    quran = _surah14()
    prefix_len = len(q.underrun_intro_prefix_norms(quran, 103))
    mid = 5
    mapped = []
    for i, w in enumerate(quran):
        m = {"norm": w["norm"], "word": w["word"]}
        if i == mid:
            m["start_ms"] = 5000       # correctly placed real word
            m["end_ms"] = 5400
            m["estimated"] = False
        else:
            m["start_ms"] = None
            m["end_ms"] = None
            m["estimated"] = True
        mapped.append(m)

    def fake_align(wav_path, transcript, device, model_name, window_start_ms=0, window_sec=None):
        # forced-align places the preserved word at 6500ms -> only 1500ms drift
        # (< 3000ms gap) -> NOT an interior misanchor; the rest spread plausibly.
        spans = []
        for k in range(len(transcript)):
            surah_idx = k - prefix_len
            if surah_idx < 0:
                start = 100 * k
            elif surah_idx == mid:
                start = 6500                       # 1500ms later than preserved 5000
            elif surah_idx < mid:
                start = 1000 + surah_idx * 700     # opening, before the preserved word
            else:
                start = 7500 + (surah_idx - mid - 1) * 700
            spans.append({"start_ms": start, "end_ms": start + 400, "score": 0.5})
        return spans

    q.ctc_align_window = fake_align
    result, report = q.recover_underrun_surah_ctc(
        mapped_words=mapped, quran_words=quran, args=_ctc_args(),
        wav_path="dummy.wav", chapter_id=103, duration_ms=30000,
    )

    # small drift -> no interior override, preserved word kept exactly as-is
    assert report.get("interior_misanchor_overridden") == 0, report
    assert result[mid]["start_ms"] == 5000 and result[mid]["end_ms"] == 5400
    assert not result[mid].get("interior_misanchor_overridden")
    assert not result[mid].get("ctc_underrun_recovered")


def test_underrun_ctc_preserves_real_word_and_stays_monotonic():
    """A real-timed word sits in the MIDDLE with a much larger timestamp than the
    (small) CTC timeline. It must be preserved untouched, and recovered neighbours
    must clamp around it so the final timeline is monotonic & non-overlapping."""
    quran = _surah14()
    prefix_len = len(q.underrun_intro_prefix_norms(quran, 103))
    mapped = []
    for i, w in enumerate(quran):
        m = {"norm": w["norm"], "word": w["word"]}
        if i == 5:
            m["start_ms"] = 5000
            m["end_ms"] = 5400
            m["estimated"] = False
        else:
            m["start_ms"] = None
            m["end_ms"] = None
            m["estimated"] = True
        mapped.append(m)

    def fake_align(wav_path, transcript, device, model_name, window_start_ms=0, window_sec=None):
        # A realistic recovery spreads its words across the clip (well under the
        # degenerate rate ceiling). EXCEPT the word right after the preserved one,
        # which is deliberately given a tiny early span so the forward clamp around
        # the preserved 5000ms word is still exercised.
        spans = []
        for k in range(len(transcript)):
            surah_idx = k - prefix_len
            if surah_idx < 0:              # intro prefix tokens, before the surah
                start = 100 * k
            elif surah_idx <= 4:           # opening words, before the preserved 5000ms
                start = 700 * surah_idx
            elif surah_idx == 6:           # conflicts with preserved word -> must clamp
                start = 1000
            else:                          # tail words, spread out after the preserved word
                start = 6000 + (surah_idx - 7) * 700
            spans.append({"start_ms": start, "end_ms": start + 400, "score": 0.5})
        return spans

    q.ctc_align_window = fake_align
    result, report = q.recover_underrun_surah_ctc(
        mapped_words=mapped, quran_words=quran, args=_ctc_args(),
        wav_path="dummy.wav", chapter_id=103, duration_ms=30000,
    )

    assert report["attempted"] is True
    assert report["recovered"] == 13, report
    # preserved word untouched
    assert result[5]["start_ms"] == 5000 and result[5]["end_ms"] == 5400
    assert not result[5].get("ctc_underrun_recovered")
    # every word real-timed, monotonic, non-overlapping
    assert all(w.get("start_ms") is not None and not w.get("estimated") for w in result)
    for a, b in zip(result, result[1:]):
        assert a["end_ms"] <= b["start_ms"], (a, b)
    # the recovered word right after the preserved one is clamped past 5400ms
    assert result[6]["start_ms"] >= 5400


def test_underrun_ctc_skips_healthy_coverage():
    quran = _surah14()
    healthy = []
    for i, w in enumerate(quran):
        healthy.append({"norm": w["norm"], "word": w["word"],
                        "start_ms": i * 500, "end_ms": i * 500 + 400, "estimated": False})

    def boom(*a, **k):
        raise AssertionError("CTC must not run for healthy coverage")

    q.ctc_align_window = boom
    result, report = q.recover_underrun_surah_ctc(
        mapped_words=healthy, quran_words=quran, args=_ctc_args(),
        wav_path="dummy.wav", chapter_id=103, duration_ms=30000,
    )
    assert report["attempted"] is False
    assert report["recovered"] == 0
    assert result[0]["start_ms"] == 0


def test_underrun_ctc_disabled_flag():
    quran = _surah14()
    mapped = _underrun_mapped(quran)

    def boom(*a, **k):
        raise AssertionError("disabled flag must short-circuit before CTC")

    q.ctc_align_window = boom
    args = types.SimpleNamespace(device_resolved="cpu", no_underrun_ctc=True)
    result, report = q.recover_underrun_surah_ctc(
        mapped_words=mapped, quran_words=quran, args=args,
        wav_path="dummy.wav", chapter_id=103, duration_ms=30000,
    )
    assert report["attempted"] is False


def test_underrun_ctc_skips_long_clip():
    quran = _surah14()
    mapped = _underrun_mapped(quran)

    def boom(*a, **k):
        raise AssertionError("long clip must be skipped before CTC")

    q.ctc_align_window = boom
    args = types.SimpleNamespace(device_resolved="cpu", underrun_ctc_max_duration_ms=120000)
    result, report = q.recover_underrun_surah_ctc(
        mapped_words=mapped, quran_words=quran, args=args,
        wav_path="dummy.wav", chapter_id=103, duration_ms=300000,
    )
    assert report.get("skipped") == "duration_exceeds_max"
    assert report["recovered"] == 0


def test_underrun_ctc_low_score_words_kept_estimated_when_threshold_set():
    """When the caller RAISES --underrun-ctc-min-score above 0, low-score spans are
    rejected and those words keep their prior (estimated) state. The default is 0.0
    (see test_underrun_ctc_accepts_low_score_by_default), so the threshold is opt-in."""
    quran = _surah14()
    mapped = _underrun_mapped(quran)
    prefix_len = len(q.underrun_intro_prefix_norms(quran, 103))

    def fake_align(wav_path, transcript, device, model_name, window_start_ms=0, window_sec=None):
        # v84: the pass now tries multiple intro prefix hypotheses, so derive
        # the prefix length from THIS call's transcript instead of assuming
        # the full prefix.
        local_prefix_len = len(transcript) - len(quran)
        spans = []
        for k in range(len(transcript)):
            surah_idx = k - local_prefix_len
            # last two surah words come back below the configured min_score
            score = 0.02 if surah_idx >= len(quran) - 2 else 0.5
            if k < local_prefix_len:
                start, end = 100 * k, 100 * k + 90
            else:
                start = 3000 + surah_idx * 600
                end = start + 400
            spans.append({"start_ms": start, "end_ms": end, "score": score})
        return spans

    args = types.SimpleNamespace(device_resolved="cpu", underrun_ctc_min_score=0.10)
    q.ctc_align_window = fake_align
    result, report = q.recover_underrun_surah_ctc(
        mapped_words=mapped, quran_words=quran, args=args,
        wav_path="dummy.wav", chapter_id=103, duration_ms=30000,
    )
    # 11 lacking words; last 2 fall below min_score -> 9 recovered (front 3 were
    # already real and are preserved, not re-counted).
    assert report["recovered"] == 9, report
    assert result[-1]["estimated"] is True and result[-1]["start_ms"] is None
    assert result[-2]["estimated"] is True


def test_underrun_ctc_accepts_low_score_by_default():
    """Catastrophic under-transcription: the text is KNOWN and forced alignment
    never drops a token, so the default min_score (0.0) accepts EVERY non-None span.
    Even a 0.01-score word must get a real (clamped) timestamp so a tiny surah can
    reach the >=99.55% gate instead of failing on a single un-recovered word."""
    quran = _surah14()
    mapped = _underrun_mapped(quran)
    prefix_len = len(q.underrun_intro_prefix_norms(quran, 103))

    def fake_align(wav_path, transcript, device, model_name, window_start_ms=0, window_sec=None):
        spans = []
        for k in range(len(transcript)):
            surah_idx = k - prefix_len
            score = 0.01 if surah_idx == len(quran) - 1 else 0.5
            if k < prefix_len:
                start, end = 100 * k, 100 * k + 90
            else:
                start = 3000 + surah_idx * 600
                end = start + 400
            spans.append({"start_ms": start, "end_ms": end, "score": score})
        return spans

    q.ctc_align_window = fake_align
    result, report = q.recover_underrun_surah_ctc(
        mapped_words=mapped, quran_words=quran, args=_ctc_args(),
        wav_path="dummy.wav", chapter_id=103, duration_ms=30000,
    )
    # all 11 lacking words recovered (incl. the 0.01-score one); none left estimated
    assert report["recovered"] == 11, report
    assert all(w.get("start_ms") is not None and not w.get("estimated") for w in result)
    assert q.recount_matched_after_repairs(result) == 14
    for a, b in zip(result, result[1:]):
        assert a["end_ms"] <= b["start_ms"] + 1


def test_underrun_ctc_reports_measured_intro_end():
    """The pass returns intro_end_ms = end of the forced-aligned intro tokens,
    capped at the first surah word's start. This is the REAL intro boundary used to
    override a bogus stripped-intro span on under-transcribed clips."""
    quran = _surah14()
    mapped = _underrun_mapped(quran)
    prefix_len = len(q.underrun_intro_prefix_norms(quran, 103))
    assert prefix_len > 0

    def fake_align(wav_path, transcript, device, model_name, window_start_ms=0, window_sec=None):
        # intro tokens occupy 0..(prefix_len*600); surah tokens start right after.
        return [{"start_ms": 600 * k, "end_ms": 600 * k + 400, "score": 0.5}
                for k in range(len(transcript))]

    q.ctc_align_window = fake_align
    result, report = q.recover_underrun_surah_ctc(
        mapped_words=mapped, quran_words=quran, args=_ctc_args(),
        wav_path="dummy.wav", chapter_id=103, duration_ms=30000,
    )
    intro_end = report["intro_end_ms"]
    assert intro_end is not None
    # boundary never lands after the earliest real surah word start
    earliest = min(w["start_ms"] for w in result if w.get("start_ms") is not None)
    assert intro_end <= earliest


def test_verify_opening_intro_override_suppresses_false_overlap():
    """A bogus stripped intro spanning nearly the whole clip would flag every surah
    opening word as overlapping the intro. The underrun-CTC measured boundary, when
    passed as an override, replaces it so no false opening WARNING is raised."""
    quran = _surah14()
    mapped = []
    for i, w in enumerate(quran):
        mapped.append({"norm": w["norm"], "word": w["word"], "text": w["word"],
                       "start_ms": 760 + i * 500, "end_ms": 760 + i * 500 + 300,
                       "estimated": False})
    # stripped report claims a basmala "intro" ending at ~29980ms (whole 30s clip)
    bogus_stripped = [{"side": "start", "phrase": ["بسم", "الله", "الرحمن", "الرحيم"],
                       "start_ms": 0, "end_ms": 29980}]

    # without override: the bogus 29980ms boundary flags the opening words
    _, problems_bad = q.verify_opening_intro(mapped, bogus_stripped, quran_words=quran, chapter_id=103)
    assert problems_bad, "bogus intro should produce a false overlap warning"

    # with the measured override (intro ends ~ first surah word): no false warning
    report, problems_ok = q.verify_opening_intro(
        mapped, bogus_stripped, quran_words=quran, chapter_id=103,
        intro_audio_end_override=760,
    )
    assert report["intro_audio_end_ms"] == 760
    assert report.get("intro_audio_end_source") == "underrun_ctc"
    assert not problems_ok, problems_ok


def test_underrun_intro_misanchor_overridden_when_intro_confident():
    """v77: a preserved pre-existing 'real' opening word is itself intro-misaligned
    (parked on the basmala audio at ~300ms, far before where forced alignment places
    the surah). When the intro was CONFIDENTLY recited (prefix force-aligns onto a
    real, decent-score region), that mis-anchored word must be OVERRIDDEN with its
    full-surah CTC span instead of being left wrong-but-flagged — this is the root
    fix for catastrophic under-transcription (e.g. Al-Masad 111 reciter-B)."""
    quran = _surah14()
    prefix_len = len(q.underrun_intro_prefix_norms(quran, 103))
    assert prefix_len > 0
    mapped = []
    for i, w in enumerate(quran):
        m = {"norm": w["norm"], "word": w["word"], "text": w["word"]}
        if i == 0:
            # misaligned: a real-timed word sitting at 300ms, on the intro audio
            m["start_ms"] = 300
            m["end_ms"] = 600
            m["estimated"] = False
        else:
            m["start_ms"] = None
            m["end_ms"] = None
            m["estimated"] = True
        mapped.append(m)

    def fake_align(wav_path, transcript, device, model_name, window_start_ms=0, window_sec=None):
        # Confident intro: prefix tokens span a real ~900ms region with decent score.
        spans = []
        for k in range(len(transcript)):
            if k < prefix_len:
                spans.append({"start_ms": 100 * k, "end_ms": 100 * k + 90, "score": 0.5})
            else:
                base = 2000 + (k - prefix_len) * 200
                spans.append({"start_ms": base, "end_ms": base + 150, "score": 0.5})
        return spans

    q.ctc_align_window = fake_align
    result, report = q.recover_underrun_surah_ctc(
        mapped_words=mapped, quran_words=quran, args=_ctc_args(),
        wav_path="dummy.wav", chapter_id=103, duration_ms=30000,
    )
    assert report.get("intro_confident") is True, report
    assert report.get("intro_misanchor_overridden") == 1, report
    # the previously mis-anchored word now carries its real CTC span (~2000ms),
    # is flagged as an override, and no longer sits on the intro audio at 300ms.
    assert result[0]["start_ms"] >= 2000, result[0]
    assert result[0].get("intro_misanchor_overridden") is True
    assert result[0].get("estimated") is False
    # timeline monotonic & non-overlapping after the override + clamp
    for a, b in zip(result, result[1:]):
        assert a["end_ms"] <= b["start_ms"], (a, b)
    # the opening overlap is now RESOLVED, not merely flagged
    rep, problems = q.verify_opening_intro(
        result, [], quran_words=quran, chapter_id=103,
        intro_audio_end_override=report["intro_end_ms"],
    )
    assert not any(ow["start_ms"] == 300 for ow in rep.get("overlap_words", []))


def test_underrun_intro_misanchor_kept_when_intro_not_recited():
    """v77 no-regression guard: when the intro was NOT recited, the prefix tokens
    collapse into tiny low-score spans so intro_end is meaningless. A correctly-early
    opening word (basmala-skipping reciter) must be PRESERVED, never clobbered."""
    quran = _surah14()
    prefix_len = len(q.underrun_intro_prefix_norms(quran, 103))
    assert prefix_len > 0
    mapped = []
    for i, w in enumerate(quran):
        m = {"norm": w["norm"], "word": w["word"], "text": w["word"]}
        if i == 0:
            # a genuinely-early, correctly-aligned opening word
            m["start_ms"] = 300
            m["end_ms"] = 600
            m["estimated"] = False
        else:
            m["start_ms"] = None
            m["end_ms"] = None
            m["estimated"] = True
        mapped.append(m)

    def fake_align(wav_path, transcript, device, model_name, window_start_ms=0, window_sec=None):
        # Intro NOT recited: prefix tokens collapse into tiny, near-zero-score spans.
        spans = []
        for k in range(len(transcript)):
            if k < prefix_len:
                spans.append({"start_ms": k, "end_ms": k + 1, "score": 0.01})
            else:
                base = 2000 + (k - prefix_len) * 200
                spans.append({"start_ms": base, "end_ms": base + 150, "score": 0.5})
        return spans

    q.ctc_align_window = fake_align
    result, report = q.recover_underrun_surah_ctc(
        mapped_words=mapped, quran_words=quran, args=_ctc_args(),
        wav_path="dummy.wav", chapter_id=103, duration_ms=30000,
    )
    assert report.get("intro_confident") is False, report
    assert report.get("intro_misanchor_overridden") == 0, report
    # the correctly-early preserved opening word is untouched
    assert result[0]["start_ms"] == 300 and result[0]["end_ms"] == 600
    assert not result[0].get("intro_misanchor_overridden")


# ---- v66 opening-scoped CTC recovery --------------------------------------

def _openctc_args():
    return types.SimpleNamespace(
        no_underrun_ctc=False,
        no_opening_ctc=False,
        opening_estimate_forgive_buffer=4,
        opening_ctc_pad_ms=1500,
        underrun_ctc_max_duration_ms=120000,
        underrun_ctc_min_score=0.0,
        underrun_ctc_min_word_ms=80,
        ctc_model="dummy-model",
        device_resolved="cpu",
        device="cpu",
    )


def _qword(norm):
    return {"norm": norm}


def test_opening_ctc_leading_run_stops_at_first_real():
    mw = (
        [{"estimated": True}] * 4
        + [{"start_ms": None, "end_ms": None}]  # missing counts as not-real
        + [{"start_ms": 900, "end_ms": 1000, "estimated": False}]
        + [{"estimated": True}]
    )
    assert q.opening_ctc_leading_estimated_run(mw) == 5


def test_opening_ctc_intro_budget_basmala_surah():
    # Non-muqatta'at surah whose text does NOT start with the basmala: budget is
    # istiadha(5) + basmala(4) + muqattaat(0) + buffer(4) = 13.
    quran = [_qword(n) for n in "تبارك الذي نزل الفرقان على".split()]
    assert q.opening_ctc_intro_budget(quran, _openctc_args(), chapter_id=25) == 13


def test_opening_ctc_skips_intro_sized_leading_run():
    """v81: a leading estimated run within the intro budget is presumed a
    normal (unrecited) lead-in ONLY when every leading word is intro vocabulary
    (isti'adha/basmala/muqatta'at) — e.g. an Al-Fatiha take whose basmala verse
    was never recited. Then the pass must NOT fire (no CTC, no error)."""
    quran = [_qword(n) for n in ["بسم", "الله", "الرحمن", "الرحيم"]] + [
        _qword(f"w{i}") for i in range(16)
    ]
    mapped = [dict(estimated=True) for _ in range(4)] + [
        dict(start_ms=5000 + i * 100, end_ms=5090 + i * 100, estimated=False)
        for i in range(16)
    ]
    result, report = q.recover_underrun_opening_ctc(
        mapped_words=mapped, quran_words=quran, args=_openctc_args(),
        wav_path="dummy.wav", chapter_id=25, duration_ms=60000,
    )
    assert report["attempted"] is False
    assert report["recovered"] == 0
    assert report["leading_estimated"] == 4
    assert "error" not in report  # returned before touching the CTC deps


def test_opening_ctc_fires_on_non_intro_vocab_budget_sized_lead():
    """v81 (reciter-C Al-Kafirun class): a budget-sized leading estimated run of
    ARBITRARY surah text (not intro vocabulary) is dropped recitation, not an
    unrecited intro — the pass must fire even though lead_end <= intro budget,
    so those words get REAL measured spans instead of v64-forgiven
    interpolations."""
    quran = [_qword(f"w{i}") for i in range(20)]
    mapped = [dict(estimated=True) for _ in range(4)] + [
        dict(start_ms=5000 + i * 100, end_ms=5090 + i * 100, estimated=False)
        for i in range(16)
    ]
    result, report = q.recover_underrun_opening_ctc(
        mapped_words=mapped, quran_words=quran, args=_openctc_args(),
        wav_path="dummy.wav", chapter_id=25, duration_ms=60000,
    )
    assert report["attempted"] is True
    assert report["leading_estimated"] == 4


def test_opening_ctc_island_includes_flagged_real_leading_word():
    """v82 (reciter-C Al-Kafirun regression): char projection flipped the FIRST
    opening word (قل) to real at the intro boundary, but the rest of the
    opening-safe-repaired words stayed estimated. The leading island must still
    include the flagged-real word so the pass fires; a real word WITHOUT the
    opening_region_estimated flag still terminates the island."""
    words = [
        dict(start_ms=3520, end_ms=3700, estimated=False, opening_region_estimated=True),
        dict(estimated=True, opening_region_estimated=True),
        dict(estimated=True, opening_region_estimated=True),
        dict(estimated=True, opening_region_estimated=True),
        dict(estimated=True, opening_region_estimated=True),
        dict(start_ms=9000, end_ms=9300, estimated=False),
    ]
    assert q.opening_ctc_leading_estimated_run(words) == 5
    # trusted recovery pops the flag -> island terminates at the real word
    words[0].pop("opening_region_estimated")
    assert q.opening_ctc_leading_estimated_run(words) == 0


def test_opening_ctc_all_real_but_flagged_lead_is_safe_noop():
    """v82: when every leading word is real but still flagged
    opening_region_estimated, the island is non-zero so the pass may run, but
    the commit loop only touches words lacking real timestamps — nothing may be
    overwritten and no span may be fabricated."""
    quran = [_qword(f"w{i}") for i in range(20)]
    mapped = [
        dict(start_ms=500 + i * 1000, end_ms=1300 + i * 1000,
             estimated=False, opening_region_estimated=True)
        for i in range(4)
    ] + [
        dict(start_ms=5000 + i * 100, end_ms=5090 + i * 100, estimated=False)
        for i in range(16)
    ]
    before = [dict(w) for w in mapped]

    def fake_align(wav_path, transcript, device, model_name,
                   window_start_ms=0, window_sec=None):
        return [
            {"start_ms": 100 * k, "end_ms": 100 * k + 80, "score": 0.7}
            for k in range(len(transcript))
        ]

    saved = q.ctc_align_window
    q.ctc_align_window = fake_align
    try:
        result, report = q.recover_underrun_opening_ctc(
            mapped_words=mapped, quran_words=quran, args=_openctc_args(),
            wav_path="dummy.wav", chapter_id=25, duration_ms=60000,
        )
    finally:
        q.ctc_align_window = saved

    assert report["leading_estimated"] == 4
    assert report["recovered"] == 0
    for i in range(4):
        assert result[i]["start_ms"] == before[i]["start_ms"]
        assert result[i]["end_ms"] == before[i]["end_ms"]
        assert result[i]["estimated"] is False
        assert not result[i].get("ctc_opening_recovered")


def _small_lead_fixture():
    quran = [_qword(f"w{i}") for i in range(20)]
    mapped = [dict(estimated=True) for _ in range(4)] + [
        dict(start_ms=5000 + i * 100, end_ms=5090 + i * 100, estimated=False)
        for i in range(16)
    ]
    return quran, mapped


def _run_small_lead_with_fake_align(mapped, quran, span_fn):
    def fake_align(wav_path, transcript, device, model_name,
                   window_start_ms=0, window_sec=None):
        prefix_len = len(transcript) - 4
        spans = []
        for k in range(len(transcript)):
            if k < prefix_len:
                spans.append({"start_ms": 100 * k, "end_ms": 100 * k + 80, "score": 0.7})
            else:
                spans.append(span_fn(k - prefix_len))
        return spans

    saved = q.ctc_align_window
    q.ctc_align_window = fake_align
    try:
        return q.recover_underrun_opening_ctc(
            mapped_words=mapped, quran_words=quran, args=_openctc_args(),
            wav_path="dummy.wav", chapter_id=25, duration_ms=60000,
        )
    finally:
        q.ctc_align_window = saved


def test_opening_ctc_small_lead_strict_rejects_low_confidence():
    """v81 strict acceptance: on a budget-sized lead there is no positive
    evidence the words exist in the audio, so near-zero-confidence spans
    (a truly absent opening aligns near zero) must be rejected — every word
    keeps its safe interpolation, no fabricated real-flagged timestamps."""
    quran, mapped = _small_lead_fixture()
    result, report = _run_small_lead_with_fake_align(
        mapped, quran,
        lambda j: {"start_ms": 500 + j * 1000, "end_ms": 1300 + j * 1000, "score": 0.01},
    )
    assert report["small_lead_rejected"] == "low_confidence"
    assert report["recovered"] == 0
    for i in range(4):
        assert result[i].get("estimated") is True
        assert not result[i].get("ctc_opening_recovered")


def test_opening_ctc_small_lead_strict_rejects_width_collapse():
    """v81 strict acceptance: confident-looking but minimum-width-collapsed
    spans (CTC failed to localize) must also be rejected on a small lead."""
    quran, mapped = _small_lead_fixture()
    result, report = _run_small_lead_with_fake_align(
        mapped, quran,
        lambda j: {"start_ms": 500 + j * 60, "end_ms": 550 + j * 60, "score": 0.7},
    )
    assert report["small_lead_rejected"] == "width_collapse"
    assert report["recovered"] == 0
    for i in range(4):
        assert result[i].get("estimated") is True


def test_opening_ctc_prefix_hypothesis_picks_best_scoring_intro():
    """v83 (reciter-C: basmala recited, isti'adha NOT): forcing absent isti'adha
    tokens drags the short-window alignment and kills the surah scores. The
    pass must try intro hypotheses (full / basmala-only / none) and keep the
    one whose opening words score best — here the basmala-only (4-token)
    prefix — so the genuine dropped opening is recovered instead of rejected."""
    quran, mapped = _small_lead_fixture()

    def fake_align(wav_path, transcript, device, model_name,
                   window_start_ms=0, window_sec=None):
        prefix_len = len(transcript) - 4
        good = prefix_len == 4  # only the basmala-only hypothesis aligns well
        spans = []
        for k in range(len(transcript)):
            if k < prefix_len:
                spans.append({"start_ms": 100 * k, "end_ms": 100 * k + 80, "score": 0.7})
            else:
                j = k - prefix_len
                spans.append({
                    "start_ms": 500 + j * 1000,
                    "end_ms": 1300 + j * 1000,
                    "score": 0.6 if good else 0.01,
                })
        return spans

    saved = q.ctc_align_window
    q.ctc_align_window = fake_align
    try:
        result, report = q.recover_underrun_opening_ctc(
            mapped_words=mapped, quran_words=quran, args=_openctc_args(),
            wav_path="dummy.wav", chapter_id=25, duration_ms=60000,
        )
    finally:
        q.ctc_align_window = saved

    assert report["prefix_len"] == 4, report
    assert "small_lead_rejected" not in report, report
    assert report["recovered"] == 4, report
    for i in range(4):
        assert result[i]["estimated"] is False
        assert result[i].get("ctc_opening_recovered") is True


def test_opening_ctc_prefix_hypothesis_near_tie_keeps_longer_prefix():
    """v83 guard: a SHORTER prefix hypothesis must beat the longer one by a
    clear margin. On a near-tie (here: empty prefix mean 0.61 vs basmala-only
    0.60) the longer, more conservative prefix wins so the empty hypothesis
    cannot steal real intro audio for the surah words."""
    quran, mapped = _small_lead_fixture()

    def fake_align(wav_path, transcript, device, model_name,
                   window_start_ms=0, window_sec=None):
        prefix_len = len(transcript) - 4
        if prefix_len == 9:
            opening_score = 0.30
        elif prefix_len == 4:
            opening_score = 0.60
        else:  # empty prefix: marginally higher, within the 0.02 margin
            opening_score = 0.61
        spans = []
        for k in range(len(transcript)):
            if k < prefix_len:
                spans.append({"start_ms": 100 * k, "end_ms": 100 * k + 80, "score": 0.7})
            else:
                j = k - prefix_len
                spans.append({
                    "start_ms": 500 + j * 1000,
                    "end_ms": 1300 + j * 1000,
                    "score": opening_score,
                })
        return spans

    saved = q.ctc_align_window
    q.ctc_align_window = fake_align
    try:
        result, report = q.recover_underrun_opening_ctc(
            mapped_words=mapped, quran_words=quran, args=_openctc_args(),
            wav_path="dummy.wav", chapter_id=25, duration_ms=60000,
        )
    finally:
        q.ctc_align_window = saved

    assert report["prefix_len"] == 4, report
    assert report["recovered"] == 4, report


def test_opening_ctc_reject_with_window_squeeze_flags_anchor_suspect():
    """v84 (reciter-C): when the strict gate rejects AND some hypothesis was
    squeezed against the window end, the first-real anchor bounding the window
    is itself suspect (Whisper timing compression) — flag anchor_suspect so
    the caller escalates to the full-surah CTC pass."""
    quran, mapped = _small_lead_fixture()

    def fake_align(wav_path, transcript, device, model_name,
                   window_start_ms=0, window_sec=None):
        n = len(transcript)
        window_end = int((window_sec or 6.0) * 1000)
        spans = []
        for k in range(n):
            start = max(0, window_end - (n - k) * 120)
            spans.append({"start_ms": start, "end_ms": start + 110, "score": 0.01})
        spans[-1]["end_ms"] = window_end  # squeezed against the boundary
        return spans

    saved = q.ctc_align_window
    q.ctc_align_window = fake_align
    try:
        result, report = q.recover_underrun_opening_ctc(
            mapped_words=mapped, quran_words=quran, args=_openctc_args(),
            wav_path="dummy.wav", chapter_id=25, duration_ms=60000,
        )
    finally:
        q.ctc_align_window = saved

    assert report.get("small_lead_rejected"), report
    assert report.get("anchor_suspect") is True, report
    assert report["recovered"] == 0
    for i in range(4):
        assert result[i].get("estimated") is True


def test_opening_ctc_squeeze_from_uncredible_losing_hypothesis_no_escalation():
    """v84 refinement: a window-end squeeze only counts when it comes from a
    CREDIBLE hypothesis (the selected best, or one whose intro prefix aligned
    with high confidence). Here a LOSING hypothesis with a garbage prefix score
    touches the boundary while the winning hypothesis is well clear of it —
    no anchor_suspect, no escalation."""
    quran, mapped = _small_lead_fixture()

    def fake_align(wav_path, transcript, device, model_name,
                   window_start_ms=0, window_sec=None):
        window_end = int((window_sec or 6.0) * 1000)
        n_opening = 4
        prefix_len = len(transcript) - n_opening
        spans = []
        if prefix_len > 0:
            # non-empty prefix hypothesis: garbage prefix scores (0.05) and its
            # opening squeezed against the window end; low opening scores so it
            # LOSES the mean-score selection.
            for k in range(prefix_len):
                spans.append({"start_ms": 50 * k, "end_ms": 50 * k + 40, "score": 0.05})
            for j in range(n_opening):
                start = window_end - (n_opening - j) * 120
                spans.append({"start_ms": start, "end_ms": start + 110, "score": 0.02})
            spans[-1]["end_ms"] = window_end
        else:
            # empty-prefix hypothesis: well inside the window, clearly beats
            # the squeezed hypotheses on mean score (0.045 > 0.02 + margin)
            # so it WINS selection, yet stays below the 0.05 strict floor so
            # the gate still rejects.
            for j in range(n_opening):
                start = 500 + j * 600
                spans.append({"start_ms": start, "end_ms": start + 400, "score": 0.045})
        return spans

    saved = q.ctc_align_window
    q.ctc_align_window = fake_align
    try:
        result, report = q.recover_underrun_opening_ctc(
            mapped_words=mapped, quran_words=quran, args=_openctc_args(),
            wav_path="dummy.wav", chapter_id=25, duration_ms=60000,
        )
    finally:
        q.ctc_align_window = saved

    # strict gate rejects (scores 0.02 < 0.05 floor) but NO anchor_suspect:
    # the only squeezed hypothesis lost AND its prefix score is not credible.
    assert report.get("small_lead_rejected"), report
    assert not report.get("anchor_suspect"), report


def test_underrun_ctc_force_requires_confident_mean_score():
    """v84 escalation: with the bad-fraction gate bypassed (force=True), the
    whole-surah forced alignment is adopted ONLY when every word places with a
    confident mean score; a low-confidence alignment changes nothing."""
    quran = _surah14()
    mapped = _underrun_mapped(quran)

    def fake_align(wav_path, transcript, device, model_name,
                   window_start_ms=0, window_sec=None):
        local_prefix_len = len(transcript) - len(quran)
        spans = []
        for k in range(len(transcript)):
            if k < local_prefix_len:
                spans.append({"start_ms": 100 * k, "end_ms": 100 * k + 90, "score": 0.5})
            else:
                j = k - local_prefix_len
                spans.append({"start_ms": 3000 + j * 600, "end_ms": 3400 + j * 600, "score": 0.02})
        return spans

    saved = q.ctc_align_window
    q.ctc_align_window = fake_align
    try:
        result, report = q.recover_underrun_surah_ctc(
            mapped_words=mapped, quran_words=quran,
            args=types.SimpleNamespace(device_resolved="cpu"),
            wav_path="dummy.wav", chapter_id=103, duration_ms=30000,
            force=True, force_reason="opening_window_squeeze",
        )
    finally:
        q.ctc_align_window = saved

    assert report.get("forced") == "opening_window_squeeze"
    assert report.get("error") == "force_low_confidence", report
    assert report["recovered"] == 0
    assert result == mapped


def test_opening_ctc_small_lead_strict_accepts_clean_alignment():
    """v81 (reciter-C Al-Kafirun class, happy path): an all-placed, confident,
    well-shaped alignment of a budget-sized non-intro lead IS adopted — the
    dropped opening words get REAL measured spans before the first anchor."""
    quran, mapped = _small_lead_fixture()
    result, report = _run_small_lead_with_fake_align(
        mapped, quran,
        lambda j: {"start_ms": 500 + j * 1000, "end_ms": 1300 + j * 1000, "score": 0.7},
    )
    assert "small_lead_rejected" not in report
    assert report["recovered"] == 4, report
    for i in range(4):
        assert result[i]["estimated"] is False
        assert result[i].get("ctc_opening_recovered") is True
        assert result[i]["end_ms"] <= 5000
    starts = [result[i]["start_ms"] for i in range(4)]
    assert starts == sorted(starts)


def test_opening_ctc_recovers_dropped_opening(monkeypatch=None):
    """The Al-Furqan class: a long, otherwise-healthy surah whose first 15 words
    were interpolated behind an over-detected intro. The opening window is
    force-aligned and those words get REAL spans clamped before the first real
    anchor."""
    lead = 15
    total = 20
    quran = [_qword(f"w{i}") for i in range(total)]
    first_real = 20000
    mapped = [dict(estimated=True, start_ms=19000 + i, end_ms=19100 + i) for i in range(lead)]
    mapped += [
        dict(start_ms=first_real + i * 300, end_ms=first_real + 150 + i * 300, estimated=False)
        for i in range(total - lead)
    ]

    def fake_align(wav_path, transcript, device, model_name, window_start_ms=0, window_sec=None):
        # prefix (isti'adha + basmala) then the 15 opening words, all inside 0..first_real.
        prefix_len = len(transcript) - lead
        spans = []
        for k in range(len(transcript)):
            if k < prefix_len:
                spans.append({"start_ms": 200 * k, "end_ms": 200 * k + 120, "score": 0.7})
            else:
                base = 3000 + (k - prefix_len) * 1000
                spans.append({"start_ms": base, "end_ms": base + 800, "score": 0.7})
        return spans

    saved = q.ctc_align_window
    q.ctc_align_window = fake_align
    try:
        result, report = q.recover_underrun_opening_ctc(
            mapped_words=mapped, quran_words=quran, args=_openctc_args(),
            wav_path="dummy.wav", chapter_id=25, duration_ms=1200000,
        )
    finally:
        q.ctc_align_window = saved

    assert report["attempted"] is True, report
    assert report["recovered"] == lead, report
    for i in range(lead):
        assert result[i]["estimated"] is False
        assert result[i].get("ctc_opening_recovered") is True
        assert result[i]["end_ms"] <= first_real
        assert result[i]["end_ms"] > result[i]["start_ms"]
    # monotonic non-decreasing starts across the recovered opening
    starts = [result[i]["start_ms"] for i in range(lead)]
    assert starts == sorted(starts), starts
    # interior healthy words untouched
    assert result[lead]["start_ms"] == first_real


def test_opening_ctc_gate_met_but_ctc_deps_missing():
    """Gate is met (long dropped opening) but the CTC deps are unavailable: the
    pass must degrade gracefully — no mutation, attempted stays False, and the
    report records the ctc-deps-missing reason."""
    lead = 15
    total = 20
    quran = [_qword(f"w{i}") for i in range(total)]
    first_real = 20000
    mapped = [dict(estimated=True, start_ms=19000 + i, end_ms=19100 + i) for i in range(lead)]
    mapped += [
        dict(start_ms=first_real + i * 300, end_ms=first_real + 150 + i * 300, estimated=False)
        for i in range(total - lead)
    ]
    before = [dict(w) for w in mapped]
    saved_torch = sys.modules.pop("torch", None)
    try:
        result, report = q.recover_underrun_opening_ctc(
            mapped_words=mapped, quran_words=quran, args=_openctc_args(),
            wav_path="dummy.wav", chapter_id=25, duration_ms=1200000,
        )
    finally:
        if saved_torch is not None:
            sys.modules["torch"] = saved_torch
    assert report["attempted"] is False
    assert report["recovered"] == 0
    assert "ctc-deps-missing" in report.get("error", "")
    assert result == before, "no word may be mutated when CTC deps are missing"


def test_opening_ctc_disabled_flag():
    quran = [_qword(f"w{i}") for i in range(20)]
    mapped = [dict(estimated=True) for _ in range(15)] + [
        dict(start_ms=20000, end_ms=20200, estimated=False) for _ in range(5)
    ]
    args = _openctc_args()
    args.no_opening_ctc = True
    result, report = q.recover_underrun_opening_ctc(
        mapped_words=mapped, quran_words=quran, args=args,
        wav_path="dummy.wav", chapter_id=25, duration_ms=60000,
    )
    assert report["attempted"] is False and report["recovered"] == 0


# ---------------------------------------------------------------------------
# v68: Whisper 30s-decode-chunk intro-inflation guard (reciter-B Al-Furqan/Al-Kahf)
# ---------------------------------------------------------------------------

def _inflated_mapped():
    """Reproduce the reciter-B artifact shape: an inflated ~30s basmala block, a
    few surah words mis-aligned at ~0ms inside it, the true opening left
    estimated, and a solid genuine-recitation anchor resuming at 30000ms."""
    total = 25
    mapped = []
    # 0..11: true opening, all estimated (fake interpolated timestamps 0..3300).
    for i in range(12):
        mapped.append(dict(start_ms=i * 300, end_ms=i * 300 + 300, estimated=True))
    # 12..14: mis-aligned REAL words parked on the basmala audio at ~0ms.
    mapped.append(dict(start_ms=0, end_ms=420, estimated=False))
    mapped.append(dict(start_ms=420, end_ms=720, estimated=False))
    mapped.append(dict(start_ms=720, end_ms=14000, estimated=False))
    # 15..18: more mis-aligned real words still inside the 30s intro span.
    for base in (20779, 21961, 24053, 29017):
        mapped.append(dict(start_ms=base, end_ms=base + 900, estimated=False))
    # 19..24: genuine recitation resumes at/after 30000ms (the anchor).
    for j in range(6):
        s = 30000 + j * 1000
        mapped.append(dict(start_ms=s, end_ms=s + 800, estimated=False))
    assert len(mapped) == total
    return mapped


def test_inflation_plan_detects_artifact():
    plan = q.opening_ctc_intro_inflation_plan(_inflated_mapped(), 29980)
    assert plan is not None
    # first solid post-intro anchor is index 19 @ 30000ms
    assert plan["anchor_index"] == 19
    assert plan["anchor_start_ms"] == 30000
    # every real word before the anchor that sits inside the intro is demoted
    assert plan["demote_indices"] == [12, 13, 14, 15, 16, 17, 18]


def test_inflation_plan_ignores_normal_intro():
    """A real ~5s basmala end must never trigger the guard, even with a clean
    opening that starts right after it."""
    mapped = [dict(start_ms=5200 + i * 300, end_ms=5400 + i * 300, estimated=False)
              for i in range(20)]
    assert q.opening_ctc_intro_inflation_plan(mapped, 5000) is None
    # None intro end is a no-op too
    assert q.opening_ctc_intro_inflation_plan(mapped, None) is None


def test_inflation_plan_no_misaligned_word():
    """Inflated intro end but the opening correctly starts AFTER it (no surah
    word parked inside the intro): nothing to re-align, so no plan."""
    mapped = [dict(start_ms=30200 + i * 300, end_ms=30400 + i * 300, estimated=False)
              for i in range(20)]
    assert q.opening_ctc_intro_inflation_plan(mapped, 29980) is None


def test_inflation_plan_requires_solid_anchor():
    """A lone post-intro real word (shorter than the anchor run) is not accepted
    as the re-alignment boundary."""
    mapped = [dict(start_ms=0, end_ms=420, estimated=False)]          # mis-aligned
    mapped += [dict(estimated=True) for _ in range(5)]                # estimated
    mapped.append(dict(start_ms=30000, end_ms=30500, estimated=False))  # lone real
    mapped += [dict(estimated=True) for _ in range(5)]                # then estimated again
    assert q.opening_ctc_intro_inflation_plan(mapped, 29980, min_anchor_run=3) is None


def test_inflation_guard_fires_opening_ctc():
    """End-to-end (with a stubbed CTC): the inflated-intro guard demotes the
    mis-aligned words so the opening-CTC gate — which previously never fired
    (leading run 12 <= budget 13, first_real_start 0) — now re-aligns the whole
    opening window and every opening word before the anchor gets a REAL span."""
    mapped = _inflated_mapped()
    quran = [_qword(f"w{i}") for i in range(len(mapped))]
    anchor = 19
    anchor_start = 30000

    def fake_align(wav_path, transcript, device, model_name, window_start_ms=0, window_sec=None):
        prefix_len = len(transcript) - anchor
        spans = []
        for k in range(len(transcript)):
            if k < prefix_len:
                spans.append({"start_ms": 200 * k, "end_ms": 200 * k + 120, "score": 0.7})
            else:
                base = 4000 + (k - prefix_len) * 1200
                spans.append({"start_ms": base, "end_ms": base + 900, "score": 0.7})
        return spans

    saved = q.ctc_align_window
    q.ctc_align_window = fake_align
    try:
        result, report = q.recover_underrun_opening_ctc(
            mapped_words=mapped, quran_words=quran, args=_openctc_args(),
            wav_path="dummy.wav", chapter_id=25, duration_ms=1200000,
            intro_audio_end_ms=29980,
        )
    finally:
        q.ctc_align_window = saved

    assert report.get("intro_inflation", {}).get("demoted") == 7, report
    assert report["attempted"] is True, report
    assert report["leading_estimated"] == anchor, report
    assert report["recovered"] == anchor, report
    for i in range(anchor):
        assert result[i]["estimated"] is False
        assert result[i].get("ctc_opening_recovered") is True
        assert result[i]["end_ms"] <= anchor_start
        assert result[i]["end_ms"] > result[i]["start_ms"]
    starts = [result[i]["start_ms"] for i in range(anchor)]
    assert starts == sorted(starts), starts
    # genuine post-intro recitation is untouched
    assert result[anchor]["start_ms"] == anchor_start


def test_inflation_plan_finds_late_anchor():
    """v68: the genuine resume anchor can appear late (e.g. a long, very fast
    opening leaves >60 leading words mis-timed/estimated). The default scan
    limit (120) must still find it so the artifact is not left unrepaired."""
    anchor = 90
    mapped = []
    # a mis-aligned real word parked in the intro at ~0ms
    mapped.append(dict(start_ms=0, end_ms=500, estimated=False))
    # then a long stretch of estimated opening words up to the anchor
    for _ in range(anchor - 1):
        mapped.append(dict(estimated=True))
    # genuine recitation resumes at/after 30000ms
    for j in range(5):
        s = 30000 + j * 800
        mapped.append(dict(start_ms=s, end_ms=s + 600, estimated=False))
    plan = q.opening_ctc_intro_inflation_plan(mapped, 29980)
    assert plan is not None
    assert plan["anchor_index"] == anchor
    assert plan["demote_indices"] == [0]
    # with the old 60-word scan window the late anchor is missed → no plan
    assert q.opening_ctc_intro_inflation_plan(mapped, 29980, scan_limit=60) is None


def test_inflation_plan_boundary_word_at_intro_end():
    """v69 (reciter-B At-Taghabun 064): the mis-aligned opening word can be parked
    EXACTLY at the inflated intro end, not scattered at ~0ms. Here the 30s (well,
    ~17.7s) basmala block swallowed the whole first verse; its last real word
    (وَهُوَ) landed right on the boundary (start == intro end) while the true
    opening stayed estimated, leaving the leading run one short of the intro
    budget so the opening-CTC gate never fired. Demoting every real word before
    the genuine resume anchor — keyed on the anchor start, not the intro cutoff —
    now catches the boundary word."""
    intro_end = 17740
    mapped = []
    # 0..11: true opening (verse 1), all estimated with fake interpolated stamps.
    for i in range(12):
        mapped.append(dict(start_ms=i * 100, end_ms=i * 100 + 100, estimated=True))
    # 12: last real word of verse 1 parked EXACTLY at the inflated intro end
    # (start == intro_end, i.e. >= cutoff, so the old cutoff filter missed it).
    mapped.append(dict(start_ms=intro_end, end_ms=intro_end + 200, estimated=False))
    # 13..16: rest of verse 1 still estimated (fake stamps after the boundary).
    for i in range(4):
        mapped.append(dict(start_ms=18900 + i * 90, end_ms=18990 + i * 90, estimated=True))
    # 17..19: genuine recitation (verse 2) resumes — the solid anchor.
    for j in range(3):
        s = 18623 + j * 200
        mapped.append(dict(start_ms=s, end_ms=s + 150, estimated=False))

    # Old cutoff-based rule saw the boundary word at 17740 >= cutoff (17540) and
    # demoted nothing → returned None (the bug). The anchor-start rule fixes it.
    plan = q.opening_ctc_intro_inflation_plan(mapped, intro_end)
    assert plan is not None
    assert plan["anchor_index"] == 17
    assert plan["anchor_start_ms"] == 18623
    assert plan["demote_indices"] == [12]


# ---------------------------------------------------------------------------
# v75: shorter basmala-inflation — Whisper merges the basmala AND the entire
# first ayah into one ~14s "basmala" block (Al-Insan 076 / reciter-C).
# ---------------------------------------------------------------------------
def _basmala_inflated_076_mapped():
    """Al-Insan 076: Whisper merged the basmala and the whole 11-word verse 1
    into a single ~14s 'basmala' block, so every verse-1 word sits inside
    0..14120ms (some estimated, some real but parked on the basmala audio) and
    verse 2 resumes cleanly at 14120ms (the solid post-intro anchor)."""
    mapped = []
    # 0..4: verse-1 words with fake interpolated stamps (estimated).
    for i in range(5):
        mapped.append(dict(start_ms=i * 300, end_ms=i * 300 + 300, estimated=True))
    # 5..8: verse-1 REAL words parked inside the basmala audio.
    mapped.append(dict(start_ms=0, end_ms=860, estimated=False))
    mapped.append(dict(start_ms=860, end_ms=1300, estimated=False))
    mapped.append(dict(start_ms=1300, end_ms=12120, estimated=False))
    mapped.append(dict(start_ms=12120, end_ms=12787, estimated=False))
    # 9: verse-1 estimated.
    mapped.append(dict(start_ms=12787, end_ms=13453, estimated=True))
    # 10: verse-1 REAL word, still inside the intro span.
    mapped.append(dict(start_ms=12300, end_ms=14120, estimated=False))
    # 11..13: verse 2 resumes at the intro end — the solid anchor.
    for s in (14120, 15600, 16640):
        mapped.append(dict(start_ms=s, end_ms=s + 900, estimated=False))
    return mapped


def test_inflation_plan_detects_basmala_merged_first_ayah():
    """v75: a ~14s merged basmala+verse-1 block (below the old 15000ms floor) is
    now detected; every real verse-1 word parked before verse 2 is demoted."""
    mapped = _basmala_inflated_076_mapped()
    plan = q.opening_ctc_intro_inflation_plan(mapped, 14120)
    assert plan is not None
    assert plan["anchor_index"] == 11
    assert plan["anchor_start_ms"] == 14120
    assert plan["demote_indices"] == [5, 6, 7, 8, 10]
    # The old 15000ms floor missed it entirely — the bug the user reported.
    assert q.opening_ctc_intro_inflation_plan(mapped, 14120, inflated_intro_ms=15000) is None


def test_inflation_plan_ignores_long_clean_basmala():
    """A genuinely long (~13s) isti'adha+basmala intro with the opening resuming
    correctly right after it must NOT fire, even below the old 15000ms floor:
    the opening IS the anchor at index 0, so nothing is parked inside the intro."""
    mapped = [
        dict(start_ms=13000 + i * 300, end_ms=13200 + i * 300, estimated=False)
        for i in range(20)
    ]
    assert q.opening_ctc_intro_inflation_plan(mapped, 12800) is None


def test_inflation_bypasses_intro_budget_gate():
    """v75 end-to-end (stubbed CTC): after demoting the merged verse-1 words the
    leading run (11) is <= the intro budget (13), which previously made the
    opening-CTC gate skip and forgave the whole dropped first ayah as 'intro'.
    With positive inflation evidence the budget gate is bypassed and the entire
    opening is re-aligned; verse 2 is left untouched."""
    mapped = _basmala_inflated_076_mapped()
    quran = [_qword(f"w{i}") for i in range(len(mapped))]  # 14 words, no basmala/muqatta'at
    first_real = 14120

    def fake_align(wav_path, transcript, device, model_name,
                   window_start_ms=0, window_sec=None):
        prefix_len = len(transcript) - 11  # 11 opening words after the basmala prefix
        spans = []
        for k in range(len(transcript)):
            if k < prefix_len:
                spans.append({"start_ms": 150 * k, "end_ms": 150 * k + 120, "score": 0.7})
            else:
                i = k - prefix_len
                base = 5000 + i * 800  # 5000..13000, all end < first_real
                spans.append({"start_ms": base, "end_ms": base + 600, "score": 0.7})
        return spans

    saved = q.ctc_align_window
    q.ctc_align_window = fake_align
    try:
        result, report = q.recover_underrun_opening_ctc(
            mapped_words=mapped, quran_words=quran, args=_openctc_args(),
            wav_path="dummy.wav", chapter_id=76, duration_ms=320296,
            intro_audio_end_ms=first_real,
        )
    finally:
        q.ctc_align_window = saved

    assert report.get("intro_inflation", {}).get("demoted") == 5, report
    assert report["leading_estimated"] == 11, report
    assert report["intro_budget"] == 13, report   # leading (11) <= budget (13) ...
    assert report["attempted"] is True, report    # ... yet it still fired
    assert report["recovered"] == 11, report
    for i in range(11):
        assert result[i]["estimated"] is False, (i, result[i])
        assert result[i].get("ctc_opening_recovered") is True, (i, result[i])
        assert result[i]["end_ms"] <= first_real, (i, result[i])
    # verse 2 anchor is genuine recitation — must stay exactly as measured.
    assert result[11]["start_ms"] == first_real
    assert not result[11].get("ctc_opening_recovered")


def test_inflation_guard_absent_when_intro_normal():
    """With a normal intro end, the guard is a no-op and the pre-existing
    intro-budget gate still governs (here: normal short intro with an
    all-intro-vocab lead, no firing)."""
    quran = [_qword(n) for n in ["بسم", "الله", "الرحمن", "الرحيم"]] + [
        _qword(f"w{i}") for i in range(16)
    ]
    mapped = [dict(estimated=True) for _ in range(4)] + [
        dict(start_ms=5000 + i * 100, end_ms=5090 + i * 100, estimated=False)
        for i in range(16)
    ]
    result, report = q.recover_underrun_opening_ctc(
        mapped_words=mapped, quran_words=quran, args=_openctc_args(),
        wav_path="dummy.wav", chapter_id=25, duration_ms=60000,
        intro_audio_end_ms=5000,
    )
    assert "intro_inflation" not in report
    assert report["attempted"] is False and report["recovered"] == 0


# ---------------------------------------------------------------------------
# v67: duplicate consecutive-ayah CTC realignment (Ash-Sharh 94:5-6)
# ---------------------------------------------------------------------------

def _dup_args():
    return types.SimpleNamespace(device_resolved="cpu")


def _ashsharh_mapped():
    """A clean, fully-real Ash-Sharh word list (27 words) with verse keys.

    Ayat 5 (فَإِنَّ مَعَ ٱلۡعُسۡرِ يُسۡرًا) and 6 (إِنَّ مَعَ ٱلۡعُسۡرِ يُسۡرًا)
    are the consecutive near-identical pair (differ only by the leading فَ)."""
    ayat = [
        ("94:1", ["الم", "نشرح", "لك", "صدرك"]),
        ("94:2", ["ووضعنا", "عنك", "وزرك"]),
        ("94:3", ["الذي", "انقض", "ظهرك"]),
        ("94:4", ["ورفعنا", "لك", "ذكرك"]),
        ("94:5", ["فان", "مع", "العسر", "يسرا"]),
        ("94:6", ["ان", "مع", "العسر", "يسرا"]),
        ("94:7", ["فاذا", "فرغت", "فانصب"]),
        ("94:8", ["والي", "ربك", "فارغب"]),
    ]
    mapped = []
    t = 0
    for vk, norms in ayat:
        for pos, nm in enumerate(norms, 1):
            mapped.append({
                "verse_key": vk, "position": pos, "text": nm, "norm": nm,
                "start_ms": t, "end_ms": t + 500, "score": 1.0, "estimated": False,
                "method": "dp_global",
            })
            t += 500
    return mapped


def _ashsharh_mapped_bugged():
    """Reproduce the real reciter-A/094 bug: Whisper transcribed ONE copy of
    مع العسر يسرا, the DP assigned it to ayah 6, so ayah 5's مع/العسر/يسرا are
    ESTIMATED on a fake sub-second window and the dropped ayah-6 audio sits in a
    large unused gap before ayah 7 (فاذا @ 23300ms)."""
    m = _ashsharh_mapped()
    m[12]["start_ms"], m[12]["end_ms"] = 16360, 17080          # 94:4 ذكرك (anchor)
    m[13]["start_ms"], m[13]["end_ms"] = 17080, 17348          # 94:5 فان (split)
    for off, k in enumerate((14, 15, 16)):                     # 94:5 مع/العسر/يسرا
        m[k]["start_ms"] = 17348 + off * 268
        m[k]["end_ms"] = m[k]["start_ms"] + 268
        m[k]["estimated"] = True
        m[k]["score"] = 0
        m[k]["method"] = "main_split_prev"
    m[17]["start_ms"], m[17]["end_ms"] = 18088, 18420          # 94:6 ان (recovered)
    m[18]["start_ms"], m[18]["end_ms"] = 18420, 18840          # 94:6 مع
    m[19]["start_ms"], m[19]["end_ms"] = 18840, 19440          # 94:6 العسر
    m[20]["start_ms"], m[20]["end_ms"] = 19440, 20170          # 94:6 يسرا
    m[21]["start_ms"], m[21]["end_ms"] = 23300, 24700          # 94:7 فاذا (anchor)
    return m


def test_ayat_near_identical_conjunction():
    assert q.ayat_near_identical(["فان", "مع", "العسر", "يسرا"],
                                 ["ان", "مع", "العسر", "يسرا"])
    assert q.ayat_near_identical(["ان", "مع", "العسر", "يسرا"],
                                 ["ان", "مع", "العسر", "يسرا"])
    # differing interior token -> not the same recitation
    assert not q.ayat_near_identical(["فان", "مع", "اليسر", "يسرا"],
                                     ["ان", "مع", "العسر", "يسرا"])
    # different lengths / unrelated ayat
    assert not q.ayat_near_identical(["فان", "مع", "العسر"],
                                     ["ان", "مع", "العسر", "يسرا"])
    assert not q.ayat_near_identical(["الم", "نشرح", "لك", "صدرك"],
                                     ["ووضعنا", "عنك", "وزرك"])
    # below min_len (single-token coincidence must never trigger)
    assert not q.ayat_near_identical(["مع"], ["مع"])


def test_adjacent_dup_regions_detects_ashsharh():
    regs = q.adjacent_duplicate_ayah_regions(_ashsharh_mapped())
    assert (13, 20, ["94:5", "94:6"]) in regs
    # exactly one region for this surah
    assert len(regs) == 1


def test_adjacent_dup_regions_ignores_distinct_and_separated():
    # No two adjacent ayat are near-identical here.
    m = [
        {"verse_key": "2:1", "position": 1, "norm": "الم", "text": "الم"},
        {"verse_key": "2:2", "position": 1, "norm": "ذلك", "text": "ذلك"},
        {"verse_key": "2:2", "position": 2, "norm": "الكتاب", "text": "الكتاب"},
        {"verse_key": "2:3", "position": 1, "norm": "لا", "text": "لا"},
        {"verse_key": "2:3", "position": 2, "norm": "ريب", "text": "ريب"},
    ]
    assert q.adjacent_duplicate_ayah_regions(m) == []


def _kafirun_mapped():
    """Al-Kafirun-shaped word map where 109:3 == 109:5 with the DIFFERENT 109:4
    between them (the A [X] A pattern the adjacent detector cannot see)."""
    def toks(vk, norms):
        return [{"verse_key": vk, "position": p + 1, "norm": t, "text": t}
                for p, t in enumerate(norms)]
    m = []
    m += toks("109:1", ["قل", "يايها", "الكافرون"])          # idx 0..2
    m += toks("109:2", ["لا", "اعبد", "ما", "تعبدون"])        # idx 3..6
    m += toks("109:3", ["ولا", "انتم", "عابدون", "ما", "اعبد"])   # idx 7..11
    m += toks("109:4", ["ولا", "انا", "عابد", "ما", "عبدتم"])     # idx 12..16
    m += toks("109:5", ["ولا", "انتم", "عابدون", "ما", "اعبد"])   # idx 17..21 (== 109:3)
    m += toks("109:6", ["لكم", "دينكم", "ولي", "دين"])       # idx 22..25
    return m


def test_nonadjacent_dup_regions_detects_kafirun():
    regs = q.nonadjacent_duplicate_ayah_regions(_kafirun_mapped())
    # spans the first copy (109:3 @ idx7) through the last copy (109:5 @ idx21),
    # inclusive of the intervening 109:4
    assert (7, 21, ["109:3", "109:4", "109:5"]) in regs
    assert len(regs) == 1


def test_nonadjacent_dup_regions_respects_max_intervening():
    # 109:3 and 109:5 are 1 apart; with 0 allowed intervening (clamped to >=1 it
    # still allows exactly one) they are found, but a copy 2 ayat away is not.
    m = _kafirun_mapped()
    # make 109:6 a third identical copy of 109:3 -> 3 ayat away from 109:3,
    # 1 ayah (109:5) away from... it is separate; with default max_intervening=1
    # only the (109:3,109:5) pair forms a region, never a 109:3..109:6 giant span.
    m2 = m[:22] + [
        {"verse_key": "109:6", "position": p + 1, "norm": t, "text": t}
        for p, t in enumerate(["ولا", "انتم", "عابدون", "ما", "اعبد"])
    ]
    regs = q.nonadjacent_duplicate_ayah_regions(m2, max_intervening_ayat=1)
    # 109:3/109:5 consume ayat, leaving 109:6 unpaired -> single compact region
    assert (7, 21, ["109:3", "109:4", "109:5"]) in regs
    for (a, b, _vks) in regs:
        assert b - a + 1 <= 40


def test_nonadjacent_dup_regions_span_cap():
    # A span wider than the word cap is skipped entirely.
    regs = q.nonadjacent_duplicate_ayah_regions(_kafirun_mapped(), max_span_words=5)
    assert regs == []


def test_nonadjacent_dup_regions_ignores_true_refrain():
    # A refrain separated by MORE than max_intervening ayat is never a region.
    def toks(vk, norms):
        return [{"verse_key": vk, "position": p + 1, "norm": t, "text": t}
                for p, t in enumerate(norms)]
    m = []
    m += toks("55:13", ["فباي", "الاء", "ربكما", "تكذبان"])
    m += toks("55:14", ["خلق", "الانسان"])
    m += toks("55:15", ["وخلق", "الجان"])
    m += toks("55:16", ["فباي", "الاء", "ربكما", "تكذبان"])  # 3 ayat away
    assert q.nonadjacent_duplicate_ayah_regions(m, max_intervening_ayat=1) == []


def _always_estimate(_w):
    return True


def _never_estimate(_w):
    return False


def test_merge_adjacent_wins_over_grown_nonadjacent():
    """A grown non-adjacent span that reaches into an adjacent-duplicate region
    must be dropped, and the adjacent region must survive (precedence)."""
    result = [{} for _ in range(30)]
    adj = [(10, 15, ["x:1", "x:2"])]
    # non-adjacent span (2..4) that, once grown across estimates, would overlap
    # the adjacent region at 10..15
    na = [(2, 4, ["y:1", "y:2", "y:3"])]
    merged = q._merge_dup_ayah_regions(
        result, adj, na, _always_estimate, max_span_words=40
    )
    # adjacent region kept; the grown non-adjacent span is dropped (it overlapped)
    assert (10, 15, ["x:1", "x:2"]) in merged
    assert all(not (b < 10 or a > 15) for (a, b, _v) in [adj[0]])
    assert not any(vks == ["y:1", "y:2", "y:3"] for (_a, _b, vks) in merged)


def test_merge_grows_and_keeps_disjoint_nonadjacent():
    """With no adjacent regions and no estimates to grow into, the non-adjacent
    span is kept unchanged."""
    result = [{} for _ in range(30)]
    na = [(7, 21, ["109:3", "109:4", "109:5"])]
    merged = q._merge_dup_ayah_regions(
        result, [], na, _never_estimate, max_span_words=40
    )
    assert merged == [(7, 21, ["109:3", "109:4", "109:5"])]


def test_nonadjacent_dup_regions_rejects_numbering_gap():
    """An unparseable / missing intermediate verse key breaks strict +1 stepping,
    so no region forms even if the endpoints are identical."""
    def toks(vk, norms):
        return [{"verse_key": vk, "position": p + 1, "norm": t, "text": t}
                for p, t in enumerate(norms)]
    m = []
    m += toks("7:1", ["الم", "ذلك"])
    m += toks(None, ["intro", "token"])          # unparseable middle
    m += toks("7:3", ["الم", "ذلك"])             # identical endpoints
    assert q.nonadjacent_duplicate_ayah_regions(m, max_intervening_ayat=1) == []


def test_dup_ayat_ctc_realigns_region():
    m = _ashsharh_mapped_bugged()
    assert q.recount_matched_after_repairs(m) == 24  # 3 estimated (94:5 tail)

    def fake_align(wav_path, transcript_texts, device, model_name,
                   window_start_ms=0, window_sec=None):
        assert len(transcript_texts) == 8
        return [{"start_ms": 17080 + k * 750, "end_ms": 17080 + k * 750 + 650,
                 "score": 0.6} for k in range(8)]

    saved = q.ctc_align_window
    q.ctc_align_window = fake_align
    try:
        result, report = q.recover_duplicate_adjacent_ayat_ctc(
            mapped_words=m, args=_dup_args(), wav_path="d.wav",
            chapter_id=94, duration_ms=30000,
        )
    finally:
        q.ctc_align_window = saved

    assert report["attempted"] is True
    assert report["recovered"] == 3, report
    for k in range(13, 21):
        assert result[k].get("estimated") is False
        assert result[k].get("start_ms") is not None
        assert result[k].get("ctc_dup_ayat_recovered") is True
    # region timeline monotonic & non-overlapping
    for k in range(13, 20):
        assert result[k]["end_ms"] <= result[k + 1]["start_ms"] + 1
    assert q.recount_matched_after_repairs(result) == 27
    assert any(r.get("adopted") for r in report["regions"])


def test_dup_ayat_ctc_incomplete_alignment_not_adopted():
    m = _ashsharh_mapped_bugged()

    def fake_align(wav_path, transcript_texts, device, model_name,
                   window_start_ms=0, window_sec=None):
        # first token unplaceable -> region must NOT be adopted (no partial)
        spans = [None]
        spans += [{"start_ms": 17080 + k * 750, "end_ms": 17080 + k * 750 + 650,
                   "score": 0.6} for k in range(1, 8)]
        return spans

    saved = q.ctc_align_window
    q.ctc_align_window = fake_align
    try:
        result, report = q.recover_duplicate_adjacent_ayat_ctc(
            mapped_words=m, args=_dup_args(), wav_path="d.wav",
            chapter_id=94, duration_ms=30000,
        )
    finally:
        q.ctc_align_window = saved

    assert report["recovered"] == 0
    assert any(r.get("skipped") == "incomplete_alignment" for r in report["regions"])
    # untouched: the estimated words stay estimated (no corruption)
    for k in (14, 15, 16):
        assert result[k]["estimated"] is True
        assert not result[k].get("ctc_dup_ayat_recovered")


def test_dup_ayat_ctc_requires_estimated_word():
    # A cleanly-aligned duplicate region (no estimated words) must not fire.
    result, report = q.recover_duplicate_adjacent_ayat_ctc(
        mapped_words=_ashsharh_mapped(), args=_dup_args(), wav_path="d.wav",
        chapter_id=94, duration_ms=30000,
    )
    assert report["attempted"] is False
    assert report["recovered"] == 0


def test_dup_ayat_ctc_disabled_flag():
    args = _dup_args()
    args.no_dup_ayat_ctc = True
    result, report = q.recover_duplicate_adjacent_ayat_ctc(
        mapped_words=_ashsharh_mapped_bugged(), args=args, wav_path="d.wav",
        chapter_id=94, duration_ms=30000,
    )
    assert report["attempted"] is False and report["recovered"] == 0


def test_dup_ayat_ctc_deps_missing_graceful():
    m = _ashsharh_mapped_bugged()
    before = [dict(w) for w in m]
    saved_torch = sys.modules.get("torch")
    sys.modules["torch"] = None  # force `import torch` to raise
    try:
        result, report = q.recover_duplicate_adjacent_ayat_ctc(
            mapped_words=m, args=_dup_args(), wav_path="d.wav",
            chapter_id=94, duration_ms=30000,
        )
    finally:
        if saved_torch is not None:
            sys.modules["torch"] = saved_torch
        else:
            sys.modules.pop("torch", None)
    assert report["attempted"] is False
    assert report.get("skipped") == "deps_missing"
    assert result == before, "no word may be mutated when CTC deps are missing"


def test_opening_summary_leading_estimated_run_is_opening():
    # v70: Al-Fatiha begins directly on ٱلۡحَمۡدُ; the basmala (1:1, 4 words) is
    # absent from the audio so it is estimated at ~0ms with NO opening_region
    # flag (the leading-gap repair bails: first real word is at 0ms). The
    # leading contiguous estimated run must still be classified as the opening so
    # the v64 intro-budget forgiveness applies, not as estimated_after_opening.
    basmala = ["بسم", "الله", "الرحمن", "الرحيم"]
    mapped = (
        [{"estimated": True, "norm": basmala[i], "start_ms": i * 300} for i in range(4)]
        + [{"estimated": False, "norm": "الحمد", "start_ms": 0}]
        + [{"estimated": False, "norm": "لله", "start_ms": 740}]
    )
    s = q.opening_estimate_summary(mapped)
    assert s["estimated_total"] == 4
    assert s["opening_estimated"] == 4, s
    assert s["estimated_after_opening"] == 0, s
    assert s["estimated_after_opening_indices"] == []


def test_opening_summary_dropped_first_ayah_not_forgiven():
    # v71: a genuinely dropped FIRST ayah (arbitrary Quran text, not intro
    # vocabulary) at index 0 must NOT be swept into the opening by position. Its
    # norm is not an isti'adha/basmala/muqatta'at token, so it breaks the leading
    # run and is counted against quality (estimated_after_opening), even though
    # the drop is short enough to fit inside the intro budget.
    mapped = [
        {"estimated": True, "norm": "الحمد", "start_ms": 0},
        {"estimated": True, "norm": "لله", "start_ms": 300},
        {"estimated": False, "norm": "رب", "start_ms": 800},
    ]
    s = q.opening_estimate_summary(mapped)
    assert s["opening_estimated"] == 0, s
    assert s["estimated_after_opening"] == 2, s
    assert s["estimated_after_opening_indices"] == [0, 1]


def test_opening_summary_basmala_then_dropped_ayah_partial():
    # v71: the basmala is dropped (intro vocab -> forgiven) AND the first ayah
    # word is also dropped (non-intro -> breaks the run). Only the basmala joins
    # the opening; the dropped ayah word remains counted against quality.
    basmala = ["بسم", "الله", "الرحمن", "الرحيم"]
    mapped = (
        [{"estimated": True, "norm": basmala[i], "start_ms": i * 300} for i in range(4)]
        + [{"estimated": True, "norm": "الحمد", "start_ms": 1400}]
        + [{"estimated": False, "norm": "لله", "start_ms": 1900}]
    )
    s = q.opening_estimate_summary(mapped)
    assert s["opening_estimated"] == 4, s
    assert s["estimated_after_opening"] == 1, s
    assert s["estimated_after_opening_indices"] == [4]


def test_opening_summary_leading_muqattaat_estimated_is_opening():
    # v71: an unrecited muqatta'at at index 0 (e.g. حم estimated) is intro
    # vocabulary and joins the leading run.
    mapped = [
        {"estimated": True, "norm": "حم", "start_ms": 0},
        {"estimated": False, "norm": "عسق", "start_ms": 900},
    ]
    s = q.opening_estimate_summary(mapped)
    assert s["opening_estimated"] == 1, s
    assert s["estimated_after_opening"] == 0, s


def test_opening_summary_interior_estimate_not_opening():
    # A real first word then a later estimated word: the interior estimate is
    # NOT part of the opening (no leading estimated run to absorb it).
    mapped = [
        {"estimated": False, "start_ms": 0},
        {"estimated": False, "start_ms": 500},
        {"estimated": True, "start_ms": 1000},
    ]
    s = q.opening_estimate_summary(mapped)
    assert s["opening_estimated"] == 0, s
    assert s["estimated_after_opening"] == 1, s
    assert s["estimated_after_opening_indices"] == [2]


def test_opening_summary_real_muqattaat_breaks_leading_run():
    # A measured (real) muqatta'at at index 0 must break the leading run so its
    # timestamp is preserved; a later estimated interior word stays after_opening.
    mapped = [
        {"estimated": False, "start_ms": 1200},  # real muqatta'at الم
        {"estimated": False, "start_ms": 2600},
        {"estimated": True, "start_ms": 5000},   # dropped interior word
    ]
    s = q.opening_estimate_summary(mapped)
    assert s["opening_estimated"] == 0, s
    assert s["estimated_after_opening"] == 1, s


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)


# ---------------------------------------------------------------------------
# v72: validator opening-safe EFFECTIVE-rate gate.
# The validator must judge a surah by the same opening-safe effective match rate
# the generator reports (forgiving a plausible intro's worth of opening
# estimates), NOT by the raw matched/expected rate plus a standalone
# zero-estimated gate. A surah clearing >=min_match effective is OK even with a
# few interpolated words; only a genuine effective shortfall is flagged.
# ---------------------------------------------------------------------------

def _write_validator_case(
    tmpdir,
    estimated_indices,
    opening_estimated,
    estimated_after_opening,
    n=1000,
    chapter_id=3,
):
    import json
    from pathlib import Path as _P

    tmpdir = _P(tmpdir)
    timing_path = tmpdir / f"{chapter_id}.json"

    segments = [[i + 1, i * 100, i * 100 + 90] for i in range(n)]
    output = {
        "chapter_id": chapter_id,
        "duration": n * 100 + 5000,
        "verse_timings": [
            {
                "verse_key": f"{chapter_id}:1",
                "timestamp_from": 0,
                "timestamp_to": (n - 1) * 100 + 90,
                "segments": segments,
            }
        ],
    }
    timing_path.write_text(json.dumps(output), encoding="utf-8")

    est = set(estimated_indices)
    debug = [
        {
            "norm": f"w{i}",
            "text": f"w{i}",
            "estimated": (i in est),
            "score": 0.9,
        }
        for i in range(n)
    ]
    (tmpdir / f"{chapter_id}.debug.json").write_text(json.dumps(debug), encoding="utf-8")

    osum = {
        "estimated_total": opening_estimated + estimated_after_opening,
        "opening_estimated": opening_estimated,
        "estimated_after_opening": estimated_after_opening,
        "estimated_after_opening_indices": [],
    }
    (tmpdir / f"{chapter_id}.opening_summary.json").write_text(
        json.dumps(osum), encoding="utf-8"
    )

    return timing_path


def _validate(timing_path, n=1000, chapter_id=3):
    return q.validate_one_timing_file(
        timing_path,
        expected_counts={chapter_id: n},
        use_logs=False,
        use_debug_quality=True,
    )


def test_validator_opening_estimates_within_budget_are_ok():
    """A few OPENING estimates (<= intro budget) with no interior drops must be
    forgiven: raw rate 99.2% would fail the old gate, but effective rate is 100%
    so the file is OK (no match_rate_below_threshold review, no needs_fix)."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        # 8 leading opening estimates, all forgiven; nothing after the opening.
        tp = _write_validator_case(
            d,
            estimated_indices=range(8),
            opening_estimated=8,
            estimated_after_opening=0,
        )
        item = _validate(tp)
        assert item["status"] == "ok", item
        assert not any(
            "match_rate_below_threshold" in x for x in item["review_issues"]
        ), item
        # estimated words alone are NOT a standalone review trigger anymore.
        assert not any("estimated_words" in x for x in item["review_issues"]), item
        assert item["metrics"].get("effective_match_rate") == 1.0, item


def test_validator_estimated_words_alone_no_longer_flagged():
    """Estimated words that still clear the effective bar must not be flagged.
    3 interior estimates out of 1000 -> effective 99.7% >= 99.55% -> OK."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        tp = _write_validator_case(
            d,
            estimated_indices=[100, 200, 300],
            opening_estimated=0,
            estimated_after_opening=3,
        )
        item = _validate(tp)
        assert item["status"] == "ok", item
        assert item["review_issues"] == [], item


def test_validator_interior_drops_below_bar_needs_review():
    """Interior drops that push the effective rate under min_match must still be
    flagged needs_review (real work remains)."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        tp = _write_validator_case(
            d,
            estimated_indices=range(100, 110),  # 10 interior drops -> 99.0%
            opening_estimated=0,
            estimated_after_opening=10,
        )
        item = _validate(tp)
        assert item["status"] == "needs_review", item
        assert any(
            "match_rate_below_threshold" in x for x in item["review_issues"]
        ), item


def test_validator_opening_excess_beyond_budget_counts_against_quality():
    """Opening estimates far beyond a plausible intro are NOT all forgiven: the
    excess counts against quality, dropping the effective rate below the bar."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        # 40 leading opening estimates; only the intro budget (~13) is forgiven,
        # the ~27 excess count against quality -> effective ~97.3% < 99.55%.
        tp = _write_validator_case(
            d,
            estimated_indices=range(40),
            opening_estimated=40,
            estimated_after_opening=0,
        )
        item = _validate(tp)
        assert item["status"] == "needs_review", item
        assert item["metrics"]["effective_estimated_words"] > 0, item
        assert item["metrics"]["effective_estimated_words"] < 40, item


def test_validator_severe_effective_shortfall_needs_fix():
    """A catastrophic effective shortfall (< severe_match) is RED / needs_fix."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        tp = _write_validator_case(
            d,
            estimated_indices=range(100, 250),  # 150 interior drops -> 85.0%
            opening_estimated=0,
            estimated_after_opening=150,
        )
        item = _validate(tp)
        assert item["status"] == "needs_fix", item
        assert any("severe_under_transcription" in x for x in item["issues"]), item


def test_targeted_fix_reads_report_and_selects_flagged_only(tmp_path, monkeypatch):
    """v73: fix_existing_bad_timings(report=None) must read args.validate_report
    from disk and re-align ONLY needs_fix (+ needs_review when fix_include_review),
    never ok files, without any full re-validation."""
    report = {
        "items": [
            {"file": "timings/r/001.json", "reciter": "r", "chapter": 1, "status": "ok", "needs_fix": False},
            {"file": "timings/r/010.json", "reciter": "r", "chapter": 10, "status": "needs_fix", "needs_fix": True},
            {"file": "timings/r/090.json", "reciter": "r", "chapter": 90, "status": "needs_review", "needs_fix": False},
        ]
    }
    report_path = tmp_path / "timings_validation_report.json"
    report_path.write_text(__import__("json").dumps(report), encoding="utf-8")

    processed = []
    monkeypatch.setattr(q, "process_chapter", lambda child, ch: processed.append((child.reciter_url, ch)) or {"chapter_id": ch})
    monkeypatch.setattr(q, "backup_existing_timing_family", lambda p: (str(tmp_path / "bak"), []))
    monkeypatch.setattr(q, "_enable_quality_validation_for_review_fix", lambda a: None)

    def _args(fix_include_review):
        return __import__("argparse").Namespace(
            validate_report=str(report_path),
            fix_include_review=fix_include_review,
            fix_backup_existing=False,
            fix_report=str(tmp_path / "fix_report.json"),
            fix_revalidate=False,
            batch_base_url="https://example.test/root",
        )

    # needs_fix only
    processed.clear()
    rep = q.fix_existing_bad_timings(_args(False))
    assert rep["fixed_count"] == 1 and rep["failed_count"] == 0
    assert processed == [("https://example.test/root/r", 10)]

    # needs_fix + needs_review
    processed.clear()
    rep = q.fix_existing_bad_timings(_args(True))
    assert rep["fixed_count"] == 2
    assert sorted(ch for _, ch in processed) == [10, 90]


# ---- v76 degenerate-span guard -------------------------------------------

def test_demote_degenerate_real_words_flags_collapsed_opening():
    """Al-Masad (111) / reciter-B: catastrophic under-transcription forces the
    full-surah CTC to recover the opening, but the monotonic clamp collapses the
    111:1 words to 0-0 behind a surah word mis-anchored onto the basmala. They
    stayed estimated=False and produced a false 100%. The guard must demote every
    impossible zero/negative-duration (or missing-bound) real word to estimated
    while leaving genuinely-timed words -- including a long madd span -- intact."""
    mapped = [
        {"norm": "تبت", "start_ms": 0, "end_ms": 0, "estimated": False},
        {"norm": "يدا", "start_ms": 0, "end_ms": 0, "estimated": False},
        {"norm": "ابي", "start_ms": 0, "end_ms": 0, "estimated": False},
        {"norm": "ما", "start_ms": 0, "end_ms": 840, "estimated": False},
        {"norm": "في", "start_ms": 5292, "end_ms": 17420, "estimated": False},
        {"norm": "من", "start_ms": 900, "end_ms": 900, "estimated": False},
        {"norm": "مسد", "start_ms": None, "end_ms": None, "estimated": False},
        {"norm": "حبل", "start_ms": 500, "end_ms": 400, "estimated": False},
    ]
    result, report = q.demote_degenerate_real_words(mapped)
    assert result is mapped
    # zero-duration (0-4 idx0,1,2), zero-duration (idx5), missing bounds (idx6),
    # negative duration (idx7) -> 6 demoted.
    assert report["demoted"] == 6, report
    assert set(report["indices"]) == {0, 1, 2, 5, 6, 7}, report
    for i in (0, 1, 2, 5, 6, 7):
        assert mapped[i]["estimated"] is True
        assert mapped[i].get("degenerate_span_demoted") is True
    # Valid words untouched: a normal span AND a long (12s) madd span both survive
    # -- the guard never judges duration LENGTH, only impossible spans.
    assert mapped[3]["estimated"] is False and "degenerate_span_demoted" not in mapped[3]
    assert mapped[4]["estimated"] is False and "degenerate_span_demoted" not in mapped[4]


def test_demote_degenerate_ignores_already_estimated():
    """Words already flagged estimated (with any bounds) are left alone -- the
    guard only re-classifies words currently claiming to be real."""
    mapped = [
        {"norm": "x", "start_ms": None, "end_ms": None, "estimated": True},
        {"norm": "y", "start_ms": 0, "end_ms": 0, "estimated": True},
    ]
    _, report = q.demote_degenerate_real_words(mapped)
    assert report["demoted"] == 0
    assert "degenerate_span_demoted" not in mapped[0]
    assert "degenerate_span_demoted" not in mapped[1]


def test_demote_degenerate_healthy_surah_noop():
    """A fully, validly aligned surah is never touched (no false positives)."""
    mapped = [
        {"norm": "a", "start_ms": 0, "end_ms": 500, "estimated": False},
        {"norm": "b", "start_ms": 500, "end_ms": 1200, "estimated": False},
        {"norm": "c", "start_ms": 1200, "end_ms": 20000, "estimated": False},
    ]
    _, report = q.demote_degenerate_real_words(mapped)
    assert report["demoted"] == 0
    assert all(w["estimated"] is False for w in mapped)
