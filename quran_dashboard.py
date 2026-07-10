#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quran_dashboard.py

A dependency-free local dashboard for the Quran forced-alignment tool.

It reads the timing JSON files, the validation report and the logs that
make_quran_timings.py / quran_forced_align_full.py produce, and shows:

  1) An overview grid (per reciter, all 114 surahs) coloured by status
     (ok / needs_review / needs_fix / missing) plus progress and errors.
  2) A player to test the timing: it streams the reciter mp3 and highlights
     each word in sync with the audio using the timing segments, so you can
     verify the alignment by ear.

It never touches the alignment scripts. Timing files are read-only by default;
the only writes happen when you explicitly save a manual edit from the player
(a .json.bak backup is kept next to the file). Pure standard library
(http.server), no third-party packages.

Run:
    python quran_dashboard.py
    # then open http://localhost:8000

Options:
    --root          timings root folder            (default: timings)
    --logs          logs folder                    (default: logs)
    --quran-words   quran_words.json path          (default: quran_words.json)
    --report        validation report json path    (default: logs/timings_validation_report.json)
    --base-url      reciter audio base url         (default: https://qurani.io/quransound)
    --host          bind host                      (default: 127.0.0.1)
    --port          bind port                      (default: 8000)
    --no-browser    do not auto-open the browser
"""

import argparse
import json
import re
import shutil
import time
import threading
import webbrowser
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

TOTAL_CHAPTERS = 114

# Display labels only (Arabic surah names). Used purely for the UI; the
# alignment logic never reads these.
SURAH_NAMES_AR = [
    "الفاتحة", "البقرة", "آل عمران", "النساء", "المائدة", "الأنعام", "الأعراف",
    "الأنفال", "التوبة", "يونس", "هود", "يوسف", "الرعد", "إبراهيم", "الحجر",
    "النحل", "الإسراء", "الكهف", "مريم", "طه", "الأنبياء", "الحج", "المؤمنون",
    "النور", "الفرقان", "الشعراء", "النمل", "القصص", "العنكبوت", "الروم",
    "لقمان", "السجدة", "الأحزاب", "سبأ", "فاطر", "يس", "الصافات", "ص", "الزمر",
    "غافر", "فصلت", "الشورى", "الزخرف", "الدخان", "الجاثية", "الأحقاف", "محمد",
    "الفتح", "الحجرات", "ق", "الذاريات", "الطور", "النجم", "القمر", "الرحمن",
    "الواقعة", "الحديد", "المجادلة", "الحشر", "الممتحنة", "الصف", "الجمعة",
    "المنافقون", "التغابن", "الطلاق", "التحريم", "الملك", "القلم", "الحاقة",
    "المعارج", "نوح", "الجن", "المزمل", "المدثر", "القيامة", "الإنسان",
    "المرسلات", "النبأ", "النازعات", "عبس", "التكوير", "الانفطار", "المطففين",
    "الانشقاق", "البروج", "الطارق", "الأعلى", "الغاشية", "الفجر", "البلد",
    "الشمس", "الليل", "الضحى", "الشرح", "التين", "العلق", "القدر", "البينة",
    "الزلزلة", "العاديات", "القارعة", "التكاثر", "العصر", "الهمزة", "الفيل",
    "قريش", "الماعون", "الكوثر", "الكافرون", "النصر", "المسد", "الإخلاص",
    "الفلق", "الناس",
]

# English (transliterated) surah names, display-only, for the English dashboard UI.
SURAH_NAMES_EN = [
    "Al-Fatihah", "Al-Baqarah", "Aal-E-Imran", "An-Nisa", "Al-Ma'idah", "Al-An'am",
    "Al-A'raf", "Al-Anfal", "At-Tawbah", "Yunus", "Hud", "Yusuf", "Ar-Ra'd",
    "Ibrahim", "Al-Hijr", "An-Nahl", "Al-Isra", "Al-Kahf", "Maryam", "Ta-Ha",
    "Al-Anbiya", "Al-Hajj", "Al-Mu'minun", "An-Nur", "Al-Furqan", "Ash-Shu'ara",
    "An-Naml", "Al-Qasas", "Al-Ankabut", "Ar-Rum", "Luqman", "As-Sajdah",
    "Al-Ahzab", "Saba", "Fatir", "Ya-Sin", "As-Saffat", "Sad", "Az-Zumar",
    "Ghafir", "Fussilat", "Ash-Shura", "Az-Zukhruf", "Ad-Dukhan", "Al-Jathiyah",
    "Al-Ahqaf", "Muhammad", "Al-Fath", "Al-Hujurat", "Qaf", "Adh-Dhariyat",
    "At-Tur", "An-Najm", "Al-Qamar", "Ar-Rahman", "Al-Waqi'ah", "Al-Hadid",
    "Al-Mujadila", "Al-Hashr", "Al-Mumtahanah", "As-Saff", "Al-Jumu'ah",
    "Al-Munafiqun", "At-Taghabun", "At-Talaq", "At-Tahrim", "Al-Mulk", "Al-Qalam",
    "Al-Haqqah", "Al-Ma'arij", "Nuh", "Al-Jinn", "Al-Muzzammil", "Al-Muddaththir",
    "Al-Qiyamah", "Al-Insan", "Al-Mursalat", "An-Naba", "An-Nazi'at", "Abasa",
    "At-Takwir", "Al-Infitar", "Al-Mutaffifin", "Al-Inshiqaq", "Al-Buruj",
    "At-Tariq", "Al-A'la", "Al-Ghashiyah", "Al-Fajr", "Al-Balad", "Ash-Shams",
    "Al-Layl", "Ad-Duha", "Ash-Sharh", "At-Tin", "Al-Alaq", "Al-Qadr",
    "Al-Bayyinah", "Az-Zalzalah", "Al-Adiyat", "Al-Qari'ah", "At-Takathur",
    "Al-Asr", "Al-Humazah", "Al-Fil", "Quraysh", "Al-Ma'un", "Al-Kawthar",
    "Al-Kafirun", "An-Nasr", "Al-Masad", "Al-Ikhlas", "Al-Falaq", "An-Nas",
]


def pad3(n):
    return str(int(n)).zfill(3)


def surah_name(chapter_id):
    try:
        return SURAH_NAMES_EN[int(chapter_id) - 1]
    except Exception:
        return ""


def surah_name_ar(chapter_id):
    try:
        return SURAH_NAMES_AR[int(chapter_id) - 1]
    except Exception:
        return ""


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


class DashboardState:
    def __init__(self, args):
        self.root = Path(args.root)
        self.logs_dir = Path(args.logs)
        self.debug_logs_dir = Path(args.debug_logs)
        self.quran_words_path = Path(args.quran_words)
        self.report_path = Path(args.report)
        self.base_url = args.base_url.rstrip("/")
        self.audio_user_agent = args.audio_user_agent
        self._log_index_cache = None
        self._log_index_sig = None
        self._log_newest_mtime_ns = 0
        self._quran_words_cache = None
        self._quran_words_mtime = None

    def report_mtime_ns(self):
        try:
            return self.report_path.stat().st_mtime_ns
        except OSError:
            return 0

    # ---- quran words (text) -------------------------------------------------
    def load_quran_words(self):
        p = self.quran_words_path
        if not p.exists():
            return None
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = None
        if self._quran_words_cache is not None and mtime == self._quran_words_mtime:
            return self._quran_words_cache
        data, err = read_json(p)
        if err or not isinstance(data, dict):
            return None
        self._quran_words_cache = data
        self._quran_words_mtime = mtime
        return data

    # ---- validation report --------------------------------------------------
    def load_report(self):
        if not self.report_path.exists():
            return None
        data, err = read_json(self.report_path)
        if err or not isinstance(data, dict):
            return None
        return data

    def report_index(self, report):
        """Map (reciter, chapter_str) -> item."""
        idx = {}
        if not report:
            return idx
        for item in report.get("items", []) or []:
            reciter = item.get("reciter")
            chapter = item.get("chapter")
            if chapter is None:
                # try to derive chapter from file name
                fname = Path(item.get("file", "")).stem
                if fname.isdigit():
                    chapter = int(fname)
            if reciter is None or chapter is None:
                continue
            idx[(str(reciter), str(int(chapter)))] = item
        return idx

    # ---- log-derived status -------------------------------------------------
    def log_index(self):
        """Parse batch/per-reciter logs -> {(reciter, chapter_str): verdict}.

        This mirrors the alignment tool's OWN verdict: a chapter that reached
        `Saved:` with no `QUALITY WARNING` is ok; with a warning it is
        needs_review; a `FAILED chapter` line is needs_fix. Status therefore
        comes from the logs, never from the timing JSON itself. Cached on the
        (path, mtime, size) signature so repeated polling stays cheap.
        """
        raw = []
        if self.logs_dir.exists():
            raw += [(f, False) for f in self.logs_dir.glob("*.log")]
        if self.debug_logs_dir.exists():
            raw += [(f, True) for f in self.debug_logs_dir.rglob("*.log")]

        # Order files by modification time so last-write-wins in parse_logs_index
        # reflects real run recency across rotated/multiple logs, not lexical name
        # order. Path is a stable tie-breaker. mtime_ns keeps sub-second changes
        # from being collapsed (important for cache invalidation during a live run).
        entries = []
        for f, from_debug in raw:
            try:
                st = f.stat()
                mtime_ns, size = st.st_mtime_ns, st.st_size
            except OSError:
                mtime_ns, size = 0, 0
            entries.append((mtime_ns, str(f), from_debug, size))
        entries.sort(key=lambda e: (e[0], e[1]))

        files = [(Path(path), from_debug) for _mt, path, from_debug, _sz in entries]
        sig = tuple((path, mt, sz) for mt, path, _fd, sz in entries)

        # Newest log mtime (entries sorted ascending by mtime) so build_overview
        # can decide whether a validation report is fresher than the logs.
        self._log_newest_mtime_ns = entries[-1][0] if entries else 0

        if sig == self._log_index_sig and self._log_index_cache is not None:
            return self._log_index_cache

        idx = parse_logs_index(files)
        self._log_index_cache = idx
        self._log_index_sig = sig
        return idx

    # ---- discovery ----------------------------------------------------------
    def list_reciters(self):
        if not self.root.exists():
            return []
        out = []
        for entry in sorted(self.root.iterdir(), key=lambda p: p.name.lower()):
            if entry.is_dir():
                out.append(entry.name)
        return out

    def present_chapters(self, reciter):
        d = self.root / reciter
        present = {}
        if not d.exists():
            return present
        for f in d.iterdir():
            if f.is_file() and f.suffix == ".json" and f.stem.isdigit():
                present[str(int(f.stem))] = f
        return present

    # ---- manual-edit lock ---------------------------------------------------
    def manual_lock_path(self, reciter, chapter):
        """<root>/<reciter>/<chapter>.manual_lock.json — the marker the editor
        drops when a human saves manual edits. Same name the alignment script's
        validator looks for, so a hand-fixed surah is skipped by --fix and never
        re-aligned to the same result."""
        return self.root / reciter / f"{pad3(int(chapter))}.manual_lock.json"

    def is_manual_locked(self, reciter, chapter):
        try:
            return self.manual_lock_path(reciter, chapter).exists()
        except (OSError, ValueError, TypeError):
            return False

    def read_manual_lock(self, reciter, chapter):
        try:
            p = self.manual_lock_path(reciter, chapter)
            if not p.exists():
                return None
            data, err = read_json(p)
            return None if err else data
        except (OSError, ValueError, TypeError):
            return None

    # ---- per-word debug quality (estimated / score) -------------------------
    def load_debug_quality(self, reciter, chapter):
        """Return {(verse_key, position): {estimated, score}} for a chapter, read
        from <root>/<reciter>/NNN.debug.json or <debug_logs>/<reciter>/NNN.debug.json.
        Lets the editor highlight the exact words the model was unsure about."""
        ch = pad3(int(chapter))
        candidates = [
            self.root / reciter / f"{ch}.debug.json",
            self.debug_logs_dir / reciter / f"{ch}.debug.json",
        ]
        for p in candidates:
            try:
                if not p.exists():
                    continue
            except (OSError, ValueError):
                continue
            data, err = read_json(p)
            if err or not isinstance(data, list):
                continue
            out = {}
            for w in data:
                if not isinstance(w, dict):
                    continue
                vk = w.get("verse_key")
                pos = w.get("position")
                if vk is None or pos is None:
                    continue
                try:
                    out[(str(vk), int(pos))] = {
                        "estimated": bool(w.get("estimated")),
                        "score": w.get("score"),
                    }
                except (TypeError, ValueError):
                    continue
            if out:
                return out
        return {}

    def count_missing_segments(self, timing_path):
        data, err = read_json(timing_path)
        if err or not isinstance(data, dict):
            return None, None
        total = 0
        missing = 0
        for verse in data.get("verse_timings", []) or []:
            for seg in verse.get("segments", []) or []:
                total += 1
                if len(seg) != 3:
                    missing += 1
        return total, missing


def metric_match_rate(item):
    m = item.get("metrics", {}) or {}
    for key in ("match_rate_from_debug", "log_match_rate", "match_rate"):
        v = m.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def metric_estimated(item):
    m = item.get("metrics", {}) or {}
    for key in ("estimated_words", "log_estimated_timings"):
        v = m.get(key)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return None


_LOG_CHAPTER_RE = re.compile(r"^Chapter:\s*(\d+)\s*$")
_LOG_OUTPUT_RE = re.compile(r"^Output:\s*(.+?)\s*$")
_LOG_SKIP_RE = re.compile(r"^SKIPPED existing:")
_LOG_MATCHED_RE = re.compile(r"^Matched:\s*(\d+)\s*/\s*(\d+)")
_LOG_RATE_RE = re.compile(r"^Match rate(?:\s+after repairs)?:\s*([0-9.]+)%")
_LOG_EFF_RATE_RE = re.compile(r"effective_match_rate=([0-9.]+)%")
_LOG_EST_AFTER_RE = re.compile(r"estimated_after_opening=(\d+)")
_LOG_FINAL_MISS_RE = re.compile(r"^Final missing segments:\s*(\d+)")
_LOG_ESTIMATED_RE = re.compile(r"^Estimated timings inside main script:\s*(\d+)")
_LOG_SAVED_RE = re.compile(r"^Saved:\s*(.+?)\s*$")
_LOG_QWARN_RE = re.compile(r"^QUALITY WARNING: output needs review")
_LOG_FAILED_RE = re.compile(r"^FAILED chapter\s+(\d+):\s*(.+?)\s*$")
# Bullet reasons the tool prints right under "QUALITY WARNING:" (e.g.
# "  - match_rate 99.40% < 99.50%", "  - estimated_timings 2 > 0"). These tell
# the user exactly why a surah was flagged for review.
_LOG_ISSUE_RE = re.compile(r"^-\s+(.+?)\s*$")

_REAL_VERDICTS = ("ok", "needs_review", "needs_fix")


def _verdict_from_block(block):
    if block.get("failed"):
        status = "needs_fix"
    elif block.get("saved"):
        status = "needs_review" if block.get("quality_warning") else "ok"
    elif block.get("skipped"):
        status = "skipped"
    else:
        status = "in_progress"

    rate = block.get("effective_match_rate")
    if rate is None:
        rate = block.get("match_rate")
    estimated = block.get("estimated_after_opening")
    if estimated is None:
        estimated = block.get("estimated")

    return {
        "status": status,
        "match_rate": rate,
        "estimated": estimated,
        "final_missing": block.get("final_missing"),
        "matched": block.get("matched"),
        "match_total": block.get("match_total"),
        "quality_warning": bool(block.get("quality_warning")),
        "quality_issues": list(block.get("quality_issues") or []),
        "error": block.get("error"),
    }


def _commit_block(index, block, default_reciter):
    if not block or block.get("chapter") is None:
        return
    reciter = block.get("reciter") or default_reciter
    if not reciter:
        return
    key = (str(reciter), str(int(block["chapter"])))
    verdict = _verdict_from_block(block)

    prev = index.get(key)
    if prev is None:
        index[key] = verdict
        return
    # A real verdict (ok/needs_review/needs_fix) always wins — logs are appended
    # so the newest run for a chapter appears last. skipped/in_progress only fill
    # in when we have no real verdict yet (e.g. the original run was rotated out).
    if verdict["status"] in _REAL_VERDICTS or prev["status"] not in _REAL_VERDICTS:
        index[key] = verdict


def parse_logs_index(files):
    """files: iterable of (Path, from_debug_dir_bool)."""
    index = {}
    for lf, from_debug in files:
        default_reciter = lf.parent.name if from_debug else None
        try:
            lines = lf.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        block = None
        for raw in lines:
            line = raw.strip()

            m = _LOG_CHAPTER_RE.match(line)
            if m:
                _commit_block(index, block, default_reciter)
                block = {"chapter": int(m.group(1))}
                continue

            if block is None:
                m = _LOG_FAILED_RE.match(line)
                if m:
                    block = {"chapter": int(m.group(1)), "failed": True,
                             "error": m.group(2)}
                    _commit_block(index, block, default_reciter)
                    block = None
                continue

            m = _LOG_OUTPUT_RE.match(line)
            if m:
                block["reciter"] = Path(m.group(1)).parent.name
                continue
            if _LOG_SKIP_RE.match(line):
                block["skipped"] = True
                continue
            m = _LOG_MATCHED_RE.match(line)
            if m:
                block["matched"] = int(m.group(1))
                block["match_total"] = int(m.group(2))
                continue
            m = _LOG_RATE_RE.match(line)
            if m:
                block["match_rate"] = float(m.group(1)) / 100.0
                continue
            m = _LOG_EFF_RATE_RE.search(line)
            if m:
                block["effective_match_rate"] = float(m.group(1)) / 100.0
            m = _LOG_EST_AFTER_RE.search(line)
            if m:
                block["estimated_after_opening"] = int(m.group(1))
            m = _LOG_FINAL_MISS_RE.match(line)
            if m:
                block["final_missing"] = int(m.group(1))
                continue
            m = _LOG_ESTIMATED_RE.match(line)
            if m:
                block["estimated"] = int(m.group(1))
                continue
            if _LOG_QWARN_RE.match(line):
                block["quality_warning"] = True
                block.setdefault("quality_issues", [])
                continue
            # Collect the "- reason" bullet lines that follow the warning, until
            # the chapter is saved, so we can show WHY it needs review.
            if block.get("quality_warning") and not block.get("saved"):
                m = _LOG_ISSUE_RE.match(line)
                if m:
                    reason = m.group(1).strip()
                    if reason and reason not in block["quality_issues"]:
                        block["quality_issues"].append(reason)
                    continue
            m = _LOG_SAVED_RE.match(line)
            if m:
                block["saved"] = True
                continue
            m = _LOG_FAILED_RE.match(line)
            if m:
                block["failed"] = True
                block["error"] = m.group(2)
                continue

        _commit_block(index, block, default_reciter)

    return index


def _apply_report_entry(entry, item):
    entry["status"] = item.get("status", "unknown") or "unknown"
    entry["match_rate"] = metric_match_rate(item)
    entry["estimated"] = metric_estimated(item)
    entry["issues"] = list(item.get("issues") or [])[:6]
    entry["source"] = "report"


def _apply_log_entry(entry, log_v):
    entry["status"] = log_v["status"]
    entry["match_rate"] = log_v.get("match_rate")
    entry["estimated"] = log_v.get("estimated")
    entry["source"] = "log"
    iss = []
    if log_v["status"] == "needs_fix":
        iss.append(log_v.get("error") or "failed (see log)")
    elif log_v["status"] == "needs_review":
        reasons = log_v.get("quality_issues") or []
        if reasons:
            # The tool's own explanation of why this surah was flagged.
            iss.extend(reasons)
        elif log_v.get("match_rate") is not None:
            iss.append("needs review — match %.2f%%" % (log_v["match_rate"] * 100))
        else:
            iss.append("needs review (from log)")
        if log_v.get("final_missing") and not any("missing" in r for r in reasons):
            iss.append("final missing %d" % log_v["final_missing"])
    entry["issues"] = iss[:6]


def build_overview(state):
    report = state.load_report()
    ridx = state.report_index(report)
    lidx = state.log_index()

    # Precedence: the validation report is an explicit on-demand re-scan of the
    # ACTUAL timing files, so when it is at least as fresh as the newest log it
    # can override the stale generation logs. BUT the default validation mode is
    # STRUCTURAL-ONLY (--validate-structural-only): it only checks that the
    # JSON/timings are well-formed and does NOT re-measure recitation match
    # quality. A structural-only report may therefore flag needs_fix (real
    # corruption) but must NOT downgrade a log's needs_review to "ok" — doing so
    # would hide a genuine quality concern behind a check that never looked at
    # quality. Only a FULL quality re-validation (--validate-quality, i.e.
    # structural_only == False) is allowed to override the log's quality verdict.
    report_fresh = report is not None and \
        state.report_mtime_ns() >= (state._log_newest_mtime_ns or 0)
    _thr = (report.get("thresholds") or {}) if report else {}
    report_quality = not bool(_thr.get("structural_only", True))
    prefer_report_full = report_fresh and report_quality

    reciters_out = []
    for reciter in state.list_reciters():
        present = state.present_chapters(reciter)
        chapters = {}
        counts = {"ok": 0, "needs_review": 0, "needs_fix": 0,
                  "in_progress": 0, "missing": 0, "unknown": 0}

        for ch in range(1, TOTAL_CHAPTERS + 1):
            key = str(ch)
            entry = {
                "chapter": ch,
                "name": surah_name(ch),
                "present": key in present,
                "status": "missing",
                "match_rate": None,
                "estimated": None,
                "total_segments": None,
                "missing_segments": None,
                "issues": [],
                "source": None,
                "manual_locked": False,
            }

            log_v = lidx.get((reciter, key))
            item = ridx.get((reciter, key))

            if key in present:
                total, missing = state.count_missing_segments(present[key])
                entry["total_segments"] = total
                entry["missing_segments"] = missing

            # Manual lock wins over every other verdict: a human hand-corrected
            # this surah, so it is ok/locked and the --fix pass leaves it alone.
            if key in present and state.is_manual_locked(reciter, key):
                entry["status"] = "ok"
                entry["manual_locked"] = True
                entry["source"] = "manual"
                entry["issues"] = []
                counts[entry["status"]] = counts.get(entry["status"], 0) + 1
                chapters[key] = entry
                continue

            # Status precedence (see prefer_report_full above):
            #   1. A full, fresh quality re-validation fully overrides the logs.
            #   2. A fresh report's needs_fix (structural corruption) always wins.
            #   3. Otherwise the log's real quality verdict wins (keeps yellow
            #      review flags that a structural-only report can't evaluate).
            #   4. Fall back to any report item, then in-progress/unknown.
            log_real = bool(log_v and log_v["status"] in _REAL_VERDICTS)
            report_fix = report_fresh and item is not None and \
                item.get("status") == "needs_fix"
            if prefer_report_full and item:
                _apply_report_entry(entry, item)
            elif report_fix:
                _apply_report_entry(entry, item)
            elif log_real:
                _apply_log_entry(entry, log_v)
            elif item:
                _apply_report_entry(entry, item)
            elif log_v and log_v["status"] == "in_progress" and key not in present:
                entry["status"] = "in_progress"
                entry["source"] = "log"
            elif key in present:
                # File exists but no verdict anywhere (e.g. generated by an
                # older run whose log rotated away). Treat as generated/unknown.
                entry["status"] = "unknown"

            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
            chapters[key] = entry

        reciters_out.append({
            "reciter": reciter,
            "present_count": len(present),
            "total": TOTAL_CHAPTERS,
            "counts": counts,
            "chapters": chapters,
        })

    if report is not None:
        report_block = {
            "present": True,
            "path": str(state.report_path),
            "summary": report.get("summary"),
            "thresholds": report.get("thresholds"),
            "script_version": report.get("script_version"),
            "preferred": prefer_report_full,
            "quality_evaluated": report_quality,
            "structural_only": bool(_thr.get("structural_only", True)),
        }
    else:
        report_block = {"present": False, "path": str(state.report_path)}

    return {
        "config": {
            "base_url": state.base_url,
            "total_chapters": TOTAL_CHAPTERS,
            "root": str(state.root),
            "quran_words_present": state.quran_words_path.exists(),
            "generated_at": int(time.time() * 1000),
        },
        "report": report_block,
        "reciters": reciters_out,
        "logs": list_logs(state),
    }


def build_timing(state, reciter, chapter):
    ch = int(chapter)
    timing_path = state.root / reciter / f"{pad3(ch)}.json"
    if not timing_path.exists():
        return {"error": f"timing file not found: {timing_path}"}, 404

    data, err = read_json(timing_path)
    if err or not isinstance(data, dict):
        return {"error": f"cannot read timing: {err}"}, 500

    # timing: verse_key -> {position: (start_ms, end_ms) or None}
    timing_map = {}
    for verse in data.get("verse_timings", []) or []:
        vk = verse.get("verse_key")
        pmap = {}
        for seg in verse.get("segments", []) or []:
            if not seg:
                continue
            try:
                pos = int(seg[0])
                if len(seg) == 3:
                    pmap[pos] = (int(seg[1]), int(seg[2]))
                else:
                    pmap[pos] = None
            except (TypeError, ValueError):
                continue
        timing_map[str(vk)] = pmap

    words_data = state.load_quran_words()
    verses_out = []
    word_count = 0
    missing_count = 0

    if words_data and str(ch) in words_data:
        # group quran words by verse_key preserving order
        by_verse = []
        seen = {}
        for w in words_data[str(ch)]:
            vk = str(w.get("verse_key"))
            if vk not in seen:
                seen[vk] = len(by_verse)
                by_verse.append((vk, []))
            by_verse[seen[vk]][1].append(w)

        for vk, ws in by_verse:
            pmap = timing_map.get(vk, {})
            words_out = []
            for w in ws:
                try:
                    pos = int(w.get("position"))
                except (TypeError, ValueError):
                    continue
                tv = pmap.get(pos)
                missing = tv is None
                if missing:
                    missing_count += 1
                words_out.append({
                    "position": pos,
                    "text": str(w.get("text", "")),
                    "start_ms": (tv[0] if tv else None),
                    "end_ms": (tv[1] if tv else None),
                    "missing": missing,
                })
                word_count += 1
            verses_out.append({"verse_key": vk, "words": words_out})
    else:
        # no quran_words.json: build from timing only (no text)
        for verse in data.get("verse_timings", []) or []:
            vk = str(verse.get("verse_key"))
            words_out = []
            for seg in verse.get("segments", []) or []:
                if not seg:
                    continue
                try:
                    pos = int(seg[0])
                except (TypeError, ValueError):
                    continue
                timed = len(seg) == 3
                if not timed:
                    missing_count += 1
                words_out.append({
                    "position": pos,
                    "text": "",
                    "start_ms": (int(seg[1]) if timed else None),
                    "end_ms": (int(seg[2]) if timed else None),
                    "missing": not timed,
                })
                word_count += 1
            verses_out.append({"verse_key": vk, "words": words_out})

    # Served through our own /api/audio proxy: a browser <audio> element cannot
    # set a custom User-Agent, and the CDN's WAF returns 403 to default browser
    # UAs (NotSameOrigin / blocked). The proxy fetches server-side with the
    # allow-listed UA and streams it back same-origin, so seeking still works.
    audio_url = f"api/audio?reciter={quote(reciter, safe='')}&chapter={ch}"
    audio_source_url = f"{state.base_url}/{reciter}/{pad3(ch)}.mp3"

    # Attach the model's own per-word confidence so the editor can spotlight the
    # exact words it estimated/interpolated (the ones a human most needs to fix).
    quality = state.load_debug_quality(reciter, ch)
    estimated_count = 0
    if quality:
        for v in verses_out:
            vk = v["verse_key"]
            for w in v["words"]:
                q = quality.get((str(vk), int(w["position"])))
                if q is not None:
                    w["estimated"] = bool(q.get("estimated"))
                    w["score"] = q.get("score")
                    if w["estimated"]:
                        estimated_count += 1

    duration_ms = data.get("duration_ms")
    if duration_ms is None:
        d = data.get("duration")
        if isinstance(d, (int, float)):
            duration_ms = int(d if d > 3600 else d * 1000)

    lock = state.read_manual_lock(reciter, ch)

    return {
        "reciter": reciter,
        "chapter_id": ch,
        "name": surah_name(ch),
        "duration": data.get("duration"),
        "duration_ms": duration_ms,
        "format": data.get("format", "mp3"),
        "audio_url": audio_url,
        "audio_source_url": audio_source_url,
        "word_count": word_count,
        "missing_count": missing_count,
        "estimated_count": estimated_count,
        "has_text": bool(words_data and str(ch) in words_data),
        "manual_locked": lock is not None,
        "manual_lock": lock,
        "verses": verses_out,
    }, 200


def save_timing(state, reciter, chapter, edits):
    """Apply manual per-word timing edits to a timing JSON file.

    edits: list of {verse_key, position, start_ms, end_ms}. Only segments that
    already exist for a (verse_key, position) are updated; a previously-missing
    word (a segment with just [position]) is promoted to a full [pos, s, e]
    segment. A .json.bak backup is written before the atomic overwrite.
    """
    ch = int(chapter)
    timing_path = state.root / reciter / f"{pad3(ch)}.json"
    # Path-traversal guard: the resolved target must live under the timings root.
    root = state.root.resolve()
    try:
        tp = timing_path.resolve()
    except OSError:
        return {"error": "invalid path"}, 400
    if root != tp and root not in tp.parents:
        return {"error": "invalid path"}, 400
    if not timing_path.exists():
        return {"error": f"timing file not found: {timing_path}"}, 404

    data, err = read_json(timing_path)
    if err or not isinstance(data, dict):
        return {"error": f"cannot read timing: {err}"}, 500

    emap = {}
    for e in (edits or []):
        try:
            vk = str(e["verse_key"])
            pos = int(e["position"])
            s = int(e["start_ms"])
            en = int(e["end_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if s < 0 or en < 0:
            continue
        if en < s:
            s, en = en, s
        emap[(vk, pos)] = (s, en)
    if not emap:
        return {"error": "no valid edits"}, 400

    changed = 0
    for verse in data.get("verse_timings", []) or []:
        vk = str(verse.get("verse_key"))
        segs = verse.get("segments")
        if not isinstance(segs, list):
            continue
        for i, seg in enumerate(segs):
            if not seg:
                continue
            try:
                pos = int(seg[0])
            except (TypeError, ValueError, IndexError):
                continue
            k = (vk, pos)
            if k in emap:
                s, en = emap[k]
                segs[i] = [pos, s, en]
                changed += 1
    if changed == 0:
        return {"error": "no matching segments found for the given edits"}, 400

    try:
        bak = timing_path.with_suffix(".json.bak")
        shutil.copyfile(timing_path, bak)
        tmp = timing_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(timing_path)
    except OSError as ex:
        return {"error": f"write failed: {ex}"}, 500

    # Drop the manual-lock marker so the alignment script's --fix pass will skip
    # this hand-corrected surah instead of re-aligning it to the same result and
    # overwriting the human's edits. The dashboard shows it green/locked.
    locked = False
    try:
        lock_path = timing_path.with_suffix(".manual_lock.json")
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump({
                "manual_edit": True,
                "source": "dashboard",
                "edited_at": int(time.time() * 1000),
                "edits": changed,
                "chapter": ch,
                "reciter": reciter,
            }, f, ensure_ascii=False)
        locked = True
    except OSError:
        locked = False

    return {"ok": True, "changed": changed, "backup": bak.name, "locked": locked}, 200


def set_manual_lock(state, reciter, chapter, locked):
    """Lock or unlock a chapter's manual-edit marker. Unlocking lets the next
    --fix run re-align the surah again."""
    ch = int(chapter)
    lock_path = state.manual_lock_path(reciter, ch)
    root = state.root.resolve()
    try:
        lp = lock_path.resolve()
    except OSError:
        return {"error": "invalid path"}, 400
    if root != lp and root not in lp.parents:
        return {"error": "invalid path"}, 400
    try:
        if locked:
            with open(lock_path, "w", encoding="utf-8") as f:
                json.dump({
                    "manual_edit": True, "source": "dashboard",
                    "edited_at": int(time.time() * 1000),
                    "chapter": ch, "reciter": reciter,
                }, f, ensure_ascii=False)
        else:
            if lock_path.exists():
                lock_path.unlink()
    except OSError as ex:
        return {"error": f"lock update failed: {ex}"}, 500
    return {"ok": True, "locked": bool(locked)}, 200


def list_logs(state):
    d = state.logs_dir
    out = []
    if not d.exists():
        return out
    files = list(d.rglob("*.log"))
    files.sort(key=lambda p: (p.stat().st_mtime if p.exists() else 0), reverse=True)
    for f in files:
        try:
            st = f.stat()
            out.append({
                "file": str(f.relative_to(d)),
                "size": st.st_size,
                "mtime": int(st.st_mtime * 1000),
            })
        except (OSError, ValueError):
            continue
    return out


def tail_log(state, rel, lines=300):
    d = state.logs_dir.resolve()
    target = (state.logs_dir / rel).resolve()
    # prevent path traversal outside the logs dir
    if d != target and d not in target.parents:
        return {"error": "invalid log path"}, 400
    if not target.exists() or not target.is_file():
        return {"error": "log not found"}, 404
    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            content = f.readlines()
    except OSError as e:
        return {"error": str(e)}, 500
    tail = content[-int(lines):]
    return {"file": rel, "lines": len(tail), "text": "".join(tail)}, 200


class Handler(BaseHTTPRequestHandler):
    state = None  # set on the class before serving

    def log_message(self, fmt, *a):
        pass  # keep the console clean

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        try:
            if path == "/" or path == "/index.html":
                self._send_html(INDEX_HTML)
                return

            if path == "/api/overview":
                self._send_json(build_overview(self.state))
                return

            if path == "/api/timing":
                reciter = (qs.get("reciter") or [""])[0]
                chapter = (qs.get("chapter") or [""])[0]
                if not reciter or not chapter:
                    self._send_json({"error": "reciter and chapter required"}, 400)
                    return
                if not chapter.isdigit() or not (1 <= int(chapter) <= TOTAL_CHAPTERS):
                    self._send_json({"error": "chapter must be 1..114"}, 400)
                    return
                obj, code = build_timing(self.state, reciter, chapter)
                self._send_json(obj, code)
                return

            if path == "/api/report":
                report = self.state.load_report()
                if report is None:
                    self._send_json({"present": False}, 200)
                else:
                    self._send_json({"present": True, "report": report}, 200)
                return

            if path == "/api/logs":
                rel = (qs.get("file") or [""])[0]
                if rel:
                    raw_lines = (qs.get("lines") or ["300"])[0]
                    try:
                        lines = int(raw_lines)
                    except (TypeError, ValueError):
                        lines = 300
                    lines = max(1, min(lines, 5000))
                    obj, code = tail_log(self.state, rel, lines)
                    self._send_json(obj, code)
                else:
                    self._send_json({"logs": list_logs(self.state)})
                return

            if path == "/api/audio":
                reciter = (qs.get("reciter") or [""])[0]
                chapter = (qs.get("chapter") or [""])[0]
                if not reciter or not chapter.isdigit():
                    self._send_json({"error": "reciter and numeric chapter required"}, 400)
                    return
                if not (1 <= int(chapter) <= TOTAL_CHAPTERS):
                    self._send_json({"error": "chapter must be 1..114"}, 400)
                    return
                self._proxy_audio(reciter, int(chapter))
                return

            self._send_json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._send_json({"error": f"server error: {e}"}, 500)
            except Exception:
                pass

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/timing":
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length > 0 else b""
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                except (ValueError, UnicodeDecodeError):
                    self._send_json({"error": "invalid JSON body"}, 400)
                    return
                if not isinstance(payload, dict):
                    self._send_json({"error": "invalid JSON body"}, 400)
                    return
                reciter = str(payload.get("reciter") or "")
                chapter = payload.get("chapter")
                if not reciter or chapter is None:
                    self._send_json({"error": "reciter and chapter required"}, 400)
                    return
                if "/" in reciter or "\\" in reciter or reciter in (".", ".."):
                    self._send_json({"error": "invalid reciter"}, 400)
                    return
                try:
                    chi = int(chapter)
                except (TypeError, ValueError):
                    self._send_json({"error": "chapter must be numeric"}, 400)
                    return
                if not (1 <= chi <= TOTAL_CHAPTERS):
                    self._send_json({"error": "chapter must be 1..114"}, 400)
                    return
                obj, code = save_timing(self.state, reciter, chi, payload.get("edits"))
                self._send_json(obj, code)
                return

            if path == "/api/manual_lock":
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length > 0 else b""
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                except (ValueError, UnicodeDecodeError):
                    self._send_json({"error": "invalid JSON body"}, 400)
                    return
                if not isinstance(payload, dict):
                    self._send_json({"error": "invalid JSON body"}, 400)
                    return
                reciter = str(payload.get("reciter") or "")
                chapter = payload.get("chapter")
                if not reciter or chapter is None:
                    self._send_json({"error": "reciter and chapter required"}, 400)
                    return
                if "/" in reciter or "\\" in reciter or reciter in (".", ".."):
                    self._send_json({"error": "invalid reciter"}, 400)
                    return
                try:
                    chi = int(chapter)
                except (TypeError, ValueError):
                    self._send_json({"error": "chapter must be numeric"}, 400)
                    return
                if not (1 <= chi <= TOTAL_CHAPTERS):
                    self._send_json({"error": "chapter must be 1..114"}, 400)
                    return
                obj, code = set_manual_lock(self.state, reciter, chi, bool(payload.get("locked")))
                self._send_json(obj, code)
                return

            self._send_json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._send_json({"error": f"server error: {e}"}, 500)
            except Exception:
                pass

    def _proxy_audio(self, reciter, chapter):
        """Stream the reciter mp3 through this server.

        The browser cannot set a custom User-Agent on an <audio> element, and
        the CDN's WAF blocks default browser UAs (403 / NotSameOrigin). We fetch
        server-side with the allow-listed UA and relay the bytes same-origin,
        forwarding the Range header so scrubbing/seeking keeps working.
        """
        url = f"{self.state.base_url}/{quote(reciter, safe='')}/{pad3(chapter)}.mp3"
        headers = {
            "User-Agent": self.state.audio_user_agent,
            "Accept": "*/*",
        }
        rng = self.headers.get("Range")
        if rng:
            headers["Range"] = rng

        req = urllib.request.Request(url, headers=headers)
        try:
            upstream = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            self._send_json({"error": f"upstream {e.code}", "url": url}, e.code if 400 <= e.code < 600 else 502)
            return
        except Exception as e:
            self._send_json({"error": f"audio fetch failed: {e}", "url": url}, 502)
            return

        try:
            status = getattr(upstream, "status", 200) or 200
            hdrs = upstream.headers
            self.send_response(status)
            ctype = hdrs.get("Content-Type") or "audio/mpeg"
            self.send_header("Content-Type", ctype)
            for h in ("Content-Length", "Content-Range", "Accept-Ranges"):
                v = hdrs.get(h)
                if v is not None:
                    self.send_header(h, v)
            if hdrs.get("Accept-Ranges") is None:
                self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            shutil.copyfileobj(upstream, self.wfile, length=64 * 1024)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                upstream.close()
            except Exception:
                pass


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Quran Alignment — QA Dashboard</title>
<style>
  :root {
    --bg: #0a0c10; --bg2: #0d1117; --panel: #12161d; --panel2: #171d26; --elev: #1b222c;
    --line: #222a35; --line2: #2c3542;
    --txt: #e6edf3; --txt2: #c3ccd6; --muted: #8b98a5; --faint: #5b6672;
    --accent: #3fb950; --accent2: #2ea043;
    --ok: #2ea043; --review: #d29922; --fix: #f85149; --missing: #262d38;
    --unknown: #58a6ff; --progress: #a371f7;
    --radius: 14px; --radius-sm: 9px;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.28);
    --shadow-lg: 0 24px 64px rgba(0,0,0,.55);
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0; background:
      radial-gradient(1200px 500px at 85% -10%, rgba(63,185,80,.06), transparent 60%),
      radial-gradient(900px 500px at 0% 0%, rgba(88,166,255,.05), transparent 55%),
      var(--bg);
    color: var(--txt);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 14px; line-height: 1.5; -webkit-font-smoothing: antialiased;
  }
  .grow { flex: 1; }
  .err { color: var(--fix); }
  code { background: var(--panel2); border: 1px solid var(--line); border-radius: 5px; padding: 1px 6px; font-size: 12px; color: var(--txt2); }

  /* ---------- header ---------- */
  header {
    padding: 14px 24px; background: rgba(13,17,23,.82); backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--line); display: flex; align-items: center; gap: 20px;
    flex-wrap: wrap; position: sticky; top: 0; z-index: 20;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand .mark {
    width: 38px; height: 38px; border-radius: 11px; display: grid; place-items: center;
    background: linear-gradient(145deg, #2ea043, #1a6c2c); box-shadow: inset 0 1px 0 rgba(255,255,255,.25), var(--shadow);
    font-size: 20px;
  }
  .brand .txt h1 { font-size: 16px; margin: 0; font-weight: 650; letter-spacing: -.2px; }
  .brand .txt p { margin: 0; font-size: 11.5px; color: var(--muted); letter-spacing: .3px; }
  .legend { display: flex; gap: 14px; font-size: 12px; color: var(--muted); flex-wrap: wrap; align-items: center; }
  .legend span { display: inline-flex; align-items: center; }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 3px; margin-right: 6px; }
  .toolbar { display: flex; gap: 8px; align-items: center; }
  .btn {
    background: var(--panel2); color: var(--txt2); border: 1px solid var(--line2);
    padding: 8px 13px; border-radius: 9px; cursor: pointer; font-size: 13px; font-weight: 500;
    display: inline-flex; align-items: center; gap: 7px; transition: all .14s ease;
  }
  .btn:hover { background: var(--elev); border-color: #3a4657; color: var(--txt); }
  .btn.primary { background: linear-gradient(180deg, #2ea043, #278a39); border-color: #2ea043; color: #fff; }
  .btn.primary:hover { filter: brightness(1.08); }
  .btn.active { border-color: var(--accent); color: var(--accent); background: rgba(63,185,80,.08); }
  .btn .led { width: 7px; height: 7px; border-radius: 50%; background: var(--faint); transition: background .2s; }
  .btn.active .led { background: var(--accent); box-shadow: 0 0 8px var(--accent); }

  /* ---------- summary ---------- */
  .summary { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; padding: 22px 24px 6px; }
  .card {
    position: relative; background: linear-gradient(180deg, var(--panel), var(--bg2));
    border: 1px solid var(--line); border-radius: var(--radius); padding: 16px 18px; overflow: hidden;
    box-shadow: var(--shadow);
  }
  .card::before { content: ""; position: absolute; inset: 0 0 auto 0; height: 2px; background: var(--accent-line, transparent); }
  .card .n { font-size: 28px; font-weight: 700; letter-spacing: -.5px; line-height: 1.1; font-variant-numeric: tabular-nums; }
  .card .l { font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: .7px; margin-top: 6px; font-weight: 600; }
  .card.wide { grid-column: span 2; display: flex; flex-direction: column; justify-content: center; gap: 4px; }
  .card.wide .l { text-transform: none; letter-spacing: 0; font-size: 12px; font-weight: 500; color: var(--txt2); }

  /* ---------- reciters ---------- */
  main { padding: 16px 24px 48px; }
  .reciter { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); margin-bottom: 18px; overflow: hidden; box-shadow: var(--shadow); }
  .reciter-head { padding: 15px 18px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; border-bottom: 1px solid var(--line); background: linear-gradient(180deg, rgba(255,255,255,.015), transparent); }
  .reciter-head h2 { font-size: 15px; margin: 0; font-weight: 650; letter-spacing: -.2px; text-transform: capitalize; }
  .rcount { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .progress { height: 7px; background: var(--panel2); border-radius: 20px; overflow: hidden; flex: 1; min-width: 140px; max-width: 420px; }
  .progress > div { height: 100%; background: linear-gradient(90deg, #2ea043, #3fb950); border-radius: 20px; transition: width .4s ease; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--txt2); background: var(--panel2); border: 1px solid var(--line); padding: 3px 9px; border-radius: 20px; font-variant-numeric: tabular-nums; }
  .chip i { width: 8px; height: 8px; border-radius: 50%; }

  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(58px, 1fr)); gap: 8px; padding: 16px 18px 20px; }
  .cell {
    position: relative; aspect-ratio: 1/1; border-radius: var(--radius-sm); display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 2px; cursor: pointer; border: 1px solid rgba(255,255,255,.06);
    font-weight: 700; font-size: 14px; color: #06110a; transition: transform .12s ease, box-shadow .12s ease; overflow: hidden;
  }
  .cell .num { font-variant-numeric: tabular-nums; line-height: 1; }
  .cell small { font-size: 8px; font-weight: 600; opacity: .82; max-width: 100%; padding: 0 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .cell.ok { background: linear-gradient(160deg, #35b24d, #268038); }
  .cell.needs_review { background: linear-gradient(160deg, #e0a825, #b57e13); color: #1c1400; }
  .cell.needs_fix { background: linear-gradient(160deg, #f85149, #c93832); color: #fff; }
  .cell.in_progress { background: linear-gradient(160deg, #a371f7, #7b4fd1); color: #fff; }
  .cell.unknown { background: linear-gradient(160deg, #58a6ff, #2f7ee0); color: #041022; }
  .cell.missing { background: var(--missing); color: var(--faint); cursor: default; border-color: transparent; font-weight: 500; }
  .cell:not(.missing):hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,.4); outline: 2px solid rgba(255,255,255,.85); outline-offset: -1px; }
  .hidden { display: none !important; }

  /* ---------- modals ---------- */
  .overlay { position: fixed; inset: 0; background: rgba(4,6,10,.66); backdrop-filter: blur(4px); display: flex; align-items: center; justify-content: center; z-index: 50; padding: 20px; }
  .modal { background: var(--panel); border: 1px solid var(--line2); border-radius: 18px; width: min(940px, 100%); max-height: 92vh; display: flex; flex-direction: column; box-shadow: var(--shadow-lg); }
  .modal-head { padding: 16px 20px; border-bottom: 1px solid var(--line); display: flex; align-items: center; gap: 12px; }
  .modal-head h3 { margin: 0; font-size: 16px; font-weight: 650; letter-spacing: -.2px; }
  .modal-body { padding: 18px 20px; overflow: auto; }
  .player-bar { display: flex; align-items: center; gap: 12px; padding: 14px 20px; border-top: 1px solid var(--line); background: var(--bg2); border-radius: 0 0 18px 18px; }
  audio { width: 100%; height: 40px; }
  .verse { margin-bottom: 18px; padding-bottom: 16px; border-bottom: 1px solid var(--line); }
  .verse:last-child { border-bottom: none; }
  .vk { font-size: 11px; color: var(--muted); margin-bottom: 8px; font-weight: 600; letter-spacing: .4px; font-variant-numeric: tabular-nums; }
  .words { display: flex; flex-wrap: wrap; gap: 8px; font-size: 27px; line-height: 2; direction: rtl;
    font-family: "Noto Naskh Arabic", "Amiri", "Scheherazade New", "Traditional Arabic", serif; }
  .w { padding: 2px 9px; border-radius: 8px; cursor: pointer; transition: background .12s, color .12s, transform .12s; }
  .w:hover { background: var(--panel2); }
  .w.active { background: linear-gradient(160deg, #35b24d, #268038); color: #fff; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(46,160,67,.35); }
  .w.miss { color: var(--fix); border-bottom: 2px dashed var(--fix); }
  .w.est { color: var(--review); border-bottom: 2px dotted var(--review); }
  .w-edit { display: inline-flex; flex-direction: column; gap: 6px; background: var(--panel2); border: 1px solid var(--line2); border-radius: 12px; padding: 8px 9px; transition: border-color .12s, box-shadow .12s; }
  .w-edit.miss { border-color: var(--fix); }
  .w-edit.est { border-color: var(--review); }
  .w-edit.sel { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(63,185,80,.35); }
  .w-edit.bad { border-color: var(--fix); box-shadow: 0 0 0 2px rgba(248,81,73,.4); }
  .w-edit .wt { font-size: 22px; text-align: center; line-height: 1.4; cursor: pointer; user-select: none; display: flex; align-items: center; justify-content: center; gap: 6px; }
  .w-edit .wt .tag { font-size: 8px; font-weight: 700; padding: 1px 5px; border-radius: 6px; letter-spacing: .4px; }
  .w-edit .wt .tag.e { background: var(--review); color: #1c1400; }
  .w-edit .wt .tag.m { background: var(--fix); color: #fff; }
  .w-edit .fld { display: flex; align-items: center; gap: 3px; direction: ltr; }
  .w-edit .fld .lbl { font-size: 9px; color: var(--muted); width: 10px; text-transform: uppercase; font-weight: 700; }
  .w-edit input { width: 58px; background: var(--bg); color: var(--txt); border: 1px solid var(--line2); border-radius: 6px; padding: 4px 5px; font-size: 12px; font-variant-numeric: tabular-nums; text-align: center; }
  .w-edit input:focus { outline: none; border-color: var(--accent); }
  /* professional stepper / knob editor */
  .w-edit input[type=number] { -moz-appearance: textfield; }
  .w-edit input::-webkit-inner-spin-button, .w-edit input::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
  .w-edit .fld .lbl { cursor: ew-resize; user-select: none; touch-action: none; text-align: center; border-radius: 5px; transition: color .1s, background .1s; }
  .w-edit .fld .lbl:hover { color: var(--accent); background: var(--bg); }
  .w-edit .fld.knobbing .lbl { color: var(--accent); background: rgba(63,185,80,.15); }
  .nb:active { transform: translateY(1px); background: var(--accent); color: #06110a; border-color: var(--accent); }
  .w-edit.rippled { animation: rippleflash .5s ease; }
  @keyframes rippleflash { 0% { box-shadow: 0 0 0 2px rgba(88,166,255,.6); } 100% { box-shadow: none; } }
  .nb { cursor: pointer; border: 1px solid var(--line2); background: var(--bg); color: var(--txt2); border-radius: 6px; width: 20px; height: 24px; font-size: 12px; line-height: 1; display: grid; place-items: center; padding: 0; transition: all .1s; }
  .nb:hover { color: #fff; border-color: var(--accent); background: var(--panel); }
  .nb.cap { width: auto; padding: 0 6px; font-size: 11px; }
  .w-actions { display: flex; gap: 4px; justify-content: center; direction: ltr; }
  .w-actions .nb.on { background: var(--accent); color: #06110a; border-color: var(--accent); }
  /* waveform */
  .wave-wrap { padding: 8px 20px 4px; border-top: 1px solid var(--line); background: var(--bg2); }
  .wave-head { display: flex; align-items: center; gap: 12px; font-size: 11px; color: var(--muted); margin-bottom: 5px; font-variant-numeric: tabular-nums; }
  .wave-head b { color: var(--txt); font-weight: 600; }
  #wave { width: 100%; height: 84px; display: block; border-radius: 8px; background: var(--bg); border: 1px solid var(--line); cursor: crosshair; }
  .valwarn { margin: 0 0 12px; padding: 9px 12px; border-radius: 9px; font-size: 12px; background: rgba(248,81,73,.12); border: 1px solid rgba(248,81,73,.4); color: #ffb4af; display: none; }
  .valwarn.show { display: block; }
  .valok { color: var(--ok); }
  .lockbadge { font-size: 11px; padding: 4px 11px; border-radius: 20px; background: rgba(46,160,67,.16); color: var(--ok); border: 1px solid rgba(46,160,67,.5); font-weight: 600; }
  .cell .lockico { position: absolute; top: 3px; right: 4px; font-size: 9px; opacity: .9; }
  .hint { font-size: 11px; color: var(--muted); margin: 2px 0 12px; line-height: 1.7; }
  .hint kbd { background: var(--panel2); border: 1px solid var(--line2); border-bottom-width: 2px; border-radius: 5px; padding: 1px 5px; font-size: 10px; font-family: ui-monospace, monospace; color: var(--txt); }
  .x { cursor: pointer; font-size: 20px; color: var(--muted); line-height: 1; padding: 4px 8px; border-radius: 8px; transition: all .12s; }
  .x:hover { color: #fff; background: var(--panel2); }
  .badge { font-size: 11px; padding: 4px 11px; border-radius: 20px; background: var(--panel2); color: var(--txt2); border: 1px solid var(--line2); font-weight: 600; }
  pre.log { background: var(--bg); border: 1px solid var(--line); border-radius: 10px; padding: 14px; font-size: 12px; line-height: 1.6; white-space: pre-wrap; max-height: 60vh; overflow: auto; font-family: "SF Mono", "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace; color: var(--txt2); }
  select { background: var(--panel2); color: var(--txt); border: 1px solid var(--line2); border-radius: 9px; padding: 7px 10px; font-size: 13px; }
  .empty { padding: 40px 24px; color: var(--muted); text-align: center; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="mark">&#128331;</div>
    <div class="txt">
      <h1>Quran Alignment</h1>
      <p>Forced-Alignment QA Dashboard</p>
    </div>
  </div>
  <div class="legend">
    <span><i class="dot" style="background:var(--ok)"></i>Complete</span>
    <span><i class="dot" style="background:var(--review)"></i>Review</span>
    <span><i class="dot" style="background:var(--fix)"></i>Fix</span>
    <span><i class="dot" style="background:var(--progress)"></i>In&nbsp;progress</span>
    <span><i class="dot" style="background:var(--unknown)"></i>Unchecked</span>
    <span><i class="dot" style="background:var(--missing)"></i>Missing</span>
  </div>
  <div class="grow"></div>
  <div class="toolbar">
    <button class="btn" id="logsBtn">&#128220; Logs</button>
    <button class="btn" id="autoBtn"><span class="led"></span> Auto-refresh</button>
    <button class="btn primary" id="refreshBtn">&#8635; Refresh</button>
  </div>
</header>

<div class="summary" id="summary"></div>
<main id="main"><p class="empty">Loading&hellip;</p></main>

<div class="overlay hidden" id="overlay">
  <div class="modal">
    <div class="modal-head">
      <h3 id="pTitle"></h3>
      <span class="badge" id="pBadge"></span>
      <span class="lockbadge hidden" id="pLockBadge">&#128274; Manual</span>
      <div class="grow"></div>
      <button class="btn" id="pLock">&#128274; Lock</button>
      <button class="btn hidden" id="pRipple" title="When on, moving a word pushes every following word by the same amount so the next word starts exactly where the edited word ends."><span class="led"></span> Ripple</button>
      <button class="btn" id="pEdit">&#9998; Edit</button>
      <button class="btn primary hidden" id="pSave">&#128190; Save</button>
      <span class="x" id="pClose">&#10005;</span>
    </div>
    <div class="modal-body" id="pBody"></div>
    <div class="wave-wrap">
      <div class="wave-head">
        <b id="wTime">0:00.000</b>
        <span id="wSel"></span>
        <div class="grow" style="flex:1"></div>
        <span id="wStatus"></span>
      </div>
      <canvas id="wave" height="84"></canvas>
    </div>
    <div class="player-bar">
      <audio id="audio" controls preload="none"></audio>
    </div>
  </div>
</div>

<div class="overlay hidden" id="logOverlay">
  <div class="modal">
    <div class="modal-head">
      <h3>&#128220; Logs</h3>
      <select id="logSelect"></select>
      <div class="grow"></div>
      <span class="x" id="logClose">&#10005;</span>
    </div>
    <div class="modal-body"><pre class="log" id="logBody">Select a file&hellip;</pre></div>
  </div>
</div>

<script>
let CONFIG = {};
let AUTO = null;

function fmtPct(v) { return v == null ? "\u2014" : (v * 100).toFixed(2) + "%"; }
function statusLabel(s) {
  return {ok:"Complete", needs_review:"Review", needs_fix:"Fix", in_progress:"In progress", unknown:"Unchecked", missing:"Missing"}[s] || s;
}
const STATUS_COLOR = {ok:"var(--ok)", needs_review:"var(--review)", needs_fix:"var(--fix)", in_progress:"var(--progress)", unknown:"var(--unknown)", missing:"var(--missing)"};

async function loadOverview() {
  const r = await fetch("api/overview");
  const data = await r.json();
  CONFIG = data.config;
  renderSummary(data);
  renderReciters(data);
}

function renderSummary(data) {
  const s = document.getElementById("summary");
  let totalPresent = 0, ok = 0, review = 0, fix = 0, prog = 0, unknown = 0;
  for (const rec of data.reciters) {
    totalPresent += rec.present_count;
    ok += rec.counts.ok; review += rec.counts.needs_review;
    fix += rec.counts.needs_fix; unknown += rec.counts.unknown;
    prog += (rec.counts.in_progress || 0);
  }
  const totalTarget = data.reciters.length * data.config.total_chapters;
  const reportInfo = data.report.present
    ? `Validation report: ${data.report.summary ? data.report.summary.total_files : "?"} files`
    : `<span class="err">No validation report yet</span>`;
  const warn = data.config.quran_words_present ? "" :
    "<span class='err'>&#9888; quran_words.json not found &mdash; word text will not appear in the player</span>";
  s.innerHTML = `
    <div class="card"><div class="n">${data.reciters.length}</div><div class="l">Reciters</div></div>
    <div class="card"><div class="n">${totalPresent}${totalTarget?("<span style='color:var(--muted);font-size:16px;font-weight:500'> / "+totalTarget+"</span>"):""}</div><div class="l">Surahs generated</div></div>
    <div class="card" style="--accent-line:var(--ok)"><div class="n" style="color:var(--ok)">${ok}</div><div class="l">Complete</div></div>
    <div class="card" style="--accent-line:var(--review)"><div class="n" style="color:var(--review)">${review}</div><div class="l">Review</div></div>
    <div class="card" style="--accent-line:var(--fix)"><div class="n" style="color:var(--fix)">${fix}</div><div class="l">Fix</div></div>
    <div class="card" style="--accent-line:var(--progress)"><div class="n" style="color:var(--progress)">${prog}</div><div class="l">In progress</div></div>
    <div class="card" style="--accent-line:var(--unknown)"><div class="n" style="color:var(--unknown)">${unknown}</div><div class="l">Unchecked</div></div>
    <div class="card wide"><div class="l">${reportInfo}</div>${warn?`<div class="l">${warn}</div>`:""}</div>
  `;
}

function renderReciters(data) {
  const main = document.getElementById("main");
  if (!data.reciters.length) {
    main.innerHTML = `<p class="empty">No timings found in <code>${data.config.root}</code>. Run the alignment script or change <code>--root</code>.</p>`;
    return;
  }
  main.innerHTML = "";
  // Reciters that are actively being processed (in_progress) rank to the top so
  // you can watch the running job; ties keep the original (stable) order.
  const recs = data.reciters.slice().sort(
    (a, b) => (b.counts.in_progress || 0) - (a.counts.in_progress || 0));
  for (const rec of recs) {
    const pct = Math.round(100 * rec.present_count / rec.total);
    const box = document.createElement("div");
    box.className = "reciter";
    const grid = document.createElement("div");
    grid.className = "grid";
    for (let ch = 1; ch <= rec.total; ch++) {
      const c = rec.chapters[String(ch)];
      const cell = document.createElement("div");
      cell.className = "cell " + c.status;
      cell.innerHTML = `<span class="num">${ch}</span><small>${c.name || ""}</small>`;
      let tip = `${ch}. ${c.name} \u2014 ${statusLabel(c.status)}`;
      if (c.match_rate != null) tip += ` | match ${fmtPct(c.match_rate)}`;
      if (c.estimated) tip += ` | estimated ${c.estimated}`;
      if (c.missing_segments) tip += ` | missing ${c.missing_segments}`;
      if (c.issues && c.issues.length) tip += `\n` + c.issues.join("\n");
      cell.title = tip;
      if (c.status !== "missing") {
        cell.onclick = () => openPlayer(rec.reciter, ch, c);
      }
      grid.appendChild(cell);
    }
    const cc = rec.counts;
    box.innerHTML = `
      <div class="reciter-head">
        <h2>${rec.reciter}</h2>
        <span class="rcount">${rec.present_count} / ${rec.total}</span>
        <div class="progress"><div style="width:${pct}%"></div></div>
        <span class="rcount">${pct}%</span>
        <div class="grow"></div>
        <div class="chips">
          <span class="chip"><i style="background:var(--ok)"></i>${cc.ok}</span>
          <span class="chip"><i style="background:var(--review)"></i>${cc.needs_review}</span>
          <span class="chip"><i style="background:var(--fix)"></i>${cc.needs_fix}</span>
          <span class="chip"><i style="background:var(--progress)"></i>${cc.in_progress || 0}</span>
          <span class="chip"><i style="background:var(--unknown)"></i>${cc.unknown}</span>
        </div>
      </div>`;
    box.appendChild(grid);
    main.appendChild(box);
  }
}

let audioEl, wordSpans = [], rafId = null;
let CUR = null, EDIT = false, RIPPLE = true;

async function openPlayer(reciter, chapter, meta) {
  const ov = document.getElementById("overlay");
  ov.classList.remove("hidden");
  EDIT = false;
  document.getElementById("pEdit").classList.remove("hidden");
  document.getElementById("pEdit").innerHTML = "&#9998; Edit";
  document.getElementById("pSave").classList.add("hidden");
  document.getElementById("pRipple").classList.add("hidden");
  document.getElementById("pTitle").textContent = `${chapter}. ${meta.name || ""} \u2014 ${reciter}`;
  const badge = document.getElementById("pBadge");
  badge.textContent = statusLabel(meta.status) + (meta.match_rate != null ? ` \u00b7 ${fmtPct(meta.match_rate)}` : "");
  badge.style.color = STATUS_COLOR[meta.status] || "var(--txt2)";
  const body = document.getElementById("pBody");
  body.innerHTML = `<p class="empty">Loading&hellip;</p>`;

  const r = await fetch(`api/timing?reciter=${encodeURIComponent(reciter)}&chapter=${chapter}`);
  const data = await r.json();
  if (data.error) { body.innerHTML = `<p class="err">${data.error}</p>`; return; }

  audioEl = document.getElementById("audio");
  audioEl.src = data.audio_url;
  CUR = { reciter, chapter, meta, data };
  renderPlayerBody();
}

// ---- edit helpers: spinner + ripple (cascade) ----------------------------
let EDIT_CELLS = [];      // ordered {sInput, eInput} for every editable word
function gv(inp) { const v = inp.value.trim(); if (v === "") return null; const n = parseInt(v, 10); return Number.isFinite(n) ? n : null; }
function sv(inp, n) { inp.value = (n == null ? "" : Math.max(0, Math.round(n))); }
function flashCell(inp) {
  const c = inp.closest(".w-edit"); if (!c) return;
  c.classList.remove("rippled"); void c.offsetWidth; c.classList.add("rippled");
}
// Move every subsequent word rigidly so the first timed word after `idx` starts
// exactly where the word at `idx` now ends. Inter-word spacing in the tail is
// preserved (a single delta is applied to all following words).
function rippleFromEnd(idx) {
  if (!RIPPLE) return;
  const cells = EDIT_CELLS;
  if (idx < 0 || idx >= cells.length) return;
  const editedEnd = gv(cells[idx].eInput);
  if (editedEnd == null) return;
  let k = -1;
  for (let j = idx + 1; j < cells.length; j++) { if (gv(cells[j].sInput) != null) { k = j; break; } }
  if (k < 0) return;
  const delta = editedEnd - gv(cells[k].sInput);
  if (delta === 0) return;
  for (let j = k; j < cells.length; j++) {
    const c = cells[j];
    const s = gv(c.sInput), e = gv(c.eInput);
    if (s != null) sv(c.sInput, s + delta);
    if (e != null) sv(c.eInput, e + delta);
    if (s != null || e != null) flashCell(c.sInput);
  }
}
// Build one spinner field: label knob + coarse/fine nudge buttons + number.
function makeField(kind, val, idx, isEnd) {
  const fld = document.createElement("div");
  fld.className = "fld";
  const lbl = document.createElement("span");
  lbl.className = "lbl"; lbl.textContent = kind; lbl.title = "drag left/right to scrub (hold Shift = faster)";
  const inp = document.createElement("input");
  inp.type = "number"; inp.className = isEnd ? "in-e" : "in-s"; inp.min = "0";
  inp.placeholder = isEnd ? "end" : "start"; inp.value = (val != null ? val : "");
  const mk = (txt, d) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "nb"; b.textContent = txt; b.tabIndex = -1;
    b.onclick = () => { const cur = gv(inp) || 0; sv(inp, cur + d); if (isEnd) rippleFromEnd(idx); };
    return b;
  };
  fld.appendChild(lbl);
  fld.appendChild(mk("\u00ab", -100)); fld.appendChild(mk("\u2039", -10));
  fld.appendChild(inp);
  fld.appendChild(mk("\u203a", 10)); fld.appendChild(mk("\u00bb", 100));
  // wheel over the field nudges (Shift = coarse)
  fld.addEventListener("wheel", (e) => {
    e.preventDefault();
    const step = (e.deltaY < 0 ? 1 : -1) * (e.shiftKey ? 100 : 10);
    const cur = gv(inp) || 0; sv(inp, cur + step); if (isEnd) rippleFromEnd(idx);
  }, { passive: false });
  // knob: drag the label horizontally to scrub the value (radio-dial feel)
  lbl.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    lbl.setPointerCapture(e.pointerId);
    fld.classList.add("knobbing");
    const startX = e.clientX, startVal = gv(inp) || 0;
    const move = (ev) => {
      const rate = ev.shiftKey ? 8 : 1;   // ms per pixel
      sv(inp, startVal + Math.round((ev.clientX - startX) * rate));
      if (isEnd) rippleFromEnd(idx);
    };
    const up = (ev) => {
      try { lbl.releasePointerCapture(e.pointerId); } catch (_) {}
      fld.classList.remove("knobbing");
      lbl.removeEventListener("pointermove", move);
      lbl.removeEventListener("pointerup", up);
      lbl.removeEventListener("pointercancel", up);
      lbl.removeEventListener("lostpointercapture", up);
    };
    lbl.addEventListener("pointermove", move);
    lbl.addEventListener("pointerup", up);
    lbl.addEventListener("pointercancel", up);
    lbl.addEventListener("lostpointercapture", up);
  });
  // manual typing: ripple once the value is committed (blur / Enter)
  inp.addEventListener("change", () => { if (isEnd) rippleFromEnd(idx); });
  return { fld, inp };
}

function renderPlayerBody() {
  const body = document.getElementById("pBody");
  const data = CUR.data;
  wordSpans = [];
  EDIT_CELLS = [];
  body.innerHTML = "";
  const info = document.createElement("p");
  info.className = "vk";
  info.innerHTML = `${data.word_count} words \u00b7 ${data.missing_count} missing`
    + (data.has_text ? "" : ` \u00b7 <span class="err">no text (quran_words.json missing)</span>`)
    + (EDIT ? ` \u00b7 <span style="color:var(--progress)">editing (ms) \u2014 \u00ab\u2039 \u203a\u00bb step \u00b7 wheel or drag S/E to scrub \u00b7 Ripple ${RIPPLE ? "on" : "off"}</span>` : "");
  body.appendChild(info);

  for (const v of data.verses) {
    const vd = document.createElement("div");
    vd.className = "verse";
    const vk = document.createElement("div");
    vk.className = "vk"; vk.textContent = v.verse_key;
    vd.appendChild(vk);
    const ws = document.createElement("div");
    ws.className = "words";
    for (const w of v.words) {
      if (EDIT) {
        const cell = document.createElement("div");
        cell.className = "w-edit" + (w.missing ? " miss" : "");
        cell.dataset.vk = v.verse_key;
        cell.dataset.pos = w.position;
        const idx = EDIT_CELLS.length;
        const t = document.createElement("div");
        t.className = "wt";
        t.textContent = w.text || ("#" + w.position);
        t.title = "click to preview from here";
        const sF = makeField("S", w.start_ms, idx, false);
        const eF = makeField("E", w.end_ms, idx, true);
        // whole-word shift row (advance / delay the word; ripples the tail)
        const act = document.createElement("div");
        act.className = "w-actions";
        const shift = (txt, d) => {
          const b = document.createElement("button");
          b.type = "button"; b.className = "nb cap"; b.textContent = txt; b.tabIndex = -1;
          b.title = "shift the whole word " + (d < 0 ? "earlier" : "later") + " by " + Math.abs(d) + " ms";
          b.onclick = () => {
            const s = gv(sF.inp), e = gv(eF.inp);
            if (s != null) sv(sF.inp, s + d);
            if (e != null) sv(eF.inp, e + d);
            rippleFromEnd(idx);
          };
          return b;
        };
        act.appendChild(shift("\u00ab", -100)); act.appendChild(shift("\u2039", -10));
        act.appendChild(shift("\u203a", 10)); act.appendChild(shift("\u00bb", 100));
        t.onclick = () => {
          const s = gv(sF.inp);
          if (s != null && audioEl) { audioEl.currentTime = s / 1000; audioEl.play(); }
        };
        cell.appendChild(t); cell.appendChild(sF.fld); cell.appendChild(eF.fld); cell.appendChild(act);
        ws.appendChild(cell);
        EDIT_CELLS.push({ sInput: sF.inp, eInput: eF.inp });
      } else {
        const sp = document.createElement("span");
        sp.className = "w" + (w.missing ? " miss" : "");
        sp.textContent = w.text || ("#" + w.position);
        if (!w.missing) {
          sp.dataset.start = w.start_ms;
          sp.dataset.end = w.end_ms;
          sp.onclick = () => { audioEl.currentTime = w.start_ms / 1000; audioEl.play(); };
          wordSpans.push(sp);
        }
        ws.appendChild(sp);
      }
    }
    vd.appendChild(ws);
    body.appendChild(vd);
  }
  if (EDIT) {
    cancelAnimationFrame(rafId);
  } else {
    wordSpans.sort((a, b) => a.dataset.start - b.dataset.start);
    startSync();
  }
}

async function savePlayer() {
  if (!CUR) return;
  const rows = document.querySelectorAll("#pBody .w-edit");
  const edits = [];
  let bad = 0;
  rows.forEach(r => {
    const s = r.querySelector(".in-s").value.trim();
    const e = r.querySelector(".in-e").value.trim();
    if (s === "" && e === "") return;
    const sn = parseInt(s, 10), en = parseInt(e, 10);
    if (!Number.isFinite(sn) || !Number.isFinite(en) || sn < 0 || en < 0) { bad++; return; }
    edits.push({ verse_key: r.dataset.vk, position: +r.dataset.pos, start_ms: sn, end_ms: en });
  });
  if (!edits.length) { alert(bad ? "Some values are invalid; nothing was saved." : "No timings to save."); return; }
  const btn = document.getElementById("pSave");
  const old = btn.innerHTML;
  btn.disabled = true; btn.textContent = "Saving\u2026";
  try {
    const r = await fetch("api/timing", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reciter: CUR.reciter, chapter: CUR.chapter, edits })
    });
    const res = await r.json();
    if (res.error) { alert("Save failed: " + res.error); return; }
    const rr = await fetch(`api/timing?reciter=${encodeURIComponent(CUR.reciter)}&chapter=${CUR.chapter}`);
    const fresh = await rr.json();
    if (!fresh.error) { CUR.data = fresh; }
    EDIT = false;
    document.getElementById("pEdit").innerHTML = "&#9998; Edit";
    btn.classList.add("hidden");
    document.getElementById("pRipple").classList.add("hidden");
    renderPlayerBody();
  } catch (err) {
    alert("Save failed: " + err);
  } finally {
    btn.disabled = false; btn.innerHTML = old;
  }
}

function startSync() {
  cancelAnimationFrame(rafId);
  let last = null;
  function tick() {
    if (audioEl && !audioEl.paused) {
      const t = audioEl.currentTime * 1000;
      let cur = null;
      for (const sp of wordSpans) {
        if (t >= +sp.dataset.start && t < +sp.dataset.end) { cur = sp; break; }
      }
      if (cur !== last) {
        if (last) last.classList.remove("active");
        if (cur) {
          cur.classList.add("active");
          cur.scrollIntoView({ block: "nearest", behavior: "smooth" });
        }
        last = cur;
      }
    }
    rafId = requestAnimationFrame(tick);
  }
  rafId = requestAnimationFrame(tick);
}

function closePlayer() {
  document.getElementById("overlay").classList.add("hidden");
  if (audioEl) { audioEl.pause(); audioEl.src = ""; }
  cancelAnimationFrame(rafId);
}

// logs
async function openLogs() {
  document.getElementById("logOverlay").classList.remove("hidden");
  const r = await fetch("api/logs");
  const data = await r.json();
  const sel = document.getElementById("logSelect");
  sel.innerHTML = "";
  if (!data.logs || !data.logs.length) {
    document.getElementById("logBody").textContent = "No log files";
    return;
  }
  for (const l of data.logs) {
    const o = document.createElement("option");
    o.value = l.file; o.textContent = l.file + "  (" + Math.round(l.size/1024) + " KB)";
    sel.appendChild(o);
  }
  sel.onchange = loadLog;
  loadLog();
}
async function loadLog() {
  const file = document.getElementById("logSelect").value;
  const r = await fetch("api/logs?file=" + encodeURIComponent(file) + "&lines=400");
  const data = await r.json();
  document.getElementById("logBody").textContent = data.error ? data.error : data.text;
}

document.getElementById("refreshBtn").onclick = loadOverview;
document.getElementById("pEdit").onclick = () => {
  EDIT = !EDIT;
  document.getElementById("pEdit").innerHTML = EDIT ? "&#10005; Cancel" : "&#9998; Edit";
  document.getElementById("pSave").classList.toggle("hidden", !EDIT);
  const rb = document.getElementById("pRipple");
  rb.classList.toggle("hidden", !EDIT);
  rb.classList.toggle("active", RIPPLE);
  if (EDIT && audioEl) audioEl.pause();
  if (CUR) renderPlayerBody();
};
document.getElementById("pRipple").onclick = function () {
  RIPPLE = !RIPPLE;
  this.classList.toggle("active", RIPPLE);
  if (CUR && EDIT) renderPlayerBody();
};
document.getElementById("pSave").onclick = savePlayer;
document.getElementById("pClose").onclick = closePlayer;
document.getElementById("logsBtn").onclick = openLogs;
document.getElementById("logClose").onclick = () => document.getElementById("logOverlay").classList.add("hidden");
document.getElementById("overlay").onclick = (e) => { if (e.target.id === "overlay") closePlayer(); };
document.getElementById("logOverlay").onclick = (e) => { if (e.target.id === "logOverlay") document.getElementById("logOverlay").classList.add("hidden"); };
document.getElementById("autoBtn").onclick = function () {
  if (AUTO) { clearInterval(AUTO); AUTO = null; this.classList.remove("active"); }
  else { AUTO = setInterval(loadOverview, 5000); this.classList.add("active"); }
};
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closePlayer(); document.getElementById("logOverlay").classList.add("hidden"); } });

loadOverview();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Local dashboard for Quran forced-alignment timings.")
    ap.add_argument("--root", default="timings", help="Timings root folder (default: timings)")
    ap.add_argument("--logs", default="logs", help="Logs folder (default: logs)")
    ap.add_argument("--debug-logs", default="debug_logs", help="Per-reciter debug logs folder written by the batch runner (default: debug_logs)")
    ap.add_argument("--quran-words", default="quran_words.json", help="quran_words.json path")
    ap.add_argument("--report", default="logs/timings_validation_report.json", help="Validation report json path")
    ap.add_argument("--base-url", default="https://qurani.io/quransound", help="Reciter audio base url")
    ap.add_argument("--audio-user-agent", default="qurani", help="User-Agent used by the server-side audio proxy to pass the CDN WAF (default: qurani)")
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    ap.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser")
    args = ap.parse_args()

    Handler.state = DashboardState(args)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print("=" * 60)
    print("  Quran timings dashboard")
    print(f"  Serving:      {url}")
    print(f"  Timings root: {args.root}")
    print(f"  Logs:         {args.logs}")
    print(f"  Report:       {args.report}")
    print(f"  Audio base:   {args.base_url}")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
        httpd.shutdown()


if __name__ == "__main__":
    main()
