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

## الترخيص

مجاني للاستخدام في المنتجات المجانية فقط — راجع ملف [LICENSE](LICENSE). باختصار: يمكنك استخدام الأداة وتعديلها وإعادة توزيعها دون مقابل، بشرط أن يكون أي منتج تبنيه بها مجانيًا بالكامل، دون أي وصول مدفوع ودون أي مشتريات داخل التطبيق.
