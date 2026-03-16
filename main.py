import argparse
import concurrent.futures
import os
import re
import subprocess
import time
import warnings

import av
import cv2
import numpy as np
import torch
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# ---------------------  Defaults  ---------------------
DEFAULT_THRESHOLD = (
    3.0  # mean pixel diff after downscale (lower than before since we resize)
)
DEFAULT_MODEL = "base"
DEFAULT_AUDIO_LOOKAHEAD = 1.0
DEFAULT_MIN_SLIDE_DURATION = 4.0
COARSE_JUMP = 60  # seconds between coarse probes
FINE_RESOLUTION = 1.0  # stop bisecting when interval < this (seconds)
RESIZE_WIDTH = 320  # downscale width; height auto-scales
# -------------------------------------------------------

GEMINI_PROMPT = (
    "You are a precise note-taking assistant. You will receive the raw transcription "
    "of what a professor said during a single lecture slide. "
    "Reformat it into clean bullet points using Typst syntax (- point one). "
    "Preserve every distinct idea. Do not add new ideas the professor did not mention. "
    "Rephrase freely for clarity and conciseness. Remove filler words like uh, um, so basically. "
    "Group related ideas into the same bullet, split unrelated ones. "
    "Output only the bullet points, nothing else — no dictionary wrapper, no slide key, no explanation."
)


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.1f}s"


def escape_typst(text: str) -> str:
    return text.replace("\\", "\\\\").replace("]", "\\]").replace("[", "\\[")


def check_gpu() -> str:
    print("🖥️  Checking hardware...")
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        try:
            t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            _ = t + t
            print(f"   🚀 GPU compute verified! ({torch.cuda.get_device_name(0)})")
            print(f"   ⏱  {fmt_duration(time.perf_counter() - t0)}")
            return "cuda"
        except Exception as e:
            print(
                f"   ⚠️  GPU detected but compute failed ({e.__class__.__name__}). Falling back to CPU."
            )
    else:
        print("   ⚠️  No GPU detected, using CPU.")
    print(f"   ⏱  {fmt_duration(time.perf_counter() - t0)}")
    return "cpu"


# --------------- Frame extraction via ffmpeg ---------------


def grab_frame(video_path: str, second: float) -> np.ndarray | None:
    """
    Extract a single frame at `second` using PyAV (ffmpeg C library directly,
    no subprocess spawning). Returns a downscaled grayscale numpy array.
    """
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            ts = int(second / stream.time_base)
            container.seek(ts, stream=stream)
            for frame in container.decode(stream):
                img = frame.to_ndarray(format="gray")
                h, w = img.shape
                new_h = int(h * RESIZE_WIDTH / w)
                return cv2.resize(
                    img, (RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA
                )
    except Exception:
        return None
    return None


def frames_differ(a: np.ndarray, b: np.ndarray, threshold: float) -> tuple[bool, float]:
    diff = float(cv2.absdiff(a, b).mean())
    return diff > threshold, diff


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


# --------------- Slide detection ---------------


def detect_slides(
    video_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    min_slide_duration: float = DEFAULT_MIN_SLIDE_DURATION,
) -> list[float]:
    print(
        f"\n🔍 Detecting slides (coarse={COARSE_JUMP}s → fine={FINE_RESOLUTION}s, scale={RESIZE_WIDTH}px wide)..."
    )
    t0 = time.perf_counter()

    duration = get_video_duration(video_path)
    if duration == 0.0:
        print("   ⚠️  Could not determine video duration.")
        return [0.0]
    print(f"   Video duration: {fmt_duration(duration)}")

    timestamps: list[float] = [0.0]
    prev_frame = grab_frame(video_path, 0.0)
    if prev_frame is None:
        print("   ❌ Could not read first frame.")
        return timestamps

    t = 0.0
    frame_reads = 1

    while t + COARSE_JUMP < duration:
        next_t = min(t + COARSE_JUMP, duration - 1)
        next_frame = grab_frame(video_path, next_t)
        frame_reads += 1
        if next_frame is None:
            break

        changed, diff = frames_differ(prev_frame, next_frame, threshold)

        if changed:
            # Binary search to pinpoint the transition within FINE_RESOLUTION seconds
            lo, hi = t, next_t
            lo_frame = prev_frame

            while hi - lo > FINE_RESOLUTION:
                mid = (lo + hi) / 2
                mid_frame = grab_frame(video_path, mid)
                frame_reads += 1
                if mid_frame is None:
                    break
                mid_changed, _ = frames_differ(lo_frame, mid_frame, threshold)
                if mid_changed:
                    hi = mid
                else:
                    lo = mid
                    lo_frame = mid_frame  # stable zone moved forward

            change_ts = round(hi, 1)
            if change_ts - timestamps[-1] >= min_slide_duration:
                timestamps.append(change_ts)
                print(f"   ➔ Slide change at {change_ts:.1f}s  (diff={diff:.1f})")

            prev_frame = grab_frame(video_path, hi)
            frame_reads += 1
            t = hi
        else:
            prev_frame = next_frame
            t = next_t

    print(
        f"   ⏱  Slide detection: {fmt_duration(time.perf_counter() - t0)} "
        f"— {len(timestamps)} slide(s) — {frame_reads} frame reads total"
    )
    return timestamps


# --------------- Transcription ---------------


def transcribe_audio(
    video_path: str,
    device: str,
    model_name: str = DEFAULT_MODEL,
) -> list[dict]:
    print(f"\n🎙️  Transcribing audio (model={model_name}, device={device.upper()})...")
    t0 = time.perf_counter()

    compute_type = "int8" if device == "cpu" else "float16"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    duration = get_video_duration(video_path)
    segments_gen, _ = model.transcribe(video_path, beam_size=5)

    segments = []
    last_printed_pct = -1

    for s in segments_gen:
        segments.append({"text": s.text, "start": s.start, "end": s.end})

        pct = int((s.end / duration * 100) if duration > 0 else 0)
        milestone = (pct // 10) * 10  # round down to nearest 10

        if milestone > last_printed_pct:
            last_printed_pct = milestone
            elapsed = time.perf_counter() - t0
            eta = (elapsed / (s.end + 0.001)) * (duration - s.end)
            bar = ("█" * (milestone // 10)).ljust(10)
            print(
                f"   [{bar}] {milestone:3d}%  elapsed {fmt_duration(elapsed)}  ETA {fmt_duration(eta)}",
                flush=True,
            )

    print(f"   ⏱  Transcription done: {fmt_duration(time.perf_counter() - t0)}")
    return segments


# --------------- Typst building ---------------


def build_typst(
    slides: list[float],
    segments: list[dict],
    lookahead: float = DEFAULT_AUDIO_LOOKAHEAD,
    start_slide: int = 1,  # <-- Added parameter
) -> str:
    print("\n📝 Building Typst output...")
    t0 = time.perf_counter()

    lines = ["#let notes = ("]
    for i, slide_start in enumerate(slides):
        slide_end = slides[i + 1] if i + 1 < len(slides) else float("inf")
        window_start = slide_start - lookahead

        text = " ".join(
            seg["text"] for seg in segments if window_start <= seg["start"] < slide_end
        ).strip()

        if text:
            current_slide_num = i + start_slide
            print(f"   ➔ Slide {current_slide_num}")
            lines.append(f'  "{current_slide_num}": [{escape_typst(text)}],')

    lines.append(")")
    print(f"   ⏱  Build: {fmt_duration(time.perf_counter() - t0)}")
    return "\n".join(lines) + "\n"


# --------------- Gemini formatting ---------------


def format_with_gemini(raw_typ: str, output_path: str) -> None:
    slide_pattern = re.compile(r'"(slide_\d+)":\s*\[([^\]]*)\]', re.DOTALL)
    slides = slide_pattern.findall(raw_typ)

    if not slides:
        print("   ⚠️  No slides found to format.")
        return

    t0 = time.perf_counter()
    formatted_lines = ["#let lecture_notes = ("]

    for key, text in slides:
        text = text.strip()
        print(f"   🤖 Formatting {key}...")
        result = subprocess.run(
            ["gemini", "-p", GEMINI_PROMPT],
            input=text,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode == 0:
            formatted_lines.append(f'  "{key}": [{result.stdout.strip()}],')
        else:
            print(f"   ⚠️  Gemini failed for {key}, keeping raw text.")
            print(f"       {result.stderr.strip()}")
            formatted_lines.append(f'  "{key}": [{text}],')

    formatted_lines.append(")")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted_lines) + "\n")

    print(f"   ⏱  Gemini formatting: {fmt_duration(time.perf_counter() - t0)}")
    print(f"   ✅ Formatted notes → {output_path}")


# --------------- Using provided transcriptions as .vtt ---------------


def parse_vtt(vtt_path: str) -> list[dict]:
    print(f"\n📄 Found existing transcription! Parsing {os.path.basename(vtt_path)}...")
    segments = []

    # Matches VTT timestamps like "00:01:23.456 --> 00:01:28.000" or "01:23.456 --> 01:28.000"
    pattern = re.compile(
        r"(?:(\d{2}):)?(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(?:(\d{2}):)?(\d{2}):(\d{2})\.(\d{3})"
    )

    def to_seconds(h, m, s, ms):
        h = int(h) if h else 0
        return h * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    try:
        with open(vtt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"   ⚠️ Could not read VTT file: {e}")
        return []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = pattern.search(line)
        if match:
            # Parse start and end times
            sh, sm, ss, sms = match.group(1, 2, 3, 4)
            eh, em, es, ems = match.group(5, 6, 7, 8)

            start_sec = to_seconds(sh, sm, ss, sms)
            end_sec = to_seconds(eh, em, es, ems)

            # The next line(s) contain the text until a blank line
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip() != "":
                # Remove common VTT styling tags like <c.color> or <b>
                clean_text = re.sub(r"<[^>]+>", "", lines[i].strip())
                text_lines.append(clean_text)
                i += 1

            segments.append(
                {"text": " ".join(text_lines), "start": start_sec, "end": end_sec}
            )
        else:
            i += 1

    print(f"   ✅ Parsed {len(segments)} transcription segments.")
    return segments


# --------------- Per-video pipeline ---------------


def process_one(video_path: str, args) -> None:
    if not os.path.exists(video_path):
        print(f"❌ Skipping '{video_path}' — file not found.")
        return

    total_start = time.perf_counter()
    stem = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, f"{stem}_notes.typ")
    formatted_path = os.path.join(args.output_dir, f"{stem}_formatted.typ")

    # Check for an accompanying .vtt file
    vtt_files = [
        f
        for f in os.listdir(os.path.dirname(video_path) or ".")
        if f.startswith(stem) and f.endswith(".vtt")
    ]

    vtt_path = (
        os.path.join(os.path.dirname(video_path), vtt_files[0]) if vtt_files else None
    )
    has_vtt = vtt_path is not None and os.path.exists(vtt_path)

    print(f"\n{'=' * 60}")
    print(f"▶  {video_path}")
    print(f"{'=' * 60}")

    if has_vtt:
        print("\n⚡ VTT file detected. Running slide detection only...")
        # Run slide detection sequentially since we don't need Whisper
        slides = detect_slides(video_path, args.threshold, args.min_slide_duration)
        segments = parse_vtt(vtt_path)
    else:
        # Fallback to original parallel Whisper + Slide Detection
        device = check_gpu()
        print(
            "\n⚡ No VTT file found. Running slide detection and Whisper transcription in parallel..."
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            slides_future = pool.submit(
                detect_slides, video_path, args.threshold, args.min_slide_duration
            )
            segments_future = pool.submit(
                transcribe_audio, video_path, device, args.model
            )
            slides = slides_future.result()
            segments = segments_future.result()

    typst_content = build_typst(slides, segments, args.lookahead, args.start_slide)

    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(typst_content)
    print(f"\n💾 Raw notes saved → {raw_path}")

    if not args.no_gemini:
        print(f"\n🤖 Sending slides to Gemini CLI one by one...")
        format_with_gemini(typst_content, formatted_path)
    else:
        print("   ⏭  Skipping Gemini formatting (--no-gemini).")

    print(
        f"\n⏱  Total for this video: {fmt_duration(time.perf_counter() - total_start)}"
    )


# --------------- Entry point ---------------


def main() -> None:
    examples = """
Examples of how to use this tool:
--------------------------------------------------------------------------------
1. Standard run (detect slides, Whisper transcribe, Gemini format):
    python script.py "lecture_01.mp4"

2. Process multiple videos at once:
    python script.py "lecture_01.mp4" "lecture_02.mp4"

3. Skip Gemini formatting (fastest way to get raw notes):
    python script.py "lecture_01.mp4" --no-gemini

4. Part 2 of a lecture (start numbering at slide 45):
    python script.py "lecture_part2.mp4" --start-slide 45

5. Using a downloaded YouTube VTT with custom slide sensitivity:
    python script.py "youtube_class.mp4" --threshold 10.0 --no-gemini
--------------------------------------------------------------------------------
Note: If a .vtt file with the exact same name as the video exists in the
same folder, the script will automatically use it and skip Whisper entirely!
"""
    parser = argparse.ArgumentParser(
        description="Convert lecture video(s) into Typst notes, then format with Gemini CLI.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,  # This keeps your line breaks!
    )
    parser.add_argument("videos", nargs="+", help="One or more video files to process.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Pixel-diff threshold after downscale (default: {DEFAULT_THRESHOLD}). "
        "Lower = more sensitive. Raise if too many false positives.",
    )
    parser.add_argument(
        "--min-slide-duration",
        type=float,
        default=DEFAULT_MIN_SLIDE_DURATION,
        help=f"Min seconds between slides (default: {DEFAULT_MIN_SLIDE_DURATION}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["tiny", "base", "small", "medium", "large"],
        help=f"Whisper model size (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--lookahead",
        type=float,
        default=DEFAULT_AUDIO_LOOKAHEAD,
        help=f"Seconds of audio before a slide to include (default: {DEFAULT_AUDIO_LOOKAHEAD}).",
    )
    parser.add_argument(
        "--output-dir", default="output", help="Output directory (default: output)."
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Skip Gemini formatting, only produce raw notes.",
    )
    parser.add_argument(
        "--start-slide",
        type=int,
        default=1,
        help="Starting number for the first slide (default: 1).",
    )
    args = parser.parse_args()

    grand_start = time.perf_counter()
    for video_path in args.videos:
        process_one(video_path, args)

    if len(args.videos) > 1:
        print(
            f"\n🏁 All {len(args.videos)} videos done in {fmt_duration(time.perf_counter() - grand_start)}"
        )


if __name__ == "__main__":
    main()
