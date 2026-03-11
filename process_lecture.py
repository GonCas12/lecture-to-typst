import os
import time

import cv2
import torch
import whisper

# --- Configuration ---
VIDEO_PATH = "input_videos/SC2.mp4"
SLIDE_THRESHOLD = 20.0
# ---------------------


def fmt_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.1f}s"


def check_gpu():
    print("🖥️  Checking hardware...")
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        try:
            t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            _ = t + t
            print(
                f"   🚀 SUCCESS: GPU compute verified! ({torch.cuda.get_device_name(0)})"
            )
            print(f"   ⏱  Hardware check: {fmt_duration(time.perf_counter() - t0)}")
            return "cuda"
        except Exception as e:
            print(f"   ⚠️  GPU detected but compute failed ({e.__class__.__name__}).")
            print("   ⚠️  Falling back to CPU.")
            print(f"   ⏱  Hardware check: {fmt_duration(time.perf_counter() - t0)}")
            return "cpu"
    else:
        print("   ⚠️  No GPU detected, using CPU.")
        print(f"   ⏱  Hardware check: {fmt_duration(time.perf_counter() - t0)}")
        return "cpu"


def detect_slides(video_path, threshold=SLIDE_THRESHOLD):
    print("\n🔍 Scanning video for slide changes (2 times per second)...")
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamps = [0.0]

    success, prev_frame = cap.read()
    if not success:
        return timestamps

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    step_size = int(fps)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + step_size)
        success, frame = cap.read()
        if not success:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_frame_gray, frame_gray)
        mean_diff = diff.mean()

        if mean_diff > threshold:
            timestamp = (frame_count + step_size) / fps
            timestamps.append(timestamp)
            print(f"   ➔ Slide changed at {timestamp:.1f} seconds")
            prev_frame_gray = frame_gray

        frame_count += step_size

    cap.release()
    print(
        f"   ⏱  Slide detection: {fmt_duration(time.perf_counter() - t0)} — {len(timestamps)} slide(s) found"
    )
    return timestamps


def transcribe_audio(video_path, device):
    print(f"\n🎙️ Transcribing audio with Whisper (Device: {device.upper()})...")
    t0 = time.perf_counter()
    model = whisper.load_model("base", device=device)
    result = model.transcribe(video_path)
    print(f"   ⏱  Transcription: {fmt_duration(time.perf_counter() - t0)}")
    return result["segments"]


def main():
    total_start = time.perf_counter()

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Cannot find '{VIDEO_PATH}'.")
        return

    device = check_gpu()
    slides = detect_slides(VIDEO_PATH)
    segments = transcribe_audio(VIDEO_PATH, device)

    print("\n📝 Building Typst output...")
    t0 = time.perf_counter()
    final_typst = "#let lecture_notes = (\n"

    for i, slide_time in enumerate(slides):
        next_slide_time = slides[i + 1] if i + 1 < len(slides) else float("inf")

        slide_text = ""
        for seg in segments:
            if (slide_time - 1.5) <= seg["start"] < next_slide_time:  # type: ignore
                slide_text += seg["text"] + " "  # type: ignore

        if slide_text.strip():
            print(f"   ➔ Writing Slide {i + 1}...")
            final_typst += f'  "slide_{i + 1}": [\n{slide_text.strip()}\n  ],\n'

    final_typst += ")\n"

    os.makedirs("output", exist_ok=True)
    with open("output/lecture_notes.typ", "w") as f:
        f.write(final_typst)

    print(f"   ⏱  Typst writing: {fmt_duration(time.perf_counter() - t0)}")
    print("\n✅ Done! Check the output/lecture_notes.typ file.")
    print(f"⏱  Total duration: {fmt_duration(time.perf_counter() - total_start)}")


if __name__ == "__main__":
    main()
