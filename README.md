# 🎓 Lecture-to-Typst (Auto-Scribe)

An automated, privacy-first pipeline that takes video lectures, detects slide transitions, transcribes the audio, and uses a local LLM to format the notes into a Typst dictionary. 

Built entirely to run locally on Linux Mint using Python, OpenCV, Faster-Whisper, and Ollama (leveraging an AMD 5700XT GPU).

## 🚀 Features
- **Local & Private:** No audio or video is sent to the cloud.
- **Visual Slide Sync:** Uses OpenCV to detect when the professor changes slides.
- **AI Transcription:** Uses `faster-whisper` for highly accurate, fast audio-totext.
- **LLM Summarization:** Uses Ollama (Llama 3) to summarize raw transcripts into clean bullet points.
- **Native Typst Output:** Automatically formats the final output into a native `.typ` dictionary for immediate use in Typst templates.

## 🛠️ Prerequisites
- **OS:** Linux (Mint/Ubuntu)
- **Hardware:** AMD GPU (5700XT) using ROCm (or CPU fallback)
- **Dependencies:** FFmpeg, Python 3.10+
- **LLM Engine:** [Ollama](https://ollama.com/) running locally with the `llama3` model.

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd lecture-to-typst
   ```

2. **Install system dependencies:**
   ```bash
   sudo apt update && sudo apt install ffmpeg python3-venv git -y
   ```

3. **Set up the Python Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍ How to Use

1. Place your video file (e.g., `.mp4`) into the `input_videos/` folder.

2. Open `process_lecture.py` and update the `VIDEO_PATH` variable to point to your file.

3. Make sure Ollama is running in the background (`ollama serve`).

4. Run the pipeline:
   ```bash
   python process_lecture.py
   ```

5. Grab your generated `lecture_notes.typ` from the `output/` folder!

*Save the file once you've pasted that in.*


## Prompt to use
You are a precise note-taking assistant. You will receive a Typst dictionary where each key is a slide number and the value is the raw transcription of what the professor said during that slide.
Your task is to reformat the dictionary, replacing each slide's raw text with clean bullet points in Typst syntax.

**Rules:**

* Keep the exact same Typst dictionary structure — same keys, same format
* Convert each slide's text into bullet points using Typst syntax: - point one
* Preserve every distinct idea from the original. Do not drop any concept, even minor ones
* Do not add new ideas, examples, or context that the professor did not mention
* You may rephrase freely for clarity and conciseness — exact wording does not matter, only the ideas do
* Remove filler words and speech artifacts (e.g. "uh", "um", "so basically", "right?")
* Group related ideas into the same bullet, split unrelated ones
* Output only the Typst dictionary, nothing else — no explanation, no preamble, no closing remarks

Input:
```typst
{paste the entire lecture_notes.typ content here}
```
