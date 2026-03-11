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

---

### 2. Make your first Git Commit
Now that we have our folder structure, our `.gitignore` protecting us from giant files, and our `README.md` explaining the project, it is time to save this baseline into Git.

Go back to your terminal (make sure you are inside the `lecture-to-typst` folder) and run these commands:

```bash
# Stage the README and the .gitignore files
git add README.md .gitignore

# Commit them to the repository history
git commit -m "Initial commit: Added project structure, .gitignore, and README"
```

