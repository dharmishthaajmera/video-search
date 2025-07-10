# 🎥 Video Search

Semantic search engine for video files.  
It transcribes videos using Whisper and indexes the content with LlamaIndex + HuggingFace embeddings.  
You can ask a question and get:

- ✅ Most relevant video filename  
- ✅ Exact timestamp  
- ✅ Relevant transcript snippet  

No API key needed. Fully local setup.

---

## 📦 Features

- Transcribe `.mp4`, `.mov`
- Build vector index with local sentence-transformer embeddings
- Ask natural language queries
- Get video name, timestamp, and exact spoken snippet
- No OpenAI key or cloud setup required

---

## 🧰 Requirements

- Python 3.8+
- FFmpeg (`brew install ffmpeg` or `sudo apt install ffmpeg`)
- Dependencies:
  pip install -r requirements.txt

## Folder Structure
video-search/
├── videos/               # Place your .mp4 files here
├── search_videos.py      # Main script
├── requirements.txt
└── README.md

## ▶️ How to Run
Place your video files inside the videos/ folder

## Run the script:
python search_videos.py
Ask any question when prompted. Example:

Ask a question: what is nodejs?

--- Result ---
Video: video1.mp4
Timestamp: 0:00:21
Answer Snippet:
Node.js is a JavaScript runtime built on Chrome’s V8 engine...