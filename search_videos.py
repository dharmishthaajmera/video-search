import os
import datetime
import whisper
from llama_index.core import Document, VectorStoreIndex
# from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

Settings.llm = None

# -------- SETTINGS --------
VIDEO_FOLDER = "videos"
WHISPER_MODEL = "base"
TOP_K_RESULTS = 1

# -------- STEP 1: Transcription --------
print("Loading Whisper model...")
asr_model = whisper.load_model(WHISPER_MODEL)

def format_timestamp(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def transcribe_video(video_path):
    print(f"Transcribing: {video_path}")
    result = asr_model.transcribe(video_path)
    segments = result['segments']
    return [
        {
            "text": seg["text"].strip(),
            "start": format_timestamp(seg["start"]),
            "video_name": os.path.basename(video_path)
        }
        for seg in segments
    ]

# -------- STEP 2: Process All Videos --------
all_chunks = []

for filename in os.listdir(VIDEO_FOLDER):
    if filename.lower().endswith((".mp4", ".mov")):
        full_path = os.path.join(VIDEO_FOLDER, filename)
        transcript = transcribe_video(full_path)

        for chunk in transcript:
            doc = Document(
                text=chunk["text"],
                metadata={
                    "video_name": chunk["video_name"],
                    "timestamp": chunk["start"]
                }
            )
            all_chunks.append(doc)

print(f"\nTotal chunks indexed: {len(all_chunks)}")

# -------- STEP 3: Build Vector Index --------
print("\nBuilding vector index...")
# index = VectorStoreIndex.from_documents(all_chunks)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
index = VectorStoreIndex.from_documents(all_chunks, embed_model=embed_model)

# -------- STEP 4: Query Loop --------
query_engine = index.as_query_engine(similarity_top_k=TOP_K_RESULTS)

print("\n🔍 Ready to ask questions. Type 'exit' to quit.\n")

while True:
    query = input("Ask a question: ").strip()
    if query.lower() == "exit":
        break

    response = query_engine.query(query)

    print("\n--- Result ---")
    if response.source_nodes:
        top_node = response.source_nodes[0].node
        print("Video:", top_node.metadata.get("video_name", "Unknown"))
        print("Timestamp:", top_node.metadata.get("timestamp", "Unknown"))
        print("Answer Snippet:", top_node.text)
    else:
        print("No relevant match found.")
    print("------------------------\n")
