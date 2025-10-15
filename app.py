from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os

app = FastAPI(title="Smart Feed Embedding API")

# Allow cross-origin calls (for your backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_NAME = "paraphrase-MiniLM-L3-v2"
print(f"🔹 Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print("✅ Model loaded successfully!")

# Request body model
class TextIn(BaseModel):
    text: str

@app.post("/embed")
async def embed(request: TextIn):
    text = request.text.strip()
    if not text:
        return {"error": "Empty text"}
    embedding = model.encode([text])[0].tolist()
    return {"embedding": embedding}
