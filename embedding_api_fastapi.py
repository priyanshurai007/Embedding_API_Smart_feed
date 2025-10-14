from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os

# ---------- Setup ----------
app = FastAPI(title="Embedding API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Model ----------
MODEL_NAME = os.getenv("MODEL_NAME", "paraphrase-MiniLM-L3-v2")
print(f"🔹 Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print("✅ Model loaded successfully!")

# ---------- Request Schema ----------
class TextInput(BaseModel):
    text: str

# ---------- Endpoint ----------
@app.post("/embed")
def get_embedding(data: TextInput):
    text = data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    embedding = model.encode([text])[0].tolist()
    return {"embedding": embedding}

# ---------- Startup ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("embedding_api_fastapi:app", host="0.0.0.0", port=port)
