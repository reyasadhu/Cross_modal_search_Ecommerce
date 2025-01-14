from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pinecone import Pinecone
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import io
import numpy as np

app = FastAPI()

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_5fSate_SqqXwN1s4AGoWcn5KY1VGXDe2QAEbMGSpLk2JKfqFNhHSrtBSNYgaBkgwH727Hn")
index_name = "clip-multimodal-search"
index = pc.Index(name=index_name)

# Load CLIP model and processor
model_id = "openai/clip-vit-base-patch32"
device = torch.device("cuda" if torch.cuda.is_available() 
                     else "mps" if torch.backends.mps.is_available() 
                     else "cpu")
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
tokenizer = CLIPTokenizer.from_pretrained(model_id)

# Connect to MongoDB
uri = "mongodb+srv://rsadhu:crY4RIKbBbQFXVms@cluster0.eze98.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)
db = client['amazon_fashion']
collection = db['metadata']

def get_image_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs).cpu().numpy()
    return embedding

def get_text_embedding(text: str):
    inputs = tokenizer(text, return_tensors = "pt").to(device)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs).cpu().numpy()
    return embedding

@app.post("/search_by_text/")
async def search_by_text(query: str = Query(..., min_length=1, max_length=500)):
    query_embedding = get_text_embedding(query).tolist()
    results = index.query(vector=query_embedding, top_k=5)
    response = []
    for match in results["matches"]:
        metadata = collection.find_one({"_id": int(match['id'])})
        response.append(metadata)
    return JSONResponse(content=response)

@app.post("/search_by_image/")
async def search_by_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    query_embedding = get_image_embedding(image).tolist()
    results = index.query(vector=query_embedding, top_k=5)
    response = []
    for match in results["matches"]:
        metadata = collection.find_one({"_id": int(match['id'])})
        response.append(metadata)
    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)