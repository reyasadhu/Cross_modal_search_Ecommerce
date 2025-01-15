from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pymongo import MongoClient
from pinecone import Pinecone
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import io
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, media_type="text/html")

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    logger.error("PINECONE_API_KEY environment variable not set")
    raise ValueError("PINECONE_API_KEY environment variable not set")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "clip-multimodal-search"
index = pc.Index(name=index_name)

# Load CLIP model and processor
model_id = "openai/clip-vit-base-patch32"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

try:
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Error loading model")

# Connect to MongoDB
mongo_uri = os.getenv("MONGODB_URI")
if not mongo_uri:
    logger.error("MONGODB_URI environment variable not set")
    raise ValueError("MONGODB_URI environment variable not set")

client = MongoClient(mongo_uri)
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

@app.get("/search-by-text/")
async def search_by_text(query: str , page: int = 0):
    try:
        query_embedding = get_text_embedding(query)
        results = index.query(
            vector=query_embedding.tolist(), 
            top_k=10 * (page + 1)  # Get more results based on page
        )
        response = []
        # Get only the new batch of results
        start_idx = page * 10
        end_idx = start_idx + 10
        matches = results["matches"][start_idx:end_idx]
        
        for match in matches:
            metadata = collection.find_one({"_id": int(match['id'])})
            if metadata:
                response.append(metadata)
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error in search_by_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-by-image/")
async def search_by_image(file: UploadFile = File(...), page: int = 0):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        query_embedding = get_image_embedding(image).tolist()
        results = index.query(vector=query_embedding, top_k=5 * (page + 1))
        response = []
        start_idx = page * 5
        end_idx = start_idx + 5
        matches = results["matches"][start_idx:end_idx]

        for match in matches:
            metadata = collection.find_one({"_id": int(match['id'])})
            if metadata:
                response.append(metadata)
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error in search_by_image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.get("/product/{product_id}")
async def get_product_details(product_id: int):
    try:
        # Fetch product from MongoDB
        
        product = collection.find_one({"_id": product_id})
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Format response
        response = {
            "title": product.get("title", ""),
            "images": product.get("images", []),  # Array of image URLs
            "features": product.get("features", []),
            "details": product.get("details", {}),
            "store": product.get("store", ""),
            "description": product.get("description", "")
        }
        
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error fetching product details: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
