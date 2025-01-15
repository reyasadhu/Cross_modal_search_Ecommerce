# Cross_Modal_Search_Ecommerce
A fashion image retrival system utilizing Contrastive Language-Image Pre-training (CLIP) models that enables users to search fashion products through both text descriptions and image uploads.

## Live Demo
Try out the application here: [Fashion Search Demo](https://text-and-visual-search-amazon.onrender.com/)
- Search using text descriptions
- Upload images to find similar products

### How to use the Dockerfile
- #### Build the Docker image
  - ```docker build -t fastapi-app .```

- #### Run the Docker container with environment variables from .env file
  - ```docker run -d -p 8000:8000 --env-file .env --name fastapi-container fastapi-app```

## Features

- **Text-based Search**: Search products using natural language descriptions
- **Image-based Search**: Upload images to find visually similar products
- **Product Details**: 
  - High-resolution product images with carousel
  - Product features and specifications
  - Store information
  - Detailed product descriptions
- **Responsive UI**: Amazon-inspired user interface with grid layout

## Tech Stack

- **Backend**:
  - FastAPI - Web framework
  - MongoDB - Product metadata storage
  - Pinecone - Vector database for similarity search
  - CLIP Model - Image and text embeddings
  - PyTorch - Deep learning framework

- **Frontend**:
  - Vanilla JavaScript
  - HTML5/CSS3
  - Responsive grid layout
  - Image carousel

## Performance Benchmarks
- Search: ~1s for 200K products




