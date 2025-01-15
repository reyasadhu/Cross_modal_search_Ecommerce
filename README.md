# Cross_Modal_Search_Ecommerce
A fashion image retrival system utilizing Contrastive Language-Image Pre-training (CLIP) models to perform text-to-image and image-to-image searches.

### How to use the Dockerfile
- #### Build the Docker image
  - ```docker build -t fastapi-app .```

- #### Run the Docker container with environment variables from .env file
  - ```docker run -d -p 8000:8000 --env-file .env --name fastapi-container fastapi-app```
