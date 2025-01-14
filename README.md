# Text-and-visual-search-amazon

### How to use the Dockerfile
- #### Build the Docker image
  - ```docker build -t fastapi-app .```

- #### Run the Docker container with environment variables from .env file
  - ```docker run -d -p 8000:8000 --env-file .env --name fastapi-container fastapi-app```
