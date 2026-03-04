# Vector Database POC Project 

## Vector Database - Weaviate
1. Start a Weaviate instance and the Ollama server inside Docker Containers - `docker-compose up -d`
2. Pull Embedding Model `docker compose exec ollama ollama pull nomic-embed-text`
3. Pull Generative Model `docker compose exec ollama ollama pull llama3.2`

## Start Development 
1. Start Python Development in venv - `pipenv install`   
2. Create a Collection & Import Data
