import weaviate
import json
from weaviate.classes.config import Configure
from weaviate.classes.generate import GenerativeConfig


# Step 1.1: Connect to your local Weaviate instance
def create_collection():
    with weaviate.connect_to_local() as client:
        # Step 1.2: Create a collection
        movies = client.collections.create(
            name="Movie",
            vector_config=Configure.Vectors.text2vec_ollama(  # Configure the Ollama embedding integration
                api_endpoint="http://host.docker.internal:11434",  # If using Docker you might need: http://host.docker.internal:11434
                model="nomic-embed-text",  # The model to use
            ),
        )

def import_data():
    with weaviate.connect_to_local() as client:
        # Step 1.3: Import three objects
        data_objects = [
            {"title": "The Matrix", "description": "A computer hacker learns about the true nature of reality and his role in the war against its controllers.", "genre": "Science Fiction"},
            {"title": "Spirited Away", "description": "A young girl becomes trapped in a mysterious world of spirits and must find a way to save her parents and return home.", "genre": "Animation"},
            {"title": "The Lord of the Rings: The Fellowship of the Ring", "description": "A meek Hobbit and his companions set out on a perilous journey to destroy a powerful ring and save Middle-earth.", "genre": "Fantasy"},
        ]

        movies = client.collections.use("Movie")
        with movies.batch.fixed_size(batch_size=200) as batch:
            for obj in data_objects:
                batch.add_object(properties=obj)

        print(f"Imported & vectorized {len(movies)} objects into the Movie collection")

def semantic_search():
    # Step 2.1: Connect to your local Weaviate instance
    with weaviate.connect_to_local() as client:

        # Step 2.2: Use this collection
        movies = client.collections.use("Movie")

        # Step 2.3: Perform a semantic search with NearText
        response = movies.query.near_text(
            query="sci-fi",
            limit=2
        )

        for obj in response.objects:
            print(json.dumps(obj.properties, indent=2))  # Inspect the results


def rag():
    # Step 2.1: Connect to your local Weaviate instance
    with weaviate.connect_to_local() as client:

        # Step 2.2: Use this collection
        movies = client.collections.use("Movie")

        # Step 2.3: Perform RAG with on NearText results
        response = movies.generate.near_text(
            query="sci-fi",
            limit=1,
            grouped_task="Write a tweet with emojis about this movie.",
            generative_provider=GenerativeConfig.ollama(  # Configure the Ollama generative integration
                api_endpoint="http://host.docker.internal:11434",  # If using Docker you might need: http://host.docker.internal:11434
                model="llama3.2",  # The model to use
            ),
        )

        print(response.generative.text)  # Inspect the results


def main():
    # create_collection()
    # import_data()
    semantic_search()
    rag()

main()