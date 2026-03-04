import weaviate
import weaviate.classes as wvc


def create_collection():
    with weaviate.connect_to_local() as client:

        client.collections.create(
            name="DocumentChunk",
            vector_config=wvc.config.Configure.Vectors.text2vec_ollama(
                name="content",
                source_properties=["chunk"],  # Only vectorize this field
                model="nomic-embed-text",
                api_endpoint="http://ollama:11434",
            ),
            properties=[
                wvc.config.Property(name="chunk", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(
                    name="chunk_file", data_type=wvc.config.DataType.TEXT
                ),
            ],
        )


create_collection()
