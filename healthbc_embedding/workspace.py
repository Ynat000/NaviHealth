import weaviate
import weaviate.classes as wvc


def debug():
    with weaviate.connect_to_local() as client:
        if client.collections.exists("DocumentChunk"):
            client.collections.delete("DocumentChunk")
            print("✓ Deleted old DocumentChunk collection")
        
debug()