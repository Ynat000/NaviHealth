import weaviate
import weaviate.classes as wvc


def debug():
    with weaviate.connect_to_local() as client:

        collection = client.collections.get("DocumentChunk")

        # 2. Check collection configuration
        print("\n[2] Checking collection configuration...")
        config = collection.config.get()
        print(f"    Vectorizer: {config.vectorizer}")
        print(f"    Vector index type: {config.vector_index_type}")
        
        if config.vectorizer == "none" or config.vectorizer is None:
            print("    ✗ ISSUE FOUND: No vectorizer configured!")
            print("    This means objects don't have vectors, so near_text won't work.")
            print("\n    SOLUTION: You need to either:")
            print("      a) Recreate the collection with a vectorizer (e.g., text2vec-openai)")
            print("      b) Use custom vectors when inserting objects")
            print("      c) Use .fetch_objects() instead of .near_text() for non-vector queries")
        else:
            print(f"    ✓ Vectorizer is configured: {config.vectorizer}")

debug()