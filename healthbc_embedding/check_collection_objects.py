import weaviate
import weaviate.classes as wvc


def create_collection():
    with weaviate.connect_to_local() as client:

        collection = client.collections.get("DocumentChunk")


        # Get total count
        response = collection.aggregate.over_all(total_count=True)
        count = response.total_count
        
        print("=" * 60)
        print(f"Collection: DocumentChunk")
        print("=" * 60)
        print(f"Total objects: {count:,}")
        print("=" * 60)
        
        if count == 0:
            print("\n✓ The collection is EMPTY")
        else:
            print(f"\n✗ The collection is NOT empty - it contains {count:,} objects")
            
            # Show a sample object
            print("\nFetching a sample object...")
            try:
                result = collection.query.fetch_objects(limit=1)
                if result.objects:
                    obj = result.objects[0]
                    print(f"\nSample object (UUID: {obj.uuid}):")
                    print("-" * 60)
                    for key, value in obj.properties.items():
                        # Truncate long values
                        str_value = str(value)
                        if len(str_value) > 100:
                            str_value = str_value[:100] + "..."
                        print(f"  {key}: {str_value}")
            except Exception as e:
                print(f"Could not fetch sample: {e}")


create_collection()
