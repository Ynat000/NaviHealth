import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading embedding model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded.")

# DATA_DIR = "./sample_data"
DATA_DIR = "../hlbc_data_pipeline/artifacts/hlbc_files_md"  # true data from pipeline


def load_documents(data_dir):
    """Return list of documents from all md files in data directory"""
    documents = []
    data_path = Path(data_dir)
    
    for file_path in data_path.glob("*.md"):
        content = file_path.read_text(encoding='utf-8')
        
        # YAML header
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                header = parts[1]
                body = parts[2].strip()
                
                # extract metadata
                metadata = {}
                for line in header.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
                
                documents.append({
                    'content': body,
                    'metadata': metadata,
                    'filename': file_path.name
                })
    
    return documents

# load documents
print("\nLoading documents...")
docs = load_documents(DATA_DIR)
print(f"Loaded {len(docs)} documents:")
for doc in docs:
    title = doc['metadata'].get('title', doc['filename'])
    print(f"  - {title}")

# create embedding for all documents
print("\nCreating document embeddings...")
doc_contents = [doc['content'] for doc in docs]
doc_embeddings = model.encode(doc_contents)
print(f"Created {len(doc_embeddings)} embeddings, dimension: {doc_embeddings[0].shape[0]}")

# the query function
def search(query, top_k=3):
    """Query the most relevant documents"""
    query_embedding = model.encode(query)
    
    # compute similarities
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
        similarities.append((i, sim))
    
    # sort
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # return top k
    results = []
    for i, sim in similarities[:top_k]:
        results.append({
            'document': docs[i],
            'similarity': sim
        })
    
    return results

# examples
test_queries = [
    # 中文
    ("我走路的时候腿很痛", "Should match UPCC - sprains and strains"),
    ("我发高烧三天了", "Should match UPCC - high fevers"),
    ("我有点咳嗽和流鼻涕", "Should match Walk-in - cold symptoms"),
    ("我胸口剧烈疼痛呼吸困难", "Should match Emergency - chest pain"),
    ("我不确定应该去哪里看病", "Should match 811 services"),
    
    # Francais
    ("J'ai mal à la jambe quand je marche", "Should match UPCC - leg pain"),
    ("J'ai une forte fièvre", "Should match UPCC - high fever"),
    
    # English (as baseline)
    ("I have a minor cut that needs stitches", "Should match UPCC - minor cuts"),
    ("I have chest pain and difficulty breathing", "Should match Emergency"),
]

print("\n" + "="*80)
print("RETRIEVAL TEST RESULTS")
print("="*80)

for query, expected in test_queries:
    print(f"\n🔍 Query: {query}")
    print(f"   Expected: {expected}")
    print(f"   Results:")
    
    results = search(query, top_k=3)
    for i, result in enumerate(results):
        title = result['document']['metadata'].get('title', result['document']['filename'])
        category = result['document']['metadata'].get('category', 'unknown')
        sim = result['similarity']
        
        # mark most relevant results
        marker = "→" if i == 0 else " "
        print(f"   {marker} [{i+1}] {title} (category: {category}, similarity: {sim:.3f})")

print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

# auto validate cases
validation_cases = [
    ("我走路的时候腿很痛", "upcc"),
    ("我胸口剧烈疼痛呼吸困难", "emergency"),
    ("我有点咳嗽和流鼻涕", "walkin"),
]

all_passed = True
for query, expected_category in validation_cases:
    results = search(query, top_k=1)
    actual_category = results[0]['document']['metadata'].get('category', '')
    passed = expected_category in actual_category.lower()
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: '{query}' → expected '{expected_category}', got '{actual_category}'")
    if not passed:
        all_passed = False

print("\n" + ("All validations passed!" if all_passed else "Some validations failed."))
