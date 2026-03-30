import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading embedding model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded.")

# DATA_DIR = "./sample_data"
DATA_DIR = "../hlbc_data_pipeline/artifacts/hlbc_files_md"  # true data from pipeline

def load_documents_chunked(data_dir):
    chunks = []
    data_path = Path(data_dir)
    
    for file_path in data_path.glob("*.md"):
        content = file_path.read_text(encoding='utf-8')
        
        # decipher YAML header
        metadata = {}
        body = content
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                header = parts[1]
                body = parts[2].strip()
                for line in header.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
        
        # split by double linebreaks
        paragraphs = body.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # if length of current chunk + the new paragraph is <= 500 chars, combine them
            if len(current_chunk) + len(para) < 500:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                # save current chunk & start new
                if current_chunk and len(current_chunk) > 100:
                    chunks.append({
                        'content': current_chunk,
                        'metadata': {
                            **metadata,
                            'filename': file_path.name
                        }
                    })
                current_chunk = para
        
        # save final chunk
        if current_chunk and len(current_chunk) > 100:
            chunks.append({
                'content': current_chunk,
                'metadata': {
                    **metadata,
                    'filename': file_path.name
                }
            })
    
    return chunks

print("\nLoading and chunking documents...")
chunks = load_documents_chunked(DATA_DIR)
print(f"Created {len(chunks)} chunks from documents:")

for i, chunk in enumerate(chunks):
    category = chunk['metadata'].get('category', 'unknown')
    preview = chunk['content'][:80].replace('\n', ' ')
    print(f"  [{i}] {category}: {preview}...")

print("\nCreating chunk embeddings...")
chunk_contents = [c['content'] for c in chunks]
chunk_embeddings = model.encode(chunk_contents)
print(f"Created {len(chunk_embeddings)} embeddings")

def search(query, top_k=3):
    query_embedding = model.encode(query)
    
    similarities = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        sim = np.dot(query_embedding, chunk_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb))
        similarities.append((i, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for i, sim in similarities[:top_k]:
        results.append({
            'chunk': chunks[i],
            'similarity': sim
        })
    
    return results

# 测试
test_queries = [
    ("我走路的时候腿很痛", "upcc"),
    ("我发高烧三天了", "upcc"),
    ("我有点咳嗽和流鼻涕", "walkin"),
    ("我胸口剧烈疼痛呼吸困难", "emergency"),
    ("I have a minor cut that needs stitches", "upcc"),
]

print("\n" + "="*80)
print("CHUNKED RETRIEVAL TEST")
print("="*80)

passed = 0
failed = 0

for query, expected in test_queries:
    results = search(query, top_k=3)
    top_category = results[0]['chunk']['metadata'].get('category', '')
    
    is_pass = expected in top_category.lower()
    status = "✓ PASS" if is_pass else "✗ FAIL"
    
    if is_pass:
        passed += 1
    else:
        failed += 1
    
    print(f"\n🔍 Query: {query}")
    print(f"   Expected: {expected}")
    print(f"   {status}")
    print(f"   Top 3 results:")
    for i, r in enumerate(results):
        cat = r['chunk']['metadata'].get('category', 'unknown')
        sim = r['similarity']
        preview = r['chunk']['content'][:60].replace('\n', ' ')
        marker = "→" if i == 0 else " "
        print(f"   {marker} [{cat}] (sim: {sim:.3f}) {preview}...")

print("\n" + "="*80)
print(f"SUMMARY: {passed} passed, {failed} failed")
print("="*80)
