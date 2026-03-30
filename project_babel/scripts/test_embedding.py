from sentence_transformers import SentenceTransformer
import numpy as np

# first run will download the model - will take a few mins as it's around 500 Mb
print("Loading model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded.")

# prepare different versions of the same sentence
leg_pain_en = "My leg hurts when I walk"
leg_pain_zh = "我走路的时候腿疼"
leg_pain_fr = "J'ai mal à la jambe quand je marche"

# another group: fever
fever_en = "I have had a high fever for three days"
fever_zh = "我发高烧三天了"

# totally irrelevant material
unrelated = "The weather in Vancouver is rainy today"

print("Computing embeddings...")
texts = [leg_pain_en, leg_pain_zh, leg_pain_fr, fever_en, fever_zh, unrelated]
embeddings = model.encode(texts)

# define similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

labels = ["leg_en", "leg_zh", "leg_fr", "fever_en", "fever_zh", "unrelated"]

print("\n=== Similarity Matrix ===\n")
print("         ", end="")
for label in labels:
    print(f"{label:>10}", end="")
print()

for i, label_i in enumerate(labels):
    print(f"{label_i:<9}", end="")
    for j in range(len(labels)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{sim:>10.3f}", end="")
    print()

print("\n=== Key Validations ===\n")

sim_leg_en_zh = cosine_similarity(embeddings[0], embeddings[1])
sim_leg_en_fr = cosine_similarity(embeddings[0], embeddings[2])
sim_leg_fever = cosine_similarity(embeddings[0], embeddings[3])
sim_leg_unrelated = cosine_similarity(embeddings[0], embeddings[5])

print(f"English leg pain vs Chinese leg pain: {sim_leg_en_zh:.3f}")
print(f"  Expected: > 0.8, Actual: {'PASS' if sim_leg_en_zh > 0.8 else 'FAIL'}")

print(f"\nEnglish leg pain vs French leg pain: {sim_leg_en_fr:.3f}")
print(f"  Expected: > 0.8, Actual: {'PASS' if sim_leg_en_fr > 0.8 else 'FAIL'}")

print(f"\nEnglish leg pain vs English fever: {sim_leg_fever:.3f}")
print(f"  Expected: < 0.5, Actual: {'PASS' if sim_leg_fever < 0.5 else 'FAIL'}")

print(f"\nEnglish leg pain vs Unrelated: {sim_leg_unrelated:.3f}")
print(f"  Expected: < 0.3, Actual: {'PASS' if sim_leg_unrelated < 0.3 else 'FAIL'}")
