import sentence_transformers
import transformers

print(sentence_transformers.__version__)
print(transformers.__version__)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print(model.encode(["hello"]))
