# import dependencies
import json
import torch
import faiss
from pprint import pprint
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load documents from JSON file
with open('data/data.json', 'r') as file:
    documents = json.load(file)
    
embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

corpus = [d['text'] for d in documents]

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

corpus_embeddings = corpus_embeddings.cpu()

# Create a flat Faiss index
inner_index = faiss.IndexFlatIP(768) # the size of our vector space
index = faiss.IndexIDMap(inner_index)

index.add_with_ids(corpus_embeddings.numpy(), 
                   np.array(range(0, len(corpus))))
    
encoded_query = embedder.encode(["spanish flu casualties"])
print(encoded_query.shape)

print(index.search(encoded_query, 2))
