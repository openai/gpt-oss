# Embeddings and FAISS

Embeddings are vector representations of text. FAISS is a fast library for similarity search and clustering of dense vectors. To use FAISS:

1. Generate embeddings for your text chunks using a model like sentence-transformers/all-MiniLM-L6-v2.
2. Build a FAISS index from these vectors.
3. Retrieve top-k similar chunks for a query using cosine similarity.