import pickle
import matplotlib.pyplot as plt
# Load embeddings and filenames
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('filenames.pkl', 'rb') as f:
    filenames = pickle.load(f)

import numpy as np
embeddings_array = np.array(embeddings)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_array)

# Plot the 2D projection
plt.figure(figsize=(10, 7))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=5, cmap='Spectral')
plt.title("PCA Visualization of Embeddings")
plt.show()
