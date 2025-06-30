import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def plot_tsne(vectorstore, doc_types=None, n_clusters=5):
    # Get all vectors and corresponding documents from Chroma
    result = vectorstore._collection.get(include=['embeddings', 'documents'])
    vectors = np.array(result['embeddings'])
    documents = result['documents']

    # Cluster using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(vectors)

    # Assign a color to each cluster
    color_palette = plt.cm.get_cmap('tab10', n_clusters)  # max 10 visually distinct
    colors = []
    for i in cluster_labels:
        rgb = (np.array(color_palette(i)[:3]) * 255).astype(int)
        r, g, b = map(int, rgb)  # convert to native Python int
        colors.append(f'rgba({r}, {g}, {b}, 0.8)')

    # Reduce to 3D
    n_samples = vectors.shape[0]
    perplexity = min(30, max(5, n_samples // 3))  # Safe bounds
    reduced = TSNE(n_components=3, perplexity=perplexity, random_state=42).fit_transform(vectors)

    # Create plot
    hover_text = [doc[:100] + "..." for doc in documents]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=reduced[:, 0],
            y=reduced[:, 1],
            z=reduced[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=colors,
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            text=hover_text,
            hoverinfo='text'
        )
    ])

    fig.update_layout(
        title='3D t-SNE Clustering of Document Chunks',
        width=900,
        height=650,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    return fig
