import matplotlib.pyplot as plt
import seaborn as sns

def plot_umap(embeddings_2d, words, alpha, s, save_path = None):
    """
    Plot UMAP embeddings with a color gradient based on the alpha values.
    
    Args:
        - embeddings_2d: 2D numpy array of shape (n_samples, 2) containing UMAP embeddings.
        - words: List of words corresponding to the embeddings.
        - alpha: 2D numpy array of shape (n_samples, 2) containing UMAP embeddings.
        - s: 1D numpy array of shape (n_samples,) containing the alpha values.
        - save_path: Optional; if provided, the plot will be saved to this path.
    
    Returns:
        - None
    """
    plt.figure(figsize = (12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s = s, alpha = alpha)

    for i in range(0, 50): 
        plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], words[i], fontsize = 10, weight = 'bold')

    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight') if save_path else plt.show()
    plt.close()

def plot_similarity_hist(similarities, bins, kde = False, save_path = None):
    """
    Plot a histogram of similarity scores.
    
    Args:
        - similarities: 1D numpy array of similarity scores.
        - bins: Number of bins for the histogram.
        - kde: Boolean indicating whether to plot a kernel density estimate (KDE) overlay.
        - save_path: Optional; if provided, the plot will be saved to this path.
    
    Returns:
        - None
    """
    plt.figure(figsize = (12, 8))
    sns.histplot(similarities, bins = bins, kde = kde) 
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of Similarity Scores')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight') if save_path else plt.show()
    plt.close()

