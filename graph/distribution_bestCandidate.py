import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

homefolder = '/home/USER/Documents/projects/MIND_VAE/'

def loadbestCandidate(folder_path):
    """
    Load bestCandidate.pt from a checkpoint folder
    
    Structure of bestCandidate:
    [0]: observations (S x N x 2)
    [1]: future predictions (S x N x 2)
    [2]: neighbors (S x N x N x 2)
    [3]: ADE errors (N,)
    [4]: FDE errors (N,)
    [5]: track IDs (N,)
    [6]: ncluster (N, 2) - has observation count and future count
    [7]: task IDs (N,)
    [8]: BCSR weights (N,)
    [9]: last weights (N,)
    [10]: final computed weights (N,)
    """
    alldata = []
    candidate_files = glob.glob(os.path.join(folder_path, "bestCandidate.pt"))
    
    if len(candidate_files) > 0:
        candidate = torch.load(candidate_files[0], map_location='cpu')
        alldata.append(candidate)
    
    return alldata


def plot_weight_distribution(candidate, title="Final Weight Distribution"):
    """Plot histogram of final computed weights"""
    if not candidate or len(candidate) < 11:
        print("Invalid candidate data")
        return
    
    final_weights = candidate[10]
    if torch.is_tensor(final_weights):
        final_weights = final_weights.cpu().numpy()
    
    final_weights = np.atleast_1d(final_weights).flatten()
    
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.hist(final_weights, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Final Weight', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Weight Statistics:")
    print(f"  Mean: {np.mean(final_weights):.4f}")
    print(f"  Std: {np.std(final_weights):.4f}")
    print(f"  Min: {np.min(final_weights):.4f}")
    print(f"  Max: {np.max(final_weights):.4f}")


def plot_error_distribution(candidate, error_type='ade', title=None):
    """Plot histogram of ADE or FDE errors"""
    if not candidate or len(candidate) < 5:
        print("Invalid candidate data")
        return
    
    idx = 3 if error_type.lower() == 'ade' else 4
    errors = candidate[idx]
    
    if torch.is_tensor(errors):
        errors = errors.cpu().numpy()
    
    errors = np.atleast_1d(errors).flatten()
    
    if title is None:
        title = f"{error_type.upper()} Error Distribution"
    
    plt.figure(figsize=(12, 5))
    plt.hist(errors, bins=30, color='coral', alpha=0.7, edgecolor='black')
    plt.xlabel(f'{error_type.upper()} Error Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"{error_type.upper()} Statistics:")
    print(f"  Mean: {np.mean(errors):.4f}")
    print(f"  Std: {np.std(errors):.4f}")
    print(f"  Min: {np.min(errors):.4f}")
    print(f"  Max: {np.max(errors):.4f}")


def plot_weight_vs_error(candidate, error_type='ade', title=None):
    """Scatter plot of final weights vs errors"""
    if not candidate or len(candidate) < 11:
        print("Invalid candidate data")
        return
    
    idx = 3 if error_type.lower() == 'ade' else 4
    errors = candidate[idx]
    final_weights = candidate[10]
    
    if torch.is_tensor(errors):
        errors = errors.cpu().numpy()
    if torch.is_tensor(final_weights):
        final_weights = final_weights.cpu().numpy()
    
    errors = np.atleast_1d(errors).flatten()
    final_weights = np.atleast_1d(final_weights).flatten()
    
    if title is None:
        title = f"Final Weight vs {error_type.upper()} Error"
    
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(errors, final_weights, alpha=0.6, c=final_weights, cmap='viridis', s=50)
    plt.xlabel(f'{error_type.upper()} Error', fontsize=12)
    plt.ylabel('Final Weight', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Final Weight', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_weight_components(candidate, title="Weight Components Comparison"):
    """Compare the three weight components: BCSR, Last, and Final"""
    if not candidate or len(candidate) < 11:
        print("Invalid candidate data")
        return
    
    bcsr_weights = candidate[8]
    last_weights = candidate[9]
    final_weights = candidate[10]
    
    if torch.is_tensor(bcsr_weights):
        bcsr_weights = bcsr_weights.cpu().numpy()
    if torch.is_tensor(last_weights):
        last_weights = last_weights.cpu().numpy()
    if torch.is_tensor(final_weights):
        final_weights = final_weights.cpu().numpy()
    
    bcsr_weights = np.atleast_1d(bcsr_weights).flatten()
    last_weights = np.atleast_1d(last_weights).flatten()
    final_weights = np.atleast_1d(final_weights).flatten()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # BCSR weights
    axes[0].hist(bcsr_weights, bins=25, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_title('BCSR Weights', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Weight Value')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Last weights
    axes[1].hist(last_weights, bins=25, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_title('Last Weights', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Weight Value')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Final weights
    axes[2].hist(final_weights, bins=25, color='green', alpha=0.7, edgecolor='black')
    axes[2].set_title('Final Weights', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Weight Value')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("Weight Components Statistics:")
    print(f"BCSR - Mean: {np.mean(bcsr_weights):.4f}, Std: {np.std(bcsr_weights):.4f}")
    print(f"Last - Mean: {np.mean(last_weights):.4f}, Std: {np.std(last_weights):.4f}")
    print(f"Final - Mean: {np.mean(final_weights):.4f}, Std: {np.std(final_weights):.4f}")


def plot_task_distribution(candidate, title="Distribution by Task ID"):
    """Bar chart showing data count per task"""
    if not candidate or len(candidate) < 8:
        print("Invalid candidate data")
        return
    
    task_ids = candidate[7]
    
    if torch.is_tensor(task_ids):
        task_ids = task_ids.cpu().numpy()
    
    task_ids = np.atleast_1d(task_ids).flatten().astype(int)
    
    unique_tasks = np.unique(task_ids)
    task_counts = [np.sum(task_ids == task_id) for task_id in unique_tasks]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique_tasks, task_counts, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Task ID', fontsize=12)
    plt.ylabel('Sample Count', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(unique_tasks)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, task_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def plot_kmeans_weight_clusters(candidate, n_clusters=5, title="KMeans Clusters of Final Weights"):
    """Cluster weights and show cluster distribution"""
    if not candidate or len(candidate) < 11:
        print("Invalid candidate data")
        return
    
    final_weights = candidate[10]
    
    if torch.is_tensor(final_weights):
        final_weights = final_weights.cpu().numpy()
    
    final_weights = np.atleast_1d(final_weights).flatten().reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(final_weights)
    
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(range(len(final_weights)), final_weights, c=clusters, cmap='viridis', s=50, alpha=0.6)
    
    # Plot cluster centers
    for center in kmeans.cluster_centers_:
        plt.axhline(y=center, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Final Weight', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster ID', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print cluster statistics
    print(f"\nCluster Statistics:")
    for i in range(n_clusters):
        cluster_weights = final_weights[clusters == i]
        print(f"  Cluster {i}: {len(cluster_weights)} samples, "
              f"Mean: {np.mean(cluster_weights):.4f}, "
              f"Std: {np.std(cluster_weights):.4f}")


def plot_cluster_count_distribution(candidate, title="Cluster Count Distribution"):
    """Plot distribution of ncluster values (observation and future counts)"""
    if not candidate or len(candidate) < 7:
        print("Invalid candidate data")
        return
    
    ncluster = candidate[6]
    
    if torch.is_tensor(ncluster):
        ncluster = ncluster.cpu().numpy()
    
    ncluster = np.atleast_1d(ncluster)
    
    if ncluster.ndim == 1:
        obs_counts = ncluster
        future_counts = ncluster
    else:
        obs_counts = ncluster[:, 0] if ncluster.shape[1] > 0 else ncluster[:, 0]
        future_counts = ncluster[:, 1] if ncluster.shape[1] > 1 else ncluster[:, 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(obs_counts, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_title('Observation Cluster Count', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Cluster Count')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].hist(future_counts, bins=20, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_title('Future Cluster Count', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Cluster Count')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    folder_path = os.path.join(homefolder, "log_eth/logcontinualproposed_20260428_155403/logduri_morning_cluster")
    
    print(f"Loading bestCandidate from: {folder_path}")
    candidates = loadbestCandidate(folder_path)
    
    if len(candidates) > 0:
        candidate = candidates[0]
        print(f"Loaded bestCandidate with {len(candidate[0])} samples")
        
        # Generate all plots
        plot_weight_distribution(candidate)
        plot_error_distribution(candidate, 'ade')
        plot_error_distribution(candidate, 'fde')
        plot_weight_vs_error(candidate, 'ade')
        plot_weight_vs_error(candidate, 'fde')
        plot_weight_components(candidate)
        plot_task_distribution(candidate)
        plot_kmeans_weight_clusters(candidate, n_clusters=5)
        plot_cluster_count_distribution(candidate)
    else:
        print("No bestCandidate found in the folder")
