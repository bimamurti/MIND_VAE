import os
import glob
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
homefolder='/home/USER/Documents/projects/MIND_VAE/'

def loadallpreviousmemories(folder_path):
        
        alldata=[]
        #pathall=folder
        pathmerge=os.path.join(homefolder,folder_path)
        #pathall=os.path.join(folder_path,"*")
        #pt_folder=glob.glob(pathall)
        #for folder in pt_folder:
        test_files = glob.glob(os.path.join(pathmerge, "bestXY.pt"))
        train_files = glob.glob(os.path.join(pathmerge, "bestloss.pt"))
        bcsr_files = glob.glob(os.path.join(pathmerge, "bestBCSR.pt"))
        rel_files = glob.glob(os.path.join(pathmerge, "bestREL.pt"))
        if(len(test_files)>0):
            test=torch.load(test_files[0], map_location='cuda:0')#,weights_only=False)
            train=torch.load(train_files[0],map_location='cuda:0')
            bcsr=torch.load(bcsr_files[0],map_location='cuda:0')
            if(len(rel_files)>0):
                rel=torch.load(rel_files[0],map_location='cuda:0')
                print("Loaded REL buffer with size:",len(rel))
            alldata.append([test,train,bcsr,rel])
        return alldata 
#folder_path="log_eth/logcontinualserverreal/logdata120b2/"
#folder_path="log_eth/logbilevel90g3_30dec90g3"
#folder_path="log_eth/logbilevel120b2_22dectest230percent"


# Create bar chart for sample indices vs loss values
def plot_sample_loss_distribution(matched_losses, matched_track_ids, title="Sample Loss Distribution"):
    plt.figure(figsize=(14, 6))
    # Create bar chart
    x_positions = np.arange(len(matched_track_ids))
    plt.bar(x_positions, matched_losses, color='steelblue', alpha=0.7)
    plt.xlabel('Track ID (from bcsr[5])', fontsize=12)
    plt.ylabel('Loss Value (from train[3])', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(x_positions, matched_track_ids, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    # Display the plot
    plt.show()
    print(f"Matched {len(matched_losses)} out of {len(bcsr_track_ids)} BCSR samples")

def getmatchedbytrackid(loss_values, bcsr_track_ids, train_track_ids):
    """
    Plot a bar chart of loss values for samples matched by track ID
    
    Args:
        loss_values: loss data from train[3]
        bcsr_track_ids: track IDs from bcsr[5]
        train_track_ids: track IDs from train[5]
        title: title of the plot
    """
    # Convert to numpy if they are tensors
    if torch.is_tensor(loss_values):
        loss_values = loss_values.detach().cpu().numpy()
    if torch.is_tensor(bcsr_track_ids):
        bcsr_track_ids = bcsr_track_ids.detach().cpu().numpy()
    if torch.is_tensor(train_track_ids):
        train_track_ids = train_track_ids.detach().cpu().numpy()
    
    # Flatten if needed
    loss_values = np.atleast_1d(loss_values).flatten()
    bcsr_track_ids = np.atleast_1d(bcsr_track_ids).flatten()
    train_track_ids = np.atleast_1d(train_track_ids).flatten()
    
    # Match track IDs: for each bcsr track ID, find corresponding loss value from train
    matched_losses = []
    matched_track_ids = []
    
    for bcsr_id in bcsr_track_ids:
        # Find where this track ID appears in train_track_ids
        matches = np.where(train_track_ids == bcsr_id)[0]
        if len(matches) > 0:
            # Use the first match and get its loss value
            matched_losses.append(loss_values[matches[0]])
            matched_track_ids.append(bcsr_id)
    
    matched_losses = np.array(matched_losses)
    matched_track_ids = np.array(matched_track_ids)

    # Sort by loss value (descending)
    sort_idx = np.argsort(-matched_losses)
    matched_losses = matched_losses[sort_idx]
    matched_track_ids = matched_track_ids[sort_idx]
    
    # Create figure and axis
    
    return matched_losses, matched_track_ids


def plot_kmeans_loss_clusters(matched_losses, n_clusters=5, title="KMeans Clusters of BCSR Losses"):
    """
    Cluster loss values using KMeans and show count per cluster, ordered by cluster center.
    """
    if torch.is_tensor(matched_losses):
        matched_losses = matched_losses.detach().cpu().numpy()
    matched_losses = np.atleast_1d(matched_losses).flatten()

    # Reshape for KMeans (expects 2D)
    X = matched_losses.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(X)

    # Cluster centers and counts
    centers = kmeans.cluster_centers_.flatten()
    counts = np.array([(labels == i).sum() for i in range(n_clusters)])

    # Order clusters by center (descending: highest loss first)
    order = np.argsort(-centers)
    centers_ordered = centers[order]
    counts_ordered = counts[order]

    # Plot
    plt.figure(figsize=(12, 5))
    plt.bar(range(n_clusters), counts_ordered, color='mediumseagreen', alpha=0.8)
    for idx, val in enumerate(counts_ordered):
        plt.text(idx, val + counts_ordered.max()*0.01 if counts_ordered.max() > 0 else 0.01, str(val), ha='center', va='bottom', fontsize=10)
    plt.xticks(range(n_clusters), [f"C{idx}: {centers_ordered[idx]:.4f}" for idx in range(n_clusters)], rotation=45, ha='right')
    plt.xlabel('Loss segmentation', fontsize=12)
    plt.ylabel('Count of Tracks Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Return raw labels and centers (not ordered) for reuse
    return kmeans,counts_ordered


def plot_bcsr_from_previous_kmeans(matched_losses,kmeans,n_clusters,count, title="KMeans Clusters of BCSR Losses (Reused)", descending=True):
    """
    Cluster loss values using KMeans and show count per cluster, ordered by cluster center.
    """
    if torch.is_tensor(matched_losses):
        matched_losses = matched_losses.detach().cpu().numpy()
    matched_losses = np.atleast_1d(matched_losses).flatten()

    # Reshape for KMeans (expects 2D)
    X = matched_losses.reshape(-1, 1)
    #kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    labels = kmeans.predict(X)

    # Cluster centers and counts
    centers = kmeans.cluster_centers_.flatten()
    counts = np.array([(labels == i).sum() for i in range(n_clusters)])

    # Order clusters by center (descending: highest loss first)
    order = np.argsort(-centers)
    centers_ordered = centers[order]
    counts_ordered = counts[order]
    #counts_ordered=(count-counts_ordered)/count
    # Plot
    plt.figure(figsize=(12, 5))
    plt.bar(range(n_clusters), counts_ordered, color='brown', alpha=0.8)
    for idx, val in enumerate(counts_ordered):
        plt.text(idx, val + counts_ordered.max()*0.01 if counts_ordered.max() > 0 else 0.01, str(val), ha='center', va='bottom', fontsize=10)
    plt.xticks(range(n_clusters), [f"C{idx}: {centers_ordered[idx]:.4f}" for idx in range(n_clusters)], rotation=45, ha='right')
    plt.xlabel('Loss segmentation', fontsize=12)
    plt.ylabel('Count of BCSR Tracks Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
def plot_bcsr_distribution_difference(matched_losses,kmeans,n_clusters,count, title="-", descending=True):
    """
    Cluster loss values using KMeans and show distribution percentage, ordered by cluster center.
    """
    if torch.is_tensor(matched_losses):
        matched_losses = matched_losses.detach().cpu().numpy()
    matched_losses = np.atleast_1d(matched_losses).flatten()

    # Reshape for KMeans (expects 2D)
    X = matched_losses.reshape(-1, 1)
    #kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    labels = kmeans.predict(X)

    # Cluster centers and counts
    centers = kmeans.cluster_centers_.flatten()
    counts = np.array([(labels == i).sum() for i in range(n_clusters)])

    # Order clusters by center (descending: highest loss first)
    order = np.argsort(-centers)
    centers_ordered = centers[order]
    counts_ordered = counts[order]
    counts_ordered=(counts_ordered)/count
    # Plot
    plt.figure(figsize=(12, 5))
    plt.bar(range(n_clusters), counts_ordered, color='blue', alpha=0.8)
    for idx, val in enumerate(counts_ordered):
        valshow= val + counts_ordered.max()*0.01
        plt.text(idx, valshow if counts_ordered.max() > 0 else 0.01, f"{val:.4f}", ha='center', va='bottom', fontsize=10)
    plt.xticks(range(n_clusters), [f"C{idx}: {centers_ordered[idx]:.4f}" for idx in range(n_clusters)], rotation=45, ha='right')
    plt.xlabel('Loss segmentation', fontsize=12)
    plt.ylabel('BCSR Percentage from loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
# Extract loss values and track IDs

#####main#####
#folder_path="log_eth/logbilevelrel25janv1"
#folder_path="log_eth/logbilevel26janv1data90d7"
folder_path="log_eth/logbilevel26janv1data90g3"
#dataname="120b2"
dataname="26jan90g3"
alldata=loadallpreviousmemories(folder_path)
train=alldata[0][1]
test=alldata[0][0]
bcsr=alldata[0][2]
rel=alldata[0][3]

print("len alldata:",len(alldata))
x=train[3]
loss_values = train[3]           # Loss values from train[3]
bcsr_track_ids = bcsr[5]         # Track IDs from bcsr[5]
train_track_ids = train[5]       # Track IDs from train[5]
rel_track_ids = rel[5]         # Track IDs from rel[5]
# Plot the distribution and capture matched losses/ids
matched_losses, matched_track_ids = getmatchedbytrackid(
    loss_values, bcsr_track_ids, train_track_ids
   
)
matched_lossesrel, matched_track_idsrel = getmatchedbytrackid(
    loss_values, rel_track_ids, train_track_ids
)
# Plot loss clusters histogram (adjust bins if desired)
#plot_loss_clusters(matched_losses, bins=5, title="BCSR Tracks per Loss Cluster")

# Plot KMeans clusters of loss values
ncluster=20
plot_sample_loss_distribution(matched_losses, matched_track_ids,  title="Loss Distribution on pedestrian data"+" for data "+dataname)
kmeans,count=plot_kmeans_loss_clusters(loss_values, n_clusters=ncluster, title="Distribution of Pedestrian Data Across Loss Score Clusters"+" for data "+dataname)
###BCSR PLOT BASED ON PREVIOUS KMEANS###
plot_bcsr_from_previous_kmeans(matched_losses, kmeans=kmeans, count=count, n_clusters=ncluster, title="Distribution of Selected BCSR Pedestrian Data Across Loss Score Clusters"+" for data "+dataname)
plot_bcsr_distribution_difference(matched_losses, kmeans=kmeans, count=count, n_clusters=ncluster, title="BCSR data Distribution Difference compared to Pedestrian Data"+" for data "+dataname)
###REL PLOT BASED ON PREVIOUS KMEANS###
plot_bcsr_from_previous_kmeans(matched_lossesrel, kmeans=kmeans, count=count, n_clusters=ncluster, title="Distribution of Selected REL Pedestrian Data Across Loss Score Clusters"+" for data "+dataname)
plot_bcsr_distribution_difference(matched_lossesrel, kmeans=kmeans, count=count, n_clusters=ncluster, title="REL data Distribution Difference compared to Pedestrian Data"+" for data "+dataname)
#make based on same cluster values for BCSR Losses and then get the percentage of samples in each cluster
