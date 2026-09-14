import numpy as np
import torch

# Load bcsr artifact (modify path as needed)
bcsr_path = "/home/USER/Documents/projects/MIND_VAE/log_eth/testoffrelmorning/bestloss.pt"
bcsr_obj = torch.load(bcsr_path, map_location="cpu", weights_only=False)

# Extract IDs (assuming index 5 as in violin script)
bcsr_ids = bcsr_obj[5]

# Convert to numpy array
if torch.is_tensor(bcsr_ids):
    bcsr_ids = bcsr_ids.detach().cpu().numpy()
bcsr_ids = np.atleast_1d(bcsr_ids).astype(np.int64).flatten()

# Check for duplicates
unique_ids = np.unique(bcsr_ids)
print(f"BCSR total ID entries: {len(bcsr_ids)}")
print(f"BCSR unique IDs: {len(unique_ids)}")

if len(bcsr_ids) == len(unique_ids):
    print("✓ No duplicates found - all IDs are unique")
else:
    print(f"✗ Found duplicates: {len(bcsr_ids) - len(unique_ids)} extra entries")
    
    # Find and count duplicates
    ncnt = np.bincount(bcsr_ids)
    dup_mask = ncnt > 1
    dup_ids = np.where(dup_mask)[0]
    
    print(f"\nDuplicate IDs (first 20):")
    for uid in dup_ids[:20]:
        count = ncnt[uid]
        print(f"  ID {uid}: appears {count} times")
