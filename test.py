import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from bcsr_coreset import BCSR_Coreset
# ----- 1. Create dummy data -----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.manual_seed(42)
X_train = torch.tensor([
    [0.12, 0.83, 0.45, 0.67, 0.29],
    [0.91, 0.15, 0.48, 0.22, 0.73],
    [0.37, 0.59, 0.11, 0.84, 0.33],
    [0.64, 0.25, 0.92, 0.53, 0.47],
    [0.18, 0.77, 0.31, 0.68, 0.56],
    [0.89, 0.42, 0.74, 0.35, 0.10],
    [0.26, 0.93, 0.51, 0.79, 0.62],
    [0.71, 0.12, 0.65, 0.47, 0.81],
    [0.55, 0.33, 0.98, 0.24, 0.17],
    [0.43, 0.69, 0.28, 0.54, 0.76],
    [0.94, 0.61, 0.15, 0.83, 0.32],
    [0.40, 0.20, 0.59, 0.73, 0.99],
    [0.35, 0.48, 0.66, 0.57, 0.25],
    [0.84, 0.05, 0.91, 0.38, 0.72],
    [0.60, 0.26, 0.47, 0.88, 0.41],
    [0.30, 0.81, 0.70, 0.19, 0.53],
    [0.16, 0.95, 0.63, 0.77, 0.58],
    [0.78, 0.14, 0.32, 0.94, 0.21],
    [0.46, 0.88, 0.08, 0.63, 0.34],
    [0.67, 0.36, 0.56, 0.09, 0.97],
    [0.22, 0.68, 0.43, 0.74, 0.80],
    [0.58, 0.09, 0.89, 0.49, 0.63],
    [0.33, 0.55, 0.19, 0.91, 0.46],
    [0.99, 0.74, 0.28, 0.41, 0.12],
    [0.44, 0.31, 0.95, 0.16, 0.53],
    [0.51, 0.89, 0.22, 0.68, 0.75],
    [0.85, 0.45, 0.62, 0.57, 0.20],
    [0.73, 0.27, 0.49, 0.82, 0.14],
    [0.10, 0.96, 0.35, 0.78, 0.39],
    [0.92, 0.63, 0.44, 0.30, 0.70],
    [0.28, 0.41, 0.97, 0.64, 0.55],
    [0.39, 0.19, 0.73, 0.46, 0.68],
    [0.83, 0.67, 0.24, 0.59, 0.93],
    [0.48, 0.29, 0.88, 0.35, 0.61],
    [0.56, 0.72, 0.52, 0.09, 0.44],
    [0.65, 0.11, 0.79, 0.25, 0.36],
    [0.32, 0.84, 0.67, 0.54, 0.28],
    [0.17, 0.91, 0.42, 0.71, 0.60],
    [0.70, 0.34, 0.55, 0.12, 0.82],
    [0.42, 0.75, 0.64, 0.39, 0.10],
    [0.27, 0.53, 0.81, 0.20, 0.99],
    [0.62, 0.30, 0.25, 0.89, 0.51],
    [0.36, 0.47, 0.13, 0.95, 0.85],
    [0.50, 0.64, 0.77, 0.18, 0.24],
    [0.79, 0.08, 0.60, 0.48, 0.71],
    [0.93, 0.23, 0.38, 0.81, 0.16],
    [0.14, 0.56, 0.91, 0.26, 0.87],
    [0.59, 0.37, 0.46, 0.73, 0.33],
    [0.87, 0.70, 0.19, 0.58, 0.40],
    [0.31, 0.99, 0.82, 0.04, 0.62],
], dtype=torch.float32)
print(X_train.shape)
y_train = (X_train.sum(dim=1) > 0).long()
X_test = torch.tensor([
    [12, 87, 34, 56, 23],
    [91, 15, 48, 22, 73],
    [37, 59, 11, 84, 33],
    [64, 25, 92, 53, 47],
    [18, 77, 31, 68, 56]
], dtype=torch.float32)
y_test = (X_test.sum(dim=1) > 0).long()

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)

# ----- 2. Simple model -----
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2)
    def forward(self, x):
        return self.fc(x)

model = TinyNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ----- 3. Train briefly -----
for epoch in range(10):
    gradslist=[]
    for xb, yb in train_loader:
        ids=torch.randperm(len(xb))
        if(epoch==9):
            proxy_model = TinyNet().to(device)
            bc = BCSR_Coreset(proxy_model, 10, 10, out_dim=100,
                             max_outer_it=5, max_inner_it=5,
                             weight_lr=10, candidate_batch_size=600, logging_period=1000)
            ids,loss=bc.coreset_select(model, xb[ids], yb[ids], task_id=0,  topk=5,out_loss=[])
        optimizer.zero_grad()
        out = model(xb[ids])
        loss = criterion(out, yb[ids])
        loss.backward()
        optimizer.step()
        
        if(epoch == 9):
            gradsin=[]
            for p in model.parameters():
                gradsin.append(p.grad.detach().flatten())
            gradscat=torch.cat(gradsin)
            gradslist.append(gradscat)
            


print("Selected coreset indices:", ids)
# ----- 4. Compute gradients per training example -----
def get_gradients(x, y):
    """Compute gradient vector for a single example."""
    model.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    grads = []
    for p in model.parameters():
        grads.append(p.grad.detach().flatten())
    return torch.cat(grads)

train_grads = [get_gradients(X_train[i:i+1], y_train[i:i+1]) for i in range(len(X_train))]

# ----- 5. Compute influence for one test example -----
test_index = 0
test_grad = get_gradients(X_test[test_index:test_index+1], y_test[test_index:test_index+1])

# Influence ≈ negative dot product of test and train gradients
influences = torch.tensor([-(test_grad @ g).item() for g in train_grads])

# ----- 6. Show most influential samples -----
topk = 5
sorted_idx = torch.argsort(influences)
print("Most helpful training examples (lowest influence values):", sorted_idx[:topk].tolist())
print("Most harmful training examples (highest influence values):", sorted_idx[-topk:].tolist())