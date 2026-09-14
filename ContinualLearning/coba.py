import torch

# Define a tensor with 4 columns
tensor_121 = torch.tensor([
    [[1.0, 2.0, 9.0, 10.0], [3.0, 4.0, 11.0, 12.0], [5.0, 6.0, 13.0, 14.0]],  # First batch (3x4)
    [[7.0, 8.0, 15.0, 16.0], [9.0, 10.0, 17.0, 18.0], [11.0, 12.0, 19.0, 20.0]] # Second batch (3x4)
])  # Shape: [2, 3, 4]

tensor_2 = torch.tensor([
    [1.0, 1.0],  # First batch (1x2)
    [2.0, 2.0]   # Second batch (1x2)
])  # Shape: [2, 2]

# Select the first two columns for filtering condition
tensor_121_selected = tensor_121[:, :, :2]  # Shape: [2, 3, 2]

# Subtract using broadcasting
result = tensor_121_selected - tensor_2[:, None, :]

# Sum across the last dimension (columns)
row_sums = result.sum(dim=2)  # Shape: [2, 3]

# Create a boolean mask for values less than 10
mask = row_sums < 10  # Shape: [2, 3] (True/False values)

# Use the mask to filter **entire rows** from the original `tensor_121`
filtered_tensor_121 = tensor_121[mask]  # Shape: [N, 4] where N is the number of selected rows

# Print results
print("\nOriginal tensor_121:")
print(tensor_121)

print("\nRow-wise sums:")
print(row_sums)

print("\nBoolean mask (rows where sum < 10 based on first two columns):")
print(mask)

print("\nFiltered tensor_121 (full rows where first two-column sum < 10):")
print(filtered_tensor_121)