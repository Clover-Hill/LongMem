import torch

test_tensor = torch.randn(8192,1024,768).type(torch.float16)

# Calculate the size of each element in bytes (float16)
element_size_bytes = test_tensor.element_size()

# Calculate the total number of elements in the tensor
num_elements = test_tensor.numel()

# Calculate the total memory occupied by the tensor in bytes
total_memory_bytes = element_size_bytes * num_elements

# Convert bytes to megabytes (optional)
total_memory_mb = total_memory_bytes / (1024 ** 2)

print(f"Size of each element (in bytes): {element_size_bytes}")
print(f"Number of elements: {num_elements}")
print(f"Total memory occupied (in bytes): {total_memory_bytes}")
print(f"Total memory occupied (in megabytes): {total_memory_mb} MB")

