# -----------------------------------------------------------------------------
# Torch
# -----------------------------------------------------------------------------
print(f"Importing torch ...")
import torch

print(f"Init tensors ...")
cpu_tensors = torch.zeros(5, 5)
print(
    f"{cpu_tensors}, on device: {cpu_tensors.device}, stride[0] = {cpu_tensors.stride(0)}, stride[1] = {cpu_tensors.stride(1)}"
)
gpu_tensors = cpu_tensors.to("cuda:0")
print(f"{gpu_tensors}, on device: {gpu_tensors.device}")
row, column = 1, 2
index = row * gpu_tensors.stride(0) + column * gpu_tensors.stride(1)
print(f"index = {index}")
row0 = gpu_tensors[0]
column1 = gpu_tensors[:, 1]
print(f"row0: {row0}, column1: {column1}")

ones_tensors = torch.ones(2, 3)
transpose_ones_tensors = ones_tensors.transpose(1, 0)
print(f"transpose_ones_tensors is_contiguous: {transpose_ones_tensors.is_contiguous()}")


num_gpus = torch.cuda.device_count()
print(f"num_gpus = {num_gpus}")
for i in range(num_gpus):
    properties = torch.cuda.get_device_properties(i)
    print(f"GPU device {i} properties: {properties}")


def tensor_matmul():
    x = torch.ones(16, 32)
    y = torch.ones(32, 2)
    mul_result = x @ y
    assert mul_result.size() == torch.Size([16, 2])

    x = torch.ones(2, 3, 16, 32)
    y = torch.ones(32, 2)
    mul_result = x @ y
    assert mul_result.size() == torch.Size([2, 3, 16, 2])


tensor_matmul()


def einops_einsum():
    from torch import einsum

    x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)
    y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)
    # z = einsum("batch seqA hidden, batch seqB hidden -> batch seqA seqB ", x, y)
    z = einsum("b s h, b t h -> b s t", x, y)
    print(f"z: {z}")


einops_einsum()


def gradients_basics():
    # backward pass: compute gradient
    x = torch.tensor([1.0, 2, 3])
    w = torch.tensor([1.0, 1, 1], requires_grad=True)

    # Forward pass
    # w.pow(2).sum().backward()
    # print(f"x = {x}, w = {w}, {x.pow(2)} {w.grad}")
    pred_y = x @ w
    loss = 0.5 * (pred_y - 5).pow(2)
    print(f"pred_y = {pred_y}, loss = {loss}")

    # Backward pass
    loss.backward()
    assert loss.grad is None
    assert pred_y.grad is None
    assert x.grad is None
    assert torch.equal(w.grad, torch.tensor([1, 2, 3]))


gradients_basics()
