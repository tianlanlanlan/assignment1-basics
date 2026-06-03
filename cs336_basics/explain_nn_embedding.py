import torch
import numpy as np
import torch.nn as nn
from scipy.sparse import csr_matrix
from einops import rearrange, einsum

images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
# print(f"images = {images.T}")
# matrix = torch.randn(128, 3)

dim_by = torch.linspace(start=0.0, end=1.0, steps=10)


def explain_nn_embedding():
    # Initialize a sparse matrix
    X_train = csr_matrix(np.array([[1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 1, 0]]))

    # Get first row in the training set
    row = X_train.getrow(0)
    print(
        f"X_train: {X_train}\nrow: {row}, row toarray: {row.toarray()}, row.indices: {row.indices}"
    )

    w_linear = nn.Linear(4, 3, bias=False)
    print(f"weight = {w_linear.weight}")

    embedding = nn.Embedding(10, 3)
    input_tensor = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])  # 2 * 4
    linear_output = w_linear(torch.FloatTensor(input_tensor.toarray()))
    output = embedding(input_tensor)
    print(f"Embedding output: {output}\ninput_tensor: {input_tensor}")


explain_nn_embedding()


# 不停的重复 !!!
# 不为找工作换工作为学习，而是为解疑未知而学习，对一些概念/名字/原理祛魅
# 以实际代码为导向，但同时也要多思考
# 当前的工作先继续混着？
