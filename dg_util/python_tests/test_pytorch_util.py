import pytest
import numpy as np
import torch
from dg_util.python_utils import pytorch_util as pt_util


def test_remove_dim_np():
    shape = [2, 3, 4, 5]
    arr = np.zeros(shape)
    new_arr = pt_util.remove_dim(arr, 1)
    assert new_arr.shape == (6, 4, 5)

    new_arr = pt_util.remove_dim(arr, 2)
    assert new_arr.shape == (2, 12, 5)

    new_arr = pt_util.remove_dim(arr, 3)
    assert new_arr.shape == (2, 3, 20)


def test_remove_dim_np_multi_dim():
    shape = [2, 3, 4, 5]
    arr = np.zeros(shape)
    new_arr = pt_util.remove_dim(arr, (1, 2))
    assert new_arr.shape == (24, 5)

    new_arr = pt_util.remove_dim(arr, (1, 3))
    assert new_arr.shape == (6, 20)

    new_arr = pt_util.remove_dim(arr, (3, 1))
    assert new_arr.shape == (6, 20)

    new_arr = pt_util.remove_dim(arr, (2, 3))
    assert new_arr.shape == (2, 60)


def test_remove_dim_np_neg_dim():
    shape = [2, 3, 4, 5]
    arr = np.zeros(shape)
    new_arr = pt_util.remove_dim(arr, -1)
    assert new_arr.shape == (2, 3, 20)

    new_arr = pt_util.remove_dim(arr, -2)
    assert new_arr.shape == (2, 12, 5)

    new_arr = pt_util.remove_dim(arr, (-2, -1))
    assert new_arr.shape == (2, 60)

    new_arr = pt_util.remove_dim(arr, (-1, -2))
    assert new_arr.shape == (2, 60)

    new_arr = pt_util.remove_dim(arr, (1, -1))
    assert new_arr.shape == (6, 20)

    new_arr = pt_util.remove_dim(arr, (-1, 1))
    assert new_arr.shape == (6, 20)


def test_split_dim():
    shape = [4, 6, 8, 10]
    arr = np.zeros(shape)
    new_arr = pt_util.split_dim(arr, 0, 2, 2)
    assert new_arr.shape == (2, 2, 6, 8, 10)

    new_arr = pt_util.split_dim(arr, 1, 2, 3)
    assert new_arr.shape == (4, 2, 3, 8, 10)

    new_arr = pt_util.split_dim(arr, 1, 3, 2)
    assert new_arr.shape == (4, 3, 2, 8, 10)

    new_arr = pt_util.split_dim(arr, 2, 2, -1)
    assert new_arr.shape == (4, 6, 2, 4, 10)

    new_arr = pt_util.split_dim(arr, 2, -1, 2)
    assert new_arr.shape == (4, 6, 4, 2, 10)

    new_arr = pt_util.split_dim(arr, -1, 2, 5)
    assert new_arr.shape == (4, 6, 8, 2, 5)

    new_arr = pt_util.split_dim(arr, -1, 5, -1)
    assert new_arr.shape == (4, 6, 8, 5, 2)

    new_arr = pt_util.split_dim(arr, -2, 2, -1)
    assert new_arr.shape == (4, 6, 2, 4, 10)


def test_fix_broadcast():
    shape1 = [4, 6, 8, 10]
    arr1 = np.zeros(shape1)
    shape2 = [4, 6]
    arr2 = np.zeros(shape2)
    new_arr1, new_arr2 = pt_util.fix_broadcast(arr1, arr2)
    assert new_arr1.shape == (4, 6, 8, 10)
    assert new_arr2.shape == (4, 6, 1, 1)

    new_arr2, new_arr1 = pt_util.fix_broadcast(arr2, arr1)
    assert new_arr1.shape == (4, 6, 8, 10)
    assert new_arr2.shape == (4, 6, 1, 1)

    shape1 = [4, 6, 8, 10]
    arr1 = np.zeros(shape1)
    shape2 = [8, 10]
    arr2 = np.zeros(shape2)
    new_arr1, new_arr2 = pt_util.fix_broadcast(arr1, arr2)
    assert new_arr1.shape == (4, 6, 8, 10)
    assert new_arr2.shape == (1, 1, 8, 10)

    shape1 = [4, 6, 8, 10]
    arr1 = np.zeros(shape1)
    shape2 = [4, 10]
    arr2 = np.zeros(shape2)
    new_arr1, new_arr2 = pt_util.fix_broadcast(arr1, arr2)
    assert new_arr1.shape == (4, 6, 8, 10)

    shape1 = [10, 6, 8, 10]
    arr1 = np.zeros(shape1)
    shape2 = [10]
    arr2 = np.zeros(shape2)
    new_arr1, new_arr2 = pt_util.fix_broadcast(arr1, arr2)
    assert new_arr1.shape == (10, 6, 8, 10)
    assert new_arr2.shape == (1, 1, 1, 10)


def test_lambda_layer():
    data1 = torch.rand(10, 20, 30).requires_grad_(True)
    data2 = data1.clone().detach().requires_grad_(True)
    layer = pt_util.LambdaLayer(lambda x: x.transpose(0, 1))

    output1 = data1.transpose(0, 1)
    output2 = layer(data2)

    assert output1.shape == output2.shape
    assert torch.allclose(output1, output2)

    (2 * output1.mean() + 2 * output2.mean()).backward()
    assert torch.allclose(data1.grad, data2.grad)
    assert data1.grad.abs().sum() > 0
