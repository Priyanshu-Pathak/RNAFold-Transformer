import torch

from rna3d_folding.model import DistTransformer


def test_disttransformer_forward_shape():
    model = DistTransformer(window=2, d_model=16, nhead=4, dim_feedforward=32, num_layers=1)
    x = torch.randn(3, 5, 8)
    y = model(x)
    assert y.shape == (3, 5, 5)
    assert torch.allclose(y, y.transpose(1, 2), atol=1e-6)
