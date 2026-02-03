import torch

from HighDimSpatial.data.preprocess import preprocess_spatial_data, SpatialPreprocessConfig


def test_preprocess_drops_zero_sum_gene():
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    # second gene sums to 0
    Y = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    genes = ["gene1", "gene2"]

    result = preprocess_spatial_data(X, Y, genes, SpatialPreprocessConfig())
    assert result.Y.shape[1] == 1
    assert result.gene_list == ["gene1"]
    assert result.dropped_genes == ["gene2"]
