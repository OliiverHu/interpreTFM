"""True-positive counts must not wrap: one shard can hold far more than 127 active rows per (feature, concept)."""
import numpy as np
from scipy import sparse

from interp_pipeline.get_annotation.f1_alignment import accumulate_tp_sparse_into_dense


def test_tp_counts_do_not_overflow():
    n_rows = 300
    A_bool = np.ones((n_rows, 2), dtype=bool)
    A_bool[:100, 1] = False
    G_csr = sparse.csr_matrix(np.ones((n_rows, 1), dtype=np.int8))
    tp = np.zeros((2, 1), dtype=np.int32)
    accumulate_tp_sparse_into_dense(tp, A_bool, G_csr)
    assert tp[:, 0].tolist() == [300, 200]
