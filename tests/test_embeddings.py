import os
import tempfile

import numpy as np
import pytest
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.embedding_providers import PrecomputedEmbeddingProvider
from model.esm_model import LaccaseModel


def _write_npy(dir_path: str, emb_id: str, arr: np.ndarray) -> None:
    path = os.path.join(dir_path, f"{emb_id}.npy")
    np.save(path, arr)


def _write_pt(dir_path: str, emb_id: str, tensor: torch.Tensor) -> None:
    path = os.path.join(dir_path, f"{emb_id}.pt")
    torch.save(tensor, path)


def test_precomputed_per_sequence_npy():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_npy(tmpdir, "seq1", np.random.randn(4).astype(np.float32))
        _write_npy(tmpdir, "seq2", np.random.randn(4).astype(np.float32))

        provider = PrecomputedEmbeddingProvider(
            embeddings_dir=tmpdir, precomputed_format="npy", granularity="per_sequence"
        )
        batch = {"ids": ["seq1", "seq2"], "sequences": ["AAAA", "BBBB"]}
        out = provider.encode(batch)

        assert out.emb.shape == (2, 4)
        assert out.mask is None
        assert out.din == 4


def test_precomputed_per_residue_padding_and_mask():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_pt(tmpdir, "a", torch.randn(3, 5))
        _write_pt(tmpdir, "b", torch.randn(5, 5))

        provider = PrecomputedEmbeddingProvider(
            embeddings_dir=tmpdir, precomputed_format="pt", granularity="per_residue"
        )
        batch = {"ids": ["a", "b"], "sequences": ["ABC", "ABCDE"]}
        out = provider.encode(batch)

        assert out.emb.shape == (2, 5, 5)
        assert out.mask.shape == (2, 5)
        assert out.mask[0].sum().item() == 3
    assert out.mask[1].sum().item() == 5


def test_precomputed_npz_container_multiple_arrays():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "embs.npz")
        np.savez(path, id1=np.random.randn(4).astype(np.float32), id2=np.random.randn(4).astype(np.float32))

        provider = PrecomputedEmbeddingProvider(
            embeddings_dir=path, precomputed_format="npz", granularity="per_sequence"
        )
        batch = {"ids": ["id1", "id2"]}
        out = provider.encode(batch)

        assert out.emb.shape == (2, 4)
        assert out.mask is None
        assert out.din == 4


def test_fair_esm_cpu_forward_smoke():
    if os.getenv("ESM_EZY_SKIP_ESM_DOWNLOAD") == "1":
        pytest.skip("Skipping ESM weight download in CI.")
    pytest.importorskip("esm")
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["TORCH_HOME"] = tmpdir
        model = LaccaseModel(
            embedding_source="fair_esm2",
            esm_checkpoint="esm2_t6_8M_UR50D",
            device="cpu",
            precision="fp32",
            d_model=128,
        )
        batch = {"ids": ["s1", "s2"], "sequences": ["ACDE", "ACDEFG"]}
        out, reps = model(batch, return_repr=True)

        assert out.shape == (2, 2)
        assert reps.shape[0] == 2
        assert reps.shape[1] == model.d_model


@pytest.mark.skipif(
    not (torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())),
    reason="No GPU/MPS backend available.",
)
def test_precomputed_device_smoke():
    device = "cuda" if torch.cuda.is_available() else "mps"
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_npy(tmpdir, "seq1", np.random.randn(4).astype(np.float32))
        _write_npy(tmpdir, "seq2", np.random.randn(4).astype(np.float32))

        model = LaccaseModel(
            embedding_source="precomputed",
            precomputed_embeddings_dir=tmpdir,
            precomputed_format="npy",
            precomputed_granularity="per_sequence",
        ).to(device)

        batch = {"ids": ["seq1", "seq2"], "sequences": ["AA", "BB"]}
        out = model(batch)
        assert out.device.type == device
