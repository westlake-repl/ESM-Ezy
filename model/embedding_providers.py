import os
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EmbeddingBatch:
    emb: torch.Tensor
    mask: Optional[torch.Tensor]
    ids: List[str]
    granularity: str
    din: int


class EmbeddingProvider(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, batch: dict) -> EmbeddingBatch:
        raise NotImplementedError


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _resolve_precision(device: torch.device, precision: str) -> torch.dtype:
    if precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if device.type == "cpu" and dtype != torch.float32:
        warnings.warn("CPU precision forced to fp32 for stability.", RuntimeWarning)
        return torch.float32
    if device.type == "mps" and dtype == torch.bfloat16:
        warnings.warn("MPS does not reliably support bf16; using fp16.", RuntimeWarning)
        return torch.float16
    return dtype


class FairEsm2Provider(EmbeddingProvider):
    def __init__(self, esm_checkpoint: str = "esm2_t36_3B_UR50D", device: str = "auto", precision: str = "fp32"):
        super().__init__()
        import esm

        self.device = _resolve_device(device)
        self.dtype = _resolve_precision(self.device, precision)

        if os.path.isfile(esm_checkpoint):
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(esm_checkpoint)
        else:
            if not hasattr(esm.pretrained, esm_checkpoint):
                raise ValueError(f"Unknown esm checkpoint: {esm_checkpoint}")
            model_fn = getattr(esm.pretrained, esm_checkpoint)
            model, alphabet = model_fn()

        self.model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.model.eval()
        self.model.to(self.device)
        if self.device.type in ("cuda", "mps") and self.dtype != torch.float32:
            self.model.to(self.dtype)

        self.num_layers = getattr(self.model, "num_layers", None)
        if self.num_layers is None:
            self.num_layers = getattr(getattr(self.model, "args", None), "layers", None)
        if self.num_layers is None and hasattr(self.model, "layers"):
            self.num_layers = len(self.model.layers)
        if self.num_layers is None:
            raise ValueError("Unable to determine number of layers for ESM model.")

        self.embed_dim = getattr(self.model, "embed_dim", None)
        if self.embed_dim is None:
            self.embed_dim = getattr(getattr(self.model, "args", None), "embed_dim", None)

        self.max_positions = getattr(self.model, "max_positions", None)
        if self.max_positions is None:
            self.max_positions = getattr(getattr(self.model, "args", None), "max_positions", None)
        if isinstance(self.max_positions, (list, tuple)):
            self.max_positions = self.max_positions[0]
        if self.max_positions is None:
            self.max_positions = 1024
        self.max_residues = self.max_positions - 2

    def encode(self, batch: dict) -> EmbeddingBatch:
        sequences = batch.get("sequences")
        if sequences is None:
            raise ValueError("Batch missing sequences for fair_esm2 embedding.")
        ids = batch.get("ids")
        if ids is None:
            ids = [f"seq{i}" for i in range(len(sequences))]

        max_len = max(len(s) for s in sequences) if sequences else 0
        if max_len > self.max_residues:
            raise ValueError(
                f"Sequence length {max_len} exceeds model limit {self.max_residues}. "
                "Please split sequences or use shorter inputs."
            )

        data = list(zip(ids, sequences))
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)

        last_layer = self.num_layers
        use_autocast = self.device.type in ("cuda", "mps") and self.dtype != torch.float32
        with torch.no_grad():
            if use_autocast:
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    out = self.model(tokens, repr_layers=[last_layer], return_contacts=False)
            else:
                out = self.model(tokens, repr_layers=[last_layer], return_contacts=False)
        rep = out["representations"][last_layer]

        if max_len == 0:
            rep = rep[:, 0:0, :]
            mask = torch.zeros((rep.size(0), 0), dtype=torch.bool, device=rep.device)
        else:
            rep = rep[:, 1 : 1 + max_len, :]
            mask = torch.zeros((rep.size(0), max_len), dtype=torch.bool, device=rep.device)
            for i, seq in enumerate(sequences):
                mask[i, : len(seq)] = True

        din = rep.shape[-1]
        return EmbeddingBatch(emb=rep, mask=mask, ids=ids, granularity="per_residue", din=din)


class PrecomputedEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        embeddings_dir: str,
        precomputed_format: str = "auto",
        granularity: str = "per_sequence",
    ):
        super().__init__()
        self.embeddings_dir = embeddings_dir
        self.precomputed_format = precomputed_format
        self.granularity = granularity
        self.embeddings_file = embeddings_dir if os.path.isfile(embeddings_dir) else None
        self._npz_data = None
        self._npz_container = bool(self.embeddings_file and self.embeddings_file.endswith(".npz"))
        if self.embeddings_file and not self._npz_container:
            raise ValueError("precomputed_embeddings_dir must be a directory or a .npz container file.")

    def _resolve_path(self, emb_id: str) -> str:
        if self.precomputed_format == "auto":
            for ext in (".npy", ".npz", ".pt"):
                if emb_id.endswith(ext):
                    candidate = os.path.join(self.embeddings_dir, emb_id)
                else:
                    candidate = os.path.join(self.embeddings_dir, emb_id + ext)
                if os.path.isfile(candidate):
                    return candidate
            raise FileNotFoundError(f"No embedding file found for id {emb_id} in {self.embeddings_dir}")

        ext = "." + self.precomputed_format if not self.precomputed_format.startswith(".") else self.precomputed_format
        if emb_id.endswith(ext):
            candidate = os.path.join(self.embeddings_dir, emb_id)
        else:
            candidate = os.path.join(self.embeddings_dir, emb_id + ext)
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"Embedding file not found: {candidate}")
        return candidate

    def _load_embedding(self, path: str) -> torch.Tensor:
        if path.endswith(".npy"):
            arr = np.load(path)
            return torch.from_numpy(arr)
        if path.endswith(".npz"):
            data = np.load(path, allow_pickle=False)
            if len(data.files) == 1:
                arr = data[data.files[0]]
                return torch.from_numpy(arr)
            raise ValueError(f"NPZ file has multiple arrays {data.files}; provide a single array or use id-key lookup.")
        if path.endswith(".pt"):
            tensor = torch.load(path, map_location="cpu")
            return torch.as_tensor(tensor)
        raise ValueError(f"Unsupported embedding format for {path}")

    def _load_npz_key(self, key: str) -> torch.Tensor:
        if self._npz_data is None:
            self._npz_data = np.load(self.embeddings_file, allow_pickle=False)
        if key not in self._npz_data.files:
            raise FileNotFoundError(f"Embedding id {key} not found in {self.embeddings_file}")
        arr = self._npz_data[key]
        return torch.from_numpy(arr)

    def encode(self, batch: dict) -> EmbeddingBatch:
        ids = batch.get("ids")
        if ids is None:
            raise ValueError("Batch missing ids for precomputed embeddings.")

        sequences = batch.get("sequences")
        embeddings = []
        lengths = []
        din = None

        for emb_id in ids:
            if self._npz_container:
                emb = self._load_npz_key(emb_id)
            else:
                path = self._resolve_path(emb_id)
                if path.endswith(".npz"):
                    data = np.load(path, allow_pickle=False)
                    if emb_id in data.files:
                        emb = torch.from_numpy(data[emb_id])
                    elif len(data.files) == 1:
                        emb = torch.from_numpy(data[data.files[0]])
                    else:
                        raise ValueError(
                            f"NPZ file {path} has multiple arrays {data.files}; provide id key or a single array."
                        )
                else:
                    emb = self._load_embedding(path)
            emb = emb.detach().cpu()
            if not torch.isfinite(emb).all():
                raise ValueError(f"Non-finite values found in embedding {emb_id}")

            if self.granularity == "per_sequence":
                if emb.ndim == 1:
                    emb = emb.unsqueeze(0)
                elif emb.ndim == 2 and emb.shape[0] == 1:
                    pass
                else:
                    raise ValueError(
                        f"Per-sequence embedding {emb_id} must have shape (Din,) or (1,Din); got {tuple(emb.shape)}"
                    )
                if din is None:
                    din = emb.shape[-1]
                elif emb.shape[-1] != din:
                    raise ValueError("Inconsistent embedding dimensions within batch.")
                embeddings.append(emb.squeeze(0))
            else:
                if emb.ndim == 3 and emb.shape[0] == 1:
                    emb = emb.squeeze(0)
                if emb.ndim != 2:
                    raise ValueError(
                        f"Per-residue embedding {emb_id} must have shape (L,Din); got {tuple(emb.shape)}"
                    )
                if din is None:
                    din = emb.shape[-1]
                elif emb.shape[-1] != din:
                    raise ValueError("Inconsistent embedding dimensions within batch.")
                embeddings.append(emb)
                lengths.append(emb.shape[0])

        if self.granularity == "per_sequence":
            emb_batch = torch.stack(embeddings, dim=0)
            mask = None
        else:
            max_len = max(lengths) if lengths else 0
            emb_batch = torch.zeros((len(embeddings), max_len, din), dtype=embeddings[0].dtype)
            mask = torch.zeros((len(embeddings), max_len), dtype=torch.bool)
            for i, emb in enumerate(embeddings):
                emb_batch[i, : emb.shape[0], :] = emb
                mask[i, : emb.shape[0]] = True
            if sequences is not None:
                for emb_id, seq, emb in zip(ids, sequences, embeddings):
                    if len(seq) != emb.shape[0]:
                        warnings.warn(
                            f"Sequence length mismatch for {emb_id}: seq {len(seq)} vs emb {emb.shape[0]}",
                            RuntimeWarning,
                        )

        return EmbeddingBatch(
            emb=emb_batch,
            mask=mask,
            ids=ids,
            granularity=self.granularity,
            din=din or 0,
        )
