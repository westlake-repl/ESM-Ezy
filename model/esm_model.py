import torch
import torch.nn as nn

from .embedding_providers import FairEsm2Provider, PrecomputedEmbeddingProvider


class LaccaseModel(nn.Module):
    def __init__(
        self,
        pretrained_model_path: str | None = None,
        embedding_source: str = "fair_esm2",
        esm_checkpoint: str = "esm2_t36_3B_UR50D",
        device: str = "auto",
        precision: str = "fp32",
        precomputed_embeddings_dir: str | None = None,
        precomputed_format: str = "auto",
        precomputed_granularity: str = "per_sequence",
        d_model: int = 1280,
    ):
        super().__init__()
        if pretrained_model_path is not None:
            esm_checkpoint = pretrained_model_path

        self.embedding_source = embedding_source
        self.d_model = d_model

        if embedding_source == "fair_esm2":
            self.embedding_provider = FairEsm2Provider(
                esm_checkpoint=esm_checkpoint, device=device, precision=precision
            )
        elif embedding_source == "precomputed":
            if precomputed_embeddings_dir is None:
                raise ValueError("precomputed_embeddings_dir must be set for precomputed embeddings.")
            self.embedding_provider = PrecomputedEmbeddingProvider(
                embeddings_dir=precomputed_embeddings_dir,
                precomputed_format=precomputed_format,
                granularity=precomputed_granularity,
            )
        else:
            raise ValueError(f"Unknown embedding_source: {embedding_source}")

        self.proj = nn.LazyLinear(d_model)
        self.dnn = nn.Sequential(nn.ReLU(), nn.Linear(d_model, 2))

    @property
    def device(self):
        if hasattr(self.embedding_provider, "device"):
            return self.embedding_provider.device
        for param in self.parameters():
            return param.device
        return torch.device("cpu")

    def _normalize_batch(self, data):
        if isinstance(data, dict):
            ids = data.get("ids")
            sequences = data.get("sequences")
            if sequences and isinstance(sequences[0], (tuple, list)) and len(sequences[0]) == 2:
                ids = [item[0] for item in sequences]
                sequences = [item[1] for item in sequences]
            if ids is None and sequences is not None:
                ids = [f"seq{i}" for i in range(len(sequences))]
            return {"ids": list(ids) if ids is not None else None, "sequences": list(sequences) if sequences is not None else None}

        ids, sequences = zip(*data)
        return {"ids": list(ids), "sequences": list(sequences)}

    def _pool_per_residue(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if x.dim() == 2:
            return x
        if mask is None:
            return x.mean(dim=1)
        mask = mask.to(dtype=x.dtype, device=x.device)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, data, return_repr: bool = False):
        batch = self._normalize_batch(data)
        emb_batch = self.embedding_provider.encode(batch)
        emb = emb_batch.emb
        mask = emb_batch.mask

        # If checkpoint projection shape mismatches input dim, reinitialize projection.
        if hasattr(self.proj, "weight") and self.proj.weight is not None:
            from torch.nn.parameter import UninitializedParameter
            if not isinstance(self.proj.weight, UninitializedParameter):
                if self.proj.weight.shape[-1] != emb.shape[-1]:
                    self.proj = nn.Linear(emb.shape[-1], self.d_model).to(emb.device)

        proj_device = self.proj.weight.device
        if emb.device != proj_device:
            emb = emb.to(proj_device)
        if mask is not None and mask.device != proj_device:
            mask = mask.to(proj_device)

        proj_emb = self.proj(emb)
        if emb_batch.granularity == "per_residue":
            pooled = self._pool_per_residue(proj_emb, mask)
        else:
            pooled = proj_emb

        out_put = self.dnn(pooled)
        if return_repr:
            return out_put, pooled
        return out_put

    def _get_layers(self) -> int:
        return int(getattr(self.embedding_provider, "num_layers", 0) or 0)

    @property
    def layers(self):
        return self.get_layers()

    def get_layers(self):
        return self._get_layers()

    def get_last_layer_idx(self):
        return self._get_layers() - 1

    def set_trainable_last_layers(self, last_layers: int):
        for _, param in self.named_parameters():
            param.requires_grad = False

        for name, param in self.named_parameters():
            if name.startswith("proj") or name.startswith("dnn"):
                param.requires_grad = True

        backbone = getattr(self.embedding_provider, "model", None)
        total_layers = getattr(self.embedding_provider, "num_layers", None)
        if backbone is None or total_layers is None or last_layers <= 0:
            return

        for offset in range(1, last_layers + 1):
            layer_idx = total_layers - offset
            for name, param in backbone.named_parameters():
                if name.startswith(f"layers.{layer_idx}."):
                    param.requires_grad = True

    def get_representations(self, data):
        _, reps = self.forward(data, return_repr=True)
        return reps

    def get_names(self, data):
        batch = self._normalize_batch(data)
        return batch.get("ids")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | None = None,
        state_dict_path: str | None = None,
        **kwargs,
    ):
        model = cls(pretrained_model_path=pretrained_model_path, **kwargs)
        if state_dict_path is not None:
            print(f"Loading state dict from {state_dict_path}")
            state = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
        return model

        
