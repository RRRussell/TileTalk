"""Modular text/image encoders.

Backends:
  * BiomedCLIP (open_clip hub model) -- joint 512-d image+text space, the
    primary zero-shot encoder for TileTalk.
  * Deterministic "pixel-stat" image encoder + hashed text encoder -- a
    dependency-free fallback so the full retrieval pipeline runs (and tests)
    without downloading model weights or a GPU.

All encoders return L2-normalizable float32 arrays; `encode_image` consumes a
(n, H, W, 3) uint8 array, `encode_text` consumes a list of strings.
"""
from __future__ import annotations

from typing import List

import numpy as np


# --------------------------------------------------------------------------- #
# BiomedCLIP
# --------------------------------------------------------------------------- #
class BiomedCLIPEncoder:
    embed_dim = 512

    def __init__(self,
                 hub: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 device: str = "cuda",
                 batch_size: int = 256):
        import torch
        import open_clip
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.model, self.preprocess = open_clip.create_model_from_pretrained(hub)
        self.tokenizer = open_clip.get_tokenizer(hub)
        self.model = self.model.to(self.device).eval()

    def encode_image(self, patches: np.ndarray) -> np.ndarray:
        from PIL import Image
        torch = self.torch
        feats = []
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                batch = patches[i:i + self.batch_size]
                ims = torch.stack([self.preprocess(Image.fromarray(p)) for p in batch])
                ims = ims.to(self.device)
                f = self.model.encode_image(ims)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu().float().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32)

    def encode_text(self, texts: List[str]) -> np.ndarray:
        torch = self.torch
        toks = self.tokenizer(list(texts)).to(self.device)
        with torch.no_grad():
            f = self.model.encode_text(toks)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().float().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# PLIP (pathology CLIP; open weights via transformers)
# --------------------------------------------------------------------------- #
class PLIPEncoder:
    embed_dim = 512

    def __init__(self, hub: str = "vinid/plip", device: str = "cuda",
                 batch_size: int = 256):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.model = CLIPModel.from_pretrained(hub).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(hub)

    def encode_image(self, patches: np.ndarray) -> np.ndarray:
        from PIL import Image
        torch = self.torch
        feats = []
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                ims = [Image.fromarray(p) for p in patches[i:i + self.batch_size]]
                inp = self.processor(images=ims, return_tensors="pt").to(self.device)
                f = self.model.get_image_features(**inp)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu().float().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32)

    def encode_text(self, texts):
        torch = self.torch
        inp = self.processor(text=list(texts), return_tensors="pt",
                             padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            f = self.model.get_text_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().float().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# UNI2-h (MahmoodLab gated pathology foundation encoder; image-only, timm ViT-H)
# --------------------------------------------------------------------------- #
class UNI2Encoder:
    embed_dim = 1536

    def __init__(self, device: str = "cuda", batch_size: int = 64):
        import torch
        import timm
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24,
                  init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
                  num_classes=0, no_embed_class=True,
                  mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                  reg_tokens=8, dynamic_img_size=True)
        self.model = timm.create_model("hf-hub:MahmoodLab/UNI2-h",
                                       pretrained=True, **kw).to(self.device).eval()
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**cfg)

    def encode_image(self, patches: np.ndarray) -> np.ndarray:
        from PIL import Image
        torch = self.torch
        feats = []
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                ims = torch.stack([self.transform(Image.fromarray(p))
                                   for p in patches[i:i + self.batch_size]]).to(self.device)
                feats.append(self.model(ims).cpu().float().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32)

    def encode_text(self, texts):
        raise NotImplementedError("UNI2 is image-only (no text tower)")


class GigaPathEncoder:
    """Prov-GigaPath tile encoder (gated, image-only, timm ViT-giant, 1536-d)."""
    embed_dim = 1536

    def __init__(self, device: str = "cuda", batch_size: int = 64):
        import torch
        import timm
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.model = timm.create_model("hf-hub:prov-gigapath/prov-gigapath",
                                       pretrained=True).to(self.device).eval()
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**cfg)

    def encode_image(self, patches: np.ndarray) -> np.ndarray:
        from PIL import Image
        torch = self.torch
        feats = []
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                ims = torch.stack([self.transform(Image.fromarray(p))
                                   for p in patches[i:i + self.batch_size]]).to(self.device)
                feats.append(self.model(ims).cpu().float().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32)

    def encode_text(self, texts):
        raise NotImplementedError("GigaPath is image-only (no text tower)")


# --------------------------------------------------------------------------- #
# Deterministic fallback (no weights, no GPU)
# --------------------------------------------------------------------------- #
class HashEncoder:
    """Deterministic image/text encoder for smoke tests and the random baseline.

    Images -> coarse color/texture statistics projected to a fixed random basis.
    Text   -> hashed bag-of-words projected to the same basis. The two spaces are
    *not* aligned (by design): this exists to exercise the pipeline, not to score
    well.
    """
    embed_dim = 512

    def __init__(self, embed_dim: int = 512, seed: int = 0):
        self.embed_dim = embed_dim
        rng = np.random.default_rng(seed)
        self._img_proj = rng.standard_normal((48, embed_dim)).astype(np.float32)
        self._txt_proj = rng.standard_normal((256, embed_dim)).astype(np.float32)

    def encode_image(self, patches: np.ndarray) -> np.ndarray:
        n = len(patches)
        feats = np.zeros((n, 48), dtype=np.float32)
        for i, p in enumerate(patches):
            p = p.astype(np.float32) / 255.0
            stats = []
            for c in range(p.shape[2]):
                ch = p[..., c]
                stats += [ch.mean(), ch.std(), np.median(ch), ch.min(), ch.max(),
                          np.percentile(ch, 25), np.percentile(ch, 75),
                          np.mean(np.abs(np.diff(ch, axis=0))),
                          np.mean(np.abs(np.diff(ch, axis=1)))]
            stats += [0.0] * (48 - len(stats))
            feats[i] = stats[:48]
        return (feats @ self._img_proj).astype(np.float32)

    def encode_text(self, texts: List[str]) -> np.ndarray:
        feats = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in str(t).lower().split():
                feats[i, hash(w) % 256] += 1.0
        return (feats @ self._txt_proj).astype(np.float32)


def get_encoder(name: str, **kwargs):
    name = name.lower()
    if name in ("biomedclip", "bmc"):
        return BiomedCLIPEncoder(**{k: v for k, v in kwargs.items()
                                    if k in ("hub", "device", "batch_size")})
    if name == "plip":
        return PLIPEncoder(**{k: v for k, v in kwargs.items()
                              if k in ("device", "batch_size")})
    if name in ("uni2", "uni2-h", "uni"):
        return UNI2Encoder(**{k: v for k, v in kwargs.items()
                              if k in ("device", "batch_size")})
    if name in ("gigapath", "prov-gigapath"):
        return GigaPathEncoder(**{k: v for k, v in kwargs.items()
                                  if k in ("device", "batch_size")})
    if name in ("hash", "random", "fallback"):
        return HashEncoder(**{k: v for k, v in kwargs.items()
                              if k in ("embed_dim", "seed")})
    raise ValueError(f"unknown encoder {name}")
