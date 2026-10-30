"""DeiT Tiny: inferenza CPU e pesi di attenzione per la dimostrazione della L13.

Architettura non distillata: 12 blocchi, 3 head, patch 16 × 16, input 224 × 224.
Il modello usa i pesi ImageNet-1k originali, senza addestramento aggiuntivo.

Riferimenti:
https://github.com/facebookresearch/deit/blob/main/models.py
https://github.com/facebookresearch/dino/blob/main/visualize_attention.py
Foto Chelsea: Stefan van der Walt, CC0, distribuita con scikit-image.
https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.chelsea

L'estrazione della riga CLS e la visualizzazione sono svolte nel notebook:
questo modulo espone l'intera matrice dell'ultimo blocco, prima di A @ V.
"""
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image
import torch
from torch import nn


CHECKPOINT_SHA256 = "a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97"
PHOTO_SHA256 = "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb"


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 192, kernel_size=16, stride=16, dtype=torch.float32)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(192, 192 * 3, bias=True, dtype=torch.float32)
        self.proj = nn.Linear(192, 192, dtype=torch.float32)
        self.last_attention = None

    def forward(self, x):
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, 3, 64).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # La softmax è sulle colonne: ogni query distribuisce peso sulle key.
        attention = ((q @ k.transpose(-2, -1)) * (64 ** -0.5)).softmax(dim=-1)
        self.last_attention = attention.detach()
        x = (attention @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(192, 768, dtype=torch.float32)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(768, 192, dtype=torch.float32)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(192, eps=1e-6, dtype=torch.float32)
        self.attn = Attention()
        self.norm2 = nn.LayerNorm(192, eps=1e-6, dtype=torch.float32)
        self.mlp = Mlp()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class DeiTTiny(nn.Module):
    """DeiT Tiny non distillato, compatibile con il checkpoint ufficiale."""

    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 192, dtype=torch.float32))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 192, dtype=torch.float32))
        self.patch_embed = PatchEmbed()
        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.norm = nn.LayerNorm(192, eps=1e-6, dtype=torch.float32)
        self.head = nn.Linear(192, 1000, dtype=torch.float32)

    def forward(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


def _verifica_file(path, expected_sha256, descrizione):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{descrizione} non trovato: {path}. Carica i materiali della demo "
            "nello stesso runtime e controlla il percorso; non occorre scaricare altri file."
        )
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"Integrità non verificata per {descrizione}: {path}. "
            f"SHA-256 atteso {expected_sha256}; trovato {actual}. "
            "Usa il file originale incluso nei materiali della demo."
        )
    return path, actual


def prepara_demo(image_path, weights_path):
    """Restituisce (modello, tensore_input, immagine_RGB, informazioni).

    I percorsi sono espliciti. I due file vengono verificati con SHA-256; il
    modulo non effettua download. Resize bicubico a lato corto 256 e crop
    centrale 224 × 224 coincidono con il protocollo dell'esempio in dispensa.
    Il tensore e tutti i parametri sono float32 su CPU, anche se le celle
    precedenti hanno impostato un diverso dtype predefinito. Il generatore
    casuale del notebook e le impostazioni globali di PyTorch sono conservati.
    """
    image_path, image_sha = _verifica_file(image_path, PHOTO_SHA256, "Foto Chelsea")
    weights_path, checkpoint_sha = _verifica_file(
        weights_path, CHECKPOINT_SHA256, "Checkpoint DeiT Tiny"
    )

    with Image.open(image_path) as source:
        photo = source.convert("RGB")
    old_w, old_h = photo.size
    if old_w < old_h:
        new_size = (256, int(256 * old_h / old_w))
    else:
        new_size = (int(256 * old_w / old_h), 256)
    resized = photo.resize(new_size, Image.Resampling.BICUBIC)
    left = round((new_size[0] - 224) / 2)
    top = round((new_size[1] - 224) / 2)
    crop_box = (left, top, left + 224, top + 224)
    cropped = resized.crop(crop_box)
    rgb = np.array(cropped, dtype=np.uint8, copy=True)
    x = torch.from_numpy(rgb.astype(np.float32) / np.float32(255)).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    x = ((x - mean) / std).unsqueeze(0)

    # L'inizializzazione verrà sostituita dai pesi: preserviamo lo stato RNG.
    with torch.random.fork_rng(devices=[]):
        model = DeiTTiny()
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError("Il checkpoint deve contenere il dizionario dei pesi nella chiave 'model'.")
    model.load_state_dict(state, strict=True)
    model.eval()

    info = {
        "model": "deit_tiny_patch16_224, non distillato, ImageNet-1k",
        "checkpoint_sha256": checkpoint_sha,
        "photo_sha256": image_sha,
        "photo_author": "Stefan van der Walt",
        "photo_license": "CC0",
        "device": "cpu",
        "dtype": "float32",
        "preprocessing": {
            "photo_width_height": [old_w, old_h],
            "resize_width_height": list(new_size),
            "resize": "BICUBIC",
            "crop_box": list(crop_box),
            "input_width_height": [224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "layer_1_based": 12,
        "heads": 3,
        "patch_size": 16,
        "patch_grid": [14, 14],
        "cls_index": 0,
        "patch_start_index": 1,
        "attention_shape": [1, 3, 197, 197],
        "training": False,
    }
    return model, x, rgb, info


def calcola_attention(model, input_tensor):
    """Restituisce A dell'ultimo blocco: (1, 3, 197, 197), CPU float32.

    L'asse -2 indica le query e l'asse -1 le key. È la matrice dopo la
    softmax per righe, prima del prodotto per V. Non è un attention rollout.
    Il modello viene usato in eval e inference_mode, senza training.
    """
    if not isinstance(model, DeiTTiny):
        raise TypeError("Per questa demo è richiesto il modello DeiTTiny restituito da prepara_demo.")
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("L'input della demo deve essere un tensore PyTorch.")
    if tuple(input_tensor.shape) != (1, 3, 224, 224):
        raise ValueError(f"Forma input attesa (1, 3, 224, 224); trovata {tuple(input_tensor.shape)}.")
    if input_tensor.dtype != torch.float32 or input_tensor.device.type != "cpu":
        raise ValueError("L'input deve essere float32 su CPU, come restituito da prepara_demo.")
    if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Il modello deve avere parametri float32 su CPU.")
    model.eval()
    with torch.inference_mode():
        model(input_tensor)
        attention = model.blocks[-1].attn.last_attention.clone()
    if tuple(attention.shape) != (1, 3, 197, 197):
        raise RuntimeError(f"Forma attention inattesa: {tuple(attention.shape)}.")
    if not bool(torch.isfinite(attention).all()):
        raise RuntimeError("La matrice di attenzione contiene valori non finiti.")
    if not bool(torch.allclose(attention.sum(-1), torch.ones_like(attention[..., 0]), atol=3e-7, rtol=0)):
        raise RuntimeError("Le righe della matrice di attenzione non sommano a 1 entro la tolleranza.")
    return attention


__all__ = ["DeiTTiny", "prepara_demo", "calcola_attention"]
