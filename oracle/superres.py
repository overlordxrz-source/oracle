"""AI super-resolution of Sentinel-2: 10 m -> 2.5 m with ESA OpenSR's SEN2SR.

SEN2SR (Aybar et al., 2025) is trained on Sentinel-2/NAIP pairs (SEN2NAIPv2) to add
4x detail to the red, green, blue and NIR bands. Its "hard constraint" layer forces
the output, averaged back down to 10 m, to equal the input. So it can't invent
reflectance, but it can and does invent *plausible sub-pixel structure*: edges,
roof outlines, road markings. Weights are CC0, from Hugging Face (downloaded once).

  lite  CNN, 2.3 MB, runs anywhere (CPU, Apple GPU)             <- default
  full  Mamba state-space model, 55 MB, needs the CUDA-only mamba_ssm kernels

Use it to *look*. Oracle never feeds super-resolved pixels to a detector or cites them
as evidence: the extra detail is a model's guess.
"""

from __future__ import annotations

import functools
import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import CACHE_DIR
from .geo import AOI
from .http import client, log
from .imagery import Grid, NoData
from .models import Scene

REPO = "https://huggingface.co/tacofoundation/SEN2SR/resolve/main"
VARIANTS = {"lite": "SEN2SRLite/NonReference_RGBN_x4", "full": "SEN2SR/NonReference_RGBN_x4"}
MODEL_DIR = CACHE_DIR / "models" / "sen2sr"
PATCH, OVERLAP, SCALE = 128, 16, 4
MAX_LR = 768  # input pixels per side (7.7 km); output is 4x that
NOTICE = "AI-enhanced (SEN2SR x4): plausible detail, not evidence"


def _fetch(variant: str) -> dict:
    d = MODEL_DIR / variant
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in ("model.safetensor", "hard_constraint.safetensor"):
        p = d / name
        if not p.exists():
            log(f"downloading SEN2SR {variant} {name}...")
            with client() as c, c.stream("GET", f"{REPO}/{VARIANTS[variant]}/{name}", timeout=600, follow_redirects=True) as r:
                r.raise_for_status()
                tmp = p.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                tmp.rename(p)
        out[name] = p
    return out


def pick_variant(variant: str = "auto") -> str:
    if variant != "auto":
        return variant
    try:
        import mamba_ssm  # noqa: F401
        import torch

        return "full" if torch.cuda.is_available() else "lite"
    except ImportError:
        return "lite"


@functools.lru_cache(maxsize=2)
def load(variant: str = "lite", device: str | None = None):
    try:
        import safetensors.torch
        import torch
        from sen2sr.models.tricks import HardConstraint
        from sen2sr.nonreference import srmodel
    except ImportError as exc:
        raise RuntimeError('super-resolution needs: pip install "oracle-osint[ai]"') from exc
    from .objdet import pick_device

    device = device or pick_device()
    if device.startswith("cuda") is False and device != "mps":
        device = "cpu"
    files = _fetch(variant)
    weights = safetensors.torch.load_file(files["model.safetensor"])
    if variant == "lite":
        from sen2sr.models.opensr_baseline.cnn import CNNSR

        net = CNNSR(4, 4, 24, 4, True, False, 6)
    else:
        from sen2sr.models.opensr_baseline.mamba import MambaSR

        net = MambaSR(
            img_size=(128, 128),
            in_channels=4,
            out_channels=4,
            embed_dim=96,
            depths=[8] * 6,
            num_heads=[8] * 6,
            mlp_ratio=4,
            upscale=4,
            attention_type="sigmoid_02",
            upsampler="pixelshuffle",
            resi_connection="1conv",
            operation_attention="sum",
        )
    net.load_state_dict(weights)
    net.eval()
    hc = HardConstraint(
        low_pass_mask=safetensors.torch.load_file(files["hard_constraint.safetensor"])["weights"].to(device), device=device
    )
    model = srmodel(net, hc, device).eval()
    torch.set_grad_enabled(False)
    return model, device


def upscale(x: np.ndarray, variant: str = "auto", batch: int = 8) -> np.ndarray:
    """(4, h, w) reflectance (R, G, B, NIR) -> (4, 4h, 4w), tiled with overlap blending."""
    import torch

    model, device = load(pick_variant(variant))
    c, h, w = x.shape
    H, W = max(h, PATCH), max(w, PATCH)
    pad = np.zeros((c, H, W), np.float32)
    pad[:, :h, :w] = np.nan_to_num(x, nan=0.0)
    step = PATCH - 2 * OVERLAP
    ys = list(range(0, max(H - PATCH, 0) + 1, step))
    xs = list(range(0, max(W - PATCH, 0) + 1, step))
    if ys[-1] != H - PATCH:
        ys.append(H - PATCH)
    if xs[-1] != W - PATCH:
        xs.append(W - PATCH)
    out = np.zeros((c, H * SCALE, W * SCALE), np.float32)
    weight = np.zeros((H * SCALE, W * SCALE), np.float32)
    # Feather each tile's edges so seams vanish.
    ramp = np.minimum(np.arange(PATCH * SCALE) + 1, np.arange(PATCH * SCALE)[::-1] + 1).astype(np.float32)
    win = np.minimum(np.minimum.outer(ramp, ramp) / (OVERLAP * SCALE), 1.0)
    coords = [(y, x0) for y in ys for x0 in xs]
    for i in range(0, len(coords), batch):
        chunk = coords[i : i + batch]
        t = torch.from_numpy(np.stack([pad[:, y : y + PATCH, x0 : x0 + PATCH] for y, x0 in chunk])).to(device)
        sr = model(t).float().cpu().numpy()
        for (y, x0), tile in zip(chunk, sr, strict=True):
            Y, X = y * SCALE, x0 * SCALE
            out[:, Y : Y + PATCH * SCALE, X : X + PATCH * SCALE] += tile * win
            weight[Y : Y + PATCH * SCALE, X : X + PATCH * SCALE] += win
    out /= np.maximum(weight, 1e-6)
    return out[:, : h * SCALE, : w * SCALE]


@dataclass
class SRResult:
    scene: Scene
    aoi: AOI
    grid: Grid  # 10 m input grid
    lr: np.ndarray  # (4, h, w)
    sr: np.ndarray  # (4, 4h, 4w)
    variant: str

    def rgb(self, which: str = "sr") -> Image.Image:
        a = self.sr if which == "sr" else self.lr
        rgb = np.clip(np.moveaxis(a[:3], 0, -1) / 0.3, 0, 1) ** (1 / 1.6)
        img = Image.fromarray((rgb * 255).astype(np.uint8), "RGB")
        if which != "sr":
            img = img.resize((img.width * SCALE, img.height * SCALE), Image.BICUBIC)
        return img

    def png(self, which: str = "sr") -> bytes:
        buf = io.BytesIO()
        self.rgb(which).save(buf, format="PNG")
        return buf.getvalue()

    def side_by_side(self) -> Image.Image:
        a, b = self.rgb("lr"), self.rgb("sr")
        from .imagery import draw_caption

        a = draw_caption(a, f"Sentinel-2 {self.scene.date}, 10 m (bicubic)")
        b = draw_caption(b, f"{NOTICE} [{self.variant}]")
        out = Image.new("RGB", (a.width + b.width + 8, a.height), (0, 0, 0))
        out.paste(a, (0, 0))
        out.paste(b, (a.width + 8, 0))
        return out


def enhance(scene: Scene, aoi: AOI, variant: str = "auto") -> SRResult:
    """Super-resolve the AOI of a Sentinel-2 scene."""
    from .change import _read_band

    if scene.source != "sentinel-2":
        raise ValueError("super-resolution works on Sentinel-2 scenes")
    grid = Grid.for_aoi(aoi, 10.0, max_pixels=MAX_LR)
    if grid.res > 10.5:
        raise ValueError(f"area too large for super-resolution (max ~{MAX_LR * 10 / 1000:.1f} km across)")
    scale = scene.extra.get("reflectance_scale", 0.0001)
    offset = scene.extra.get("reflectance_offset", 0.0)
    lr = np.stack([_read_band(scene, b, grid) * scale + offset for b in ("red", "green", "blue", "nir")])
    if np.isnan(lr[0]).mean() > 0.5:
        raise NoData("scene doesn't cover most of the area")
    v = pick_variant(variant)
    sr = upscale(np.clip(lr, 0, 1), v)
    return SRResult(scene, aoi, grid, lr, sr, v)


def tiles_needed(w: int, h: int) -> int:
    step = PATCH - 2 * OVERLAP
    return max(1, math.ceil(max(w - PATCH, 0) / step) + 1) * max(1, math.ceil(max(h - PATCH, 0) / step) + 1)
