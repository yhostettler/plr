"""
binary_map_gaussian.py
======================
Generates a random binary occupancy map using PyTorch, smooths it with a
Gaussian convolution (GPU-accelerated via torch.nn.functional.conv2d), and
visualises the result with Matplotlib.

NUM_RECTANGLES_MIN / MAX are derived automatically from the map area so the
occupancy density stays reasonable regardless of map size or resolution.

Requirements:
    pip install torch matplotlib numpy

Usage:
    python binary_map_gaussian.py [--seed SEED] [--sigma SIGMA] [--ksize K]
                                  [--res RES] [--map_h H] [--map_w W]
                                  [--output PATH]
"""

import argparse
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ──────────────────────────────────────────────────────────────────────────────
# Global map geometry  (all overridable via CLI)
# ──────────────────────────────────────────────────────────────────────────────
MAP_RES    = 0.1   # metres per cell
MAP_H      = 6.4   # map height in metres
MAP_W      = 6.4   # map width in metres
ADD_BORDER = False

# Rectangle size bounds (cells) — kept fixed; only *count* scales with map area
MIN_RECT_SIZE = 1
MAX_RECT_SIZE = 5

# Gaussian kernel defaults
GAUSSIAN_SIGMA = 1.5
KERNEL_SIZE    = 3  # must be odd

# Target occupancy density range (fraction of cells covered before overlap)
# Tune these two constants to taste — everything else adapts automatically.
TARGET_DENSITY_MIN = 0.05   # ≥10 % of cells touched
TARGET_DENSITY_MAX = 0.20   # ≤30 % of cells touched


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic rectangle-count calculator
# ──────────────────────────────────────────────────────────────────────────────
def compute_rect_count_range(H: int, W: int) -> tuple[int, int]:
    """
    Derive NUM_RECTANGLES_MIN / MAX so that expected coverage stays in
    [TARGET_DENSITY_MIN, TARGET_DENSITY_MAX] regardless of map dimensions.

    Expected area covered by one rectangle (ignoring overlap):
        E[rect_area] = E[rh] * E[rw]
                     = ((MIN+MAX)/2)^2   (assuming symmetric height/width)

    Rectangles needed for target density d:
        n = d * H * W / E[rect_area]
    """
    avg_side     = (MIN_RECT_SIZE + MAX_RECT_SIZE) / 2.0
    avg_rect_area = avg_side ** 2
    map_area      = H * W

    n_min = max(1, math.ceil(TARGET_DENSITY_MIN * map_area / avg_rect_area))
    n_max = max(n_min + 1, math.ceil(TARGET_DENSITY_MAX * map_area / avg_rect_area))
    return n_min, n_max


# ──────────────────────────────────────────────────────────────────────────────
# Map generation
# ──────────────────────────────────────────────────────────────────────────────
def generate_binary_map(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Create a (H, W) float32 binary map on `device` by stamping random
    axis-aligned filled rectangles.  The rectangle count is chosen
    dynamically to hit a reasonable occupancy density.

    Returns
    -------
    torch.Tensor  shape (H, W), dtype float32, values in {0, 1}
    """
    n_min, n_max = compute_rect_count_range(H, W)
    n_rects = int(torch.randint(n_min, n_max + 1, (1,)))

    grid = torch.zeros(H, W, dtype=torch.float32, device=device)

    for _ in range(n_rects):
        rh = int(torch.randint(MIN_RECT_SIZE, MAX_RECT_SIZE + 1, (1,)))
        rw = int(torch.randint(MIN_RECT_SIZE, MAX_RECT_SIZE + 1, (1,)))
        r0 = int(torch.randint(0, max(H - rh, 1) + 1, (1,)))
        c0 = int(torch.randint(0, max(W - rw, 1) + 1, (1,)))
        grid[r0:min(r0 + rh, H), c0:min(c0 + rw, W)] = 1.0

    if ADD_BORDER:
        grid[0, :]  = 1.0
        grid[-1, :] = 1.0
        grid[:, 0]  = 1.0
        grid[:, -1] = 1.0

    return grid


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian kernel
# ──────────────────────────────────────────────────────────────────────────────
def make_gaussian_kernel(size: int, sigma: float,
                         device: torch.device) -> torch.Tensor:
    """
    Build a 2-D isotropic Gaussian kernel via outer product of two 1-D
    Gaussians (separable, numerically stable).

    Returns
    -------
    torch.Tensor  shape (1, 1, size, size), dtype float32, sums to 1
    """
    assert size % 2 == 1, "Kernel size must be odd."
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g1d    = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d   /= g1d.sum()
    kernel  = g1d[:, None] * g1d[None, :]   # outer product → (size, size)
    return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian smoothing  (GPU-accelerated)
# ──────────────────────────────────────────────────────────────────────────────
def apply_gaussian(binary_map: torch.Tensor,
                   kernel: torch.Tensor) -> torch.Tensor:
    """
    Apply Gaussian blur via a single-pass depthwise conv2d (same spatial size).

    Computational notes
    -------------------
    * Dispatches to cuDNN on CUDA for maximum throughput.
    * Kernel is built once on-device; no host↔device transfer per call.
    * For very large maps, split into two separable 1×K / K×1 passes to
      reduce FLOPs from O(K²·H·W) → O(2K·H·W).

    Parameters
    ----------
    binary_map : (H, W) float32
    kernel     : (1, 1, K, K) float32

    Returns
    -------
    torch.Tensor  shape (H, W), values in [0, 1]
    """
    x   = binary_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    pad = kernel.shape[-1] // 2
    out = F.conv2d(x, kernel, padding=pad)       # (1, 1, H, W)
    return out.squeeze(0).squeeze(0)             # (H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────
def visualise(binary_np: np.ndarray, smooth_np: np.ndarray,
              map_h: float, map_w: float, map_res: float,
              n_min: int, n_max: int,
              sigma: float, kernel_size: int,
              save_path: str = "binary_map_gaussian.png") -> None:
    """Render three panels: original, smoothed (nearest), smoothed (bilinear)."""
    H, W   = binary_np.shape
    extent = [0, map_w, 0, map_h]

    fig = plt.figure(figsize=(15, 5))  # a bit wider for 3 plots
    fig.suptitle(
        "Binary Occupancy Map — Original vs. Gaussian Smoothing",
        fontsize=14, fontweight="bold",
    )

    # 3 columns: original, smooth-nearest, smooth-bilinear
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Panel 1: original ─────────────────────────────────────────────
    occupancy = binary_np.mean() * 100
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(
        binary_np, cmap="gray_r", origin="lower", extent=extent,
        vmin=0, vmax=1, interpolation="nearest"
    )
    ax1.set_title(f"Original binary  —  {occupancy:.1f}% occupied", fontsize=11)
    ax1.set_xlabel("x  [m]")
    ax1.set_ylabel("y  [m]")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Occupancy")

    # ── Panel 2: smoothed, nearest ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(
        smooth_np, cmap="gray_r", origin="lower", extent=extent,
        vmin=0, vmax=1, interpolation="nearest"
    )
    ax2.set_title(
        f"Smoothed (nearest)  σ={sigma}, k={kernel_size}×{kernel_size}",
        fontsize=11,
    )
    ax2.set_xlabel("x  [m]")
    ax2.set_ylabel("y  [m]")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Occupancy")

    # ── Panel 3: smoothed, bilinear ──────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(
        smooth_np, cmap="gray_r", origin="lower", extent=extent,
        vmin=0, vmax=1, interpolation="bilinear"
    )
    ax3.set_title(
        f"Smoothed (bilinear)  σ={sigma}, k={kernel_size}×{kernel_size}",
        fontsize=11,
    )
    ax3.set_xlabel("x  [m]")
    ax3.set_ylabel("y  [m]")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="Occupancy")

    # Footer: dynamic rect range shown explicitly
    info = (
        f"Map: {H}×{W} cells @ {map_res} m/cell  ({map_h} m × {map_w} m)  |  "
        f"Rect count range (auto): {n_min}–{n_max}  |  "
        f"Target density: {TARGET_DENSITY_MIN*100:.0f}–{TARGET_DENSITY_MAX*100:.0f}%"
    )
    fig.text(0.5, 0.01, info, ha="center", fontsize=9, color="gray")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {save_path}")
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binary occupancy map with dynamic rect count + Gaussian smoothing"
    )
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--sigma",  type=float, default=GAUSSIAN_SIGMA,
                        help="Gaussian σ in cells (default: %(default)s)")
    parser.add_argument("--ksize",  type=int,   default=KERNEL_SIZE,
                        help="Kernel side length, must be odd (default: %(default)s)")
    parser.add_argument("--res",    type=float, default=MAP_RES,
                        help="Cell size in metres (default: %(default)s)")
    parser.add_argument("--map_h",  type=float, default=MAP_H,
                        help="Map height in metres (default: %(default)s)")
    parser.add_argument("--map_w",  type=float, default=MAP_W,
                        help="Map width in metres (default: %(default)s)")
    parser.add_argument("--output", type=str,   default="binary_map_gaussian.png")
    args = parser.parse_args()

    # Device selection: CUDA → Apple MPS → CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)

    H = int(args.map_h / args.res)
    W = int(args.map_w / args.res)

    # Compute dynamic rectangle range and print it
    n_min, n_max = compute_rect_count_range(H, W)
    print(f"Device          : {device}")
    print(f"Map             : {H} × {W} cells  ({args.map_h} m × {args.map_w} m @ {args.res} m/cell)")
    print(f"Rect count range: {n_min} – {n_max}  "
          f"(auto, target density {TARGET_DENSITY_MIN*100:.0f}–{TARGET_DENSITY_MAX*100:.0f}%)")

    # Generate → smooth → visualise
    binary_map = generate_binary_map(H, W, device)
    kernel     = make_gaussian_kernel(args.ksize, args.sigma, device)
    smooth_map = apply_gaussian(binary_map, kernel)

    if not torch.allclose(binary_map, smooth_map):
        print("maps differ")

    visualise(
        binary_map.cpu().numpy(),
        smooth_map.cpu().numpy(),
        map_h=args.map_h,
        map_w=args.map_w,
        map_res=args.res,
        n_min=n_min,
        n_max=n_max,
        sigma=args.sigma,
        kernel_size=args.ksize,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
