"""
Generate architecture diagram for the report.
Run: python generate_diagram.py
Outputs: architecture_diagram.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("LandslideSegNet: UNet++ with Spectral Index Features & scSE Attention",
                  fontsize=14, fontweight="bold", pad=20)

    # Colors
    c_input = "#E3F2FD"
    c_indices = "#FFF3E0"
    c_proj = "#E8F5E9"
    c_encoder = "#BBDEFB"
    c_decoder = "#C8E6C9"
    c_attn = "#FFE0B2"
    c_output = "#F3E5F5"
    c_loss = "#FFCDD2"

    def box(x, y, w, h, text, color, fontsize=8):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Input
    box(0.5, 8.2, 3, 0.8, "Input: 128×128×14\n(12 Sentinel-2 + DEM + Slope)", c_input, 7)

    # Spectral Indices
    box(4.5, 8.2, 3, 0.8, "Spectral Index Module\nNDVI, NDWI, NDBI, BSI", c_indices, 7)
    arrow(3.5, 8.6, 4.5, 8.6)

    # Concatenate
    box(4.5, 7.0, 3, 0.6, "Concatenate → 128×128×18", c_input, 7)
    arrow(2, 8.2, 2, 7.6)
    arrow(6, 8.2, 6, 7.6)
    arrow(2, 7.6, 4.5, 7.3)

    # Input Projection
    box(4.5, 5.8, 3, 0.8, "Input Projection\nConv7×7→Conv3×3→Conv1×1\n18ch → 64 → 32 → 3ch", c_proj, 6.5)
    arrow(6, 7.0, 6, 6.6)

    # Encoder
    box(0.5, 3.5, 3.5, 1.8, "ResNet34 Encoder\n(ImageNet Pretrained)\n\nStage1: 64ch, 64×64\nStage2: 64ch, 32×32\nStage3: 128ch, 16×16\nStage4: 256ch, 8×8\nStage5: 512ch, 4×4", c_encoder, 6)
    arrow(4.5, 5.8, 2.25, 5.3)

    # Decoder
    box(5, 3.5, 3.5, 1.8, "UNet++ Decoder\n(Dense Skip Connections)\n\n256 → 128 → 64 → 32 → 16\nNested skip paths\nMulti-scale feature fusion", c_decoder, 6)
    arrow(4, 4.4, 5, 4.4)

    # Attention
    box(9.5, 3.5, 2.5, 1.8, "scSE Attention\n\nChannel SE:\nGAP → FC → Sigmoid\n\nSpatial SE:\nConv1×1 → Sigmoid\n\nApplied at each\ndecoder level", c_attn, 6)
    arrow(8.5, 4.4, 9.5, 4.4)

    # Output
    box(5, 1.5, 3.5, 0.8, "Segmentation Head\nConv 1×1 → Sigmoid\n128×128×1", c_output, 7)
    arrow(6.75, 3.5, 6.75, 2.3)

    # Loss
    box(10, 1.5, 3, 0.8, "Combined Loss\n0.5×Tversky(α=0.3,β=0.7)\n+ 0.5×Focal(α=0.75,γ=2)", c_loss, 6.5)
    arrow(8.5, 1.9, 10, 1.9)

    # Training details
    box(10, 6.2, 5.5, 2.8, "Training Configuration\n\nOptimizer: AdamW (lr=1e-3, wd=1e-4)\n"
        "Scheduler: OneCycleLR\n"
        "Augmentation: Flip, Rotate90,\n  ShiftScaleRotate, GaussNoise\n"
        "Grad Clipping: max_norm=1.0\n"
        "Val Split: 15% of train\n"
        "Early Stop: patience=20 on Z-score\n"
        "Batch Size: 16", "#F5F5F5", 6.5)

    # Output label
    box(5, 0.3, 3.5, 0.7, "Binary Mask: Landslide / Non-Landslide", "#E1BEE7", 7)
    arrow(6.75, 1.5, 6.75, 1.0)

    # Params
    ax.text(0.5, 0.5, "Total Trainable Parameters: ~25.7M", fontsize=9,
            fontweight="bold", style="italic")

    plt.tight_layout()
    plt.savefig("architecture_diagram.png", dpi=200, bbox_inches="tight")
    print("Saved architecture_diagram.png")
    plt.close()


if __name__ == "__main__":
    draw_architecture()
