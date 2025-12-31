#!/usr/bin/env python3
"""
Generate neural-themed icons for AVA Tauri application.

Creates a minimalist brain/neural network icon with:
- Dark background (#0A0A10)
- Cyan primary accent (#00D4C8)
- Purple secondary accent (#8B5CF6)
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw


def create_neural_icon(size: int) -> Image.Image:
    """
    Create a neural network themed icon.

    Design: Abstract neural network with nodes and connections
    - Dark background
    - Cyan nodes in a brain-like pattern
    - Purple connecting synapses
    - Subtle glow effect
    """
    # Colors
    bg_color = (10, 10, 16)  # #0A0A10 - neural void
    cyan = (0, 212, 200)  # #00D4C8 - primary accent
    purple = (139, 92, 246)  # #8B5CF6 - cortex color
    cyan_dim = (0, 150, 140)  # Dimmed cyan for glow
    purple_dim = (99, 52, 186)  # Dimmed purple

    # Create image with anti-aliasing (2x then downscale)
    render_size = size * 2
    img = Image.new("RGBA", (render_size, render_size), bg_color + (255,))
    draw = ImageDraw.Draw(img)

    center = render_size // 2

    # Node positions forming a brain-like pattern
    # Central node (the "cortex")
    nodes = [
        (center, center),  # Central node
    ]

    # Inner ring (6 nodes)
    inner_radius = render_size * 0.18
    for i in range(6):
        angle = (i * 60 - 30) * math.pi / 180
        x = center + int(inner_radius * math.cos(angle))
        y = center + int(inner_radius * math.sin(angle))
        nodes.append((x, y))

    # Outer ring (8 nodes)
    outer_radius = render_size * 0.35
    for i in range(8):
        angle = (i * 45 + 22.5) * math.pi / 180
        x = center + int(outer_radius * math.cos(angle))
        y = center + int(outer_radius * math.sin(angle))
        nodes.append((x, y))

    # Draw connections (synapses)
    connection_width = max(2, render_size // 64)

    # Connect central to inner ring
    for i in range(1, 7):
        draw.line([nodes[0], nodes[i]], fill=purple_dim + (180,), width=connection_width)

    # Connect inner ring to outer ring
    for i in range(1, 7):
        # Connect to 1-2 outer nodes
        outer_start = 7 + ((i - 1) * 8 // 6)
        for j in range(2):
            outer_idx = 7 + (outer_start + j) % 8
            draw.line([nodes[i], nodes[outer_idx]], fill=cyan_dim + (120,), width=connection_width)

    # Connect some outer nodes
    for i in range(8):
        next_idx = 7 + (i + 1) % 8
        draw.line([nodes[7 + i], nodes[next_idx]], fill=purple_dim + (100,), width=connection_width)

    # Draw nodes
    central_size = render_size // 8
    inner_size = render_size // 14
    outer_size = render_size // 18

    # Draw glow for central node
    for glow_offset in range(5, 0, -1):
        glow_size = central_size + glow_offset * 4
        glow_alpha = 30 - glow_offset * 5
        glow_color = cyan_dim + (max(0, glow_alpha),)
        x, y = nodes[0]
        draw.ellipse([x - glow_size, y - glow_size, x + glow_size, y + glow_size], fill=glow_color)

    # Central node (brightest, main focal point)
    x, y = nodes[0]
    draw.ellipse(
        [x - central_size, y - central_size, x + central_size, y + central_size], fill=cyan + (255,)
    )

    # Inner ring nodes (purple)
    for i in range(1, 7):
        x, y = nodes[i]
        # Small glow
        for glow_offset in range(3, 0, -1):
            glow_size = inner_size + glow_offset * 2
            glow_alpha = 20 - glow_offset * 5
            draw.ellipse(
                [x - glow_size, y - glow_size, x + glow_size, y + glow_size],
                fill=purple_dim + (max(0, glow_alpha),),
            )
        draw.ellipse(
            [x - inner_size, y - inner_size, x + inner_size, y + inner_size], fill=purple + (255,)
        )

    # Outer ring nodes (cyan, smaller)
    for i in range(7, 15):
        x, y = nodes[i]
        draw.ellipse(
            [x - outer_size, y - outer_size, x + outer_size, y + outer_size], fill=cyan + (220,)
        )

    # Downscale with anti-aliasing
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    return img


def create_icon_set(output_dir: Path):
    """Create all required icon sizes for Tauri."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate PNG icons
    sizes = {
        "32x32.png": 32,
        "128x128.png": 128,
        "128x128@2x.png": 256,  # Retina is 2x
    }

    for filename, size in sizes.items():
        icon = create_neural_icon(size)
        icon.save(output_dir / filename, "PNG")
        print(f"Created {filename} ({size}x{size})")

    # Create ICO (Windows) - multi-resolution
    ico_sizes = [16, 32, 48, 256]
    ico_images = [create_neural_icon(s) for s in ico_sizes]
    ico_images[0].save(
        output_dir / "icon.ico",
        format="ICO",
        sizes=[(s, s) for s in ico_sizes],
        append_images=ico_images[1:],
    )
    print("Created icon.ico (multi-resolution)")

    # Create ICNS (macOS) - multi-resolution
    icns_sizes = [16, 32, 64, 128, 256, 512, 1024]
    icns_images = [create_neural_icon(s) for s in icns_sizes]

    # ICNS format using Pillow
    try:
        icns_images[0].save(output_dir / "icon.icns", format="ICNS", append_images=icns_images[1:])
        print("Created icon.icns (macOS)")
    except Exception as e:
        # Fallback: just create a PNG that macOS can use
        print(f"ICNS creation failed ({e}), creating 512x512 PNG instead")
        create_neural_icon(512).save(output_dir / "icon.png", "PNG")
        print("Created icon.png (512x512 fallback)")


if __name__ == "__main__":
    script_dir = Path(__file__).parent.parent
    icons_dir = script_dir / "ui" / "src-tauri" / "icons"

    print("Generating neural-themed AVA icons...")
    print(f"Output directory: {icons_dir}")
    print()

    create_icon_set(icons_dir)

    print()
    print("Icon generation complete!")
