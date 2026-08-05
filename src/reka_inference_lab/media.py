"""Original deterministic media for a license-clean multimodal smoke test."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_demo_image(path: str | Path) -> Path:
    """Create an owned diagram with obvious objects and text for visual evaluation."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (768, 512), color=(15, 23, 42))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((55, 80, 245, 270), radius=20, fill=(37, 99, 235))
    draw.ellipse((300, 80, 490, 270), fill=(239, 68, 68))
    draw.polygon([(640, 75), (545, 270), (735, 270)], fill=(250, 204, 21))
    font = ImageFont.load_default(size=30)
    small = ImageFont.load_default(size=20)
    draw.text((55, 32), "MULTIMODAL EVALUATION CARD", fill=(241, 245, 249), font=font)
    draw.text((95, 290), "BLUE SQUARE", fill=(147, 197, 253), font=small)
    draw.text((330, 290), "RED CIRCLE", fill=(252, 165, 165), font=small)
    draw.text((560, 290), "YELLOW TRIANGLE", fill=(254, 240, 138), font=small)
    draw.line((80, 385, 688, 385), fill=(71, 85, 105), width=3)
    draw.text((80, 410), "Request ID: REKA-EDGE-LAB-001", fill=(203, 213, 225), font=small)
    image.save(destination, format="PNG", optimize=True)
    return destination
