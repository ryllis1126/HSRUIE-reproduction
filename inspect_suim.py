
from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter

root = Path(r"D:\Research\SUIM")

image_dir = root / "TEST" / "images"
mask_dir = root / "TEST" / "masks"

extensions = {".jpg", ".jpeg", ".png", ".bmp"}

images = sorted(
    p for p in image_dir.iterdir()
    if p.suffix.lower() in extensions
)

masks = sorted(
    p for p in mask_dir.iterdir()
    if p.suffix.lower() in extensions
)

image_map = {p.stem: p for p in images}
mask_map = {p.stem: p for p in masks}

common = sorted(set(image_map) & set(mask_map))

print("测试图像数：", len(images))
print("测试 Mask 数：", len(masks))
print("文件名匹配数：", len(common))

print("\n前五组文件和图像尺寸：")

for name in common[:5]:
    with Image.open(image_map[name]) as image, \
         Image.open(mask_map[name]) as mask:

        print(
            name,
            "Image:", image.size,
            "Mask:", mask.size,
            "Mask mode:", mask.mode
        )

print("\n抽样检查 Mask 颜色：")

counts = Counter()

for name in common[:5]:
    with Image.open(mask_map[name]) as mask:
        arr = np.array(mask.convert("RGB"))

    colors, numbers = np.unique(
        arr.reshape(-1, 3),
        axis=0,
        return_counts=True
    )

    for color, number in zip(colors, numbers):
        counts[tuple(map(int, color))] += int(number)

for color, count in counts.most_common(20):
    print(color, count)
