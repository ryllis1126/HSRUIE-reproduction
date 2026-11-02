from pathlib import Path
from PIL import Image
import numpy as np
import csv

# 原始 SUIM 测试图片
src_dir = Path(r"D:\Research\SUIM\TEST\images")

# 新建的填充后图片目录
dst_dir = Path(r"D:\Research\SUIM\TEST_padded_images")
dst_dir.mkdir(parents=True, exist_ok=True)

# 保存原始尺寸，以便之后裁剪回来
manifest = Path(r"D:\Research\SUIM\TEST_padded_sizes.csv")

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

images = sorted(
    p for p in src_dir.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
)

records = []
used_names = set()

for path in images:
    with Image.open(path) as img:
        array = np.array(img.convert("RGB"))

    h, w = array.shape[:2]

    # 计算需要填充的像素数
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    # 只在右侧、底部填充，不移动原有像素
    padded = np.pad(
        array,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect"
    )

    output_name = path.stem + ".png"

    if output_name.lower() in used_names:
        raise ValueError(f"输出文件名重复：{output_name}")
    used_names.add(output_name.lower())

    Image.fromarray(padded).save(dst_dir / output_name)

    records.append([
        path.name,
        output_name,
        w,
        h,
        w + pad_w,
        h + pad_h
    ])

with open(manifest, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "original_name",
        "output_name",
        "original_width",
        "original_height",
        "padded_width",
        "padded_height"
    ])
    writer.writerows(records)

print("原始图片数：", len(images))
print("填充后图片数：", len(records))
print("输出目录：", dst_dir)
print("尺寸记录：", manifest)