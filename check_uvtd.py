from pathlib import Path
from PIL import Image
from collections import Counter

# 修改为你实际的 UVTD 路径
raw_dir = Path(r"E:\BaiduNetdiskDownload\UVTD\原图")
ref_dir = Path(r"E:\BaiduNetdiskDownload\UVTD\参考图像")

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}

def get_images(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in extensions
    )

raw_files = get_images(raw_dir)
ref_files = get_images(ref_dir)

raw_map = {p.stem.lower(): p for p in raw_files}
ref_map = {p.stem.lower(): p for p in ref_files}

common = set(raw_map) & set(ref_map)

print("原图数量:", len(raw_files))
print("参考图数量:", len(ref_files))
print("同名配对数量:", len(common))

print("仅原图存在:", len(set(raw_map) - set(ref_map)))
print("仅参考图存在:", len(set(ref_map) - set(raw_map)))

sizes = Counter()

for path in raw_files:
    with Image.open(path) as img:
        sizes[img.size] += 1

print("\n原图尺寸统计:")
for size, count in sizes.most_common(10):
    print(size, count)