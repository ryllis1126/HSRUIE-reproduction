from pathlib import Path
from PIL import Image
import csv

ROOT = Path(r"D:\Research")

# 之前 prepare_suim.py 保存的尺寸记录
manifest = ROOT / "SUIM" / "TEST_padded_sizes.csv"

# HSRUIE 的增强输出
src_dir = (
    ROOT / "HSRUIE" / "results_suim_full"
    / "maps_cyclegan" / "test_latest" / "images"
)

# 恢复尺寸后的图像，存放在独立目录
dst_dir = ROOT / "HSRUIE" / "results_suim_restored"
dst_dir.mkdir(parents=True, exist_ok=True)

success = 0
missing = []
invalid = []

with open(manifest, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        name = row["output_name"]

        original_w = int(row["original_width"])
        original_h = int(row["original_height"])

        padded_w = int(row["padded_width"])
        padded_h = int(row["padded_height"])

        src_path = src_dir / name
        dst_path = dst_dir / name

        if not src_path.exists():
            missing.append(name)
            continue

        with Image.open(src_path) as img:
            if img.size != (padded_w, padded_h):
                invalid.append(
                    (name, img.size, (padded_w, padded_h))
                )
                continue

            # 由于原来只在右侧和下方填充，
            # 所以从左上角裁剪回原始尺寸
            restored = img.crop(
                (0, 0, original_w, original_h)
            )
            restored.save(dst_path)

        success += 1

print("成功恢复图像：", success)
print("缺失图像：", len(missing))
print("尺寸异常：", len(invalid))
print("输出目录：", dst_dir)

if missing:
    print("缺失示例：", missing[:10])

if invalid:
    print("尺寸异常示例：", invalid[:10])

if missing or invalid or success != 110:
    raise RuntimeError("未完成全部110张图像恢复，请检查以上信息")