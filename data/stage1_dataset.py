"""Stage 1: independent RGB domains; crop only, never resize/pad/filter."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


EXTENSIONS = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ppm'}


def image_paths(root, manifest=None):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Not an image directory: {}'.format(root))
    if manifest is None:
        # Require the actual HR leaf directory; do not sweep HR/LR subtrees.
        paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else:
        names = Path(manifest).read_text(encoding='utf-8-sig').splitlines()
        paths = []
        for name in names:
            if not name.strip():
                continue
            p = (root / name.strip()).resolve(strict=True)
            if root not in p.parents or not p.is_file() or p.suffix.lower() not in EXTENSIONS:
                raise ValueError('Manifest entry outside image domain or unsupported: {}'.format(name))
            paths.append(p)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Image domain must be nonempty, with unique paths: {}'.format(root))
    return paths


def check_images(paths):
    """Report every member; callers must reject the entire run on any issue."""
    records, small, errors = [], [], []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.load()
                w, h = image.size
                record = {'path': str(path), 'width': w, 'height': h, 'mode': image.mode}
            records.append(record)
            if min(w, h) < 256:
                small.append(record)
        except (OSError, ValueError) as exc:
            errors.append({'path': str(path), 'error': str(exc)})
    return {'count': len(paths), 'readable': len(records), 'records': records,
            'too_small': small, 'errors': errors, 'skipped': 0}


def read_crop(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        w, h = image.size
        if min(w, h) < 256:
            raise ValueError('Image is smaller than 256; no resizing/padding/discard allowed: {}'.format(path))
        x, y = random.randint(0, w - 256), random.randint(0, h - 256)
        pixels = np.array(image.crop((x, y, x + 256, y + 256)), dtype=np.uint8, copy=True)
    # Same arithmetic/order as ToTensor + Normalize(0.5, 0.5).
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous().float().div_(255)
    tensor.sub_(0.5).div_(0.5)
    return tensor


class Stage1Dataset(Dataset):
    """Engineering assumption: batch=2 means two images in EACH domain.

    Epoch=max(domain lengths); underwater cyclic, air sampled with replacement.
    The public unaligned implementation is incomplete, so this is not an
    authenticated author Stage 1 sampling protocol. Tail batches are retained.
    """
    def __init__(self, underwater_paths, inair_paths):
        self.underwater_paths = list(underwater_paths)
        self.inair_paths = list(inair_paths)
        if not self.underwater_paths or not self.inair_paths:
            raise ValueError('Both image domains must be nonempty')

    def __len__(self):
        return max(len(self.underwater_paths), len(self.inair_paths))

    def __getitem__(self, index):
        u = self.underwater_paths[index % len(self.underwater_paths)]
        a = self.inair_paths[random.randrange(len(self.inair_paths))]
        return {'Xu': read_crop(u), 'Xa': read_crop(a), 'Xu_path': str(u), 'Xa_path': str(a)}
