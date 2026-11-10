"""Stage 1: approved in-memory UVTD reflection, then crop; never resize/filter."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


EXTENSIONS = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ppm'}
UVTD_REFLECT_POLICY = 'uvtd-reflect16-352x224-224x352-v1'
# (width, height) -> (left, top, right, bottom); no inferred padding policy.
UVTD_PADDING = {(352, 224): (0, 16, 0, 16), (224, 352): (16, 0, 16, 0)}


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


def prepare_image(image, allow_uvtd_reflect=False):
    """D-11 engineering adaptation, enabled only for the underwater domain."""
    image = image.convert('RGB')
    w, h = image.size
    left = top = right = bottom = 0
    if min(w, h) < 256:
        if not allow_uvtd_reflect or (w, h) not in UVTD_PADDING:
            raise ValueError('Image is smaller than 256 without an approved reflection policy: {}'.format((w, h)))
        left, top, right, bottom = UVTD_PADDING[(w, h)]
        pixels = np.pad(np.asarray(image), ((top, bottom), (left, right), (0, 0)), mode='reflect')
        image = Image.fromarray(pixels, mode='RGB')
    metadata = {'original_size': [w, h], 'padded_size': list(image.size),
                'padding': {'left': left, 'top': top, 'right': right, 'bottom': bottom},
                'strategy': 'reflect_then_random_crop256' if any((left, top, right, bottom)) else 'random_crop256',
                'policy': UVTD_REFLECT_POLICY if allow_uvtd_reflect else 'crop256-only'}
    return image, metadata


def check_images(paths, allow_uvtd_reflect=False):
    """Report every member; callers must reject the entire run on any issue."""
    records, small, errors, padded = [], [], [], []
    no_padding = 0
    for path in paths:
        try:
            with Image.open(path) as image:
                image.load()
                w, h = image.size
                record = {'path': str(path), 'width': w, 'height': h, 'mode': image.mode}
                records.append(record)
                if min(w, h) < 256 and not (allow_uvtd_reflect and (w, h) in UVTD_PADDING):
                    record['strategy'] = 'unsupported_short_image'
                    small.append(record)
                    continue
                _, metadata = prepare_image(image, allow_uvtd_reflect=allow_uvtd_reflect)
                record.update(metadata)
                if any(metadata['padding'].values()):
                    padded.append(record)
                else:
                    no_padding += 1
        except (OSError, ValueError) as exc:
            errors.append({'path': str(path), 'error': str(exc)})
    return {'count': len(paths), 'readable': len(records), 'records': records,
            'too_small': small, 'errors': errors, 'skipped': 0,
            'padding_count': len(padded), 'no_padding_count': no_padding, 'requires_padding': padded,
            'preprocessing_policy': UVTD_REFLECT_POLICY if allow_uvtd_reflect else 'crop256-only'}


def read_crop(path, allow_uvtd_reflect=False):
    with Image.open(path) as image:
        image, _ = prepare_image(image, allow_uvtd_reflect=allow_uvtd_reflect)
        w, h = image.size
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
        return {'Xu': read_crop(u, allow_uvtd_reflect=True), 'Xa': read_crop(a),
                'Xu_path': str(u), 'Xa_path': str(a)}
