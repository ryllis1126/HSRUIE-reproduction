"""Independent Stage2, paper p.46 (7)-(14); assumptions in stage2_implementation.md."""
import hashlib
import importlib.util
import itertools
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from stage1 import make_scheduler, restore_rng, rng_state, validate_image


CLIP_RN50_SHA256 = 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'
CODE_VERSION = 'stage2-fe18-rn50-spatial234-bicubic-aa-fp32-v1'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clip_preprocess(x):
    """User decision; Tensor bicubic may overshoot [0,1], deliberately no clamp."""
    validate_image(x)
    x = F.interpolate((x + 1) * 0.5, size=(224, 224), mode='bicubic',
                      align_corners=False, antialias=True)
    mean = x.new_tensor((0.48145466, 0.4578275, 0.40821073))[None, :, None, None]
    std = x.new_tensor((0.26862954, 0.26130258, 0.27577711))[None, :, None, None]
    return (x - mean) / std


class ClipSpatialFeatures(nn.Module):
    """Capture layer2/3/4 before attention pooling; no detach or CPU copy."""
    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        self._captures = None
        for name in ('layer2', 'layer3', 'layer4'):
            getattr(visual, name).register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        # A bound method also binds correctly when the wrapper is deep-copied.
        if self._captures is not None:
            self._captures[module] = output

    def forward(self, x):
        if self._captures is not None:
            raise RuntimeError('CLIP spatial adapter is not reentrant')
        self._captures = {}
        try:
            self.visual(x)
            features = [self._captures[getattr(self.visual, name)] for name in ('layer2', 'layer3', 'layer4')]
            if any(f.ndim != 4 or f.shape[0] != x.shape[0] for f in features):
                raise ValueError('CLIP must supply three spatial feature maps, not pooled embeddings')
            return features
        finally:
            self._captures = None


def load_clip_rn50(path, device):
    """Only the verified official local archive. No loader/download/half conversion."""
    if sha256(path) != CLIP_RN50_SHA256:
        raise ValueError('CLIP RN50 SHA-256 differs from the published archive')
    archive = torch.jit.load(str(path), map_location='cpu')
    state = {k[len('visual.'):]: v for k, v in archive.state_dict().items() if k.startswith('visual.')}
    # Import only author model definitions; avoid tokenizer/ftfy and loader side effects.
    source = Path(__file__).resolve().parent / 'CLIP/clip/model.py'
    spec = importlib.util.spec_from_file_location('hsruie_stage2_clip_model', str(source))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    visual = module.ModifiedResNet(layers=(3, 4, 6, 3), output_dim=1024,
                                  heads=32, input_resolution=224, width=64).float()
    visual.load_state_dict(state, strict=True)
    del archive, state
    return ClipSpatialFeatures(visual.to(device)).eval().requires_grad_(False)


def _three(features, name):
    if not isinstance(features, (list, tuple)) or len(features) != 3:
        raise ValueError('{} must contain exactly three scales'.format(name))
    for x in features:
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or not all(x.shape) or x.dtype != torch.float32:
            raise ValueError('{} must contain nonempty FP32 spatial feature maps'.format(name))


def stage2_losses(fu, fc, fd, clip_u, clip_c, fidt, xidt, xc):
    groups = (fu, fc, fd, clip_u, clip_c, fidt)
    for name, group in zip(('Fu', 'Fc', 'Fd', 'CLIP_u', 'CLIP_c', 'Fidt'), groups):
        _three(group, name)
    for i in range(3):
        if any(group[i].shape != fu[i].shape or group[i].device != fu[i].device for group in (fc, fd, fidt)):
            raise ValueError('Fu/Fc/Fd/Fidt shapes and devices must match per scale')
        if clip_u[i].shape != clip_c[i].shape or clip_u[i].device != clip_c[i].device:
            raise ValueError('CLIP source/reconstruction shapes and devices must match')
    if (xc.ndim != 4 or xidt.shape != xc.shape or xc.shape[1] != 3 or
            xc.dtype != torch.float32 or xidt.dtype != torch.float32 or not all(xc.shape)):
        raise ValueError('Xidt/Xc must be equal-shape nonempty FP32 RGB batches')
    tensors = [x for group in groups for x in group] + [xidt, xc]
    if any(x.device != xc.device or x.shape[0] != xc.shape[0] for x in tensors):
        raise ValueError('All Stage2 tensors must share batch and device')
    if not torch.stack([torch.isfinite(x).all() for x in tensors]).all():
        raise ValueError('Nonfinite Stage2 loss inputs')
    sep = sum(F.l1_loss(d + c, u) for d, c, u in zip(fd, fc, fu))
    clip = sum(F.l1_loss(u, c) for u, c in zip(clip_u, clip_c))
    idt1 = sum(F.l1_loss(i, c) for i, c in zip(fidt, fc))
    idt2 = F.l1_loss(xidt, xc)
    losses = {'sep': sep, 'clip': clip, 'idt1': idt1, 'idt2': idt2,
              'total': sep + clip + 2 * idt1 + idt2}
    if not torch.stack([torch.isfinite(x) for x in losses.values()]).all():
        raise ValueError('Nonfinite Stage2 loss')
    return losses


class Stage2(nn.Module):
    def __init__(self, encoder, generator, clip, content=None, distortion=None):
        super().__init__()
        from models.DenseNet import FE
        self.E, self.G, self.clip = encoder, generator, clip
        self.C = content if content is not None else FE()
        self.D = distortion if distortion is not None else FE()
        if {id(p) for p in self.C.parameters()} & {id(p) for p in self.D.parameters()}:
            raise ValueError('C and D must have independent parameters')
        for module in (self.E, self.G, self.clip):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        for module in (self.E, self.G, self.clip):
            module.eval()
        return self

    def forward(self, xu):
        validate_image(xu)
        with torch.no_grad():
            fu = self.E(xu)
            clip_u = self.clip(clip_preprocess(xu))
        fc, fd = self.C(fu), self.D(fu)
        xc = self.G(fc)
        fidt = self.E(xc)
        return stage2_losses(fu, fc, fd, clip_u, self.clip(clip_preprocess(xc)),
                             fidt, self.G(fidt), xc)


def make_optimizer(model, lr=1e-4):
    return torch.optim.Adam(itertools.chain(model.C.parameters(), model.D.parameters()),
                            lr=lr, betas=(0.5, 0.999))


def train_step(model, optimizer, xu):
    expected = {id(p) for module in (model.C, model.D) for p in module.parameters()}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Optimizer must own exactly C/D parameters without duplicates')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = model(xu)
    losses['total'].backward()
    for name, module in (('C', model.C), ('D', model.D)):
        grads = [p.grad for p in module.parameters()]
        if not grads or any(g is None for g in grads):
            raise RuntimeError('{} has missing gradients'.format(name))
        flags = [torch.isfinite(g).all() for g in grads]
        finite = torch.stack(flags).all()
        nonzero = torch.stack([g.ne(0).any() for g in grads]).any()
        if not (finite & nonzero):
            raise RuntimeError('{} needs finite nonzero gradients'.format(name))
    if any(p.grad is not None for m in (model.E, model.G, model.clip) for p in m.parameters()):
        raise RuntimeError('Frozen component accumulated gradients')
    optimizer.step()
    if not torch.stack([torch.isfinite(p).all() for m in (model.C, model.D) for p in m.parameters()]).all():
        raise RuntimeError('Nonfinite trainable parameters after update')
    values = torch.stack([v.detach() for v in losses.values()]).cpu().tolist()
    return dict(zip(losses, values))


def read_stage1_checkpoint(path, vgg_sha256, diagnostic=False, acceptance=None):
    """Trusted local Stage1 package, not an arbitrary pickle or SAD bare state."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    required = {'G', 'optimizer', 'scheduler', 'epoch', 'global_step', 'rng', 'config'}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError('Expected original Stage1 training checkpoint package')
    config = payload['config']
    if config.get('vgg_sha256') != vgg_sha256:
        raise ValueError('Stage1/Stage2 VGG SHA-256 mismatch')
    if not payload['G'] or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in payload['G'].values()):
        raise ValueError('Invalid/nonfinite Stage1 G state')
    if not diagnostic:
        if (payload['epoch'] != 100 or payload['global_step'] < 1 or config.get('mode') != 'train' or
                config.get('fixed_epochs') != 50 or config.get('decay_epochs') != 50):
            raise ValueError('Formal Stage2 requires completed Stage1 epoch100, train, 50+50')
        if (not acceptance or acceptance.get('stage') != 'stage1' or acceptance.get('accepted') is not True or
                acceptance.get('completed_epoch') != 100 or not acceptance.get('basis') or
                acceptance.get('checkpoint_sha256') != sha256(path) or acceptance.get('vgg_sha256') != vgg_sha256):
            raise ValueError('Formal G requires explicit hash-bound Stage1 acceptance record')
    return payload


def build_model(vgg_path, clip_path, stage1_path, device, diagnostic=False, acceptance=None):
    from models.DenseNet import SAD
    from models.vggloss import Vgg19_out
    source = read_stage1_checkpoint(stage1_path, sha256(vgg_path), diagnostic, acceptance)
    generator = SAD()
    generator.load_state_dict(source['G'], strict=True)
    del source
    encoder = Vgg19_out(weight_path=str(vgg_path), device=device)
    clip = load_clip_rn50(clip_path, device)
    return Stage2(encoder, generator.to(device), clip).to(device)


def save_checkpoint(path, model, optimizer, scheduler, completed_epoch, global_step, config):
    if completed_epoch < 0 or global_step < 1:
        raise ValueError('Invalid checkpoint epoch/step')
    payload = {'stage': 'stage2', 'code_version': CODE_VERSION,
               'mode': config['mode'], 'C': model.C.state_dict(), 'D': model.D.state_dict(),
               'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
               'completed_epoch': completed_epoch, 'global_step': global_step,
               'rng': rng_state(), 'effective_config': config}
    with open(path, 'xb') as stream:
        torch.save(payload, stream)


def load_checkpoint(path, model, optimizer, scheduler, config, mode='train'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('stage') != 'stage2' or payload.get('code_version') != CODE_VERSION or
            payload.get('mode') != mode or payload.get('effective_config') != config):
        raise ValueError('Stage2 checkpoint stage/mode/code/config mismatch')
    epoch, step = payload['completed_epoch'], payload['global_step']
    if epoch < (1 if mode == 'train' else 0) or step < 1:
        raise ValueError('Only completed train epoch packages can resume training')
    model.C.load_state_dict(payload['C'], strict=True)
    model.D.load_state_dict(payload['D'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    restore_rng(payload['rng'])
    return epoch, step
