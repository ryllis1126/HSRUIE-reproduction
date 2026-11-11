"""Minimal Stage 1 objective, shared E/G, epoch scheduler and recovery.

Paper: printed pp.45-47, III.B (1)-(6), IV.A. Reduction/sampling/initialization
assumptions are documented in docs/stage1_implementation.md.
"""
import functools
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def validate_image(x):
    if x.ndim != 4 or x.shape[0] == 0 or x.shape[1] != 3 or x.dtype != torch.float32:
        raise ValueError('Expected a nonempty float32 Bx3xHxW image batch')
    # Combine device predicates before the single host decision.
    valid = torch.isfinite(x).all() & (x.min() >= -1) & (x.max() <= 1)
    if not valid:
        raise ValueError('Expected finite images in [-1,1]')


def ranking_loss(sa, su):
    return F.relu(1.0 - sa + su).mean()


def stage1_losses(xa, xu, reconstructed_a, reconstructed_u):
    for x in (xa, xu, reconstructed_a, reconstructed_u):
        validate_image(x)
        if x.shape != xa.shape:
            raise ValueError('All four Stage 1 image batches must have equal shapes')
    # Per-image CHW cosine, then per-image hinge, then batch mean.
    sa = F.cosine_similarity(xa.flatten(1), reconstructed_a.flatten(1), dim=1)
    su = F.cosine_similarity(xu.flatten(1), reconstructed_u.flatten(1), dim=1)
    rec_a, rec_u = F.l1_loss(reconstructed_a, xa), F.l1_loss(reconstructed_u, xu)
    sr = ranking_loss(sa, su)
    return {'rec_a': rec_a, 'rec_u': rec_u, 'sr': sr,
            'total': sr + rec_a + 0.1 * rec_u, 'sa': sa.mean(), 'su': su.mean()}


class Stage1(nn.Module):
    def __init__(self, encoder, generator):
        super().__init__()
        self.E, self.G = encoder, generator
        self.E.requires_grad_(False)
        self.E.eval()

    def train(self, mode=True):
        super().train(mode)
        self.E.eval()
        return self

    def forward(self, xa, xu):
        validate_image(xa)
        validate_image(xu)
        with torch.no_grad():
            fa, fu = self.E(xa), self.E(xu)
        return self.G(fa), self.G(fu)


def build_model(vgg_weights, device, g_init):
    from models.vggloss import Vgg19_out
    from models.DenseNet import SAD
    encoder = Vgg19_out(weight_path=str(vgg_weights), device=device)
    generator = SAD()
    if g_init == 'author-normal':
        from models.networks import init_weights
        init_weights(generator, init_type='normal', init_gain=0.02)
    elif g_init != 'constructor':
        raise ValueError('Explicit SAD initialization must be constructor or author-normal')
    return Stage1(encoder, generator.to(device))


def make_optimizer(generator, lr=1e-4, beta1=0.5):
    # Only the arguments actually passed in the published Adam call.
    return torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))


def lr_factor(completed_epochs, fixed_epochs=50, decay_epochs=50):
    return max(0.0, 1.0 - max(0, completed_epochs - fixed_epochs) / float(decay_epochs))


def make_scheduler(optimizer, fixed_epochs=50, decay_epochs=50):
    if fixed_epochs < 1 or decay_epochs < 1:
        raise ValueError('Both learning-rate phases must be positive')
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, functools.partial(lr_factor, fixed_epochs=fixed_epochs, decay_epochs=decay_epochs))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    # Worker torch RNG is seeded by DataLoader; persistent workers are disabled.
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed; exact RNG recovery unavailable')
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, config):
    payload = {'G': model.G.state_dict(), 'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(), 'epoch': epoch,
               'global_step': global_step, 'rng': rng_state(), 'config': config}
    # Exclusive creation: do not replace any previous file.
    with open(path, 'xb') as stream:
        torch.save(payload, stream)


def load_checkpoint(path, model, optimizer, scheduler, config):
    # This is a trusted local training checkpoint (contains Python/NumPy RNG).
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['config'] != config:
        raise ValueError('Checkpoint training configuration/data/VGG fingerprint mismatch')
    model.G.load_state_dict(checkpoint['G'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    restore_rng(checkpoint['rng'])
    return checkpoint['epoch'], checkpoint['global_step']


def validate_gradients(grads):
    if not grads:
        raise RuntimeError('Missing/nonfinite generator gradients')
    # Check every gradient on-device, then read one aggregate flag. Do not
    # short-circuit in Python: each device boolean would synchronize CUDA.
    finite = torch.stack([torch.isfinite(g).all() for g in grads]).all()
    if not finite:
        raise RuntimeError('Missing/nonfinite generator gradients')
    if not any(torch.count_nonzero(g).item() for g in grads):
        raise RuntimeError('All generator gradients are zero')


def train_step(model, optimizer, xa, xu):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ya, yu = model(xa, xu)
    losses = stage1_losses(xa, xu, ya, yu)
    losses['total'].backward()
    grads = [p.grad for p in model.G.parameters() if p.grad is not None]
    validate_gradients(grads)
    if any(p.grad is not None for p in model.E.parameters()):
        raise RuntimeError('Frozen encoder accumulated gradients')
    optimizer.step()
    # All six metrics are scalar float32 in the existing objective. Transfer
    # them together without retaining their autograd graphs.
    values = torch.stack([value.detach() for value in losses.values()]).cpu().tolist()
    return dict(zip(losses, values))
