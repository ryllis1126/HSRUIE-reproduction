"""Independent Stage 1 CLI. No model execution without an explicit mode."""
import argparse
import copy
import hashlib
import io
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from torch.utils.data import DataLoader

from data.stage1_dataset import Stage1Dataset, check_images, image_paths, read_crop
from stage1 import (build_model, load_checkpoint, make_optimizer, make_scheduler,
                    save_checkpoint, seed_everything, seed_worker, train_step)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=['data-check', 'smoke', 'train'])
    p.add_argument('--underwater-root', required=True, type=Path)
    p.add_argument('--inair-root', required=True, type=Path)
    p.add_argument('--underwater-manifest', type=Path)
    p.add_argument('--inair-manifest', type=Path)
    p.add_argument('--vgg-weights', type=Path)
    p.add_argument('--g-init', choices=['constructor', 'author-normal'],
                   help='Explicit assumption: author Stage 1 initialization is unavailable')
    p.add_argument('--output-root', required=True, type=Path)
    p.add_argument('--run-id', required=True)
    p.add_argument('--resume', type=Path, help='Completed-epoch checkpoint; train mode only')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=2, help='Images per domain (engineering assumption)')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--beta1', type=float, default=0.5, help='Published train_options default')
    p.add_argument('--fixed-epochs', type=int, default=50)
    p.add_argument('--decay-epochs', type=int, default=50)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=20261109, help='Engineering seed, not paper setting')
    return p


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with open(path, 'x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=str)


def provenance(args):
    repo = Path(__file__).resolve().parent
    def git(*arguments):
        result = subprocess.run(['git', '-C', str(repo), *arguments], capture_output=True, text=True)
        return {'returncode': result.returncode, 'output': result.stdout, 'error': result.stderr}
    files = ['stage1.py', 'train_stage1.py', 'data/stage1_dataset.py',
             'models/vggloss.py', 'models/DenseNet.py', 'models/NEDB_IN.py',
             'models/FEM.py', 'tests/test_stage1.py', 'docs/stage1_implementation.md', 'AGENTS.md']
    return {'start_utc': now(), 'purpose': 'Stage 1 III.B equations (1)-(6)',
            'argv': sys.argv, 'cwd': str(Path.cwd()), 'executable': sys.executable,
            'python': sys.version, 'platform': platform.platform(),
            'versions': {'torch': torch.__version__, 'torchvision': torchvision.__version__,
                         'numpy': np.__version__, 'Pillow': PIL.__version__},
            'arguments': vars(args), 'git_head': git('rev-parse', 'HEAD'),
            'git_status': git('status', '--short'), 'git_diff': git('diff', '--', '*.py'),
            'source_snapshot': {f: (repo / f).read_text(encoding='utf-8') for f in files},
            'forward': 'frozen E -> shared trainable SAD; NC/LOW/HIGH/CLIP/U-Net not loaded',
            'determinism': {'cudnn_benchmark': torch.backends.cudnn.benchmark,
                            'cudnn_deterministic': torch.backends.cudnn.deterministic,
                            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}}


def smoke(model, optimizer, scheduler, loader, device, run, config):
    batch = next(iter(loader))
    xa, xu = batch['Xa'].to(device), batch['Xu'].to(device)
    if tuple(xa.shape) != (2, 3, 256, 256) or xu.shape != xa.shape:
        raise ValueError('Smoke acceptance requires exactly two 256x256 images per domain')
    before_e = {k: v.detach().clone() for k, v in model.E.state_dict().items()}
    before_g = {k: v.detach().clone() for k, v in model.G.named_parameters()}
    features, outputs = [], []
    def encoder_hook(module, inputs, result):
        expected = [(2, 64, 256, 256), (2, 128, 128, 128), (2, 256, 64, 64)]
        if [tuple(x.shape) for x in result] != expected:
            raise AssertionError('Published VGG three-scale shape contract failed')
        features.append(result)
    def generator_hook(module, inputs, result):
        if result.shape != xa.shape:
            raise AssertionError('SAD output shape contract failed')
        result.retain_grad()
        outputs.append(result)
    eh = model.E.register_forward_hook(encoder_hook)
    gh = model.G.register_forward_hook(generator_hook)
    try:
        losses = train_step(model, optimizer, xa, xu)
    finally:
        eh.remove()
        gh.remove()
    for output in outputs:
        if output.grad is None or not torch.isfinite(output.grad).all() or not output.grad.count_nonzero():
            raise AssertionError('Both domain outputs must receive finite nonzero gradients')
    if len(outputs) != 2 or len(features) != 2:
        raise AssertionError('Expected exactly two shared E/G forwards')
    if model.E.training or any(p.requires_grad or p.grad is not None for p in model.E.parameters()):
        raise AssertionError('E freeze/mode contract failed')
    if not all(torch.equal(before_e[k], v) for k, v in model.E.state_dict().items()):
        raise AssertionError('Frozen E state changed')
    expected_unused = {name for name, _ in model.G.named_parameters()
                       if name.startswith(('trans_down_1.', 'trans_down_2.'))}
    unused = {name for name, p in model.G.named_parameters() if p.grad is None}
    if unused != expected_unused:
        raise AssertionError('Unexpected missing SAD gradients: {}'.format(unused ^ expected_unused))
    changed = [name for name, p in model.G.named_parameters() if not torch.equal(before_g[name], p)]
    if not changed or any(name in unused for name in changed):
        raise AssertionError('G parameter update contract failed')
    if not all(torch.isfinite(p).all() for p in model.G.parameters()):
        raise AssertionError('Nonfinite G after update')
    path = run / 'smoke_checkpoint.pth'
    save_checkpoint(path, model, optimizer, scheduler, 0, 1, config)
    # Reuse the frozen E; independently restore G/optimizer/scheduler.
    from stage1 import Stage1
    restored = Stage1(model.E, copy.deepcopy(model.G))
    with torch.no_grad():
        for p in restored.G.parameters():
            p.zero_()
    restored_optimizer = make_optimizer(restored.G, config['lr'], config['beta1'])
    restored_scheduler = make_scheduler(restored_optimizer, config['fixed_epochs'], config['decay_epochs'])
    epoch, step = load_checkpoint(path, restored, restored_optimizer, restored_scheduler, config)
    with torch.no_grad():
        reference, recovered = model.G(features[0]), restored.G(features[0])
    if epoch != 0 or step != 1 or not torch.allclose(reference, recovered, rtol=1e-5, atol=1e-6):
        raise AssertionError('SAD checkpoint/output round trip failed')
    return {'losses': losses, 'Xu_paths': batch['Xu_path'], 'Xa_paths': batch['Xa_path'],
            'shapes': [list(x.shape) for x in features[0]], 'changed_parameters': len(changed),
            'expected_unused_parameters': sorted(unused), 'checkpoint_sha256': sha256(path),
            'processed_underwater': 2, 'processed_inair': 2, 'global_step': 1,
            'result': 'single batch passed; not a completed training epoch'}


def run(args):
    if (not args.run_id or args.run_id in ('.', '..') or
            any(x in args.run_id for x in '/\\:')):
        raise ValueError('run-id must be a single new directory name')
    if args.batch_size < 1 or args.num_workers < 0 or args.lr <= 0 or not 0 <= args.beta1 < 1:
        raise ValueError('Invalid batch/workers/lr/beta1')
    if args.fixed_epochs < 1 or args.decay_epochs < 1:
        raise ValueError('Invalid epoch schedule')
    if args.resume and args.mode != 'train':
        raise ValueError('Only completed train epochs can be resumed')
    if args.mode != 'data-check' and (args.vgg_weights is None or args.g_init is None):
        raise ValueError('Model modes require explicit --vgg-weights and --g-init')
    if args.mode == 'smoke' and (args.underwater_manifest is None or args.inair_manifest is None):
        raise ValueError('Smoke requires explicit underwater and in-air manifests')
    if args.mode == 'train' and (args.underwater_manifest is None or args.inair_manifest is None):
        raise ValueError('Training requires explicit approved domain manifests')
    run_dir = args.output_root.resolve() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    processed_u = processed_a = global_step = 0
    try:
        write_json(run_dir / 'run.json', provenance(args))
        # Only smoke may scope its gate to an explicit diagnostic manifest.
        # data-check/train must check the full domain even with manifests.
        paths_u = image_paths(args.underwater_root, args.underwater_manifest if args.mode == 'smoke' else None)
        paths_a = image_paths(args.inair_root, args.inair_manifest if args.mode == 'smoke' else None)
        if args.mode == 'smoke' and (len(paths_u) < 2 or len(paths_a) < 2):
            raise ValueError('Smoke manifests must each contain at least 2 distinct images')
        report = {'underwater': check_images(paths_u), 'inair': check_images(paths_a)}
        if args.mode == 'smoke':
            for domain in report.values():
                for record in domain['records']:
                    if min(record['width'], record['height']) >= 256:
                        try:
                            read_crop(record['path'])  # Validate every listed RGB/crop, not just the sampled batch.
                        except (OSError, ValueError) as exc:
                            domain['errors'].append({'path': record['path'], 'error': str(exc)})
        write_json(run_dir / 'data_check.json', report)
        ready = all(not d['too_small'] and not d['errors'] for d in report.values())
        print(json.dumps({'data_ready': ready, 'report': str(run_dir / 'data_check.json')}))
        if not ready:
            raise ValueError('data-check blocked: see ALL small/unreadable images in data_check.json; none removed')
        if args.mode == 'data-check':
            write_json(run_dir / 'end.json', {'end_utc': now(), 'status': 'data-check passed',
                       'processed_images': len(paths_u) + len(paths_a), 'failed': 0, 'skipped': 0})
            return
        if args.mode == 'train':
            # Keep the actual training selection and checkpoint config.data as before;
            # the full-domain gate above cannot be bypassed by that selection.
            paths_u = image_paths(args.underwater_root, args.underwater_manifest)
            paths_a = image_paths(args.inair_root, args.inair_manifest)
            report = {'underwater': check_images(paths_u), 'inair': check_images(paths_a)}
            write_json(run_dir / 'training_data_check.json', report)
            if any(d['too_small'] or d['errors'] for d in report.values()):
                raise ValueError('Training data-check blocked: invalid images in training manifests')
        vgg = args.vgg_weights.resolve(strict=True)
        config = {k: getattr(args, k) for k in ('mode', 'g_init', 'batch_size', 'lr', 'beta1',
                  'fixed_epochs', 'decay_epochs', 'num_workers', 'seed', 'device')}
        config.update({'vgg_weights': str(vgg), 'vgg_sha256': sha256(vgg),
                       'data': report, 'preprocess': 'RGB; crop256 only; [-1,1]',
                       'loss': 'mean L1; per-image CHW cosine; mean hinge; margin1; weights1/0.1',
                       'sampling': 'max lengths; cyclic underwater; random air with replacement; keep tail'})
        seed_everything(args.seed)
        device = torch.device(args.device)
        model = build_model(vgg, device, args.g_init)
        optimizer = make_optimizer(model.G, args.lr, args.beta1)
        scheduler = make_scheduler(optimizer, args.fixed_epochs, args.decay_epochs)
        config['adam_defaults'] = optimizer.defaults.copy()
        dataset = Stage1Dataset(paths_u, paths_a)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, worker_init_fn=seed_worker, drop_last=False)
        state_buffer = io.BytesIO()
        torch.save(model.G.state_dict(), state_buffer)
        write_json(run_dir / 'effective_config.json', {
            'training': config, 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'initial_G_sha256': hashlib.sha256(state_buffer.getvalue()).hexdigest(),
            'resume': str(args.resume) if args.resume else None})
        if args.mode == 'smoke':
            result = smoke(model, optimizer, scheduler, loader, device, run_dir, config)
            write_json(run_dir / 'end.json', dict(result, status='smoke passed', end_utc=now(), failed=0, skipped=0))
            print(json.dumps(result))
            return
        epoch = 0
        if args.resume:
            epoch, global_step = load_checkpoint(args.resume, model, optimizer, scheduler, config)
        total_epochs = args.fixed_epochs + args.decay_epochs
        if epoch >= total_epochs:
            raise ValueError('Checkpoint has already completed this schedule')
        with open(run_dir / 'metrics.jsonl', 'x', encoding='utf-8') as metrics:
            for current in range(epoch + 1, total_epochs + 1):
                for batch in loader:
                    losses = train_step(model, optimizer, batch['Xa'].to(device), batch['Xu'].to(device))
                    global_step += 1
                    processed_u += len(batch['Xu'])
                    processed_a += len(batch['Xa'])
                    record = dict(losses, epoch=current, global_step=global_step,
                                  lr=optimizer.param_groups[0]['lr'])
                    metrics.write(json.dumps(record) + '\n')
                metrics.flush()
                scheduler.step()  # Only after all optimizer updates in this epoch.
                checkpoint = run_dir / 'epoch_{:04d}.pth'.format(current)
                save_checkpoint(checkpoint, model, optimizer, scheduler, current, global_step, config)
                print(json.dumps({'completed_epoch': current, 'global_step': global_step,
                                  'checkpoint_sha256': sha256(checkpoint)}))
        write_json(run_dir / 'end.json', {'end_utc': now(), 'status': 'completed', 'global_step': global_step,
                   'processed_underwater': processed_u, 'processed_inair': processed_a, 'failed': 0, 'skipped': 0})
    except Exception as exc:
        write_json(run_dir / 'failure.json', {'end_utc': now(), 'status': 'failed', 'error': repr(exc),
                   'processed_underwater': processed_u, 'processed_inair': processed_a,
                   'global_step': global_step, 'failed': 1, 'skipped': 0})
        raise


if __name__ == '__main__':
    run(parser().parse_args())
