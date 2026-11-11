"""Independent Stage2 CLI; explicit mode, fixed manifests and non-overwriting runs."""
import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from torch.utils.data import DataLoader

from data.stage1_dataset import UVTD_REFLECT_POLICY, check_images, image_paths, read_crop
from data.stage2_dataset import Stage2Dataset
from stage1 import seed_everything, seed_worker
from stage2 import (CODE_VERSION, build_model, load_checkpoint, make_optimizer, make_scheduler,
                    read_stage1_checkpoint, save_checkpoint, sha256, train_step)


SOURCE_FILES = ('stage2.py', 'train_stage2.py', 'data/stage2_dataset.py', 'stage1.py',
                'data/stage1_dataset.py', 'models/DenseNet.py', 'models/vggloss.py',
                'models/NEDB_IN.py', 'models/FEM.py', 'CLIP/clip/model.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with open(path, 'x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('data-check', 'smoke', 'train'), required=True)
    p.add_argument('--underwater-root', type=Path, required=True)
    p.add_argument('--underwater-manifest', type=Path)
    p.add_argument('--vgg-weights', type=Path)
    p.add_argument('--clip-weights', type=Path)
    p.add_argument('--stage1-checkpoint', type=Path)
    p.add_argument('--stage1-acceptance', type=Path, help='Hash-bound epoch100 acceptance JSON for formal G')
    p.add_argument('--diagnostic-stage1', action='store_true', help='Smoke only; source remains explicitly diagnostic')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--fixed-epochs', type=int, default=100)
    p.add_argument('--decay-epochs', type=int, default=100)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--resume', type=Path, help='Completed Stage2 train epoch only')
    return p


def provenance(args):
    repo = Path(__file__).resolve().parent
    def git(*command):
        result = subprocess.run(['git', '-C', str(repo), *command], capture_output=True, text=True)
        return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    return {'start_utc': now(), 'purpose': 'Stage2 paper III.C (7)-(14)',
            'argv': sys.argv, 'arguments': vars(args), 'cwd': str(Path.cwd()),
            'executable': sys.executable, 'python': sys.version, 'platform': platform.platform(),
            'versions': {'torch': torch.__version__, 'torchvision': torchvision.__version__,
                         'numpy': np.__version__, 'Pillow': PIL.__version__},
            'git_head': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short'),
            'git_diff': git('diff', '--', '*.py'),
            'resume_source': {'path': str(args.resume.resolve()), 'sha256': sha256(args.resume)} if args.resume else None,
            'source_snapshot': {f: (repo/f).read_text(encoding='utf-8') for f in SOURCE_FILES},
            'source_sha256': {f: sha256(repo/f) for f in SOURCE_FILES},
            'forward': 'E(Xu)->independent C/D; G(Fc)->CLIP234/E->G(Fidt); E/G/CLIP frozen',
            'unused_components': 'author NC/LOW/HIGH2/HIGH3/UNet/task weights not loaded'}


def state_snapshot(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def state_fingerprint(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(repr((name, str(value.dtype), tuple(value.shape))).encode('utf-8'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def smoke(model, optimizer, scheduler, loader, device, run_dir, config):
    batch = next(iter(loader))
    xu = batch['Xu'].to(device)
    if tuple(xu.shape) != (2, 3, 256, 256):
        raise ValueError('Smoke requires exactly two RGB 256x256 images')
    frozen = {name: state_snapshot(getattr(model, name)) for name in ('E', 'G', 'clip')}
    before = {name: state_snapshot(getattr(model, name)) for name in ('C', 'D')}
    outputs, content_features, clip_features, e_shapes = [], [], [], []
    def encoder_hook(module, inputs, result):
        shapes = [list(x.shape) for x in result]
        if shapes != [[2, 64, 256, 256], [2, 128, 128, 128], [2, 256, 64, 64]]:
            raise AssertionError('Real VGG three-scale contract failed')
        e_shapes.append(shapes)
    def generator_hook(module, inputs, result):
        if result.shape != xu.shape:
            raise AssertionError('SAD output shape failed')
        result.retain_grad()
        outputs.append(result)
    def content_hook(module, inputs, result):
        for x in result:
            x.retain_grad()
        content_features.extend(result)
    def clip_hook(module, inputs, result):
        if [list(x.shape) for x in result] != [[2, 512, 28, 28], [2, 1024, 14, 14], [2, 2048, 7, 7]]:
            raise AssertionError('RN50 layer2/3/4 spatial contract failed')
        if result[0].requires_grad:
            for x in result:
                x.retain_grad()
            clip_features.extend(result)
    handles = [model.E.register_forward_hook(encoder_hook), model.G.register_forward_hook(generator_hook),
               model.C.register_forward_hook(content_hook), model.clip.register_forward_hook(clip_hook)]
    try:
        losses = train_step(model, optimizer, xu)
    finally:
        for handle in handles:
            handle.remove()
    if len(outputs) != 2 or len(e_shapes) != 2 or len(content_features) != 3 or len(clip_features) != 3:
        raise AssertionError('Unexpected Stage2 forward branches')
    for x in outputs + content_features + clip_features:
        if x.grad is None or not torch.isfinite(x.grad).all() or not x.grad.ne(0).any():
            raise AssertionError('Generated branches require finite nonzero input gradients')
    del outputs, content_features, clip_features
    for name in ('E', 'G', 'clip'):
        module = getattr(model, name)
        if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise AssertionError('Frozen module mode/grad failed: '+name)
        if not equal_state(frozen[name], module.state_dict()):
            raise AssertionError('Frozen parameter/buffer changed: '+name)
    changed = {name: sum(not torch.equal(value, before[name][key])
                        for key, value in state_snapshot(getattr(model, name)).items()) for name in ('C', 'D')}
    if not all(changed.values()):
        raise AssertionError('Both C and D must update')
    expected_cd = {name: state_snapshot(getattr(model, name)) for name in ('C', 'D')}
    expected_optimizer = optimizer.state_dict()
    expected_scheduler = scheduler.state_dict()
    path = run_dir/'smoke_checkpoint.pth'
    save_checkpoint(path, model, optimizer, scheduler, 0, 1, config)
    with torch.no_grad():
        fu = model.E(xu)
        reference = model.G(model.C(fu))
        for module in (model.C, model.D):
            for parameter in module.parameters():
                parameter.zero_()
    optimizer.state.clear()
    optimizer.param_groups[0]['lr'] = 9.0
    scheduler.last_epoch = 999
    epoch, step = load_checkpoint(path, model, optimizer, scheduler, config, mode='smoke')
    with torch.no_grad():
        restored = model.G(model.C(fu))
    if (epoch != 0 or step != 1 or not torch.allclose(reference, restored, atol=1e-6, rtol=1e-5) or
            any(not equal_state(expected_cd[name], getattr(model, name).state_dict()) for name in ('C', 'D')) or
            not equal_state(expected_optimizer, optimizer.state_dict()) or not equal_state(expected_scheduler, scheduler.state_dict())):
        raise AssertionError('Stage2 checkpoint recovery failed')
    return {'losses': losses, 'Xu_paths': batch['Xu_path'], 'vgg_shapes': e_shapes[0],
            'clip_shapes': [[2, 512, 28, 28], [2, 1024, 14, 14], [2, 2048, 7, 7]],
            'changed_C': changed['C'], 'changed_D': changed['D'], 'checkpoint_recovery': True,
            'checkpoint_sha256': sha256(path), 'global_step': 1, 'processed_images': 2,
            'diagnostic_stage1': config['diagnostic_stage1'], 'note': 'one batch; no completed training epoch'}


def validate_args(args):
    if not args.run_id or args.run_id in ('.', '..') or any(c in args.run_id for c in '/\\:'):
        raise ValueError('run-id must be a single new directory name')
    if (args.batch_size < 1 or args.num_workers < 0 or not math.isfinite(args.lr) or args.lr <= 0 or
            args.fixed_epochs < 1 or args.decay_epochs < 1 or args.seed < 0):
        raise ValueError('Invalid batch/workers/lr/schedule/seed')
    if args.resume and args.mode != 'train':
        raise ValueError('Resume is only for completed Stage2 train epochs')
    if args.diagnostic_stage1 and args.mode != 'smoke':
        raise ValueError('diagnostic Stage1 weights are smoke-only')
    if args.mode in ('smoke', 'train'):
        if args.underwater_manifest is None:
            raise ValueError('Model modes require an explicit fixed underwater manifest')
        if any(getattr(args, key) is None for key in ('vgg_weights', 'clip_weights', 'stage1_checkpoint')):
            raise ValueError('Model modes require explicit VGG, CLIP and Stage1 checkpoint paths')
        if args.batch_size != 2:
            raise ValueError('Paper/real-model smoke batch size is 2')
    if args.mode == 'train' and (args.lr != 1e-4 or args.fixed_epochs != 100 or args.decay_epochs != 100):
        raise ValueError('Formal Stage2 requires paper lr1e-4 and 100+100 schedule')


def run(args):
    validate_args(args)
    run_dir = args.output_root.resolve()/args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    processed = global_step = 0
    try:
        origin = provenance(args)
        write_json(run_dir/'run.json', origin)
        paths = image_paths(args.underwater_root, args.underwater_manifest if args.mode == 'smoke' else None)
        if args.mode == 'smoke' and len(paths) < 2:
            raise ValueError('Smoke manifest must contain at least two distinct images')
        report = check_images(paths, allow_uvtd_reflect=True)
        if args.mode == 'smoke':
            old_random = random.getstate()
            try:
                random.seed(args.seed)
                for path in paths:
                    try:
                        read_crop(path, allow_uvtd_reflect=True)
                    except (OSError, ValueError) as exc:
                        report['errors'].append({'path': str(path), 'error': str(exc)})
            finally:
                random.setstate(old_random)
        write_json(run_dir/'data_check.json', report)
        ready = not report['too_small'] and not report['errors']
        print(json.dumps({'data_ready': ready, 'report': str(run_dir/'data_check.json')}))
        if not ready:
            raise ValueError('data-check blocked; no images filtered, see complete report')
        if args.mode == 'data-check':
            write_json(run_dir/'end.json', {'status': 'data-check passed', 'end_utc': now(),
                       'processed_images': len(paths), 'failed': 0, 'skipped': 0})
            return
        selected = image_paths(args.underwater_root, args.underwater_manifest)
        if args.mode == 'train' and set(selected) != set(paths):
            raise ValueError('Train manifest must cover the full domain; no implicit filtering/split change')
        selected_report = check_images(selected, allow_uvtd_reflect=True)
        write_json(run_dir/'selected_data_check.json', selected_report)
        if selected_report['too_small'] or selected_report['errors']:
            raise ValueError('Selected manifest data-check blocked')
        inventory = [{'path': str(p), 'sha256': sha256(p)} for p in selected]
        manifest_fingerprint = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode('utf-8')).hexdigest()
        write_json(run_dir/'data_inventory.json', {'images': inventory, 'fingerprint': manifest_fingerprint})
        vgg, clip, g = (getattr(args, name).resolve(strict=True) for name in ('vgg_weights', 'clip_weights', 'stage1_checkpoint'))
        acceptance = json.loads(args.stage1_acceptance.read_text(encoding='utf-8')) if args.stage1_acceptance else None
        source = read_stage1_checkpoint(g, sha256(vgg), diagnostic=args.diagnostic_stage1, acceptance=acceptance)
        if args.mode == 'train':
            source_records = source['config'].get('data', {}).get('underwater', {}).get('records', [])
            if {str(p) for p in selected} != {r.get('path') for r in source_records}:
                raise ValueError('Stage1 actual underwater manifest/config does not match Stage2 full manifest')
        source_metadata = {key: source[key] for key in ('epoch', 'global_step', 'config')}
        del source
        seed_everything(args.seed)
        # Explicit user precision constraints, not a throughput optimization.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device = torch.device(args.device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('Stage2 supports CPU validation or single CUDA GPU only')
        model = build_model(vgg, clip, g, device, diagnostic=args.diagnostic_stage1, acceptance=acceptance)
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('Stage2 requires FP32 model parameters')
        optimizer = make_optimizer(model, args.lr)
        scheduler = make_scheduler(optimizer, args.fixed_epochs, args.decay_epochs)
        config = {key: getattr(args, key) for key in ('mode', 'batch_size', 'lr', 'fixed_epochs', 'decay_epochs',
                  'num_workers', 'seed', 'device', 'diagnostic_stage1')}
        config.update({'code_version': CODE_VERSION, 'code_sha256': origin['source_sha256'],
            'git_commit': origin['git_head']['stdout'].strip(),
            'weights': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in (('G', g), ('VGG', vgg), ('CLIP', clip))},
            'stage1_metadata': source_metadata, 'stage1_acceptance': acceptance,
            'manifest': {'path': str(args.underwater_manifest.resolve()), 'file_sha256': sha256(args.underwater_manifest),
                         'fingerprint': manifest_fingerprint, 'count': len(selected)},
            'preprocess': 'RGB; {}; random crop256; FP32 [-1,1]'.format(UVTD_REFLECT_POLICY),
            'clip_preprocess': '[-1,1]->[0,1]; bicubic224 align_corners=False antialias=True; no clamp; CLIP mean/std',
            'clip_layers': ['visual.layer2', 'visual.layer3', 'visual.layer4'],
            'loss': 'per-tensor mean L1, sum scales; sep+clip+2*idt1+idt2',
            'initialization': 'independent original FE constructors; D symmetric engineering reconstruction',
            'initial_parameters_sha256': {name: state_fingerprint(getattr(model, name)) for name in ('C', 'D')},
            'software': {'python': sys.version, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
                         'numpy': np.__version__, 'Pillow': PIL.__version__, 'cuda': torch.version.cuda},
            'sampling': 'one pass fixed manifest per epoch; shuffle=True; keep tail; nonpersistent workers',
            'adam_defaults': optimizer.defaults.copy(), 'precision': 'FP32; AMP/TF32/compile/DDP disabled',
            'determinism': {'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}})
        loader = DataLoader(Stage2Dataset(selected), batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, worker_init_fn=seed_worker, drop_last=False)
        write_json(run_dir/'effective_config.json', {'training': config, 'cuda': torch.version.cuda,
                   'cudnn': torch.backends.cudnn.version(), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                   'resume': str(args.resume) if args.resume else None})
        if args.mode == 'smoke':
            result = smoke(model, optimizer, scheduler, loader, device, run_dir, config)
            write_json(run_dir/'metrics.json', result['losses'])
            write_json(run_dir/'end.json', dict(result, status='smoke passed', end_utc=now(), failed=0, skipped=0))
            print(json.dumps(result))
            return
        epoch = 0
        if args.resume:
            epoch, global_step = load_checkpoint(args.resume, model, optimizer, scheduler, config)
        total_epochs = args.fixed_epochs + args.decay_epochs
        if epoch >= total_epochs:
            raise ValueError('Stage2 checkpoint already completed this schedule')
        with open(run_dir/'metrics.jsonl', 'x', encoding='utf-8') as metrics:
            for current in range(epoch + 1, total_epochs + 1):
                for batch in loader:
                    values = train_step(model, optimizer, batch['Xu'].to(device))
                    global_step += 1
                    processed += len(batch['Xu'])
                    metrics.write(json.dumps(dict(values, epoch=current, global_step=global_step,
                                                  lr=optimizer.param_groups[0]['lr']), allow_nan=False)+'\n')
                metrics.flush()
                scheduler.step()
                checkpoint = run_dir/'epoch_{:04d}.pth'.format(current)
                save_checkpoint(checkpoint, model, optimizer, scheduler, current, global_step, config)
                print(json.dumps({'completed_epoch': current, 'global_step': global_step,
                                  'checkpoint_sha256': sha256(checkpoint)}))
        write_json(run_dir/'end.json', {'status': 'completed', 'end_utc': now(), 'global_step': global_step,
                   'processed_images': processed, 'failed': 0, 'skipped': 0})
    except Exception as exc:
        write_json(run_dir/'failure.json', {'status': 'failed', 'end_utc': now(), 'error': repr(exc),
                   'global_step': global_step, 'processed_images': processed, 'failed': 1, 'skipped': 0})
        raise


if __name__ == '__main__':
    run(parser().parse_args())
