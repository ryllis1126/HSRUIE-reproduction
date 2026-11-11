"""CPU contracts: paper arithmetic, graph paths, frozen state and epoch recovery."""
import copy
import hashlib
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data.stage2_dataset import Stage2Dataset
from stage2 import (Stage2, ClipSpatialFeatures, clip_preprocess, stage2_losses,
                    make_optimizer, make_scheduler, train_step, save_checkpoint,
                    load_checkpoint, read_stage1_checkpoint, load_clip_rn50)
from stage1 import seed_worker


class TinyE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        f = self.bn(self.conv(x))
        return [f, F.avg_pool2d(f, 2), F.avg_pool2d(f, 4)]


class TinyExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv2d(4, 4, 1) for _ in range(3)])

    def forward(self, fs):
        return [m(f) for m, f in zip(self.branches, fs)]


class TinyG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 3, 1)

    def forward(self, fs):
        x = sum(F.interpolate(f, size=fs[0].shape[-2:], mode='nearest') for f in fs)
        return self.conv(x).tanh()


class TinyVisual(nn.Module):
    """Deliberately returns only a pooled embedding, unlike required spatial maps."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 4, 1)
        self.layer2 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
        self.layer3 = nn.Conv2d(4, 4, 1)
        self.layer4 = nn.Conv2d(4, 4, 1)

    def forward(self, x):
        x = self.layer1(F.avg_pool2d(x, 28))
        x = self.layer2(F.avg_pool2d(x, 2))
        x = self.layer3(F.avg_pool2d(x, 2))
        x = self.layer4(F.avg_pool2d(x, 2))
        return x.mean((2, 3))


def tiny_model():
    return Stage2(TinyE(), TinyG(), ClipSpatialFeatures(TinyVisual()),
                  TinyExtractor(), TinyExtractor())


def state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


class Stage2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(711)
        random.seed(711)
        np.random.seed(711)

    def test_original_fe_branches_and_independence(self):
        from models.DenseNet import ResBlock
        model = Stage2(TinyE(), TinyG(), ClipSpatialFeatures(TinyVisual()))
        for extractor in (model.C, model.D):
            self.assertEqual(sum(isinstance(m, ResBlock) for m in extractor.modules()), 18)
            self.assertEqual([len(extractor.conv11), len(extractor.conv12), len(extractor.conv13)], [6, 6, 6])
            self.assertEqual(sum(p.numel() for p in extractor.parameters()), 9300480)
            fs = [torch.randn(2, 64, 8, 8), torch.randn(2, 128, 4, 4), torch.randn(2, 256, 2, 2)]
            self.assertEqual([f.shape for f in extractor(fs)], [f.shape for f in fs])
        self.assertFalse({id(p) for p in model.C.parameters()} & {id(p) for p in model.D.parameters()})

    def loss_args(self):
        fs = lambda value: [torch.full((2, 4, 4 // (2 ** i), 4 // (2 ** i)), float(value)) for i in range(3)]
        return [fs(3), fs(1), fs(1), fs(2), fs(0), fs(4),
                torch.ones(2, 3, 4, 4), torch.zeros(2, 3, 4, 4)]

    def test_four_losses_hand_calculated_mean_and_weights(self):
        result = stage2_losses(*self.loss_args())
        self.assertEqual({k: float(v) for k, v in result.items()},
                         {'sep': 3.0, 'clip': 6.0, 'idt1': 9.0, 'idt2': 1.0, 'total': 28.0})

    def test_separation_descent_reduces_residual(self):
        args = self.loss_args()
        args[1] = [x.requires_grad_() for x in args[1]]
        args[2] = [x.requires_grad_() for x in args[2]]
        before = stage2_losses(*args)['sep']
        before.backward()
        after = [x.detach() - 0.1 * x.grad for x in args[1]]
        self.assertTrue(all((x.grad < 0).all() for x in args[1] + args[2]))
        args[1] = after
        self.assertLess(float(stage2_losses(*args)['sep']), float(before))

    def test_all_scale_shape_errors_rejected_instead_of_broadcast(self):
        for group in range(6):
            for scale in range(3):
                args = self.loss_args()
                args[group][scale] = args[group][scale][:1]
                with self.subTest(group=group, scale=scale), self.assertRaises(ValueError):
                    stage2_losses(*args)
        args = self.loss_args()
        args[6] = args[6][:1]
        with self.assertRaises(ValueError):
            stage2_losses(*args)

    def test_missing_scales_and_nonfinite_features_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            for group in range(8):
                args = self.loss_args()
                tensor = args[group][0] if group < 6 else args[group]
                tensor.flatten()[0] = value
                with self.subTest(group=group, value=value), self.assertRaises(ValueError):
                    stage2_losses(*args)
        args = self.loss_args()
        args[0] = args[0][:2]
        with self.assertRaises(ValueError):
            stage2_losses(*args)

    def test_clip_range_channel_order_and_gradients(self):
        x = torch.empty(1, 3, 8, 8)
        x[:, 0], x[:, 1], x[:, 2] = -1, 0, 1
        x.requires_grad_()
        y = clip_preprocess(x)
        self.assertEqual(tuple(y.shape), (1, 3, 224, 224))
        want = [-0.48145466/0.26862954, (0.5-0.4578275)/0.26130258,
                (1-0.40821073)/0.27577711]
        for channel in range(3):
            self.assertTrue(torch.allclose(y[:, channel], torch.full_like(y[:, channel], want[channel]), atol=2e-6))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_clip_invalid_inputs_rejected(self):
        for x in (torch.zeros(1, 3, 8, 8, dtype=torch.float64), torch.full((1, 3, 8, 8), 2.0),
                  torch.full((1, 3, 8, 8), float('nan'))):
            with self.assertRaises(ValueError):
                clip_preprocess(x)

    def test_nonconstant_clip_resize_matches_torchvision_reference_and_input_gradient(self):
        from torchvision.transforms import functional as TF, InterpolationMode
        x = (torch.arange(256 * 256).reshape(1, 1, 256, 256) % 17).float().div(8).sub(1)
        x = x.expand(1, 3, -1, -1).clone().requires_grad_()
        reference_input = x.detach().clone().requires_grad_()
        actual = clip_preprocess(x)
        reference = TF.normalize(TF.resize((reference_input + 1) / 2, [224, 224],
            interpolation=InterpolationMode.BICUBIC, antialias=True),
            [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.assertTrue(torch.equal(actual, reference))
        actual.square().mean().backward()
        reference.square().mean().backward()
        self.assertTrue(torch.equal(x.grad, reference_input.grad))

    def test_spatial_hooks_preserve_graph_and_do_not_return_embedding(self):
        visual = TinyVisual().eval().requires_grad_(False)
        adapter = ClipSpatialFeatures(visual)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        fs = adapter(x)
        self.assertEqual([tuple(f.shape) for f in fs], [(2, 4, 4, 4), (2, 4, 2, 2), (2, 4, 1, 1)])
        sum(f.square().mean() for f in fs).backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in visual.parameters()))
        with torch.no_grad():
            l2 = visual.layer2(F.avg_pool2d(visual.layer1(F.avg_pool2d(x, 28)), 2))
            l3 = visual.layer3(F.avg_pool2d(l2, 2))
            l4 = visual.layer4(F.avg_pool2d(l3, 2))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(fs, (l2, l3, l4))))

    def test_each_loss_reaches_c_and_only_separation_reaches_d(self):
        for name in ('sep', 'clip', 'idt1', 'idt2'):
            model = tiny_model().train()
            generated = []
            handle = model.G.register_forward_hook(lambda module, inputs, output: generated.append(output)) if name == 'idt2' else None
            try:
                loss = model(torch.rand(2, 3, 8, 8) * 2 - 1)[name]
            finally:
                if handle is not None:
                    handle.remove()
            expected_c = None
            if name == 'idt2':
                self.assertEqual(len(generated), 2)
                xc, xidt = generated
                self.assertTrue(xc.requires_grad and xidt.requires_grad)
                # Mean L1: s=sign(Xidt-Xc)/N; dL/dXc=J_(G E)^T s-s.
                sign = (xidt.detach() - xc.detach()).sign() / xc.numel()
                observed_xidt, observed_xc = torch.autograd.grad(
                    loss, (xidt, xc), retain_graph=True, allow_unused=True)
                self.assertIsNotNone(observed_xidt, 'idt2 generated Xidt argument was disconnected')
                self.assertIsNotNone(observed_xc, 'idt2 must reach Xc')
                self.assertTrue(torch.equal(observed_xidt, sign))
                generated_at_xc = torch.autograd.grad(xidt, xc, grad_outputs=sign, retain_graph=True)[0]
                self.assertTrue(torch.isfinite(generated_at_xc).all())
                self.assertGreater(generated_at_xc.abs().sum().item(), 0)
                self.assertTrue(torch.allclose(observed_xc, generated_at_xc - sign, atol=1e-7, rtol=1e-5),
                                'idt2 must include both the direct -s and generated J^T s terms')
                parameters = list(model.C.parameters())
                direct_c = torch.autograd.grad(xc, parameters, grad_outputs=-sign, retain_graph=True)
                generated_c = torch.autograd.grad(xidt, parameters, grad_outputs=sign, retain_graph=True)
                for branch in (direct_c, generated_c):
                    self.assertTrue(all(torch.isfinite(g).all() for g in branch))
                    self.assertGreater(sum(g.abs().sum().item() for g in branch), 0)
                expected_c = [direct + indirect for direct, indirect in zip(direct_c, generated_c)]
            loss.backward()
            if expected_c is not None:
                for parameter, expected in zip(model.C.parameters(), expected_c):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.allclose(parameter.grad, expected, atol=1e-7, rtol=1e-5))
            cgrads = [p.grad for p in model.C.parameters() if p.grad is not None]
            self.assertTrue(cgrads, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in cgrads), name)
            self.assertGreater(sum(g.abs().sum().item() for g in cgrads), 0, name)
            dgrads = [p.grad for p in model.D.parameters() if p.grad is not None]
            if name == 'sep':
                self.assertGreater(sum(g.abs().sum().item() for g in dgrads), 0)
            else:
                self.assertEqual(dgrads, [])
            for fixed in (model.E, model.G, model.clip):
                self.assertFalse(fixed.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in fixed.parameters()))

    def test_optimizer_updates_c_d_and_preserves_all_frozen_buffers(self):
        model = tiny_model()
        before = copy.deepcopy(model.state_dict())
        optimizer = make_optimizer(model)
        owned = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertEqual(owned, {id(p) for m in (model.C, model.D) for p in m.parameters()})
        metrics = train_step(model, optimizer, torch.rand(2, 3, 8, 8) * 2 - 1)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        for prefix in ('E.', 'G.', 'clip.'):
            for name, value in model.state_dict().items():
                if name.startswith(prefix):
                    self.assertTrue(torch.equal(value, before[name]), name)
        for prefix in ('C.', 'D.'):
            self.assertTrue(any(not torch.equal(value, before[name]) for name, value in model.state_dict().items() if name.startswith(prefix)))

    def test_idt1_generated_encoder_arm_matches_full_reference_gradient(self):
        actual = tiny_model()
        reference = copy.deepcopy(actual)
        x = torch.rand(2, 3, 8, 8) * 2 - 1
        actual(x)['idt1'].backward()
        fc = reference.C(reference.E(x))
        fidt = reference.E(reference.G(fc))
        correct = sum(F.l1_loss(i, c) for i, c in zip(fidt, fc))
        detached_arm = sum(F.l1_loss(i.detach(), c) for i, c in zip(fidt, fc))
        parameters = list(reference.C.parameters())
        detached_grads = torch.autograd.grad(detached_arm, parameters, retain_graph=True)
        correct.backward()
        self.assertTrue(any(not torch.equal(p.grad, detached) for p, detached in zip(parameters, detached_grads)))
        for observed, expected in zip(actual.C.parameters(), parameters):
            self.assertTrue(torch.equal(observed.grad, expected.grad))

    def test_train_step_rejects_optimizer_containing_frozen_parameters(self):
        model = tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with self.assertRaises(ValueError):
            train_step(model, optimizer, torch.zeros(2, 3, 8, 8))

    def test_raw_target_no_grad_matches_reference_values_and_gradients(self):
        optimized = tiny_model()
        reference = copy.deepcopy(optimized)
        x = torch.rand(2, 3, 8, 8) * 2 - 1
        a = optimized(x)
        fu = reference.E(x)
        fc, fd = reference.C(fu), reference.D(fu)
        xc = reference.G(fc)
        fidt = reference.E(xc)
        b = stage2_losses(fu, fc, fd, reference.clip(clip_preprocess(x)),
                          reference.clip(clip_preprocess(xc)), fidt, reference.G(fidt), xc)
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)
        a['total'].backward()
        b['total'].backward()
        for p, q in zip(optimized.C.parameters(), reference.C.parameters()):
            self.assertTrue(torch.equal(p.grad, q.grad))
        for p, q in zip(optimized.D.parameters(), reference.D.parameters()):
            self.assertTrue(torch.equal(p.grad, q.grad))

    def test_dataset_rgb_crop_reflection_and_source_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index, size in enumerate(((352, 224), (224, 352), (300, 280))):
                path = Path(tmp)/('{}.png'.format(index))
                Image.fromarray(np.arange(size[0]*size[1], dtype=np.uint8).reshape(size[1], size[0])).save(path)
                paths.append(path)
            original = [p.read_bytes() for p in paths]
            ds = Stage2Dataset(paths)
            self.assertEqual(len(ds), 3)
            for i in range(3):
                sample = ds[i]
                self.assertEqual(tuple(sample['Xu'].shape), (3, 256, 256))
                self.assertEqual(sample['Xu'].dtype, torch.float32)
                self.assertGreaterEqual(sample['Xu'].min().item(), -1)
                self.assertLessEqual(sample['Xu'].max().item(), 1)
                self.assertTrue(torch.equal(sample['Xu'][0], sample['Xu'][1]))
            self.assertEqual(original, [p.read_bytes() for p in paths])

    def test_dataset_rejects_unknown_short_and_corrupt_without_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            small, bad = Path(tmp)/'small.png', Path(tmp)/'bad.png'
            Image.new('RGB', (255, 300)).save(small)
            bad.write_bytes(b'not an image')
            ds = Stage2Dataset([small, bad])
            self.assertEqual(len(ds), 2)
            for i in range(2):
                with self.assertRaises((OSError, ValueError)):
                    ds[i]

    def test_epoch_checkpoint_true_next_step_and_rng_equivalence(self):
        model = tiny_model()
        clone = copy.deepcopy(model)
        optimizer = make_optimizer(model)
        scheduler = make_scheduler(optimizer, 100, 100)
        train_step(model, optimizer, torch.rand(2, 3, 8, 8) * 2 - 1)
        scheduler.step()
        config = {'mode': 'train', 'fingerprints': {'g': 'g', 'vgg': 'v', 'clip': 'c', 'manifest': 'm'}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'epoch.pth'
            save_checkpoint(path, model, optimizer, scheduler, 1, 1, config)
            expected_random = (random.random(), np.random.rand())
            expected = train_step(model, optimizer, torch.rand(2, 3, 8, 8) * 2 - 1)
            scheduler.step()
            op2 = make_optimizer(clone)
            sc2 = make_scheduler(op2, 100, 100)
            self.assertEqual(load_checkpoint(path, clone, op2, sc2, config), (1, 1))
            self.assertEqual((random.random(), np.random.rand()), expected_random)
            recovered = train_step(clone, op2, torch.rand(2, 3, 8, 8) * 2 - 1)
            sc2.step()
            self.assertEqual(expected, recovered)
            self.assertTrue(state_equal(model.state_dict(), clone.state_dict()))
            self.assertTrue(state_equal(optimizer.state_dict(), op2.state_dict()))
            self.assertTrue(state_equal(scheduler.state_dict(), sc2.state_dict()))
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model, optimizer, scheduler, 2, 2, config)
            with self.assertRaises(ValueError):
                load_checkpoint(path, clone, op2, sc2, {'mode': 'train', 'fingerprints': {}})
            payload = torch.load(path, map_location='cpu', weights_only=False)
            payload['C'].pop(next(iter(payload['C'])))
            broken = Path(tmp)/'broken.pth'
            torch.save(payload, broken)
            with self.assertRaises(RuntimeError):
                load_checkpoint(broken, clone, op2, sc2, config)

    def test_smoke_checkpoint_cannot_resume_train(self):
        model = tiny_model()
        optimizer = make_optimizer(model)
        scheduler = make_scheduler(optimizer, 100, 100)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'smoke.pth'
            save_checkpoint(path, model, optimizer, scheduler, 0, 1, {'mode': 'smoke'})
            with self.assertRaises(ValueError):
                load_checkpoint(path, model, optimizer, scheduler, {'mode': 'smoke'})

    def test_shuffled_random_crop_epoch_resume_with_zero_and_one_worker(self):
        for workers in (0, 1):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as tmp:
                paths = []
                pixels = np.random.RandomState(71).randint(0, 256, (270, 272, 3), dtype=np.uint8)
                for index in range(3):
                    path = Path(tmp)/('{}.png'.format(index))
                    Image.fromarray(np.roll(pixels, index * 9, axis=0)).save(path)
                    paths.append(path)
                def loader():
                    return DataLoader(Stage2Dataset(paths), batch_size=2, shuffle=True,
                                      num_workers=workers, worker_init_fn=seed_worker, drop_last=False)
                model = tiny_model()
                restored = copy.deepcopy(model)
                op = make_optimizer(model)
                sch = make_scheduler(op, 100, 100)
                original_loader = loader()
                for batch in original_loader:
                    train_step(model, op, batch['Xu'])
                sch.step()
                config = {'mode': 'train', 'workers': workers}
                checkpoint = Path(tmp)/'epoch1.pth'
                save_checkpoint(checkpoint, model, op, sch, 1, 2, config)
                expected = []
                for batch in original_loader:
                    expected.append((batch['Xu_path'], batch['Xu'].clone(), train_step(model, op, batch['Xu'])))
                sch.step()
                resumed_loader = loader()
                op2 = make_optimizer(restored)
                sch2 = make_scheduler(op2, 100, 100)
                load_checkpoint(checkpoint, restored, op2, sch2, config)
                recovered = []
                for batch in resumed_loader:
                    recovered.append((batch['Xu_path'], batch['Xu'].clone(), train_step(restored, op2, batch['Xu'])))
                sch2.step()
                self.assertEqual(len(expected), 2)
                self.assertEqual(len(recovered), 2)
                for a, b in zip(expected, recovered):
                    self.assertEqual(a[0], b[0])
                    self.assertTrue(torch.equal(a[1], b[1]))
                    self.assertEqual(a[2], b[2])
                self.assertTrue(state_equal(model.state_dict(), restored.state_dict()))
                self.assertTrue(state_equal(op.state_dict(), op2.state_dict()))
                self.assertTrue(state_equal(sch.state_dict(), sch2.state_dict()))

    def test_lr_epoch_boundaries_leave_final_update_nonzero(self):
        parameter = nn.Parameter(torch.ones(()))
        optimizer = torch.optim.Adam([parameter], lr=1e-4)
        scheduler = make_scheduler(optimizer, 100, 100)
        used = []
        for _ in range(200):
            used.append(optimizer.param_groups[0]['lr'])
            optimizer.zero_grad()
            parameter.square().backward()
            optimizer.step()
            scheduler.step()
        for epoch, expected in ((1, 1e-4), (100, 1e-4), (101, 1e-4), (102, 9.9e-5), (200, 1e-6)):
            self.assertAlmostEqual(used[epoch-1], expected, places=12)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0)

    def test_stage1_source_diagnostic_vs_formal_and_vgg_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'g.pth'
            config = {'mode': 'train', 'fixed_epochs': 50, 'decay_epochs': 50, 'vgg_sha256': 'vgg'}
            payload = {'G': {'weight': torch.ones(1)}, 'optimizer': {}, 'scheduler': {},
                       'epoch': 99, 'global_step': 200, 'rng': {}, 'config': config}
            torch.save(payload, path)
            self.assertIn('weight', read_stage1_checkpoint(path, 'vgg', diagnostic=True)['G'])
            with self.assertRaises(ValueError):
                read_stage1_checkpoint(path, 'vgg')
            payload['epoch'] = 100
            torch.save(payload, path)
            acceptance = {'stage': 'stage1', 'accepted': True, 'completed_epoch': 100,
                          'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'vgg_sha256': 'vgg', 'basis': 'synthetic test acceptance, not a real G'}
            self.assertEqual(read_stage1_checkpoint(path, 'vgg', acceptance=acceptance)['epoch'], 100)
            with self.assertRaises(ValueError):
                read_stage1_checkpoint(path, 'other', diagnostic=True)
            acceptance['checkpoint_sha256'] = 'wrong'
            with self.assertRaises(ValueError):
                read_stage1_checkpoint(path, 'vgg', acceptance=acceptance)

    def test_clip_weight_hash_failure_precedes_archive_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'clip.pt'
            path.write_bytes(b'not a trusted archive')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                load_clip_rn50(path, 'cpu')

    def test_actual_sad_stage1_package_load_is_strict_and_does_not_load_full_package_as_state(self):
        from models.DenseNet import SAD
        from stage2 import build_model
        generator = SAD()
        with tempfile.TemporaryDirectory() as tmp:
            vgg, clip, source = (Path(tmp)/name for name in ('vgg.pth', 'clip.pt', 'source.pth'))
            vgg.write_bytes(b'synthetic encoder fixture')
            clip.write_bytes(b'synthetic clip fixture')
            payload = {'G': generator.state_dict(), 'optimizer': {}, 'scheduler': {}, 'rng': {},
                       'epoch': 0, 'global_step': 1,
                       'config': {'vgg_sha256': hashlib.sha256(vgg.read_bytes()).hexdigest()}}
            torch.save(payload, source)
            with patch('models.vggloss.Vgg19_out', return_value=TinyE()), patch('stage2.load_clip_rn50', return_value=ClipSpatialFeatures(TinyVisual())):
                model = build_model(vgg, clip, source, 'cpu', diagnostic=True)
                self.assertTrue(state_equal(generator.state_dict(), model.G.state_dict()))
                self.assertTrue(all(not p.requires_grad for p in model.G.parameters()))
                payload['G'].pop(next(iter(payload['G'])))
                torch.save(payload, source)
                with self.assertRaises(RuntimeError):
                    build_model(vgg, clip, source, 'cpu', diagnostic=True)


if __name__ == '__main__':
    unittest.main()
