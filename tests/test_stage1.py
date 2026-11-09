"""CPU synthetic contracts, not a paper reproduction/3090 acceptance run."""
import copy
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data.stage1_dataset import Stage1Dataset, check_images, image_paths, read_crop
from stage1 import (Stage1, load_checkpoint, make_optimizer, make_scheduler,
                    ranking_loss, save_checkpoint, seed_everything, stage1_losses, train_step)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        return self.conv(F.adaptive_avg_pool2d(x, (8, 8)))


class TinyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 3, 1)

    def forward(self, features):
        return F.interpolate(self.conv(features), size=(256, 256), mode='nearest').tanh()


def tiny_model():
    return Stage1(TinyEncoder(), TinyGenerator())


class Stage1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        seed_everything(123)

    def test_dataset_formats_domains_and_batch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            u, a = root / 'u', root / 'a'
            u.mkdir()
            a.mkdir()
            Image.new('RGB', (300, 280), (255, 0, 0)).save(u / 'u.bmp')
            Image.new('RGB', (256, 256), (0, 255, 0)).save(u / 'v.jpg')
            Image.new('L', (280, 300), 0).save(a / 'a.png')
            ds = Stage1Dataset(image_paths(u), image_paths(a))
            batch = next(iter(DataLoader(ds, batch_size=2)))
            self.assertEqual(len(ds), 2)
            for key in ('Xu', 'Xa'):
                self.assertEqual(tuple(batch[key].shape), (2, 3, 256, 256))
                self.assertEqual(batch[key].dtype, torch.float32)
                self.assertTrue(torch.isfinite(batch[key]).all())
                self.assertGreaterEqual(float(batch[key].min()), -1)
                self.assertLessEqual(float(batch[key].max()), 1)
            self.assertTrue(torch.equal(batch['Xa'], -torch.ones_like(batch['Xa'])))
            for path in batch['Xu_path']:
                self.assertEqual(Path(path).parent, u)
            for path in batch['Xa_path']:
                self.assertEqual(Path(path).parent, a)

    def test_crop_is_exact_without_resize_or_flip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'pattern.png'
            pixels = np.random.RandomState(8).randint(0, 256, (270, 310, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(p)
            random.seed(7)
            x, y = random.randint(0, 54), random.randint(0, 14)
            expected = torch.from_numpy(pixels[y:y+256, x:x+256].copy()).permute(2, 0, 1).float()
            expected = (expected / 255 - 0.5) / 0.5
            random.seed(7)
            self.assertTrue(torch.equal(read_crop(p), expected))

    def test_small_and_bad_images_are_reported_not_filtered(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            Image.new('RGB', (352, 224)).save(root / 'short.bmp')
            Image.new('RGB', (256, 256)).save(root / 'good.png')
            (root / 'bad.png').write_bytes(b'not an image')
            paths = image_paths(root)
            report = check_images(paths)
            self.assertEqual(report['count'], 3)
            self.assertEqual(len(report['too_small']), 1)
            self.assertEqual(len(report['errors']), 1)
            self.assertEqual(report['skipped'], 0)
            self.assertEqual(len(paths), 3)
            with self.assertRaisesRegex(ValueError, 'smaller than 256'):
                read_crop(root / 'short.bmp')

    def test_manifest_and_nonrecursive_hr_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            Image.new('RGB', (256, 256)).save(root / 'one.png')
            (root / 'LR').mkdir()
            Image.new('RGB', (256, 256)).save(root / 'LR' / 'low.png')
            self.assertEqual(image_paths(root), [root / 'one.png'])
            manifest = root / 'list.txt'
            manifest.write_text('one.png\n', encoding='utf-8')
            self.assertEqual(image_paths(root, manifest), [root / 'one.png'])
            manifest.write_text('one.png\none.png', encoding='utf-8')
            with self.assertRaises(ValueError):
                image_paths(root, manifest)
            manifest.write_text('../list.txt', encoding='utf-8')
            with self.assertRaises((ValueError, FileNotFoundError)):
                image_paths(root, manifest)
            empty = root / 'empty'
            empty.mkdir()
            with self.assertRaises(ValueError):
                image_paths(empty)

    def test_three_losses_against_manual_values(self):
        x = torch.tensor([1., 0., 0.]).reshape(1, 3, 1, 1)
        ya = torch.tensor([0., 1., 0.]).reshape_as(x)
        yu = torch.tensor([0., -1., 0.]).reshape_as(x)
        losses = stage1_losses(x, x, ya, yu)
        self.assertAlmostEqual(float(losses['rec_a']), 2 / 3, places=6)
        self.assertAlmostEqual(float(losses['rec_u']), 2 / 3, places=6)
        self.assertEqual(float(losses['sr']), 1)
        self.assertAlmostEqual(float(losses['total']), 1 + 2 / 3 + 0.1 * 2 / 3, places=6)

    def test_ranking_direction_and_derivative(self):
        self.assertEqual(float(ranking_loss(torch.tensor([1.]), torch.tensor([0.]))), 0)
        self.assertEqual(float(ranking_loss(torch.tensor([0.]), torch.tensor([1.]))), 2)
        sa = torch.tensor([0.2], requires_grad=True)
        su = torch.tensor([0.3], requires_grad=True)
        ranking_loss(sa, su).backward()
        self.assertEqual(float(sa.grad), -1)
        self.assertEqual(float(su.grad), 1)
        baseline = ranking_loss(sa.detach(), su.detach())
        self.assertLess(ranking_loss(sa.detach() + 0.1, su.detach()), baseline)
        self.assertGreater(ranking_loss(sa.detach(), su.detach() + 0.1), baseline)

    def test_per_image_hinge_not_hinge_of_batch_means(self):
        x = torch.tensor([1., 0., 0.]).reshape(1, 3, 1, 1).repeat(2, 1, 1, 1)
        ya, yu = x.clone(), x.clone()
        ya[1] *= -1
        yu[0] *= -1
        losses = stage1_losses(x, x, ya, yu)
        self.assertEqual(float(losses['sr']), 1.5)

    def test_zero_cosine_and_invalid_contracts(self):
        zero = torch.zeros(2, 3, 1, 1)
        losses = stage1_losses(zero, zero, zero, zero)
        self.assertEqual(float(losses['sa']), 0)
        self.assertEqual(float(losses['total']), 1)
        for bad in (zero.double(), torch.full_like(zero, float('nan')),
                    torch.full_like(zero, 1.1), zero[:1]):
            with self.assertRaises(ValueError):
                stage1_losses(zero, zero, bad, zero)

    def test_frozen_encoder_shared_generator_both_domain_gradients_and_update(self):
        model = tiny_model().train()
        optimizer = make_optimizer(model.G)
        self.assertFalse(model.E.training)
        self.assertTrue(all(not p.requires_grad for p in model.E.parameters()))
        self.assertEqual({id(p) for group in optimizer.param_groups for p in group['params']},
                         {id(p) for p in model.G.parameters()})
        self.assertEqual(optimizer.defaults['betas'], (0.5, 0.999))
        xa, xu = torch.rand(2, 3, 256, 256) * 2 - 1, torch.rand(2, 3, 256, 256) * 2 - 1
        ya, yu = model(xa, xu)
        ya.retain_grad()
        yu.retain_grad()
        stage1_losses(xa, xu, ya, yu)['total'].backward()
        for value in (ya, yu):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(torch.count_nonzero(value.grad).item(), 0)
        old_e, old_g = copy.deepcopy(model.E.state_dict()), copy.deepcopy(model.G.state_dict())
        train_step(model, optimizer, xa, xu)
        self.assertTrue(all(p.grad is None for p in model.E.parameters()))
        self.assertTrue(all(torch.equal(v, old_e[k]) for k, v in model.E.state_dict().items()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.G.parameters()))
        self.assertTrue(any(not torch.equal(v, old_g[k]) for k, v in model.G.state_dict().items()))

    def test_epoch_checkpoint_rng_optimizer_next_update_and_no_overwrite(self):
        model = tiny_model()
        optimizer = make_optimizer(model.G)
        scheduler = make_scheduler(optimizer, 1, 2)
        x = torch.rand(2, 3, 256, 256) * 2 - 1
        train_step(model, optimizer, x, -x)
        scheduler.step()
        config = {'fixture': 'synthetic', 'seed': 123}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'epoch_0001.pth'
            save_checkpoint(path, model, optimizer, scheduler, 1, 1, config)
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model, optimizer, scheduler, 1, 1, config)
            random_values = (random.random(), np.random.rand(), torch.rand(4))
            next_x = torch.rand_like(x) * 2 - 1
            train_step(model, optimizer, next_x, -next_x)
            scheduler.step()
            restored = tiny_model()
            restored.E.load_state_dict(model.E.state_dict())  # external fixed E, not in checkpoint
            restored_optimizer = make_optimizer(restored.G)
            restored_scheduler = make_scheduler(restored_optimizer, 1, 2)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                load_checkpoint(path, restored, restored_optimizer, restored_scheduler, {'fixture': 'wrong'})
            self.assertEqual(load_checkpoint(path, restored, restored_optimizer, restored_scheduler, config), (1, 1))
            self.assertEqual(random.random(), random_values[0])
            self.assertEqual(np.random.rand(), random_values[1])
            self.assertTrue(torch.equal(torch.rand(4), random_values[2]))
            self.assertTrue(torch.equal(torch.rand_like(x) * 2 - 1, next_x))
            train_step(restored, restored_optimizer, next_x, -next_x)
            restored_scheduler.step()
            self.assertTrue(all(torch.equal(v, model.G.state_dict()[k]) for k, v in restored.G.state_dict().items()))
            self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
            for p, q in zip(model.G.parameters(), restored.G.parameters()):
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    self.assertTrue(torch.equal(optimizer.state[p][key], restored_optimizer.state[q][key]))

    def test_lr_full_schedule_all_updates_nonzero_final_boundary_zero(self):
        parameter = nn.Parameter(torch.tensor(0.))
        optimizer = torch.optim.SGD([parameter], lr=1e-4)
        scheduler = make_scheduler(optimizer)
        used = []
        for _ in range(100):
            used.append(optimizer.param_groups[0]['lr'])
            old = parameter.detach().clone()
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            self.assertFalse(torch.equal(old, parameter))
            scheduler.step()
        self.assertTrue(all(lr == 1e-4 for lr in used[:50]))
        self.assertEqual(used[50], 1e-4)
        self.assertAlmostEqual(used[51], 9.8e-5)
        self.assertAlmostEqual(used[99], 2e-6)
        self.assertTrue(all(lr > 0 for lr in used))
        self.assertEqual(optimizer.param_groups[0]['lr'], 0)

    def test_original_sad_synthetic_features_active_gradients_and_strict_keys(self):
        from models.DenseNet import SAD
        generator = SAD()
        optimizer = make_optimizer(generator)
        features = [torch.randn(2, 64, 32, 32), torch.randn(2, 128, 16, 16), torch.randn(2, 256, 8, 8)]
        before = copy.deepcopy(generator.state_dict())
        y = generator(features)
        self.assertEqual(tuple(y.shape), (2, 3, 32, 32))
        self.assertTrue(torch.isfinite(y).all())
        self.assertGreaterEqual(float(y.min()), -1)
        self.assertLessEqual(float(y.max()), 1)
        losses = stage1_losses(torch.zeros_like(y), torch.ones_like(y), y, generator(features))
        losses['total'].backward()
        unused = {n for n, p in generator.named_parameters() if p.grad is None}
        expected = {n for n, _ in generator.named_parameters() if n.startswith(('trans_down_1.', 'trans_down_2.'))}
        self.assertEqual(unused, expected)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in generator.parameters() if p.grad is not None))
        self.assertTrue(any(p.grad.count_nonzero() for p in generator.parameters() if p.grad is not None))
        optimizer.step()
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in generator.state_dict().items()))
        restored = SAD()
        self.assertEqual(restored.load_state_dict(generator.state_dict(), strict=True).missing_keys, [])
        with torch.no_grad():
            self.assertTrue(torch.equal(generator(features), restored(features)))

    def test_explicit_initialization_reuses_constructor_or_author_initializer(self):
        from stage1 import build_model
        from models.DenseNet import SAD
        from models.networks import init_weights
        for choice in ('constructor', 'author-normal'):
            seed_everything(9)
            TinyEncoder()  # Same encoder construction RNG use as the mocked loader.
            reference = SAD()
            if choice == 'author-normal':
                init_weights(reference, 'normal', 0.02)
            seed_everything(9)
            with patch('models.vggloss.Vgg19_out', side_effect=lambda **kw: TinyEncoder()):
                actual = build_model('synthetic-only.pth', 'cpu', choice)
            self.assertTrue(all(torch.equal(v, actual.G.state_dict()[k]) for k, v in reference.state_dict().items()))

    def test_vgg_explicit_loader_preserves_original_slices_forward_and_freeze(self):
        from models.vggloss import Vgg19_out
        from torchvision.models.vgg import make_layers, cfgs
        # Real torchvision VGG19 feature architecture with synthetic fixed weights;
        # omit the unused classifier to keep this fixture light, never claim pretrained acceptance.
        class FeatureFixture(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = make_layers(cfgs['E'])
        state = FeatureFixture().state_dict()
        with patch('models.vggloss.models.vgg19', side_effect=lambda **kw: FeatureFixture()), \
                patch('models.vggloss.torch.load', return_value=state) as load, \
                patch.object(nn.Module, 'cuda', lambda self: self):
            original = Vgg19_out()
            explicit = Vgg19_out(weight_path='synthetic-vgg.pth', device='cpu')
            self.assertEqual(load.call_args_list[1].kwargs, {'map_location': 'cpu'})
        x = torch.rand(2, 3, 256, 256) * 2 - 1
        with torch.no_grad():
            old, new = original(x), explicit(x)
        self.assertEqual([tuple(v.shape) for v in new], [(2,64,256,256), (2,128,128,128), (2,256,64,64)])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old, new)))
        self.assertTrue(all(not p.requires_grad for p in explicit.parameters()))
        with self.assertRaises(ValueError):
            Vgg19_out(device='cpu')

    def test_cli_data_check_blocks_entire_run_and_preserves_existing_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            u, a = root / 'u', root / 'a'
            u.mkdir()
            a.mkdir()
            Image.new('RGB', (352, 224)).save(u / 'short.bmp')
            Image.new('RGB', (256, 256)).save(a / 'good.png')
            command = [sys.executable, '-B', 'train_stage1.py', '--mode', 'data-check',
                       '--underwater-root', str(u), '--inair-root', str(a),
                       '--output-root', str(root / 'runs'), '--run-id', 'check']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            output = root / 'runs' / 'check'
            report = json.loads((output / 'data_check.json').read_text(encoding='utf-8'))
            self.assertEqual(report['underwater']['too_small'][0]['height'], 224)
            self.assertFalse(list(output.glob('*.pth')))
            before = (output / 'data_check.json').read_bytes()
            rerun = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual(before, (output / 'data_check.json').read_bytes())

    def test_entry_epoch_recovery_reproduces_next_shuffled_crops_and_updates(self):
        from train_stage1 import parser, run
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for domain in ('u', 'a'):
                (root / domain).mkdir()
                for i in range(3):
                    pixels = np.random.RandomState(i + (domain == 'a') * 10).randint(
                        0, 256, (270, 280, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(root / domain / '{}.png'.format(i))
                (root / (domain + '.txt')).write_text('0.png\n1.png\n2.png\n', encoding='utf-8')
            weights = root / 'synthetic-fixture.pth'
            weights.write_bytes(b'fixture only; build_model is mocked')
            base = ['--mode', 'train', '--underwater-root', str(root / 'u'),
                    '--inair-root', str(root / 'a'), '--underwater-manifest', str(root / 'u.txt'),
                    '--inair-manifest', str(root / 'a.txt'), '--vgg-weights', str(weights),
                    '--g-init', 'constructor', '--output-root', str(root / 'runs'),
                    '--fixed-epochs', '1', '--decay-epochs', '1']
            with patch('train_stage1.build_model', side_effect=lambda *args: tiny_model()):
                run(parser().parse_args(base + ['--run-id', 'continuous']))
                def stop_at_epoch_boundary(*args):
                    save_checkpoint(*args)
                    raise RuntimeError('synthetic interruption after completed epoch')
                with patch('train_stage1.save_checkpoint', side_effect=stop_at_epoch_boundary):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                        run(parser().parse_args(base + ['--run-id', 'interrupted']))
                parent = root / 'runs' / 'interrupted' / 'epoch_0001.pth'
                run(parser().parse_args(base + ['--run-id', 'resumed', '--resume', str(parent)]))
            full = torch.load(root / 'runs' / 'continuous' / 'epoch_0002.pth', weights_only=False)
            resumed = torch.load(root / 'runs' / 'resumed' / 'epoch_0002.pth', weights_only=False)
            self.assertEqual(full['epoch'], 2)
            self.assertEqual(full['global_step'], 4)  # Includes the final single-image batches.
            self.assertEqual(full['global_step'], resumed['global_step'])
            self.assertTrue(all(torch.equal(v, resumed['G'][k]) for k, v in full['G'].items()))
            self.assertEqual(full['scheduler'], resumed['scheduler'])
            self.assertEqual(full['rng']['python'], resumed['rng']['python'])
            self.assertTrue(torch.equal(full['rng']['torch'], resumed['rng']['torch']))

    def test_smoke_acceptance_wrapper_with_synthetic_shape_fixture(self):
        from train_stage1 import smoke
        class ShapeEncoder(TinyEncoder):
            def forward(self, x):
                f = self.conv(x)
                return [f.repeat(1,16,1,1), F.avg_pool2d(f,2).repeat(1,32,1,1),
                        F.avg_pool2d(f,4).repeat(1,64,1,1)]
        class ShapeGenerator(TinyGenerator):
            def forward(self, features):
                return self.conv(features[0][:, :4]).tanh()
        model = Stage1(ShapeEncoder(), ShapeGenerator())
        optimizer = make_optimizer(model.G)
        scheduler = make_scheduler(optimizer)
        x = torch.rand(2, 3, 256, 256) * 2 - 1
        loader = [{'Xa': x, 'Xu': -x, 'Xa_path': ['synthetic:a0', 'synthetic:a1'],
                   'Xu_path': ['synthetic:u0', 'synthetic:u1']}]
        with tempfile.TemporaryDirectory() as d:
            result = smoke(model, optimizer, scheduler, loader, torch.device('cpu'), Path(d),
                           {'lr': 1e-4, 'beta1': 0.5, 'fixed_epochs': 50, 'decay_epochs': 50})
        self.assertEqual(result['global_step'], 1)
        self.assertGreater(result['changed_parameters'], 0)
        self.assertEqual(result['shapes'], [[2,64,256,256], [2,128,128,128], [2,256,64,64]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
