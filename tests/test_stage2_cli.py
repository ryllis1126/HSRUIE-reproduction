"""Real image/manifest/CLI gates; model builder replaced only in CPU smoke."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

import train_stage2 as cli
from stage2 import Stage2


class ContractE(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.7))

    def forward(self, x):
        x = x.mean(1, keepdim=True) * self.scale
        return [F.avg_pool2d(x, 2 ** i).expand(-1, c, -1, -1) for i, c in enumerate((64, 128, 256))]


class ContractExtractor(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.ParameterList([nn.Parameter(torch.full((1, c, 1, 1), scale)) for c in (64, 128, 256)])
        self.bias = nn.ParameterList([nn.Parameter(torch.full((1, c, 1, 1), 0.13)) for c in (64, 128, 256)])

    def forward(self, fs):
        return [f * s + b for f, s, b in zip(fs, self.scale, self.bias)]


class ContractG(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.4))

    def forward(self, fs):
        result = sum(F.interpolate(f.mean(1, keepdim=True), size=(256, 256), mode='nearest') for f in fs)
        return (result * self.scale).tanh().expand(-1, 3, -1, -1)


class ContractClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        x = x.mean(1, keepdim=True) * self.scale
        return [F.adaptive_avg_pool2d(x, (s, s)).expand(-1, c, -1, -1)
                for c, s in ((512, 28), (1024, 14), (2048, 7))]


def contract_model(*args, **kwargs):
    return Stage2(ContractE(), ContractG(), ContractClip(), ContractExtractor(0.5), ContractExtractor(0.2))


class Stage2CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.data = self.base/'raw'
        self.data.mkdir()
        self.images = []
        for index, size in enumerate(((352, 224), (224, 352), (280, 270))):
            name = '{}.png'.format(index)
            Image.new('RGB', size, (80 + 10 * index, 120, 160)).save(self.data/name)
            self.images.append(name)
        self.manifest = self.base/'manifest.txt'
        self.manifest.write_text('\n'.join(self.images[:2])+'\n', encoding='utf-8')

    def args(self, mode, run_id='new'):
        return cli.parser().parse_args(['--mode', mode, '--underwater-root', str(self.data),
            '--underwater-manifest', str(self.manifest), '--output-root', str(self.base/'out'), '--run-id', run_id])

    def report(self, run_id, filename):
        return json.loads((self.base/'out'/run_id/filename).read_text(encoding='utf-8'))

    def test_explicit_mode_is_required(self):
        with self.assertRaises(SystemExit):
            cli.parser().parse_args([])

    def test_data_check_scans_full_domain_approved_reflection_and_no_model(self):
        with patch.object(cli, 'build_model', side_effect=AssertionError('must not load model')):
            cli.run(self.args('data-check'))
        report = self.report('new', 'data_check.json')
        self.assertEqual(report['count'], 3)
        self.assertEqual(report['padding_count'], 2)
        self.assertEqual(report['no_padding_count'], 1)
        self.assertEqual(report['skipped'], 0)
        self.assertEqual(self.report('new', 'end.json')['status'], 'data-check passed')

    def test_full_data_check_and_train_cannot_hide_corrupt_unlisted_image(self):
        (self.data/'bad.png').write_bytes(b'broken')
        for mode in ('data-check', 'train'):
            args = self.args(mode, mode)
            if mode == 'train':
                self.model_args(args, diagnostic=False)
            with patch.object(cli, 'build_model', side_effect=AssertionError('gate must stop first')):
                with self.assertRaisesRegex(ValueError, 'data-check'):
                    cli.run(args)
            self.assertEqual(len(self.report(mode, 'data_check.json')['errors']), 1)
            self.assertEqual(self.report(mode, 'failure.json')['status'], 'failed')

    def model_args(self, args, diagnostic=True):
        args.diagnostic_stage1 = diagnostic
        for name in ('vgg_weights', 'clip_weights', 'stage1_checkpoint'):
            path = self.base/(name+'.pth')
            if not path.exists():
                path.write_bytes(b'synthetic placeholder; model builder/source loader patched')
            setattr(args, name, path)

    def test_smoke_requires_manifest_and_two_distinct_members(self):
        for manifest in (None, 'single', 'duplicate'):
            args = self.args('smoke', str(manifest))
            self.model_args(args)
            if manifest is None:
                args.underwater_manifest = None
            else:
                self.manifest.write_text('0.png\n' if manifest == 'single' else '0.png\n0.png\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                cli.run(args)

    def test_smoke_checks_every_manifest_member_and_crop_legality(self):
        for name, content in (('bad.png', b'bad'), ('short.png', None)):
            if content:
                (self.data/name).write_bytes(content)
            else:
                Image.new('RGB', (250, 270)).save(self.data/name)
            self.manifest.write_text('0.png\n1.png\n'+name+'\n', encoding='utf-8')
            args = self.args('smoke', name)
            self.model_args(args)
            with patch.object(cli, 'build_model', side_effect=AssertionError('must reject third member')):
                with self.assertRaisesRegex(ValueError, 'data-check'):
                    cli.run(args)

    def test_run_directory_never_overwritten(self):
        cli.run(self.args('data-check'))
        before = (self.base/'out/new/end.json').read_bytes()
        with self.assertRaises(FileExistsError):
            cli.run(self.args('data-check'))
        self.assertEqual((self.base/'out/new/end.json').read_bytes(), before)

    def test_diagnostic_weights_forbidden_in_train_before_run_creation(self):
        args = self.args('train')
        self.model_args(args)
        with self.assertRaisesRegex(ValueError, 'diagnostic'):
            cli.run(args)
        self.assertFalse((self.base/'out').exists())

    def test_smoke_scope_ignores_unlisted_bad_data_and_roundtrips_checkpoint(self):
        (self.data/'unlisted.png').write_bytes(b'bad')
        args = self.args('smoke')
        self.model_args(args)
        source = {'epoch': 0, 'global_step': 1, 'config': {'mode': 'smoke'}}
        with patch.object(cli, 'build_model', side_effect=contract_model), patch.object(cli, 'read_stage1_checkpoint', return_value=source):
            cli.run(args)
        result = self.report('new', 'end.json')
        self.assertEqual(result['status'], 'smoke passed')
        self.assertEqual(result['processed_images'], 2)
        self.assertEqual(result['global_step'], 1)
        self.assertTrue(result['checkpoint_recovery'])
        self.assertGreater(result['changed_C'], 0)
        self.assertGreater(result['changed_D'], 0)
        self.assertEqual(self.report('new', 'data_check.json')['count'], 2)
        with self.assertRaises(ValueError):
            from stage2 import load_checkpoint, make_optimizer, make_scheduler
            model = contract_model()
            optimizer = make_optimizer(model)
            load_checkpoint(self.base/'out/new/smoke_checkpoint.pth', model, optimizer,
                            make_scheduler(optimizer, 100, 100), {'mode': 'smoke'})

    def test_train_rejects_manifest_subset_even_if_full_data_healthy(self):
        args = self.args('train')
        self.model_args(args, diagnostic=False)
        with self.assertRaisesRegex(ValueError, 'full domain'):
            cli.run(args)

    def test_train_enforces_paper_schedule_and_batch(self):
        for key, value in (('batch_size', 1), ('fixed_epochs', 1), ('decay_epochs', 1), ('lr', 2e-4)):
            args = self.args('train')
            self.model_args(args, diagnostic=False)
            setattr(args, key, value)
            with self.assertRaises(ValueError):
                cli.run(args)

    def test_train_checks_actual_stage1_manifest_before_model_loading(self):
        self.manifest.write_text('\n'.join(self.images)+'\n', encoding='utf-8')
        args = self.args('train')
        self.model_args(args, diagnostic=False)
        source = {'epoch': 100, 'global_step': 200, 'config': {'data': {'underwater': {'records': []}}}}
        with patch.object(cli, 'read_stage1_checkpoint', return_value=source), patch.object(cli, 'build_model', side_effect=AssertionError('must stop')):
            with self.assertRaisesRegex(ValueError, 'Stage1 actual underwater manifest'):
                cli.run(args)

    def test_resume_source_sha256_is_recorded_even_when_later_validation_fails(self):
        args = self.args('train')
        self.model_args(args, diagnostic=False)
        args.resume = self.base/'resume.pth'
        args.resume.write_bytes(b'synthetic source for provenance, not a loadable checkpoint')
        with self.assertRaisesRegex(ValueError, 'full domain'):
            cli.run(args)
        from hashlib import sha256
        self.assertEqual(self.report('new', 'run.json')['resume_source']['sha256'],
                         sha256(args.resume.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
