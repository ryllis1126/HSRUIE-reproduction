"""Real data-gate tests. Stop at model construction; never execute a GPU model."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from data.stage1_dataset import read_crop
from train_stage1 import parser, run


class ReachedModelConstruction(AssertionError):
    """A test boundary, after real path/decode/size/crop validation."""


class SmokeGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for domain in ('u', 'a'):
            directory = self.root / domain
            directory.mkdir()
            Image.new('RGB', (256, 256), (255, 0, 0)).save(directory / 'one.png')
            Image.new('RGB', (280, 300), (0, 255, 0)).save(directory / 'two.png')
            Image.new('RGB', (352, 224)).save(directory / 'small.png')
            (self.root / (domain + '.txt')).write_text('one.png\ntwo.png\n', encoding='utf-8')
        self.weights = self.root / 'placeholder.pth'
        self.weights.write_bytes(b'not loaded: stop before model construction')

    def args(self, mode, run_id='check', manifests=True):
        command = ['--mode', mode, '--run-id', run_id, '--output-root', str(self.root / 'runs'),
                   '--underwater-root', str(self.root / 'u'), '--inair-root', str(self.root / 'a'),
                   '--vgg-weights', str(self.weights), '--g-init', 'constructor', '--device', 'cpu']
        if manifests:
            command += ['--underwater-manifest', str(self.root / 'u.txt'),
                        '--inair-manifest', str(self.root / 'a.txt')]
        return parser().parse_args(command)

    def report(self, run_id='check'):
        return json.loads((self.root / 'runs' / run_id / 'data_check.json').read_text(encoding='utf-8'))

    def test_data_check_full_domain_small_images_fail_even_with_manifest(self):
        for manifests in (False, True):
            with self.subTest(manifests=manifests):
                run_id = 'full-{}'.format(manifests)
                with self.assertRaisesRegex(ValueError, 'data-check blocked'):
                    run(self.args('data-check', run_id, manifests))
                report = self.report(run_id)
                for domain in ('underwater', 'inair'):
                    self.assertEqual(report[domain]['count'], 3)
                    self.assertEqual(len(report[domain]['too_small']), 1)

    def test_valid_smoke_manifests_pass_despite_unlisted_small_and_corrupt_images(self):
        for domain in ('u', 'a'):
            (self.root / domain / 'outside-bad.png').write_bytes(b'not an image')
        before = {p: p.read_bytes() for domain in ('u', 'a') for p in (self.root / domain).iterdir()}
        with patch('train_stage1.build_model', side_effect=ReachedModelConstruction):
            with self.assertRaises(ReachedModelConstruction):
                run(self.args('smoke'))
        report = self.report()
        for domain in ('underwater', 'inair'):
            self.assertEqual(report[domain]['count'], 2)
            self.assertEqual(report[domain]['too_small'], [])
            self.assertEqual(report[domain]['errors'], [])
            self.assertEqual({Path(r['path']).name for r in report[domain]['records']}, {'one.png', 'two.png'})
        self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))
        self.assertFalse(list((self.root / 'runs' / 'check').glob('*.pth')))

    def test_smoke_requires_both_explicit_manifests(self):
        for missing in ('underwater_manifest', 'inair_manifest', 'both'):
            with self.subTest(missing=missing):
                args = self.args('smoke', missing)
                if missing == 'both':
                    args.underwater_manifest = args.inair_manifest = None
                else:
                    setattr(args, missing, None)
                with self.assertRaisesRegex(ValueError, 'Smoke requires explicit'):
                    run(args)
                self.assertFalse((self.root / 'runs' / missing).exists())

    def test_smoke_requires_two_distinct_images_in_each_manifest(self):
        for domain in ('u', 'a'):
            with self.subTest(domain=domain):
                manifest = self.root / (domain + '.txt')
                manifest.write_text('one.png\n', encoding='utf-8')
                with patch('train_stage1.build_model', side_effect=ReachedModelConstruction):
                    with self.assertRaisesRegex(ValueError, 'at least 2'):
                        run(self.args('smoke', domain))
                manifest.write_text('one.png\ntwo.png\n', encoding='utf-8')

    def test_smoke_rejects_any_listed_small_image_in_either_domain(self):
        for domain, report_key in (('u', 'underwater'), ('a', 'inair')):
            with self.subTest(domain=domain):
                manifest = self.root / (domain + '.txt')
                manifest.write_text('one.png\ntwo.png\nsmall.png\n', encoding='utf-8')
                with patch('train_stage1.build_model', side_effect=ReachedModelConstruction):
                    with self.assertRaisesRegex(ValueError, 'data-check blocked'):
                        run(self.args('smoke', domain))
                self.assertEqual(self.report(domain)[report_key]['too_small'][0]['height'], 224)
                manifest.write_text('one.png\ntwo.png\n', encoding='utf-8')

    def test_smoke_rejects_any_listed_unreadable_image_in_either_domain(self):
        for domain, report_key in (('u', 'underwater'), ('a', 'inair')):
            with self.subTest(domain=domain):
                (self.root / domain / 'bad.png').write_bytes(b'not an image')
                manifest = self.root / (domain + '.txt')
                manifest.write_text('one.png\ntwo.png\nbad.png\n', encoding='utf-8')
                with patch('train_stage1.build_model', side_effect=ReachedModelConstruction):
                    with self.assertRaisesRegex(ValueError, 'data-check blocked'):
                        run(self.args('smoke', domain))
                self.assertEqual(len(self.report(domain)[report_key]['errors']), 1)
                manifest.write_text('one.png\ntwo.png\n', encoding='utf-8')

    def test_smoke_rejects_missing_outside_and_duplicate_manifest_paths(self):
        outside = self.root / 'outside.png'
        Image.new('RGB', (256, 256)).save(outside)
        entries = ('one.png\nmissing.png\n', 'one.png\n../outside.png\n', 'one.png\none.png\n')
        for i, text in enumerate(entries):
            with self.subTest(entries=text):
                (self.root / 'u.txt').write_text(text, encoding='utf-8')
                with patch('train_stage1.build_model', side_effect=ReachedModelConstruction):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        run(self.args('smoke', 'bad-path-{}'.format(i)))

    def test_train_manifests_cannot_bypass_full_domain_gate(self):
        with patch('train_stage1.build_model', side_effect=ReachedModelConstruction):
            with self.assertRaisesRegex(ValueError, 'data-check blocked'):
                run(self.args('train'))
        report = self.report()
        self.assertEqual(report['underwater']['count'], 3)
        self.assertEqual(len(report['underwater']['too_small']), 1)
        self.assertFalse(list((self.root / 'runs' / 'check').glob('*.pth')))

    def test_smoke_checks_crop_for_every_listed_image(self):
        visited = []
        for domain in ('u', 'a'):
            Image.new('L', (270, 280)).save(self.root / domain / 'third.png')
            (self.root / (domain + '.txt')).write_text('one.png\ntwo.png\nthird.png\n', encoding='utf-8')
        def real_crop(path):
            tensor = read_crop(path)
            self.assertEqual(tuple(tensor.shape), (3, 256, 256))
            visited.append(Path(path))
            return tensor
        with patch('train_stage1.read_crop', side_effect=real_crop), \
                patch('train_stage1.build_model', side_effect=ReachedModelConstruction):
            with self.assertRaises(ReachedModelConstruction):
                run(self.args('smoke'))
        self.assertCountEqual(visited, [self.root / domain / name for domain in ('u', 'a')
                                       for name in ('one.png', 'two.png', 'third.png')])

    def test_train_checks_selected_nested_images_after_full_domain_passes(self):
        for domain in ('u', 'a'):
            Image.new('RGB', (256, 256)).save(self.root / domain / 'small.png')
        nested = self.root / 'u' / 'nested'
        nested.mkdir()
        (nested / 'bad.png').write_bytes(b'not an image')
        (self.root / 'u.txt').write_text('one.png\ntwo.png\nnested/bad.png\n', encoding='utf-8')
        with patch('train_stage1.build_model', side_effect=ReachedModelConstruction):
            with self.assertRaisesRegex(ValueError, 'Training data-check blocked'):
                run(self.args('train'))
        self.assertEqual(self.report()['underwater']['errors'], [])
        selected = json.loads((self.root / 'runs' / 'check' / 'training_data_check.json').read_text(encoding='utf-8'))
        self.assertEqual(len(selected['underwater']['errors']), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
