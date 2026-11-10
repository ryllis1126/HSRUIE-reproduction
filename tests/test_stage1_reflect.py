"""Approved UVTD reflection policy; CPU synthetic images, no real training."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import data.stage1_dataset as data
from train_stage1 import parser, run


class ModelBoundary(AssertionError):
    """Stop after real data checks and before any model construction."""


def pattern(width, height):
    y, x = np.indices((height, width))
    return np.stack((y % 256, x % 251, (y + 2*x) % 256), axis=-1).astype(np.uint8)


class ReflectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def prepare(self, width, height):
        pixels = pattern(width, height)
        image = Image.fromarray(pixels)
        padded, meta = data.prepare_image(image, allow_uvtd_reflect=True)
        return pixels, np.asarray(padded), meta

    def domains(self):
        for domain in ('u', 'a'):
            (self.root / domain).mkdir()
        for name, size in [('landscape.png', (352, 224)), ('portrait.png', (224, 352)), ('regular.png', (300, 280))]:
            Image.fromarray(pattern(*size)).save(self.root / 'u' / name)
        for name in ('one.png', 'two.png'):
            Image.new('RGB', (256, 256)).save(self.root / 'a' / name)
        (self.root / 'u.txt').write_text('landscape.png\nportrait.png\nregular.png\n', encoding='utf-8')
        (self.root / 'a.txt').write_text('one.png\ntwo.png\n', encoding='utf-8')
        (self.root / 'weights.pth').write_bytes(b'placeholder; never loaded')

    def args(self, mode, run_id='check'):
        return parser().parse_args(['--mode', mode, '--run-id', run_id, '--output-root', str(self.root / 'runs'),
            '--underwater-root', str(self.root / 'u'), '--inair-root', str(self.root / 'a'),
            '--underwater-manifest', str(self.root / 'u.txt'), '--inair-manifest', str(self.root / 'a.txt'),
            '--vgg-weights', str(self.root / 'weights.pth'), '--g-init', 'constructor', '--device', 'cpu'])

    def report(self, run_id='check', filename='data_check.json'):
        return json.loads((self.root / 'runs' / run_id / filename).read_text(encoding='utf-8'))

    def test_352x224_exact_top_bottom_reflection_and_metadata(self):
        original, padded, meta = self.prepare(352, 224)
        self.assertEqual(padded.shape, (256, 352, 3))
        self.assertTrue(np.array_equal(padded[16:240], original))
        self.assertTrue(np.array_equal(padded[:16], original[16:0:-1]))
        self.assertTrue(np.array_equal(padded[240:], original[222:206:-1]))
        self.assertEqual(meta['original_size'], [352, 224])
        self.assertEqual(meta['padded_size'], [352, 256])
        self.assertEqual(meta['padding'], {'left': 0, 'top': 16, 'right': 0, 'bottom': 16})
        self.assertEqual(meta['strategy'], 'reflect_then_random_crop256')

    def test_224x352_exact_left_right_reflection_and_metadata(self):
        original, padded, meta = self.prepare(224, 352)
        self.assertEqual(padded.shape, (352, 256, 3))
        self.assertTrue(np.array_equal(padded[:,16:240], original))
        self.assertTrue(np.array_equal(padded[:,:16], original[:,16:0:-1]))
        self.assertTrue(np.array_equal(padded[:,240:], original[:,222:206:-1]))
        self.assertEqual(meta['original_size'], [224, 352])
        self.assertEqual(meta['padded_size'], [256, 352])
        self.assertEqual(meta['padding'], {'left': 16, 'top': 0, 'right': 16, 'bottom': 0})

    def test_reflect_excludes_edge_and_is_not_replicate_or_zero(self):
        original, padded, _ = self.prepare(352, 224)
        self.assertEqual(int(padded[15,20,0]), 1)   # neighbor of original row0 reflects row1
        self.assertEqual(int(padded[0,20,0]), 16)
        self.assertEqual(int(padded[240,20,0]), 222)
        self.assertNotEqual(int(padded[15,20,0]), int(original[0,20,0]))
        self.assertFalse(np.array_equal(padded[:16], np.zeros_like(padded[:16])))
        original, padded, _ = self.prepare(224, 352)
        self.assertEqual(int(padded[20,15,1]), 1)
        self.assertNotEqual(int(padded[20,15,1]), int(original[20,0,1]))

    def test_short_crops_use_padded_pixels_and_preserve_original_files(self):
        for width, height in ((352,224), (224,352)):
            with self.subTest(size=(width,height)):
                pixels = pattern(width,height)
                path = self.root / '{}x{}.png'.format(width,height)
                Image.fromarray(pixels).save(path)
                before = path.read_bytes()
                # Independent reflection oracle, without np.pad or production preparation.
                if height == 224:
                    padded = np.concatenate((pixels[16:0:-1],pixels,pixels[222:206:-1]),axis=0)
                else:
                    padded = np.concatenate((pixels[:,16:0:-1],pixels,pixels[:,222:206:-1]),axis=1)
                random.seed(19)
                x,y = random.randint(0,padded.shape[1]-256), random.randint(0,padded.shape[0]-256)
                expected = torch.from_numpy(padded[y:y+256,x:x+256].copy()).permute(2,0,1).float()
                expected = (expected/255-0.5)/0.5
                random.seed(19)
                actual = data.read_crop(path, allow_uvtd_reflect=True)
                self.assertEqual(tuple(actual.shape),(3,256,256))
                self.assertEqual(actual.dtype,torch.float32)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.equal(actual,expected))
                self.assertGreaterEqual(float(actual.min()),-1)
                self.assertLessEqual(float(actual.max()),1)
                self.assertEqual(path.read_bytes(),before)

    def test_regular_image_is_unchanged_before_existing_random_crop(self):
        pixels = pattern(310,270)
        prepared,meta = data.prepare_image(Image.fromarray(pixels), allow_uvtd_reflect=True)
        self.assertTrue(np.array_equal(np.asarray(prepared),pixels))
        self.assertEqual(meta['padding'],dict(left=0,top=0,right=0,bottom=0))
        self.assertEqual(meta['padded_size'],[310,270])
        self.assertEqual(meta['strategy'],'random_crop256')

    def test_check_counts_and_records_padding_without_filtering(self):
        self.domains()
        report = data.check_images(data.image_paths(self.root/'u'), allow_uvtd_reflect=True)
        self.assertEqual(report['count'],3)
        self.assertEqual(report['readable'],3)
        self.assertEqual(report['padding_count'],2)
        self.assertEqual(report['no_padding_count'],1)
        self.assertEqual(len(report['requires_padding']),2)
        self.assertEqual(report['too_small'],[])
        self.assertEqual(report['errors'],[])
        self.assertEqual(report['skipped'],0)
        self.assertEqual(len(report['records']),3)

    def test_full_data_check_accepts_approved_short_images(self):
        self.domains()
        run(self.args('data-check'))
        report = self.report()
        self.assertEqual(report['underwater']['padding_count'],2)
        self.assertEqual(report['inair']['padding_count'],0)
        self.assertEqual(report['underwater']['too_small'],[])
        self.assertTrue((self.root/'runs'/'check'/'end.json').is_file())

    def test_bad_image_still_blocks_train_even_if_manifest_excludes_it(self):
        self.domains()
        (self.root/'u'/'unlisted-bad.png').write_bytes(b'not an image')
        with patch('train_stage1.build_model',side_effect=ModelBoundary):
            with self.assertRaisesRegex(ValueError,'data-check blocked'):
                run(self.args('train'))
        report = self.report()['underwater']
        self.assertEqual(report['count'],4)
        self.assertEqual(len(report['errors']),1)
        self.assertEqual(report['padding_count'],2)
        self.assertEqual(report['skipped'],0)

    def test_smoke_and_train_preflight_use_same_uvtd_policy(self):
        self.domains()
        reports=[]
        for mode in ('smoke','train'):
            with patch('train_stage1.build_model',side_effect=ModelBoundary):
                with self.assertRaises(ModelBoundary):
                    run(self.args(mode,mode))
            reports.append(self.report(mode)['underwater'])
        self.assertEqual(reports[0],reports[1])
        self.assertEqual(reports[0]['padding_count'],2)
        self.assertEqual(self.report('train','training_data_check.json')['underwater'],reports[0])

    def test_dataset_yields_both_padded_orientations(self):
        self.domains()
        underwater=[self.root/'u'/'landscape.png',self.root/'u'/'portrait.png']
        dataset=data.Stage1Dataset(underwater,data.image_paths(self.root/'a'))
        for i in (0,1):
            batch=dataset[i]
            self.assertEqual(tuple(batch['Xu'].shape),(3,256,256))
            self.assertEqual(tuple(batch['Xa'].shape),(3,256,256))
            self.assertTrue(torch.isfinite(batch['Xu']).all())

    def test_no_unapproved_padding_for_air_or_other_small_sizes(self):
        with self.assertRaises(ValueError):
            data.prepare_image(Image.new('RGB',(352,224)),allow_uvtd_reflect=False)
        for size in ((352,223),(223,352),(128,128)):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    data.prepare_image(Image.new('RGB',size),allow_uvtd_reflect=True)


if __name__=='__main__':
    unittest.main(verbosity=2)
