"""CPU equivalence and synchronization-budget tests; no CUDA experiment."""
import copy
import unittest
from unittest.mock import patch
import torch
import stage1
from test_stage1 import tiny_model


def reference_step(model, optimizer, xa, xu):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ya, yu = model(xa, xu)
    losses = stage1.stage1_losses(xa, xu, ya, yu)
    losses['total'].backward()
    grads = [p.grad for p in model.G.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError('Missing/nonfinite generator gradients')
    if not any(torch.count_nonzero(g).item() for g in grads):
        raise RuntimeError('All generator gradients are zero')
    if any(p.grad is not None for p in model.E.parameters()):
        raise RuntimeError('Frozen encoder accumulated gradients')
    optimizer.step()
    return {key: float(value.detach()) for key, value in losses.items()}


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        stage1.seed_everything(123)

    def test_gradient_validation_uses_one_host_boolean_for_all_finite_flags(self):
        grads = [torch.ones(3) for _ in range(40)]
        original = torch.Tensor.__bool__
        reads = []
        def counted(tensor):
            reads.append(1)
            return original(tensor)
        with patch.object(torch.Tensor, '__bool__', counted):
            stage1.validate_gradients(grads)
        self.assertEqual(len(reads), 1)

    def test_gradient_guards_still_reject_missing_zero_and_late_nonfinite(self):
        for grads, message in (([], 'Missing/nonfinite'), ([torch.zeros(4)], 'zero'),
                               ([torch.ones(4)] * 30 + [torch.tensor([float('nan')])], 'Missing/nonfinite'),
                               ([torch.ones(4), torch.tensor([float('inf')])], 'Missing/nonfinite')):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                stage1.validate_gradients(grads)
        stage1.validate_gradients([torch.zeros(4), torch.ones(4)])

    def test_identical_losses_gradients_adam_state_and_rng(self):
        model = tiny_model()
        other = copy.deepcopy(model)
        optimizer = stage1.make_optimizer(model.G)
        other_optimizer = stage1.make_optimizer(other.G)
        xa, xu = torch.rand(2,3,256,256)*2-1, torch.rand(2,3,256,256)*2-1
        rng = torch.get_rng_state().clone()
        expected = reference_step(model, optimizer, xa, xu)
        actual = stage1.train_step(other, other_optimizer, xa, xu)
        self.assertEqual(expected, actual)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for p, q in zip(model.G.parameters(), other.G.parameters()):
            self.assertTrue(torch.equal(p.grad, q.grad))
            self.assertTrue(torch.equal(p, q))
            for key in ('step','exp_avg','exp_avg_sq'):
                self.assertTrue(torch.equal(optimizer.state[p][key], other_optimizer.state[q][key]))
        self.assertTrue(all(torch.equal(v, other.E.state_dict()[k]) for k,v in model.E.state_dict().items()))

    def test_metrics_do_not_extract_individual_device_scalars(self):
        model = tiny_model()
        optimizer = stage1.make_optimizer(model.G)
        x = torch.rand(2,3,256,256)*2-1
        with patch.object(torch.Tensor, '__float__', side_effect=AssertionError('batch metric transfer required')):
            metrics = stage1.train_step(model, optimizer, x, -x)
        self.assertEqual(set(metrics), {'rec_a','rec_u','sr','total','sa','su'})
        self.assertTrue(all(isinstance(v,float) for v in metrics.values()))

    def test_image_contract_still_rejects_all_invalid_values(self):
        for v in (float('nan'),float('inf'),-1.01,1.01):
            with self.subTest(value=v), self.assertRaises(ValueError):
                stage1.validate_image(torch.full((2,3,4,4),v))
        stage1.validate_image(torch.tensor([-1.,0.,1.]).view(1,3,1,1))
