"""Run: python -m unittest discover -s tests -v (CPU; no dataset download)."""
import unittest
import torch
from common import windows
from student import build_model


class ContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.model = build_model(dict(vocab=2048,width=32,heads=4,depth=2,context=256)).eval()

    def test_future_inputs_cannot_change_earlier_predictions(self):
        x = torch.randint(0,2048,(2,12))
        changed = x.clone(); changed[:,7:] = (changed[:,7:]+19)%2048
        with torch.no_grad():
            a, b = self.model.predict_log_probs(x), self.model.predict_log_probs(changed)
        torch.testing.assert_close(a[:,:7],b[:,:7],atol=1e-6,rtol=1e-6)

    def test_probabilities_are_normalized_and_examples_independent(self):
        x = torch.randint(0,2048,(2,12))
        with torch.no_grad():
            together = self.model.predict_log_probs(x)
            alone = self.model.predict_log_probs(x[:1])
        self.assertEqual(tuple(together.shape),(2,12,2048))
        torch.testing.assert_close(together.logsumexp(-1),torch.zeros(2,12),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(together[:1],alone,atol=1e-5,rtol=1e-5)

    def test_state_resets_between_windows(self):
        x = torch.randint(0,2048,(1,12))
        with torch.no_grad():
            first = self.model.predict_log_probs(x)
            self.model.predict_log_probs((x+31)%2048)
            again = self.model.predict_log_probs(x)
        torch.testing.assert_close(first,again,atol=1e-6,rtol=1e-6)

    def test_shifted_loss_produces_gradients(self):
        x = torch.randint(0,2048,(2,13))
        loss = torch.nn.functional.cross_entropy(self.model(x[:,:-1]).flatten(0,1),x[:,1:].flatten())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(gradients, 'The trainable model must receive gradients.')
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients),0)

    def test_every_target_is_counted_once_including_last_short_window(self):
        tokens = torch.arange(2*256+7)
        targets = torch.cat([y[y!=-100] for _,y in windows(tokens,batch_size=2)])
        torch.testing.assert_close(targets,tokens[1:])


if __name__ == '__main__':
    unittest.main()
