import unittest
import torch
from model import create_model

class TestModel(unittest.TestCase):
    def test_tensorflow_model(self):
        model = create_model('tensorflow')
        self.assertIsNotNone(model)
        # Add more specific tests for TensorFlow model

    def test_pytorch_model(self):
        model = create_model('pytorch')
        self.assertIsNotNone(model)
        # Test forward pass
        x = torch.randn(1, 28, 28)
        output = model(x)
        self.assertEqual(output.shape, torch.Size([1, 10]))

if __name__ == '__main__':
    unittest.main()

