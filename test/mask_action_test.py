from torch_ac.utils.helper import mask_tensor
from torch.distributions.categorical import Categorical
import torch
import unittest

EPS = 1e-8

class TestMaskTensor(unittest.TestCase):
    
    def test_basic_masking(self):
        """Test basic masking with binary mask"""
        logits = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([1.0, 0.0, 1.0])
        categorical = Categorical(logits=logits)
        
        result = mask_tensor(categorical, mask)
        
        # Check that masked position has very low probability
        probs = result.probs
        self.assertLess(probs[1].item(), 1e-6)
        # Check that unmasked positions are positive
        self.assertGreater(probs[0].item(), 0.0)
        self.assertGreater(probs[2].item(), 0.0)
    
    def test_all_ones_mask(self):
        """Test that all-ones mask doesn't change distribution significantly"""
        logits = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.ones(3)
        categorical = Categorical(logits=logits)
        
        result = mask_tensor(categorical, mask)
        
        # Probabilities should be very close to original
        original_probs = categorical.probs
        result_probs = result.probs
        torch.testing.assert_close(original_probs, result_probs, rtol=1e-5, atol=1e-5)
    
    def test_all_zeros_mask(self):
        """Test behavior with all-zeros mask (edge case)"""
        logits = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.zeros(3)
        categorical = Categorical(logits=logits)
        
        result = mask_tensor(categorical, mask)
        
        # All probabilities should be almost the same as original (uniform distribution)
        probs = result.probs
        self.assertTrue(torch.allclose(probs, categorical.probs, atol=1e-5))
    
    def test_batch_dimensions(self):
        """Test with batched logits and masks"""
        logits = torch.tensor([[1.0, 2.0, 3.0], 
                               [4.0, 5.0, 6.0]])
        mask = torch.tensor([[1.0, 0.0, 1.0],
                            [1.0, 1.0, 0.0]])
        categorical = Categorical(logits=logits)
        
        result = mask_tensor(categorical, mask)
        
        # Check batch dimension is preserved
        self.assertEqual(result.logits.shape, logits.shape)
        
        # Check masking works for each batch
        probs = result.probs
        self.assertLess(probs[0, 1].item(), 1e-6)  # First batch, second position
        self.assertLess(probs[1, 2].item(), 1e-6)  # Second batch, third position
    
    def test_partial_mask_values(self):
        """Test with mask values between 0 and 1"""
        logits = torch.tensor([1.0, 1.0, 1.0])
        mask = torch.tensor([1.0, 0.5, 0.1])
        categorical = Categorical(logits=logits)
        
        result = mask_tensor(categorical, mask)
        
        # Probabilities should be reduced proportionally
        probs = result.probs
        p1_p2_ratio = probs[0].item() / probs[1].item()
        p1_p3_ratio = probs[0].item() / probs[2].item()
        p2_p3_ratio = probs[1].item() / probs[2].item()

        self.assertAlmostEqual(p1_p2_ratio, 2.0, delta=1e-4)
        self.assertAlmostEqual(p1_p3_ratio, 10.0, delta=1e-4)
        self.assertAlmostEqual(p2_p3_ratio, 5.0, delta=1e-4)
    
    def test_output_type(self):
        """Test that output is a Categorical distribution"""
        logits = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([1.0, 0.0, 1.0])
        categorical = Categorical(logits=logits)
        
        result = mask_tensor(categorical, mask)
        
        self.assertIsInstance(result, Categorical)
    
    def test_logits_shape_preserved(self):
        """Test that logits shape is preserved"""
        logits = torch.randn(5, 10)
        mask = torch.ones(5, 10)
        categorical = Categorical(logits=logits)
        
        result = mask_tensor(categorical, mask)
        
        self.assertEqual(result.logits.shape, logits.shape)
    
    def test_single_unmasked_element(self):
        """Test with only one unmasked element"""
        logits = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([0.0, 1.0, 0.0])
        categorical = Categorical(logits=logits)
        
        result = mask_tensor(categorical, mask)
        
        # Only middle element should have high probability
        probs = result.probs
        self.assertGreater(probs[1].item(), 0.99)
        self.assertLess(probs[0].item(), 0.01)
        self.assertLess(probs[2].item(), 0.01)


if __name__ == '__main__':
    unittest.main()
