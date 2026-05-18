import unittest
import torch

from project_name.data.preprocessors import normalize_RGB_img


class PreprocessingTest(unittest.TestCase):
    def test_RGB(self):
        # Generate a random tensor with shape (3, 64, 64)
        rgb_img = torch.randn(3, 64, 64)
        preprocessed_img = normalize_RGB_img(rgb_img)
        self.assertTrue(torch.equal(preprocessed_img, rgb_img / 255))


if __name__ == "__main__":
    unittest.main()
