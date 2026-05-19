import unittest

import torch
from project_name.data.preprocessors import normalize_MS_img


class PreprocessingTest(unittest.TestCase):
    def test_MS(self):
        # Generate a random tensor with shape (13, 64, 64),
        # filled with somewhat realistic values.
        ms_img = torch.randint(
            low=0,
            high=10000,
            size=(13, 64, 64),
        )

        preprocessed_img = normalize_MS_img(ms_img)

        # For each band, check that the values are between the
        # expected minimum and expected maximum values.
        # TODO

        # Sanity check that all values are between -3 and 3,
        # should be consistent because of the clipping.
        self.assertTrue(torch.all(preprocessed_img >= -3))
        self.assertTrue(torch.all(preprocessed_img <= 3))


if __name__ == "__main__":
    unittest.main()
