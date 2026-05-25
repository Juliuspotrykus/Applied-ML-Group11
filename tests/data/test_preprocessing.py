import unittest

import torch
from eurosat_classification.data.preprocessors import (
    clip_maxs,
    clip_mins,
    means,
    normalize_MS_img,
    stdevs,
)


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        # Generate a random tensor with shape (13, 64, 64),
        # filled with somewhat realistic values.
        self.ms_img = torch.randint(
            low=0,
            high=10000,
            size=(13, 64, 64),
        )
        self.preprocessed_ms_img = normalize_MS_img(self.ms_img)

    def test_shape(self):
        # Check that shapes are identical after preprocessing.
        self.assertTrue(self.ms_img.shape == self.preprocessed_ms_img.shape)

    def test_expected_values(self):
        # For each band, check that the values are between the
        # expected minimum and expected maximum values.
        for band in range(13):
            band_vals = self.preprocessed_ms_img[band, :, :]
            expected_minimum = (clip_mins[band] - means[band]) / stdevs[band]
            expected_maximum = (clip_maxs[band] - means[band]) / stdevs[band]
            self.assertTrue(torch.all(band_vals >= expected_minimum))
            self.assertTrue(torch.all(band_vals <= expected_maximum))

    def test_extreme_values(self):
        # Sanity check that all values are between -3 and 3,
        # should be consistent because of the clipping.
        self.assertTrue(torch.all(self.preprocessed_ms_img >= -3))
        self.assertTrue(torch.all(self.preprocessed_ms_img <= 3))


if __name__ == "__main__":
    unittest.main()
