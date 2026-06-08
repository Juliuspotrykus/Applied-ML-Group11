import io
import random
import unittest

import numpy as np
import tifffile
from fastapi.testclient import TestClient
from main import app
from PIL import Image


class APITest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_predict_rgb(self):
        """Test for RGB prediction endpoint"""
        # Create a dummy RGB image of the right
        # shape with a random color
        color = tuple(random.randint(0, 255) for _ in range(3))
        img = Image.new("RGB", (64, 64), color=color)

        # Convert image to correct format for POST request,
        # necessary when not loading a specific image from path
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        response = self.client.post(
            "/predict_rgb",
            files={"image": ("test.jpg", buf)},
        )

        # POST request should be succesful
        assert response.status_code == 200
        data = response.json()

        # Response should have a predictions field containing
        # 10 confidence values, one for each possible class
        assert len(data["predictions"]) == 10

    def test_rgb_error(self):
        """Tests error cases in RGB prediction endpoint"""
        # Send a string with the image argument
        response = self.client.post(
            "/predict_rgb",
            files={"image": ("string")},
        )

        # POST request should throw an error.
        assert response.status_code == 415

    def test_predict_ms(self):
        """Test for MS prediction endpoint"""
        img = np.random.randint(
            low=0,
            high=10000,
            size=(64, 64, 13),
        )

        # Convert image to correct format for POST request,
        # necessary when not loading a specific image from path
        buf = io.BytesIO()
        tifffile.imwrite(buf, img)

        response = self.client.post(
            "/predict_ms",
            files={"image": ("test.tif", buf)},
        )

        assert response.status_code == 200

        data = response.json()

        assert "predictions" in data
        assert len(data["predictions"]) == 10

    def test_ms_error(self):
        """Tests error cases in MS prediction endpoint"""
        # Send a string with the image argument
        response = self.client.post(
            "/predict_rgb",
            files={"image": ("string")},
        )

        # POST request should throw an error.
        assert response.status_code == 415

        # Generate image with incorrect number of bands
        img = np.random.randint(
            low=0,
            high=10000,
            size=(64, 64, 3),
        )

        buf = io.BytesIO()
        tifffile.imwrite(buf, img)

        response = self.client.post(
            "/predict_ms",
            files={"image": ("test.tif", buf)},
        )

        # POST request should throw an error indicating
        # that the image has the wrong number of bands.
        assert response.status_code == 422
