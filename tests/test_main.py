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

        # Create a dummy RGB image.
        color = tuple(random.randint(0, 255) for _ in range(3))
        img = Image.new("RGB", (64, 64), color=color)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        self.rgb_buf = buf

        # Create a dummy TIF image.
        img = np.random.randint(
            low=0,
            high=10000,
            size=(64, 64, 13),
        )
        buf = io.BytesIO()
        tifffile.imwrite(buf, img)
        self.ms_buf = buf

        # Create dummy TIF image with wrong number of bands
        # Generate image with incorrect number of bands
        img = np.random.randint(
            low=0,
            high=10000,
            size=(64, 64, 3),
        )
        buf = io.BytesIO()
        tifffile.imwrite(buf, img)
        self.ms_buf_incorrect = buf

    def test_predict_rgb(self):
        """Test for RGB prediction endpoint"""
        response = self.client.post(
            "/predict_rgb",
            files={"image": ("test.jpg", self.rgb_buf)},
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
        response = self.client.post(
            "/predict_ms",
            files={"image": ("test.tif", self.ms_buf)},
        )

        assert response.status_code == 200

        data = response.json()
        assert len(data["predictions"]) == 10

    def test_ms_error(self):
        """Tests error cases in MS prediction endpoint"""
        # Send a string with the image argument
        response = self.client.post(
            "/predict_ms",
            files={"image": ("string")},
        )

        # POST request should throw an error.
        assert response.status_code == 415

        response = self.client.post(
            "/predict_ms",
            files={"image": ("test.tif", self.ms_buf_incorrect)},
        )

        # POST request should throw an error indicating
        # that the image has the wrong number of bands.
        assert response.status_code == 422

    def test_rgb_explainability(self):
        """Tests for RGB explainability endpoint"""
        # Response should throw no errors.
        response = self.client.post(
            "/explain_rgb",
            files={"image": ("test.jpg", self.rgb_buf)},
        )

        # Check for correct response and look for image output
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_rgb_explainability_error(self):
        """Tests error cases in RGB explainability endpoint"""
        # Send string instead of image to endpoint.
        response = self.client.post(
            "/explain_rgb",
            files={"image": ("test.jpg", "string")},
        )

        assert response.status_code == 415

        # Send incorrect target class to endpoint.
        response = self.client.post(
            "/explain_rgb",
            files={"target_class": 14, "image": ("test.jpg", "string")},
        )

        assert response.status_code == 400

    def test_ms_explainability(self):
        """Tests for MS explainability endpoint"""
        # Response should throw no errors.
        response = self.client.post(
            "/explain_ms",
            files={"image": ("test.jpg", self.ms_buf)},
        )

        # Check for correct response and look for image output
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_ms_explainability_errors(self):
        """Test errors for MS explainability endpoint"""
        # Send incorrect target class to endpoint.
        response = self.client.post(
            "/explain_ms",
            files={"target_class": 14, "image": ("test.jpg", "string")},
        )
        assert response.status_code == 400

        # Send string instead of image to endpoint.
        response = self.client.post(
            "/explain_ms",
            files={"image": ("test.jpg", "string")},
        )
        assert response.status_code == 415

        # Send non-TIFF image to endpoint:
        response = self.client.post(
            "/explain_ms",
            files={"image": ("test.jpg", self.rgb_buf)}
        )
        assert response.status_code == 415
