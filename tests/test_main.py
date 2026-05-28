import io
import random
import unittest

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
