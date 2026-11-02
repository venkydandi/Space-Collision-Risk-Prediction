"""Verification tests for Dockerfile and Render-only deployment configuration."""
from __future__ import annotations

import unittest
from pathlib import Path
import yaml

from config.settings import MODELS_DIR, PROJECT_ROOT


class DockerAndRenderConfigTests(unittest.TestCase):
    def test_dockerfile_exists_and_configured(self):
        dockerfile_path = PROJECT_ROOT / "Dockerfile"
        self.assertTrue(dockerfile_path.is_file(), "Dockerfile must exist at project root.")
        content = dockerfile_path.read_text(encoding="utf-8")

        # 1. Base image
        self.assertIn("FROM python:3.11-slim", content)

        # 2. Dependency install
        self.assertIn("pip install", content)
        self.assertIn("requirements.txt", content)

        # 3. Model and app copy
        self.assertIn("COPY . .", content)

        # 4. Render PORT dynamic expansion
        self.assertIn("${PORT:-8501}", content)
        self.assertIn("sh", content)
        self.assertIn("streamlit run app.py", content)
        self.assertIn("--server.address=0.0.0.0", content)

        # 5. Healthcheck endpoint
        self.assertIn("/_stcore/health", content)

    def test_render_yaml_specification(self):
        render_yaml_path = PROJECT_ROOT / "render.yaml"
        self.assertTrue(render_yaml_path.is_file(), "render.yaml must exist at project root.")
        
        with open(render_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertIn("services", data)
        self.assertEqual(len(data["services"]), 1)
        svc = data["services"][0]
        self.assertEqual(svc["type"], "web")
        self.assertEqual(svc["runtime"], "docker")
        self.assertEqual(svc["healthCheckPath"], "/_stcore/health")

    def test_model_artifacts_present_in_repository(self):
        m1 = MODELS_DIR / "rf_unweighted.joblib"
        m2 = MODELS_DIR / "rf_smotenc_025.joblib"
        meta = MODELS_DIR / "metadata.json"

        self.assertTrue(m1.is_file(), "rf_unweighted.joblib must be in models/")
        self.assertTrue(m2.is_file(), "rf_smotenc_025.joblib must be in models/")
        self.assertTrue(meta.is_file(), "metadata.json must be in models/")

        self.assertGreater(m1.stat().st_size, 1_000_000, "Unweighted model file must be complete")
        self.assertGreater(m2.stat().st_size, 3_000_000, "SMOTENC model file must be complete")


if __name__ == "__main__":
    unittest.main()
