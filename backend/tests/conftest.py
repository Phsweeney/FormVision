"""Shared pytest fixtures.

The key trick: the API tests run against a temporary data directory and an
injected synthetic pose estimator, so the full HTTP flow — upload, analyze,
poll, fetch, stream — is exercised with no MediaPipe, no model download, and no
real squat footage.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import Settings, get_settings


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch) -> Settings:
    """Settings pointed at a throwaway data directory.

    `get_settings` is `lru_cache`d, so the cache is cleared before *and* after:
    before so this test gets fresh settings, after so a later test does not
    inherit a temp path that has since been deleted.
    """
    monkeypatch.setenv("FORMVISION_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv(
        "FORMVISION_DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    )
    get_settings.cache_clear()

    settings = get_settings()
    settings.ensure_directories()
    yield settings

    get_settings.cache_clear()


@pytest.fixture
def client(isolated_settings, monkeypatch):
    """A `TestClient` wired to the isolated database and a synthetic estimator."""
    from fastapi.testclient import TestClient

    from app.db.database import reset_engine_cache
    from app.services import analysis_service
    from tests.synthetic import SyntheticPoseEstimator, build_squat_series

    # The engine is cached per process and would otherwise still point at the
    # previous test's database file.
    reset_engine_cache()

    from app.main import create_app

    application = create_app()

    analysis_service.set_estimator_override(
        SyntheticPoseEstimator(build_squat_series(reps=3))
    )

    # FastAPI's BackgroundTasks run after the response is sent but before
    # TestClient returns, so `POST /analyze` completes synchronously here. That
    # is what lets these tests assert on finished results without polling.
    with TestClient(application) as test_client:
        yield test_client

    analysis_service.set_estimator_override(None)
    reset_engine_cache()


@pytest.fixture
def sample_video(tmp_path):
    """A small, genuinely decodable video file.

    Real bytes, so upload validation and the overlay renderer both run their
    actual code paths. The landmarks come from the synthetic estimator, so the
    picture itself is irrelevant.
    """
    path = tmp_path / "squat.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 480))
    assert writer.isOpened()
    for _ in range(180):
        writer.write(np.full((480, 320, 3), 50, dtype=np.uint8))
    writer.release()
    assert path.stat().st_size > 0
    return path


@pytest.fixture
def uploaded(client, sample_video):
    """Upload the sample video and return the parsed response body."""
    with sample_video.open("rb") as handle:
        response = client.post(
            "/upload", files={"file": ("squat.mp4", handle, "video/mp4")}
        )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
def analysed(client, uploaded):
    """Upload, analyse to completion, and return the final analysis body."""
    response = client.post("/analyze", json={"analysis_id": uploaded["id"]})
    assert response.status_code == 202, response.text

    result = client.get(f"/analysis/{uploaded['id']}")
    assert result.status_code == 200
    body = result.json()
    assert body["status"] == "completed", body.get("error_message")
    return body
