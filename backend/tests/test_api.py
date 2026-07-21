"""End-to-end API tests.

Exercises the full HTTP surface — upload, analyze, poll, fetch results, stream
media — against an isolated database and a synthetic pose estimator. No
MediaPipe, no model download, no real footage.
"""

from __future__ import annotations

import io

import pytest


class TestHealth:
    def test_health_reports_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["version"]

    def test_openapi_documents_every_endpoint(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        for expected in (
            "/health",
            "/upload",
            "/analyze",
            "/analysis/{analysis_id}",
            "/video/{analysis_id}",
            "/overlay/{analysis_id}",
        ):
            assert expected in paths, f"{expected} missing from the OpenAPI schema"


class TestUpload:
    def test_accepts_a_valid_video(self, uploaded):
        assert uploaded["status"] == "uploaded"
        assert uploaded["filename"] == "squat.mp4"
        assert uploaded["size_bytes"] > 0
        assert len(uploaded["id"]) == 32

    def test_ids_are_unguessable(self, client, sample_video):
        """Sequential ids would let anyone enumerate other people's uploads."""
        ids = set()
        for _ in range(3):
            with sample_video.open("rb") as handle:
                response = client.post(
                    "/upload", files={"file": ("squat.mp4", handle, "video/mp4")}
                )
            ids.add(response.json()["id"])
        assert len(ids) == 3
        assert all(len(i) == 32 for i in ids)

    def test_rejects_a_disallowed_extension(self, client):
        response = client.post(
            "/upload", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
        )
        assert response.status_code == 415
        assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA"

    def test_rejects_a_text_file_renamed_as_mp4(self, client):
        """The case header validation cannot catch.

        Extension and content type both look fine; only opening the file
        reveals it is not a video.
        """
        response = client.post(
            "/upload",
            files={"file": ("fake.mp4", io.BytesIO(b"not a video at all"), "video/mp4")},
        )
        assert response.status_code == 415
        assert "video" in response.json()["error"]["message"].lower()

    def test_rejects_an_empty_file(self, client):
        response = client.post(
            "/upload", files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")}
        )
        assert response.status_code == 400

    def test_rejects_an_oversized_file(self, client, isolated_settings, monkeypatch):
        monkeypatch.setattr(isolated_settings, "max_upload_bytes", 1024)
        response = client.post(
            "/upload",
            files={"file": ("big.mp4", io.BytesIO(b"x" * 5000), "video/mp4")},
        )
        assert response.status_code == 413

    def test_a_rejected_upload_leaves_no_file_behind(self, client, isolated_settings):
        """Otherwise the disk accumulates junk nothing references."""
        before = list(isolated_settings.upload_dir.glob("*"))
        client.post(
            "/upload",
            files={"file": ("fake.mp4", io.BytesIO(b"still not a video"), "video/mp4")},
        )
        assert list(isolated_settings.upload_dir.glob("*")) == before

    def test_probes_real_video_properties(self, uploaded, client):
        """Dimensions come from decoding the file, not from the client."""
        record = client.get(f"/analysis/{uploaded['id']}").json()
        assert record["status"] == "uploaded"


class TestAnalyze:
    def test_returns_202_and_starts_work(self, client, uploaded):
        response = client.post("/analyze", json={"analysis_id": uploaded["id"]})
        assert response.status_code == 202
        assert response.json()["id"] == uploaded["id"]

    def test_unknown_id_is_404(self, client):
        response = client.post("/analyze", json={"analysis_id": "0" * 32})
        assert response.status_code == 404

    def test_malformed_id_is_rejected(self, client):
        """The path-traversal guard: ids must be strict hex."""
        for bad in ("../../etc/passwd", "abc", "x" * 32):
            response = client.post("/analyze", json={"analysis_id": bad})
            assert response.status_code in (400, 404), bad

    def test_missing_body_is_422(self, client):
        assert client.post("/analyze", json={}).status_code == 422


class TestAnalysisResults:
    def test_completes_and_reports_reps(self, analysed):
        assert analysed["status"] == "completed"
        assert analysed["metrics"]["total_reps"] == 3
        assert analysed["processing_seconds"] > 0

    def test_returns_every_result_section(self, analysed):
        for section in ("video", "metrics", "reps", "feedback", "series"):
            assert analysed[section] is not None, f"{section} missing"

    def test_reports_the_camera_view(self, analysed):
        """The dashboard needs it to explain which cards are blank and why."""
        assert analysed["metrics"]["camera_view"] in {
            "side",
            "front",
            "oblique",
            "unknown",
        }

    def test_reps_are_fully_described(self, analysed):
        assert len(analysed["reps"]) == 3
        for rep in analysed["reps"]:
            assert rep["start_time_s"] < rep["bottom_time_s"] < rep["end_time_s"]
            assert rep["duration_s"] > 0
            assert rep["depth_percent"] is not None

    def test_feedback_items_carry_an_explanation(self, analysed):
        assert analysed["feedback"]
        for item in analysed["feedback"]:
            assert item["rule_id"] and item["title"]
            assert item["message"] and item["explanation"]
            assert item["severity"] in {"good", "info", "warning", "critical"}

    def test_series_are_index_aligned(self, analysed):
        """Charts plot these against a shared axis; unequal lengths would
        silently misalign every curve."""
        series = analysed["series"]
        length = len(series["time_s"])
        for key in (
            "left_knee_deg",
            "right_knee_deg",
            "hip_deg",
            "torso_lean_deg",
            "hip_height",
        ):
            assert len(series[key]) == length, key
        assert series["sample_count"] == length

    def test_series_are_decimated(self, analysed, isolated_settings):
        series = analysed["series"]
        assert series["sample_count"] <= isolated_settings.max_series_points
        assert series["source_frame_count"] >= series["sample_count"]

    def test_time_axis_is_monotonic(self, analysed):
        times = analysed["series"]["time_s"]
        assert times == sorted(times)

    def test_media_urls_are_present(self, analysed):
        assert analysed["video_url"].endswith(analysed["id"])
        assert analysed["overlay_url"] is not None

    def test_unknown_id_is_404(self, client):
        assert client.get(f"/analysis/{'0' * 32}").status_code == 404

    def test_malformed_id_is_rejected(self, client):
        assert client.get("/analysis/not-a-valid-id").status_code in (400, 404)

    def test_listing_returns_summaries(self, client, analysed):
        body = client.get("/analyses").json()
        assert any(item["id"] == analysed["id"] for item in body)


class TestMedia:
    def test_original_video_streams(self, client, analysed):
        response = client.get(f"/video/{analysed['id']}")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("video/")
        assert len(response.content) > 0

    def test_overlay_streams_as_mp4(self, client, analysed):
        response = client.get(f"/overlay/{analysed['id']}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"
        assert len(response.content) > 0

    def test_range_requests_are_supported(self, client, analysed):
        """Without Range, the player cannot seek - it can only play from 0."""
        response = client.get(
            f"/video/{analysed['id']}", headers={"Range": "bytes=0-1023"}
        )
        assert response.status_code == 206
        assert "content-range" in response.headers
        assert len(response.content) == 1024

    def test_video_is_served_inline(self, client, analysed):
        """`attachment` would make the browser download rather than play it."""
        response = client.get(f"/video/{analysed['id']}")
        assert "inline" in response.headers.get("content-disposition", "")

    def test_media_is_cacheable(self, client, analysed):
        response = client.get(f"/video/{analysed['id']}")
        assert "max-age" in response.headers.get("cache-control", "")

    def test_overlay_404s_before_analysis(self, client, uploaded):
        assert client.get(f"/overlay/{uploaded['id']}").status_code == 404

    def test_unknown_media_is_404(self, client):
        assert client.get(f"/video/{'0' * 32}").status_code == 404

    def test_traversal_attempt_is_blocked(self, client):
        """An id is used to build a filesystem path, so it must be validated."""
        for attempt in ("..%2f..%2fetc%2fpasswd", "....//....//secret"):
            response = client.get(f"/video/{attempt}")
            assert response.status_code in (400, 404), attempt


class TestFailureHandling:
    def test_a_failing_pipeline_marks_the_record_failed(
        self, client, uploaded, monkeypatch
    ):
        """A background task must never leave a record stuck on `processing`.

        If it did, the client would poll forever with no explanation.
        """
        from app.services import analysis_service

        def explode(*_args, **_kwargs):
            raise RuntimeError("pipeline exploded")

        monkeypatch.setattr(analysis_service, "run_pipeline", explode)

        client.post("/analyze", json={"analysis_id": uploaded["id"]})
        body = client.get(f"/analysis/{uploaded['id']}").json()

        assert body["status"] == "failed"
        assert body["error_code"] == "INTERNAL_ERROR"
        assert body["error_message"]

    def test_a_domain_failure_surfaces_its_message(self, client, uploaded, monkeypatch):
        """Known failures carry text written for a human, safe to display."""
        from app.core.exceptions import VideoProcessingError
        from app.services import analysis_service

        def explode(*_args, **_kwargs):
            raise VideoProcessingError("This video could not be decoded.")

        monkeypatch.setattr(analysis_service, "run_pipeline", explode)

        client.post("/analyze", json={"analysis_id": uploaded["id"]})
        body = client.get(f"/analysis/{uploaded['id']}").json()

        assert body["status"] == "failed"
        assert body["error_code"] == "VIDEO_PROCESSING_ERROR"
        assert "decoded" in body["error_message"]

    def test_errors_use_a_consistent_envelope(self, client):
        """One shape for every error keeps the client's handling simple."""
        body = client.get(f"/analysis/{'0' * 32}").json()
        assert set(body) == {"error"}
        assert set(body["error"]) == {"code", "message", "detail"}


class TestFullFlow:
    def test_upload_analyze_and_retrieve(self, client, sample_video):
        """The complete user journey in one test."""
        with sample_video.open("rb") as handle:
            upload = client.post(
                "/upload", files={"file": ("my_squat.mp4", handle, "video/mp4")}
            )
        assert upload.status_code == 201
        analysis_id = upload.json()["id"]

        assert (
            client.post("/analyze", json={"analysis_id": analysis_id}).status_code == 202
        )

        result = client.get(f"/analysis/{analysis_id}").json()
        assert result["status"] == "completed"
        assert result["metrics"]["total_reps"] == 3
        assert result["feedback"]

        assert client.get(f"/video/{analysis_id}").status_code == 200
        assert client.get(f"/overlay/{analysis_id}").status_code == 200

    @pytest.mark.parametrize("reps", [1, 2, 5])
    def test_rep_count_reaches_the_api_intact(
        self, client, sample_video, reps, monkeypatch
    ):
        """Guards the whole chain from estimator to JSON response."""
        from app.services import analysis_service
        from tests.synthetic import SyntheticPoseEstimator, build_squat_series

        analysis_service.set_estimator_override(
            SyntheticPoseEstimator(build_squat_series(reps=reps))
        )

        with sample_video.open("rb") as handle:
            analysis_id = client.post(
                "/upload", files={"file": ("s.mp4", handle, "video/mp4")}
            ).json()["id"]
        client.post("/analyze", json={"analysis_id": analysis_id})

        body = client.get(f"/analysis/{analysis_id}").json()
        assert body["metrics"]["total_reps"] == reps
        assert len(body["reps"]) == reps
