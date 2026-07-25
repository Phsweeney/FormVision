"""Tests for the offline training pipeline.

Kept out of `backend/tests/` because they need pandas, which is a training-only
dependency. Run them explicitly:

    .venv/Scripts/python -m pytest training/tests/
"""
