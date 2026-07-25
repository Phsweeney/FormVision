"""Offline model training. Not imported by the application.

Everything here runs by hand to produce the artifact in `app/ml/artifacts/`,
and nothing in `app/` imports this package. That separation is why pandas is a
training-only dependency and never reaches the Docker image.

The one thing this package must not do is define its own idea of a feature. It
imports `app.ml.features` like the runtime does, so the two cannot drift.
"""
