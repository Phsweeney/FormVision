"""The machine-learning layer.

Everything here is optional. With no trained artifact on disk, or with
scikit-learn absent, the predictor resolves to a null implementation and the
pipeline produces exactly what it produced before this package existed. The
rules remain the floor; the model only ever adds.
"""
