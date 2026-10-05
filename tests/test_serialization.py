"""PyTorch vs ONNX parity. Placeholder until the ONNX export exists."""

import pytest


@pytest.mark.skip(reason="TODO: ONNX export not implemented yet")
def test_onnx_matches_torch(sample_reviews):
    # Plan:
    # 1. Export the trained BertClassifier to ONNX (needs onnx + onnxruntime).
    # 2. Run sample_reviews through both the torch model and an onnxruntime session.
    # 3. assert np.allclose(torch_logits, onnx_logits, atol=1e-4)
    ...
