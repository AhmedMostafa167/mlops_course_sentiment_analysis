"""Docker image checks. Placeholder until the Dockerfile exists."""

import pytest


@pytest.mark.skip(reason="TODO: Dockerfile not written yet")
def test_container_healthcheck_reports_healthy():
    # Plan:
    # 1. Build the image and start a container.
    # 2. Wait for the Docker HEALTHCHECK (which calls /health) to report "healthy".
    # 3. assert GET /health returns 200 and {"status": "healthy", ...}
    ...
