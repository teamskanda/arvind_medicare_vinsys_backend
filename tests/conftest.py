"""Test configuration.

Dummy settings are injected before any test module imports `main`, which
reads configuration at import time. No real service is contacted: the tests
that use the app exercise paths that stop before any database or network
call.
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/skanda_test")
os.environ["DEMO_AUTH_ENABLED"] = "false"
os.environ["DEMO_AUTH_SECRET"] = "unit-test-secret-not-for-deployment-12345"
os.environ["DEBUG"] = "false"
os.environ.setdefault("FIRECRAWL_API_KEY", "fc-test-key")
os.environ.setdefault("ENVIRONMENT", "test")
