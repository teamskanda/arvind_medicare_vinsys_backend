"""Verify TLS connection options without contacting a database."""

import ssl
from types import SimpleNamespace
from unittest.mock import Mock

from sqlalchemy.engine import make_url

from helpers import database


def test_verify_full_uses_system_certificates(monkeypatch):
    settings = SimpleNamespace(
        database_url="postgresql+asyncpg://user:password@db.example.com/db?ssl=verify-full",
        db_echo=False,
        db_pool_size=5,
        db_max_overflow=5,
        db_command_timeout_seconds=15.0,
    )
    factory = Mock()
    monkeypatch.setattr(database, "get_settings", lambda: settings)
    monkeypatch.setattr(database, "create_async_engine", factory)

    assert database._build_engine() is factory.return_value

    args, kwargs = factory.call_args
    url = make_url(args[0])
    assert "ssl" not in url.query
    assert url.host == "db.example.com"
    context = kwargs["connect_args"]["ssl"]
    assert isinstance(context, ssl.SSLContext)
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED


def test_local_database_does_not_force_tls(monkeypatch):
    settings = SimpleNamespace(
        database_url="postgresql+asyncpg://user:password@localhost/db",
        db_echo=False,
        db_pool_size=5,
        db_max_overflow=5,
        db_command_timeout_seconds=15.0,
    )
    factory = Mock()
    monkeypatch.setattr(database, "get_settings", lambda: settings)
    monkeypatch.setattr(database, "create_async_engine", factory)

    database._build_engine()

    assert "ssl" not in factory.call_args.kwargs["connect_args"]
