from pathlib import Path

import steklov_arap.steklov as steklov


def test_steklov_cache_dir_defaults_to_home_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / ".cache" / "steklov_arap"
    monkeypatch.setattr(steklov, "STEKLOV_CACHE_DIR", cache_dir)
    monkeypatch.delenv(steklov.STEKLOV_CACHE_ENV_VAR, raising=False)

    assert steklov.steklov_cache_dir() == cache_dir
    assert cache_dir.is_dir()


def test_steklov_cache_dir_uses_env_override(monkeypatch, tmp_path):
    cache_dir = tmp_path / "operator-cache"
    monkeypatch.setenv(steklov.STEKLOV_CACHE_ENV_VAR, str(cache_dir))

    assert steklov.steklov_cache_dir() == cache_dir
    assert cache_dir.is_dir()


def test_steklov_cache_dir_expands_user(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(steklov.STEKLOV_CACHE_ENV_VAR, "~/steklov-arap-test-cache")

    cache_dir = Path.home() / "steklov-arap-test-cache"
    assert steklov.steklov_cache_dir() == cache_dir
    assert cache_dir.is_dir()
