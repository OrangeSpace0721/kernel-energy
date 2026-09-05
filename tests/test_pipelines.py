"""Resolving where model weights come from.

Weights are large enough that re-downloading them is the most expensive mistake
available, so the override has to be reliable and its failures have to be legible. These
tests need no weights and no GPU -- they exercise the resolution logic against temporary
directories.
"""

from __future__ import annotations

import json

import pytest

from kernelenergy.trace.pipelines import PIPELINES, env_var_for, resolve_source


def test_env_var_names_are_predictable():
    assert env_var_for("flux1-dev") == "KE_MODEL_FLUX1_DEV"
    assert env_var_for("sd35-large") == "KE_MODEL_SD35_LARGE"
    assert env_var_for("qwen-image") == "KE_MODEL_QWEN_IMAGE"


@pytest.mark.parametrize("key", sorted(PIPELINES))
def test_every_pipeline_has_a_documentable_override(key):
    """The name must be derivable from the key alone, so config.sh can list them."""
    var = env_var_for(key)
    assert var.startswith("KE_MODEL_")
    assert var.replace("KE_MODEL_", "").replace("_", "").isalnum()


def test_default_is_the_hub_repo_id(monkeypatch):
    monkeypatch.delenv("KE_MODEL_FLUX1_DEV", raising=False)
    source, how = resolve_source("flux1-dev")
    assert (source, how) == (PIPELINES["flux1-dev"].repo, "hub")


def test_override_accepts_a_pipeline_directory(tmp_path, monkeypatch):
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "FluxPipeline"}))
    monkeypatch.setenv("KE_MODEL_FLUX1_DEV", str(tmp_path))
    source, how = resolve_source("flux1-dev")
    assert (source, how) == (str(tmp_path), "local")


def test_missing_path_is_reported_before_loading(tmp_path, monkeypatch):
    """Fail here, in a second, rather than minutes into a job with a URL in the error."""
    monkeypatch.setenv("KE_MODEL_FLUX1_DEV", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_source("flux1-dev")


def test_directory_without_model_index_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("KE_MODEL_FLUX1_DEV", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no model_index.json"):
        resolve_source("flux1-dev")


@pytest.mark.parametrize("marker", ["hub", "models--black-forest-labs--FLUX.1-dev"])
def test_a_hub_cache_in_the_wrong_variable_says_so(tmp_path, monkeypatch, marker):
    """The likely mistake, given a folder of weights and two ways to point at it.

    A hub cache handed to KE_MODEL_* fails the model_index.json check, and the bare
    message would send someone looking for a corrupt download. Naming the actual fix --
    use HF_HOME instead -- is the difference between a one-minute correction and an
    afternoon.
    """
    (tmp_path / marker).mkdir()
    monkeypatch.setenv("KE_MODEL_FLUX1_DEV", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="HF_HOME"):
        resolve_source("flux1-dev")


def _snapshot(root, components=("transformer", "vae", "text_encoder", "scheduler")):
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "FluxPipeline",
        "_diffusers_version": "0.35.0",
        **{c: ["diffusers", "Something"] for c in components},
    }))
    for c in components:
        (root / c).mkdir(exist_ok=True)
    return root


def test_complete_snapshot_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("KE_MODEL_FLUX1_DEV", str(_snapshot(tmp_path / "flux")))
    assert resolve_source("flux1-dev")[1] == "local"


def test_incomplete_snapshot_names_what_is_absent(tmp_path, monkeypatch):
    """model_index.json can list components an interrupted sync never wrote.

    Without this check the job loads for minutes and then fails reaching for the VAE,
    which on a queued allocation is an expensive way to learn the copy was truncated.
    """
    root = _snapshot(tmp_path / "flux")
    (root / "vae").rmdir()
    monkeypatch.setenv("KE_MODEL_FLUX1_DEV", str(root))
    with pytest.raises(FileNotFoundError, match=r"not on disk: \['vae'\]"):
        resolve_source("flux1-dev")


def test_component_only_directory_says_what_it_is(tmp_path, monkeypatch):
    """A distillation project may keep only the denoiser -- that is not a pipeline."""
    root = tmp_path / "flux1_teacher"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "diffusion_pytorch_model.safetensors").write_text("")
    monkeypatch.setenv("KE_MODEL_FLUX1_DEV", str(root))
    with pytest.raises(FileNotFoundError, match="single component"):
        resolve_source("flux1-dev")


def test_corrupt_model_index_is_reported_as_such(tmp_path, monkeypatch):
    root = tmp_path / "flux"
    root.mkdir()
    (root / "model_index.json").write_text("{not json")
    monkeypatch.setenv("KE_MODEL_FLUX1_DEV", str(root))
    with pytest.raises(FileNotFoundError, match="not valid JSON"):
        resolve_source("flux1-dev")


def test_overrides_are_independent_per_pipeline(tmp_path, monkeypatch):
    """Mixing is the common case: one model already local, the others from the cache."""
    local = tmp_path / "flux"
    local.mkdir()
    (local / "model_index.json").write_text("{}")
    monkeypatch.setenv("KE_MODEL_FLUX1_DEV", str(local))
    monkeypatch.delenv("KE_MODEL_SD35_LARGE", raising=False)

    assert resolve_source("flux1-dev")[1] == "local"
    assert resolve_source("sd35-large")[1] == "hub"
