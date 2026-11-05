"""Path-independent, CPU-only configuration for the standalone deployment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def state_dir():
    path = Path(os.environ.get("OMLX_DS_STATE", "~/.local/share/omlx-ds-256")).expanduser().resolve()
    if path == REPO or path.is_relative_to(REPO):
        raise ValueError("OMLX_DS_STATE must be outside the source repository")
    return path


def profile_path():
    return state_dir() / "profile.json"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)
    path.chmod(0o600)


def configure(model_path, metadata_proof, *, host="127.0.0.1", port=8080):
    state = state_dir()
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    state.chmod(0o700)
    model = Path(model_path).expanduser().resolve(strict=True)
    proof = Path(metadata_proof).expanduser().resolve(strict=True)
    if not model.is_dir() or not proof.is_file() or proof.is_relative_to(model):
        raise ValueError("Model must be a directory; metadata proof must be a file outside it")
    if not 1 <= port <= 65535 or not host:
        raise ValueError("A valid host and port are required")
    profile = json.loads((HERE / "profile.template.json").read_text())
    profile.update(source=str(REPO), python=str(state / "venv/bin/python"),
                   entry=str(HERE / "launch.py"), state_path=str(state),
                   base_path=str(state / "service"), api_key_file=str(state / "api-key"),
                   model_path=str(model), metadata_proof=str(proof), host=host, port=port)
    profile["environment"]["DS41_NATIVE_METADATA"] = str(proof)
    profile["cache"].update(directory=str(state / "cache/exact-prefix"),
                            native_ssd_cache_dir=str(state / "cache/ssd"))
    for directory in ("logs", "service", "cache", "model-registry"):
        (state / directory).mkdir(mode=0o700, exist_ok=True)
    key = state / "api-key"
    try:
        fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if not key.read_text().strip():
            raise ValueError("Existing API key is empty")
        key.chmod(0o600)
    else:
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_urlsafe(36) + "\n")
    atomic_json(profile_path(), profile)
    return profile


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_configuration(profile, *, check_build=True):
    """Fail closed before importing MLX or allocating model weights."""
    from omlx_physical_memory_guard import MemoryPolicy
    template = json.loads((HERE / "profile.template.json").read_text())
    MemoryPolicy.from_profile(profile)
    if profile["memory_policy"] != template["memory_policy"]:
        raise ValueError("Physical memory policy differs from the preserved deployment profile")
    for field in ("effective_settings", "model_settings", "web_asset_refs"):
        if profile[field] != template[field]:
            raise ValueError("Preserved deployment policy differs: " + field)
    expected_env = dict(template["environment"], DS41_NATIVE_METADATA=profile["metadata_proof"])
    if profile["environment"] != expected_env:
        raise ValueError("Optimization or prefill guard environment differs from deployment profile")
    if Path(profile["source"]).resolve() != REPO or Path(profile["state_path"]).resolve() != state_dir():
        raise ValueError("Configuration belongs to another source checkout or state directory")
    if Path(profile["python"]).resolve() != Path(sys.executable).resolve():
        raise ValueError("Use this deployment's isolated Python environment")
    model = Path(profile["model_path"]).resolve(strict=True)
    proof = Path(profile["metadata_proof"]).resolve(strict=True)
    if proof.is_relative_to(model):
        raise ValueError("Exact metadata must live outside the checkpoint directory")
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"):
        if not (model / name).is_file():
            raise ValueError("Missing required checkpoint metadata: " + name)
    for name in ("base_path", "api_key_file"):
        if not Path(profile[name]).resolve().is_relative_to(state_dir()):
            raise ValueError("Service configuration escapes its private state: " + name)
    for name in ("directory", "native_ssd_cache_dir"):
        if not Path(profile["cache"][name]).resolve().is_relative_to(state_dir()):
            raise ValueError("Cache directory escapes its private state")
    for name in ("capacity_gib", "l2_enabled", "persistence_enabled"):
        if profile["cache"][name] != template["cache"][name]:
            raise ValueError("Preserved cache policy differs: " + name)
    if check_build:
        if Path(sys.prefix).resolve() != (state_dir() / "venv").resolve():
            raise ValueError("Use this deployment's isolated virtual environment")
        receipt = json.loads((state_dir() / "build-receipt.json").read_text())
        if receipt["dependency_lock_sha256"] != sha(HERE / "dependency-lock.json"):
            raise ValueError("Build dependencies changed; run bootstrap again")
        if not receipt["native_files"]:
            raise ValueError("No native kernel build was recorded")
        for filename, digest in receipt["native_files"].items():
            if sha(REPO / filename) != digest:
                raise ValueError("Native kernel changed after bootstrap: " + filename)
        for filename, digest in receipt["mlx_files"].items():
            if sha(filename) != digest:
                raise ValueError("Patched MLX binary changed after bootstrap")
    return True
