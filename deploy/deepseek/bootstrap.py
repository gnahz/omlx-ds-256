#!/usr/bin/env python3
"""Rebuild the pinned source dependencies and custom kernels; never start a service."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from portable import HERE, REPO, atomic_json, sha, state_dir


def run(args, *, cwd=None, env=None):
    subprocess.run([str(value) for value in args], cwd=cwd, env=env, check=True)


def mlx_binary_files():
    distribution = importlib.metadata.distribution("mlx")
    return {str(distribution.locate_file(path).resolve()): sha(distribution.locate_file(path))
            for path in distribution.files or [] if Path(path).suffix in {".so", ".dylib", ".metallib"}}


def record_build():
    lock = json.loads((HERE / "dependency-lock.json").read_text())
    distribution = importlib.metadata.distribution("mlx")
    if distribution.version != lock["mlx"]["version"]:
        raise RuntimeError("Installed MLX version differs from the patched build")
    for name, expected in lock["git_dependencies"].items():
        package = importlib.metadata.distribution(name)
        direct = json.loads(package.read_text("direct_url.json") or "{}")
        if direct.get("vcs_info", {}).get("commit_id") != expected["revision"]:
            raise RuntimeError("Installed Git dependency differs from its pinned revision: " + name)
    native = {str(path.relative_to(REPO)): sha(path)
              for path in (REPO / "omlx/custom_kernels").rglob("*")
              if path.is_file() and path.suffix in {".so", ".dylib", ".metallib"}}
    mlx = {str(distribution.locate_file(path).resolve()): sha(distribution.locate_file(path))
           for path in distribution.files or [] if Path(path).suffix in {".so", ".dylib", ".metallib"}}
    original = json.loads((state_dir() / "build/mlx-installed.json").read_text())
    if original != mlx:
        raise RuntimeError("Dependency installation replaced the patched MLX binaries")
    if not native or not mlx:
        raise RuntimeError("Native oMLX or MLX binaries were not built")
    atomic_json(state_dir() / "build-receipt.json", {
        "dependency_lock_sha256": sha(HERE / "dependency-lock.json"),
        "native_files": native, "mlx_files": mlx,
        "validation": "Build provenance only; no model was loaded or benchmarked"})


def build_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-mlx", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--record-build", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.record_mlx:
        atomic_json(state_dir() / "build/mlx-installed.json", mlx_binary_files())
        return
    if args.record_build:
        record_build()
        return
    lock = json.loads((HERE / "dependency-lock.json").read_text())
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("This deployment requires Apple Silicon macOS")
    if f"{sys.version_info.major}.{sys.version_info.minor}" != lock["python"]:
        parser.error("Use Python " + lock["python"])
    target = lock["build"]["deployment_target"]
    if tuple(map(int, platform.mac_ver()[0].split(".")[:2])) < tuple(map(int, target.split("."))):
        parser.error("This preserved native profile requires macOS " + target + " or later")
    state = state_dir()
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    record = state / "process.json"
    if record.exists():
        pid = json.loads(record.read_text())["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            parser.error("Stop this deployment before rebuilding its environment")
    sdk = subprocess.check_output(["xcrun", "--show-sdk-version"], text=True).strip()
    if tuple(map(int, sdk.split(".")[:2])) < tuple(map(int, target.split("."))):
        parser.error("Select an Xcode SDK >= " + target + " before building")
    venv = state / "venv"
    if not (venv / "bin/python").exists():
        run([sys.executable, "-m", "venv", venv])
    python = venv / "bin/python"
    env = dict(os.environ, OMLX_DS_STATE=str(state), PATH=str(venv / "bin") + os.pathsep + os.environ.get("PATH", ""),
               CMAKE_BUILD_PARALLEL_LEVEL=str(lock["build"]["parallel_jobs"]),
               MACOSX_DEPLOYMENT_TARGET=target, PYPI_RELEASE="1",
               OMLX_WITH_CUSTOM_KERNEL="1", OMLX_CUSTOM_KERNEL_DEPLOYMENT_TARGET=target)
    # No isolated build environment is allowed to silently replace patched MLX.
    run([python, "-m", "pip", "install", *[f"{name}=={version}" for name, version in lock["build_packages"].items()], "wheel", "setuptools-scm>=9"], env=env)
    mlx = lock["mlx"]
    patch = HERE / mlx["patch"]
    if sha(patch) != mlx["patch_sha256"]:
        raise RuntimeError("MLX patch checksum differs from the dependency lock")
    source = state / "build" / ("mlx-" + mlx["revision"][:12])
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--no-checkout", mlx["repository"], source])
        run(["git", "checkout", "--detach", mlx["revision"]], cwd=source)
        run(["git", "apply", "--check", patch], cwd=source)
        run(["git", "apply", patch], cwd=source)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != mlx["revision"] or any(sha(source / item["path"]) != item["sha256"] for item in mlx["files"]):
        raise RuntimeError("Existing MLX build tree differs from the locked patched source; preserve it and select a fresh state")
    run([python, "-m", "pip", "install", "--no-build-isolation", "--no-deps", "--force-reinstall", source], env=env)
    run([python, __file__, "--record-mlx"], env=env)
    constraints = state / "build" / "constraints.txt"
    constraints.write_text("mlx==" + mlx["version"] + "\n" + "\n".join(
        f"{name}=={version}" for name, version in {**lock["build_packages"], **lock["runtime_packages"]}.items()) + "\n")
    # Resolve oMLX's pinned Git requirements with exact runtime constraints.
    run([python, "-m", "pip", "install", "--no-build-isolation", "-c", constraints,
         *[f"{name}=={version}" for name, version in lock["runtime_packages"].items()], "-e", REPO], env=env)
    run([python, "setup.py", "build_ext", "--inplace", "--force"], cwd=REPO, env=env)
    # CPU-only metadata helpers compile separately from all MLX/GPU code.
    for module in ("private_os_telemetry", "private_os_telemetry_thermal"):
        report = state / "telemetry" / (module.replace("_", "-") + "-abi.json")
        if not report.exists():
            run([python, HERE / "telemetry" / (module + ".py"), "--prepare"], env=env)
    run([python, "-m", "pip", "check"], env=env)
    run([python, __file__, "--record-build"], env=env)
    print("Build complete. Configure the model with service.sh; no service has been started.")


def main():
    if "--record-build" in sys.argv or "--record-mlx" in sys.argv:
        build_main()
    else:
        from control import lifecycle_lock
        with lifecycle_lock():
            build_main()


if __name__ == "__main__":
    main()
