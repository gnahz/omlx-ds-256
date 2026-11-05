"""CPU-only lifecycle safety: all service operations and sockets are mocked."""
from contextlib import ExitStack
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock

DEPLOY = Path(__file__).resolve().parents[2] / "deploy/deepseek"
sys.path.insert(0, str(DEPLOY))
import control
import portable


class PortableLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.env = patch.dict(os.environ, {"OMLX_DS_STATE": str(self.root / "state")})
        self.env.start()
        self.model = self.root / "checkpoint"
        self.model.mkdir()
        for filename in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"):
            (self.model / filename).write_text("{}")
        self.proof = self.root / "metadata-proof.json"
        self.proof.write_text("{}")
        self.profile = portable.configure(self.model, self.proof)
        self.profile["python"] = sys.executable

    def tearDown(self):
        self.env.stop()
        self.temporary.cleanup()

    def record(self):
        return {"pid": 1234, "created": 42.1, "executable": "/test/python",
                "source": str(portable.REPO), "token": "unique-child-token"}

    def test_configure_private_external_state_and_stable_secret(self):
        key = Path(self.profile["api_key_file"])
        original = key.read_text()
        portable.configure(self.model, self.proof)
        self.assertEqual(original, key.read_text())
        self.assertEqual(key.stat().st_mode & 0o777, 0o600)
        self.assertEqual(portable.state_dir().stat().st_mode & 0o777, 0o700)
        self.assertEqual(self.profile["port"], 8080)
        self.assertFalse(Path(self.profile["base_path"]).is_relative_to(portable.REPO))

    def test_rejects_state_inside_checkout(self):
        with patch.dict(os.environ, {"OMLX_DS_STATE": str(portable.REPO / "state")}):
            with self.assertRaises(ValueError):
                portable.state_dir()

    def test_rejects_overlay_inside_checkpoint(self):
        inside = self.model / "proof.json"
        inside.write_text("{}")
        with self.assertRaises(ValueError):
            portable.configure(self.model, inside)

    def test_preserved_profile_passes_without_importing_mlx(self):
        self.assertTrue(portable.validate_configuration(self.profile, check_build=False))
        self.assertNotIn("mlx.core", sys.modules)

    def test_guard_weakening_is_rejected(self):
        for mutation in (lambda p: p["memory_policy"].update(system_budget_gib=0),
                         lambda p: p["effective_settings"].update(memory_guard_gb=256),
                         lambda p: p["environment"].update(OMLX_PIPELINE_MEMORY_BUDGET="0"),
                         lambda p: p["cache"].update(capacity_gib=20)):
            profile = copy.deepcopy(self.profile)
            mutation(profile)
            with self.assertRaises(ValueError):
                portable.validate_configuration(profile, check_build=False)

    def test_reused_pid_never_owned_or_signalled(self):
        record = self.record()
        current = {name: record[name] for name in ("pid", "created", "executable")}
        current["created"] += 1
        with patch.object(control, "identity", return_value=current), patch.object(control, "read_record", return_value=record), patch.object(control.os, "kill") as kill:
            self.assertFalse(control.owns(record))
            with self.assertRaises(RuntimeError):
                control.stop()
            kill.assert_not_called()

    def test_other_checkout_never_owned(self):
        record = self.record()
        record["source"] = "/another/checkout"
        with patch.object(control, "identity") as identity:
            self.assertFalse(control.owns(record))
            identity.assert_not_called()

    def test_lifecycle_lock_rejects_concurrent_operation(self):
        with control.lifecycle_lock():
            with self.assertRaises(RuntimeError):
                with control.lifecycle_lock():
                    self.fail("second command acquired the lock")

    def test_stop_waits_for_matching_process_only(self):
        record = self.record()
        with patch.object(control, "read_record", return_value=record), patch.object(control, "owns", side_effect=[True, True, False, False]), patch.object(control.os, "kill") as kill, patch.object(control.time, "sleep"):
            control.stop()
            kill.assert_called_once_with(1234, control.signal.SIGTERM)

    def test_occupied_port_never_spawns_or_signals(self):
        with patch.object(control, "read_record", return_value=None), patch.object(control, "validate_configuration"), patch.object(control, "check_port_available", side_effect=RuntimeError("occupied")), patch.object(control.subprocess, "Popen") as spawn, patch.object(control.os, "kill") as kill:
            with self.assertRaises(RuntimeError):
                control.start()
            spawn.assert_not_called()
            kill.assert_not_called()

    def test_existing_owned_process_never_duplicates(self):
        with patch.object(control, "read_record", return_value=self.record()), patch.object(control, "owns", return_value=True), patch.object(control.subprocess, "Popen") as spawn:
            control.start()
            spawn.assert_not_called()

    def test_readiness_never_contacts_unowned_process(self):
        with patch.object(control, "owns", return_value=False), patch.object(control.urllib.request, "build_opener") as opener:
            self.assertFalse(control.ready(self.profile, self.record()))
            opener.assert_not_called()

    def test_readiness_requires_pid_token_and_source(self):
        record = self.record()
        good = {"ready": True, "pid": record["pid"], "instance_token": record["token"],
                "launch_identity": {"source": str(portable.REPO)}}
        for mutate, expected in (({}, True), ({"instance_token": "other-instance"}, False), ({"pid": 999}, False), ({"ready": False}, False)):
            opener = Mock()
            response = io.StringIO(json.dumps({**good, **mutate}))
            opener.open.return_value = response
            with patch.object(control, "owns", return_value=True), patch.object(control.urllib.request, "build_opener", return_value=opener):
                self.assertEqual(control.ready(self.profile, record), expected)

    def test_restart_order_stops_then_starts(self):
        events = []
        with patch.object(sys, "argv", ["control.py", "restart"]), patch.object(control, "stop", side_effect=lambda: events.append("stop")), patch.object(control, "start", side_effect=lambda: events.append("start")):
            control.main()
        self.assertEqual(events, ["stop", "start"])

    def test_shell_entrypoints_work_from_unrelated_directory(self):
        interpreter = portable.state_dir() / "venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
        interpreter.chmod(0o755)
        for script, entry in (("service.sh", "control.py"), ("recorder.sh", "recorder_control.py")):
            output = subprocess.check_output([str(DEPLOY / script), "status"], cwd=self.root, text=True)
            self.assertEqual(output.splitlines(), [str(DEPLOY / entry), "status"])

    def test_archive_disabled_by_default(self):
        import request_archive
        self.assertEqual(request_archive.load_config(portable.state_dir()), {"enabled": False})

    def test_recorder_config_is_portable_and_metadata_only(self):
        import recorder
        config = recorder.make_config()
        config.validate()
        self.assertFalse(config.enable_incident_enrichment)
        self.assertFalse(config.enable_target_subprocess_snapshots)
        self.assertEqual(config.pid_file, str(portable.state_dir() / "process.json"))


if __name__ == "__main__":
    unittest.main()
