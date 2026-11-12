"""Metadata guards only; no credentials, server launch or installed-profile update."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "vault", Path(__file__).with_name("agentbridge_vault.py")
)
vault = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vault)


class ProfileTests(unittest.TestCase):
    def test_serve_rejects_mismatched_installed_framework_before_session_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            versions = {"next": "16.4.5", "react": "19.3.1", "react-dom": "19.3.1"}
            (root / "package.json").write_text(json.dumps({"dependencies": versions}))
            for name, version in versions.items():
                target = root / "node_modules" / name / "package.json"
                target.parent.mkdir(parents=True)
                target.write_text(json.dumps({"name": name, "version": version}))
            with patch.object(vault, "ROOT", root):
                vault.verify_app_dependencies()
                (root / "node_modules/next/package.json").write_text(
                    json.dumps({"name": "next", "version": "16.4.1"})
                )
                with self.assertRaisesRegex(
                    ValueError, "differ from the reviewed manifest"
                ):
                    vault.verify_app_dependencies()

    def test_exact_scopes_sources_and_runtimes_are_required(self):
        if not vault.PROFILE.exists():
            self.skipTest("Requires installed metadata, not service credentials")
        profile = json.loads(vault.PROFILE.read_text())
        profile["infisical"] = str(
            Path.home() / ".config/infisical/cli/0.45.6/infisical"
        )
        profile["code_sha256"] = {
            name: hashlib.sha256((vault.ROOT / name).read_bytes()).hexdigest()
            for name in vault.source_files()
        }
        profile["runtime_sha256"] = {
            name: hashlib.sha256(Path(profile[name]).read_bytes()).hexdigest()
            for name in ("node", "infisical", "ca_file")
        }
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve() / "profile.json"

            def write(value):
                target.write_text(json.dumps(value))
                target.chmod(0o600)

            with patch.object(vault, "PROFILE", target):
                write(profile)
                self.assertEqual(
                    vault.load_profile()["bundle"], "local-modelarchviz-agentbridge-read"
                )
                changes = [
                    {"source_path": "/unreviewed"},
                    {"infisical": "/bin/sh"},
                    {"version": True},
                    {"extra": "unreviewed"},
                    {"code_sha256": {}},
                    {"code_sha256": {**profile["code_sha256"], "../escape": "0" * 64}},
                    {
                        "code_sha256": {
                            **profile["code_sha256"],
                            "scripts/infisical/agentbridge_vault.py": "0" * 64,
                        }
                    },
                ]
                changes += [
                    {"runtime_sha256": {**profile["runtime_sha256"], name: "0" * 64}}
                    for name in ("node", "infisical", "ca_file")
                ]
                for change in changes:
                    with self.subTest(fields=list(change)):
                        write({**profile, **change})
                        with self.assertRaises(ValueError):
                            vault.load_profile()
                write(profile)
                target.chmod(0o644)
                with self.assertRaises(ValueError):
                    vault.load_profile()

    def test_scope_overrides_fail_before_profile_read_or_native_launch(self):
        with patch.object(vault, "load_profile") as load:
            with self.assertRaises(ValueError):
                vault.main(["check", "--project=other"])
            load.assert_not_called()

    def test_native_command_uses_reviewed_project_and_bounded_sessions(self):
        profile = {
            "infisical": "/trusted/infisical",
            "bundle": "reviewed",
            "proxy": "127.0.0.1:8283",
            "ca_fingerprint": vault.FINGERPRINT,
            "base_url": "http://127.0.0.1:8082/api/v1",
            "node": str(Path.home() / vault.NODE_RELATIVE),
            "ca_file": "/reviewed/public.pem",
        }
        for mode, ttl in [("check", "10m"), ("serve", "30m")]:
            with self.subTest(mode=mode):
                args = vault.session_command(profile, mode)
                self.assertIn("--project-id=5078c850-3f93-46d8-8afe-82107df8b931", args)
                self.assertIn("--ttl=" + ttl, args)
                self.assertIn("--no-skills", args)
                self.assertIn("--no-ca-trust", args)


if __name__ == "__main__":
    unittest.main()
