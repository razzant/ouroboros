"""Exercise compiler orchestration without device access or real signing material."""
from contextlib import redirect_stderr
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


SOURCE = Path(__file__).resolve().parents[1] / "host" / "build.py"
SPEC = importlib.util.spec_from_file_location("android_host_build", SOURCE)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class AndroidBuildTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="obo-android-build-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        (self.project / "src").mkdir(parents=True)
        (self.project / "res").mkdir()
        self.manifest = self.project / "AndroidManifest.xml"
        self.manifest.write_text('<manifest package="test.app" versionCode="1"/>')
        (self.project / "src" / "App.java").write_text("class App {}")
        self.key = self.root / "host.keystore"
        self.key.write_bytes(b"not a real signing key")
        self.password = self.root / "host-password"
        self.password.write_text("private-test-value\n")
        self.output = self.root / "build"
        self.commands = []

    def argv(self, *extra):
        return [str(SOURCE), "--sdk", str(self.root / "sdk"),
                "--java-home", str(self.root / "jdk"), "--out", str(self.output),
                "--project", str(self.project), "--keystore", str(self.key),
                "--keystore-pass-file", str(self.password), *extra]

    def run_tool(self, argv, *, env, check):
        self.assertTrue(check)
        self.commands.append(argv)
        if "-genkeypair" in argv:
            self.key.write_bytes(b"explicit test key")
        if "link" in argv:
            with zipfile.ZipFile(self.output / "unsigned.apk", "w") as apk:
                apk.writestr("AndroidManifest.xml", b"compiled manifest")
        if "com.android.tools.r8.D8" in argv:
            (self.output / "dex" / "classes.dex").write_bytes(b"dex")
        if "sign" in argv:
            (self.output / "app.apk").write_bytes(b"signed test APK")

    def invoke(self, argv):
        with patch.object(sys, "argv", argv), patch.object(BUILD.subprocess, "run", self.run_tool):
            BUILD.main()

    def test_existing_key_and_output_versions_preserve_source(self):
        before = self.manifest.read_bytes()
        key_before = self.key.read_bytes()
        self.invoke(self.argv("--version-code", "17", "--version-name", "7.0.0"))
        link = next(argv for argv in self.commands if "link" in argv)
        self.assertEqual(link[link.index("--version-code") + 1], "17")
        self.assertEqual(link[link.index("--version-name") + 1], "7.0.0")
        self.assertIn("--replace-version", link)
        sign = next(argv for argv in self.commands if "sign" in argv)
        self.assertEqual(sign[sign.index("--ks-pass") + 1], "file:" + str(self.password))
        self.assertEqual(sign[sign.index("--ks-key-alias") + 1], "ouroboros-host")
        self.assertEqual(sign[sign.index("--out") + 1], str((self.output / "app.apk").resolve()))
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertEqual(self.key.read_bytes(), key_before)
        self.assertFalse(any("-genkeypair" in argv for argv in self.commands))
        self.assertFalse(any("private-test-value" in item for argv in self.commands for item in argv))

    def test_missing_key_fails_before_compilation_and_does_not_replace_it(self):
        self.key.unlink()
        with redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit) as failure:
            self.invoke(self.argv())
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("Restore this installation's original key", error.getvalue())
        self.assertEqual(self.commands, [])
        self.assertFalse(self.key.exists())
        self.assertFalse(self.output.exists())

    def test_explicit_development_creation_uses_password_file_and_keeps_existing_key(self):
        self.key.unlink()
        self.invoke(self.argv("--create-development-key"))
        keygen = next(argv for argv in self.commands if "-genkeypair" in argv)
        self.assertEqual(keygen[keygen.index("-storepass:file") + 1], str(self.password))
        self.assertEqual(self.key.stat().st_mode & 0o777, 0o600)
        self.commands.clear()
        self.invoke(self.argv("--create-development-key"))
        self.assertFalse(any("-genkeypair" in argv for argv in self.commands))

    def test_nonpositive_version_refuses_without_compilation(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke(self.argv("--version-code", "0"))
        self.assertEqual(self.commands, [])


if __name__ == "__main__":
    unittest.main()
