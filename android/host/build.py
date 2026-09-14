"""Build the Android host or a supplied Java project with an existing signing key.

Build outputs and signing material use caller-supplied paths outside source.
Key creation belongs to installation: replacing a lost key breaks APK updates.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--java-home", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keystore", type=Path, required=True)
    parser.add_argument("--key-alias", default="ouroboros-host")
    password = parser.add_mutually_exclusive_group()
    password.add_argument("--keystore-pass-file", type=Path,
                          help="Private file containing the existing keystore password.")
    password.add_argument("--ks-pass", default="env:OUROBOROS_ANDROID_KEYSTORE_PASSWORD",
                          help="apksigner password source: env:NAME, file:/path, or stdin.")
    parser.add_argument("--create-development-key", action="store_true",
                        help="Explicitly create a new development key; never use to repair a lost install key.")
    parser.add_argument("--version-code", type=int,
                        help="Override only the compiled APK's Android build number.")
    parser.add_argument("--version-name", help="Override only the compiled APK's displayed version.")
    parser.add_argument("--project", type=Path,
                        help="Build an ordinary AndroidManifest.xml + src/ + res/ project.")
    args = parser.parse_args()
    if not args.keystore.is_file() and not args.create_development_key:
        parser.error("The signing keystore is missing. Restore this installation's original key; "
                     "the compiler never creates a replacement.")
    if args.keystore_pass_file is not None and not args.keystore_pass_file.is_file():
        parser.error("--keystore-pass-file must name an existing private password file")
    if args.ks_pass != "stdin" and not args.ks_pass.startswith(("env:", "file:")):
        parser.error("Use a password file or environment source, never a literal password argument")
    if args.create_development_key and args.keystore_pass_file is None:
        parser.error("--create-development-key requires --keystore-pass-file")
    if args.version_code is not None and not 1 <= args.version_code <= 2147483647:
        parser.error("--version-code must be a positive signed 32-bit integer")
    source = args.project.resolve() if args.project else Path(__file__).resolve().parent
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    classes, dex, res, generated = out / "classes", out / "dex", out / "res", out / "generated"
    for directory in (classes, dex, res, generated):
        directory.mkdir(parents=True, exist_ok=True)
    tools = args.sdk / "build-tools" / "36.0.0"
    platform = args.sdk / "platforms" / "android-36" / "android.jar"
    java = args.java_home / "bin"
    env = os.environ.copy()
    env["JAVA_HOME"] = str(args.java_home)
    env["PATH"] = str(java) + os.pathsep + env.get("PATH", "")

    def run(*argv: object) -> None:
        subprocess.run([str(item) for item in argv], env=env, check=True)

    if args.create_development_key and not args.keystore.exists():
        args.keystore.parent.mkdir(parents=True, exist_ok=True)
        run(java / "keytool", "-genkeypair", "-keystore", args.keystore,
            "-storepass:file", args.keystore_pass_file, "-keypass:file", args.keystore_pass_file,
            "-alias", args.key_alias, "-dname", "CN=Ouroboros Development",
            "-keyalg", "RSA", "-keysize", "2048", "-validity", "10000")
        args.keystore.chmod(0o600)
    if (source / "res").exists():
        shutil.copytree(source / "res", res, dirs_exist_ok=True)
    if not args.project:
        (res / "drawable").mkdir(exist_ok=True)
        # source is <repo>/android/host; the shared asset lives at <repo>/assets.
        shutil.copyfile(source.parents[1] / "assets" / "icon_1024.png", res / "drawable" / "icon.png")
    run(tools / "aapt2", "compile", "--dir", res, "-o", out / "resources.zip")
    versions = []
    if args.version_code is not None:
        versions.extend(("--version-code", args.version_code))
    if args.version_name is not None:
        versions.extend(("--version-name", args.version_name))
    if versions:
        versions.append("--replace-version")
    run(tools / "aapt2", "link", "-o", out / "unsigned.apk", "-I", platform,
        "--manifest", source / "AndroidManifest.xml", "--java", generated,
        *versions, out / "resources.zip")
    run(java / "javac", "-encoding", "UTF-8", "-source", "8", "-target", "8",
        "-classpath", platform, "-d", classes, *sorted((source / "src").rglob("*.java")),
        *sorted(generated.rglob("*.java")))
    with zipfile.ZipFile(out / "classes.jar", "w") as jar:
        for path in sorted(classes.rglob("*.class")):
            jar.write(path, path.relative_to(classes))
    run(java / "java", "-Xmx384m", "-cp", tools / "lib" / "d8.jar", "com.android.tools.r8.D8",
        "--lib", platform, "--min-api", "26", "--output", dex, out / "classes.jar")
    with zipfile.ZipFile(out / "unsigned.apk", "a") as apk:
        for path in dex.glob("*.dex"):
            apk.write(path, path.name)
    run(tools / "zipalign", "-f", "4", out / "unsigned.apk", out / "aligned.apk")
    apk_output = out / ("app.apk" if args.project else "Ouroboros.apk")
    password_source = ("file:" + str(args.keystore_pass_file)
                       if args.keystore_pass_file is not None else args.ks_pass)
    run(java / "java", "-Xmx256m", "-jar", tools / "lib" / "apksigner.jar", "sign",
        "--ks", args.keystore, "--ks-key-alias", args.key_alias, "--ks-pass", password_source,
        "--out", apk_output, out / "aligned.apk")
    run(java / "java", "-Xmx256m", "-jar", tools / "lib" / "apksigner.jar", "verify", apk_output)
    print(apk_output)


if __name__ == "__main__":
    main()
