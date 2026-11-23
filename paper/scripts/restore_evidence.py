"""Materialize the original evidence bytes from the pinned publication revision."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("source_pins.json")
MANIFEST_SHA256 = "d52b5df5f903361eb12e85d039da2f93083d5d58ff5525294f8ae085859834d2"
EVIDENCE_COMMIT = "834128d6870fab5dcd2a4c6a39c91d3750632602"
PIN_COUNT = 58


def read_pins():
    payload = MANIFEST.read_bytes()
    if sha256(payload).hexdigest() != MANIFEST_SHA256:
        raise ValueError("source_pins.json differs from the original manifest bytes")
    pins = json.loads(payload)
    if len(pins) != PIN_COUNT:
        raise ValueError(f"expected exactly {PIN_COUNT} original pins")
    names = {key.casefold() for key in pins}
    if len(names) != len(pins):
        raise ValueError("case-insensitive path collision in source_pins.json")
    for key in pins:
        parts = key.split("/")
        if any(not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", p) for p in parts):
            raise ValueError(f"unsafe source pin path: {key}")
        if any(parent.as_posix().casefold() in names for parent in PurePosixPath(key).parents):
            raise ValueError(f"file/directory collision in source_pins.json: {key}")
    return pins


def recover(pins):
    files, records = {}, []
    for key, expected in sorted(pins.items()):
        blob = subprocess.check_output([
            "git", "--no-replace-objects", "-C", str(ROOT), "cat-file", "blob", f"{EVIDENCE_COMMIT}:{key}",
        ])
        blob_sha = sha256(blob).hexdigest()
        data, method = blob, "blob"
        if blob_sha != expected:
            # Accept the reconstruction only when it restores the recorded bytes.
            data = blob.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            method = "crlf"
            reconstructed_sha = sha256(data).hexdigest()
            if reconstructed_sha != expected:
                raise ValueError(f"cannot restore {key}: expected {expected}, "
                                 f"blob {blob_sha}, CRLF {reconstructed_sha}")
        files[key] = data
        records.append({"path": key, "method": method, "blob_sha256": blob_sha,
                        "restored_sha256": expected, "bytes": len(data)})
    return files, records


def restore(destination):
    if not destination.name or destination.name == ".." or destination.is_symlink():
        raise ValueError("choose a new evidence directory")
    output = destination.parent.resolve(strict=True) / destination.name
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError("choose a new evidence directory outside the checkout")
    files, records = recover(read_pins())
    with TemporaryDirectory(prefix="mttt-evidence-", dir=output.parent) as temp:
        staging = Path(temp) / "evidence"
        staging.mkdir()
        for key, data in files.items():
            target = staging / key
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
        if output.exists() or output.is_symlink():
            raise ValueError(f"evidence destination already exists: {output}")
        staging.rename(output)
    return {
        "evidence_root": str(output),
        "git_commit": EVIDENCE_COMMIT,
        "manifest_sha256": MANIFEST_SHA256,
        "verified_pins": len(records),
        "expected_pins": PIN_COUNT,
        "blob_count": sum(row["method"] == "blob" for row in records),
        "crlf_count": sum(row["method"] == "crlf" for row in records),
        "files": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new directory outside the source checkout")
    options = parser.parse_args()
    try:
        result = restore(options.output)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"evidence restoration failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
