"""Verified and resumable acquisition from the ESA authors' official archive."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from .data import sha256

RECORD_URL = "https://zenodo.org/api/records/4463683"
EXPECTED_MD5 = "d19dc8875229f2f6893253c38adddc87"
CHUNK_SIZE = 8 * 1024**2


def get_bytes(url, start, end, total):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "SpaceDebrisResearch/0.1"})
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                    raise ValueError("Server did not honour byte range")
                payload = response.read(end - start + 2)
            if len(payload) != end - start + 1:
                raise ValueError("Incomplete range response")
            return payload
        except urllib.error.HTTPError as error:
            if error.code == 429:
                raise RuntimeError("Archive server rate limit: retry after cooldown") from error
            if attempt == 2:
                raise
        except (OSError, ValueError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)


def download_archive(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = data_dir / "zenodo_metadata.json"
    if not metadata_path.exists():
        with urllib.request.urlopen(RECORD_URL, timeout=30) as response:
            metadata = json.load(response)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = [f for f in metadata["files"] if f["key"] == "Collision Avoidance Challenge - Dataset.zip"]
    if len(entries) != 1 or entries[0]["checksum"] != "md5:" + EXPECTED_MD5:
        raise ValueError("Unexpected archive identity; inspect provenance")
    entry = entries[0]
    archive = data_dir / "official_dataset.zip"
    if not archive.exists():
        parts = data_dir / "archive_parts"
        parts.mkdir(exist_ok=True)
        total = entry["size"]
        ranges = [(s, min(s + CHUNK_SIZE, total) - 1) for s in range(0, total, CHUNK_SIZE)]
        def fetch_part(pair):
            start, end = pair
            path = parts / f"{start:012d}.bin"
            if not path.exists() or path.stat().st_size != end - start + 1:
                payload = get_bytes(entry["links"]["self"], start, end, total)
                partial = path.with_suffix(".partial")
                partial.write_bytes(payload)
                partial.replace(path)
            return path
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [pool.submit(fetch_part, pair) for pair in ranges]
            for completed, job in enumerate(as_completed(jobs), 1):
                job.result()
                print(f"Archive chunks verified: {completed}/{len(jobs)}", flush=True)
        temporary = archive.with_suffix(".part")
        with temporary.open("wb") as output:
            for start, _ in ranges:
                with (parts / f"{start:012d}.bin").open("rb") as source:
                    shutil.copyfileobj(source, output)
        if temporary.stat().st_size != total:
            raise ValueError("Assembled archive size mismatch")
        with temporary.open("rb") as source:
            digest = hashlib.file_digest(source, "md5").hexdigest()
        if digest != EXPECTED_MD5:
            raise ValueError("Archive checksum mismatch; inspect cached chunks")
        temporary.replace(archive)
    with archive.open("rb") as source:
        if hashlib.file_digest(source, "md5").hexdigest() != EXPECTED_MD5:
            raise ValueError("Cached archive checksum mismatch")
    return archive, metadata


def download(data_dir: Path) -> Path:
    archive, metadata = download_archive(data_dir)
    csv = data_dir / "train_data.csv"
    with zipfile.ZipFile(archive) as zipped:
        inventory = [{"name": m.filename, "bytes": m.file_size} for m in zipped.infolist()]
        (data_dir / "archive_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        training_archives = [m for m in zipped.infolist() if Path(m.filename).name == "train_data.zip"]
        if len(training_archives) != 1:
            raise ValueError("Expected one nested training archive; inspect archive_inventory.json")
        training_member = training_archives[0]
        training_archive = data_dir / "train_data.zip"
        temporary_archive = training_archive.with_suffix(".part")
        with zipped.open(training_member) as source, temporary_archive.open("wb") as output:
            shutil.copyfileobj(source, output)
        if temporary_archive.stat().st_size != training_member.file_size:
            raise ValueError("Nested archive size mismatch")
        temporary_archive.replace(training_archive)
    with zipfile.ZipFile(training_archive) as zipped:
        members = [m for m in zipped.infolist() if Path(m.filename).name == "train_data.csv"]
        if len(members) != 1:
            raise ValueError("Expected one train_data.csv; inspect archive_inventory.json")
        member = members[0]
        temporary = csv.with_suffix(".part")
        with zipped.open(member) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output)
        if temporary.stat().st_size != member.file_size:
            raise ValueError("Extracted CSV size mismatch")
        temporary.replace(csv)
    manifest = {
        "source_doi": "10.5281/zenodo.4463683",
        "source_record": "https://zenodo.org/records/4463683",
        "archive_md5": EXPECTED_MD5, "archive_sha256": sha256(archive),
        "csv_sha256": sha256(csv), "archive_bytes": archive.stat().st_size,
        "csv_bytes": csv.stat().st_size,
        "csv_member": training_member.filename + " :: " + member.filename,
        "license": metadata["metadata"]["license"]["id"],
        "attribution": "Uriot, Izzo, Martinez-Heras, Letizia, Siminski, and Merz / ESA; Collision Avoidance Challenge dataset (2021), doi:10.5281/zenodo.4463683.",
        "additional_labels": "The author archive includes original competition labels; this milestone uses training data only.",
    }
    (data_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return csv
