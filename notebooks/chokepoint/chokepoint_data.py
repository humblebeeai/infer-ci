"""Download ChokePoint archives and load frame, pass, and attendance labels.

ChokePoint (Zenodo 815657) is CC BY-NC 4.0. Cite Wong et al., CVPRW 2011.
"""

import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import pandas as pd

_ZENODO_FILES = "https://zenodo.org/records/815657/files"
_ZENODO_RECORD = "https://zenodo.org/api/records/815657"
FPS = 30
MAX_GAP_FRAMES = 30
_XML_NAME = re.compile(r"(?P<sequence>P(?P<portal>\d+)(?P<direction>[EL])_S\d+)_(?P<camera>C\d+)(?P<day>\.\d+)?\.xml")
_KEYS = ["sequence", "session", "camera", "person_id"]
_FRAME_COLUMNS = [
    "sequence", "portal", "direction", "session", "camera", "frame", "time_s",
    "person_id", "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
]
_PASS_COLUMNS = [*_KEYS, "start_frame", "end_frame", "start_time_s", "end_time_s", "n_frames"]
_ATTENDANCE_COLUMNS = [*_KEYS, "start_time_s"]


def _extract_archive(archive, destination):
    with tarfile.open(archive, "r:xz") as tar:
        for member in tar:
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported member in {archive.name}: {member.name}")
            # CPython added the extraction filter in 3.10.12 and 3.11.4.
            tar.extract(member, destination, filter="data")


def _unpack(archive, destination, stamp):
    # Publish only after both levels extract successfully. An incomplete run
    # can start again without discarding files that are already present.
    with tempfile.TemporaryDirectory(dir=destination) as scratch:
        scratch = Path(scratch)
        _extract_archive(archive, scratch)
        for inner in sorted(scratch.rglob("*.tar.xz")):
            _extract_archive(inner, inner.parent)
            inner.unlink()

        for source in scratch.rglob("*"):
            target = destination / source.relative_to(scratch)
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(source, target)
    stamp.touch()


def _zenodo_checksums():
    response = subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "20", _ZENODO_RECORD],
        check=True,
        capture_output=True,
        text=True,
    )
    return {entry["key"]: entry["checksum"] for entry in json.loads(response.stdout)["files"]}


def _file_md5(path):
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as archive:
        for block in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(block)
    return f"md5:{digest.hexdigest()}"


def download_chokepoint(names, dest="datasets/chokepoint"):
    """Download named Zenodo archives, resume partial files, and extract to dest.

    Requires curl on PATH. New and resumed downloads are checked against the
    Zenodo record before promotion. Completed archives stay in dest so the
    images and labels can be extracted again by removing the .extracted stamp.
    Returns the destination Path.
    """
    destination = Path(dest)
    destination.mkdir(parents=True, exist_ok=True)
    checksums = None
    for name in [names] if isinstance(names, str) else names:
        if not re.fullmatch(r"[A-Za-z0-9_]+\.tar\.xz", name):
            raise ValueError(f"Expected a ChokePoint .tar.xz filename, got {name!r}")
        archive = destination / name
        if not archive.is_file():
            if checksums is None:
                checksums = _zenodo_checksums()
            expected = checksums.get(name, "")
            if not re.fullmatch(r"md5:[0-9a-f]{32}", expected):
                raise ValueError(f"No valid Zenodo MD5 checksum for {name}")
            partial = archive.with_suffix(".xz.part")
            subprocess.run(
                [
                    "curl", "--fail", "--location", "--silent", "--show-error",
                    "--connect-timeout", "20", "--speed-time", "60", "--speed-limit", "1024",
                    "--continue-at", "-", "--output", str(partial),
                    f"{_ZENODO_FILES}/{quote(name)}?download=1",
                ],
                check=True,
            )
            if _file_md5(partial) != expected:
                # A later attempt must start without the corrupt bytes.
                partial.unlink()
                raise ValueError(f"MD5 mismatch for {name}; retry the download")
            partial.replace(archive)

        stamp = destination / f".{name}.extracted"
        if not stamp.is_file():
            _unpack(archive, destination, stamp)
    return destination


def _eye_coordinates(person, eye_tag):
    eye = person.find(eye_tag)
    if eye is None:
        return float("nan"), float("nan")
    return float(eye.attrib["x"]), float(eye.attrib["y"])


def load_ground_truth(xml_dir):
    """Return (frames, passes, attendance) from ChokePoint XML files.

    Sessions are sequence names with the recording-day suffix when present.
    Camera labels use C1/C2/C3. Attendance keeps the earliest pass for each
    person in each session and camera. Times use frame / FPS. An unlabelled
    eye has NaN coordinates while the labelled person is retained.
    """
    directory = Path(xml_dir)
    rows = []
    files = sorted(directory.rglob("P*.xml"))
    if not files:
        raise FileNotFoundError(f"No ChokePoint XML files in {directory}")

    for file in files:
        match = _XML_NAME.fullmatch(file.name)
        if match is None:
            raise ValueError(f"Unexpected ChokePoint XML filename: {file.name}")
        sequence, portal, direction, camera, day = match.groups()
        session = sequence + (day or "")
        for frame in ET.parse(file).getroot().iterfind("frame"):
            number = int(frame.attrib["number"])
            for person in frame.iterfind("person"):
                rows.append((
                    sequence, int(portal), direction, session, camera, number,
                    number / FPS, person.attrib["id"],
                    *_eye_coordinates(person, "leftEye"), *_eye_coordinates(person, "rightEye"),
                ))

    frames = pd.DataFrame.from_records(rows, columns=_FRAME_COLUMNS).sort_values(
        ["sequence", "session", "camera", "frame", "person_id"], ignore_index=True
    )
    if frames.empty:
        return frames, pd.DataFrame(columns=_PASS_COLUMNS), pd.DataFrame(columns=_ATTENDANCE_COLUMNS)

    pass_starts = frames.groupby(_KEYS)["frame"].diff().gt(MAX_GAP_FRAMES)
    pass_number = pass_starts.groupby([frames[key] for key in _KEYS]).cumsum()
    passes = (
        frames.assign(pass_number=pass_number).groupby([*_KEYS, "pass_number"], as_index=False)
        .agg(
            start_frame=("frame", "min"),
            end_frame=("frame", "max"),
            start_time_s=("time_s", "min"),
            end_time_s=("time_s", "max"),
            n_frames=("frame", "size"),
        )
        .drop(columns="pass_number")
        .sort_values(["sequence", "session", "camera", "start_frame", "person_id"], ignore_index=True)
    )
    attendance = passes.loc[:, _ATTENDANCE_COLUMNS].drop_duplicates(_KEYS, ignore_index=True)
    return frames, passes, attendance
