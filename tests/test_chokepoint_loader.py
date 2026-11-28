import hashlib
import io
import json
import subprocess
import tarfile

import pandas as pd
import pytest

from notebooks.chokepoint import chokepoint_data as data


def test_loader_preserves_days_cameras_ids_and_pass_boundaries(tmp_path):
    def write(name, numbers):
        frames = ['<frame number="0"/>'] + [
            f'<frame number="{n}"><person id="0005">'
            '<leftEye x="10" y="20"/><rightEye x="30" y="40"/>'
            '</person></frame>'
            for n in numbers
        ]
        (tmp_path / name).write_text(f"<dataset>{''.join(frames)}</dataset>")

    write("P2E_S1_C1.1.xml", [91, 30, 60])
    write("P2E_S1_C2.1.xml", [33])
    write("P2E_S1_C1.2.xml", [45])
    write("P1L_S2_C3.xml", [120])
    frames, passes, attendance = data.load_ground_truth(tmp_path)

    assert frames.columns.tolist() == [
        "sequence", "portal", "direction", "session", "camera", "frame",
        "time_s", "person_id", "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
    ]
    assert passes.columns.tolist() == [
        "sequence", "session", "camera", "person_id", "start_frame", "end_frame",
        "start_time_s", "end_time_s", "n_frames",
    ]
    assert attendance.columns.tolist() == ["sequence", "session", "camera", "person_id", "start_time_s"]
    assert frames[["portal", "direction", "frame", "time_s", "person_id"]].iloc[0].tolist() == [
        1, "L", 120, 4.0, "0005",
    ]
    assert frames.iloc[0][["left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y"]].tolist() == [10, 20, 30, 40]
    assert passes[["session", "camera", "start_frame", "end_frame", "n_frames"]].values.tolist() == [
        ["P1L_S2", "C3", 120, 120, 1],
        ["P2E_S1.1", "C1", 30, 60, 2],
        ["P2E_S1.1", "C1", 91, 91, 1],
        ["P2E_S1.1", "C2", 33, 33, 1],
        ["P2E_S1.2", "C1", 45, 45, 1],
    ]
    assert attendance[["session", "camera", "person_id", "start_time_s"]].values.tolist() == [
        ["P1L_S2", "C3", "0005", 4.0],
        ["P2E_S1.1", "C1", "0005", 1.0],
        ["P2E_S1.1", "C2", "0005", 1.1],
        ["P2E_S1.2", "C1", "0005", 1.5],
    ]


def archive(name, payload):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w:xz") as tar:
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    return result.getvalue()


def checksum_response(command, content):
    checksum = hashlib.md5(content, usedforsecurity=False).hexdigest()
    metadata = {"files": [{"key": "sample.tar.xz", "checksum": f"md5:{checksum}"}]}
    return subprocess.CompletedProcess(command, 0, stdout=json.dumps(metadata))


def test_download_resumes_partial_and_extracts_both_levels(tmp_path, monkeypatch):
    content = archive("sample/C1.tar.xz", archive("C1/frame.jpg", b"fake-frame"))
    partial = tmp_path / "sample.tar.xz.part"
    partial.write_bytes(content[:30])
    calls = []

    def fake_curl(command, *, check, **kwargs):
        assert check
        calls.append(command)
        if command[-1] == "https://zenodo.org/api/records/815657":
            assert kwargs == {"capture_output": True, "text": True}
            return checksum_response(command, content)
        assert command[command.index("--continue-at") + 1] == "-"
        path = command[command.index("--output") + 1]
        assert path == str(partial)
        with partial.open("ab") as file:
            assert file.tell() == 30
            file.write(content[30:])

    monkeypatch.setattr(data.subprocess, "run", fake_curl)
    data.download_chokepoint(["sample.tar.xz"], tmp_path)
    assert (tmp_path / "sample/C1/frame.jpg").read_bytes() == b"fake-frame"
    assert (tmp_path / "sample.tar.xz").read_bytes() == content
    assert not partial.exists()
    (tmp_path / "sample/C1/frame.jpg").write_bytes(b"existing")
    data.download_chokepoint(["sample.tar.xz"], tmp_path)
    assert [call[-1] for call in calls] == [
        "https://zenodo.org/api/records/815657",
        "https://zenodo.org/records/815657/files/sample.tar.xz?download=1",
    ]
    assert (tmp_path / "sample/C1/frame.jpg").read_bytes() == b"existing"


def test_download_rejects_bad_checksum_before_publishing_and_can_retry(tmp_path, monkeypatch):
    valid = archive("frame.jpg", b"correct frame")
    payload = archive("frame.jpg", b"corrupt frame")

    def fake_curl(command, *, check, **kwargs):
        if command[-1] == "https://zenodo.org/api/records/815657":
            return checksum_response(command, valid)
        (tmp_path / "sample.tar.xz.part").write_bytes(payload)

    monkeypatch.setattr(data.subprocess, "run", fake_curl)
    with pytest.raises(ValueError, match=r"MD5 mismatch for sample.tar.xz"):
        data.download_chokepoint("sample.tar.xz", tmp_path)
    assert list(tmp_path.iterdir()) == []

    payload = valid
    data.download_chokepoint("sample.tar.xz", tmp_path)
    assert (tmp_path / "frame.jpg").read_bytes() == b"correct frame"


@pytest.mark.parametrize("failed_step", ["metadata", "download"])
def test_download_keeps_partial_after_network_error(tmp_path, monkeypatch, failed_step):
    partial = tmp_path / "sample.tar.xz.part"
    partial.write_bytes(b"resumable bytes")

    def fake_curl(command, *, check, **kwargs):
        if command[-1] == "https://zenodo.org/api/records/815657" and failed_step == "download":
            return checksum_response(command, b"complete archive")
        raise subprocess.CalledProcessError(28, command)

    monkeypatch.setattr(data.subprocess, "run", fake_curl)
    with pytest.raises(subprocess.CalledProcessError, match="exit status 28"):
        data.download_chokepoint("sample.tar.xz", tmp_path)
    assert partial.read_bytes() == b"resumable bytes"
    assert list(tmp_path.iterdir()) == [partial]


def test_loader_keeps_person_with_only_one_labelled_eye(tmp_path):
    (tmp_path / "P2E_S3_C1.1.xml").write_text(
        '<dataset><frame number="30"><person id="0002">'
        '<rightEye x="529" y="174"/></person></frame></dataset>'
    )
    frames, passes, attendance = data.load_ground_truth(tmp_path)
    assert frames.person_id.tolist() == ["0002"]
    assert pd.isna(frames.loc[0, "left_eye_x"])
    assert pd.isna(frames.loc[0, "left_eye_y"])
    assert frames.loc[0, ["right_eye_x", "right_eye_y"]].tolist() == [529, 174]
    assert passes.n_frames.tolist() == [1]
    assert attendance.start_time_s.tolist() == [1.0]
