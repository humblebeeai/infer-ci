import io
import tarfile

import pandas as pd

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


def test_download_resumes_partial_and_extracts_both_levels(tmp_path, monkeypatch):
    def archive(name, payload):
        result = io.BytesIO()
        with tarfile.open(fileobj=result, mode="w:xz") as tar:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
        return result.getvalue()

    content = archive("sample/C1.tar.xz", archive("C1/frame.jpg", b"fake-frame"))
    partial = tmp_path / "sample.tar.xz.part"
    partial.write_bytes(content[:30])
    calls = []

    def fake_curl(command, *, check):
        assert check
        assert command[command.index("--continue-at") + 1] == "-"
        path = command[command.index("--output") + 1]
        assert path == str(partial)
        with partial.open("ab") as file:
            assert file.tell() == 30
            file.write(content[30:])
        calls.append(command)

    monkeypatch.setattr(data.subprocess, "run", fake_curl)
    data.download_chokepoint(["sample.tar.xz"], tmp_path)
    assert (tmp_path / "sample/C1/frame.jpg").read_bytes() == b"fake-frame"
    assert (tmp_path / "sample.tar.xz").read_bytes() == content
    assert not partial.exists()
    (tmp_path / "sample/C1/frame.jpg").write_bytes(b"existing")
    data.download_chokepoint(["sample.tar.xz"], tmp_path)
    assert len(calls) == 1
    assert (tmp_path / "sample/C1/frame.jpg").read_bytes() == b"existing"


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
