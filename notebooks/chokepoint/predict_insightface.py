"""Enrol ChokePoint Portal 1 subjects and write Portal 2 predictions locally.

Run from the repository root with insightface, onnxruntime and pandas installed.
This script uses the companion chokepoint_data loader. The model archive,
gallery, frames and prediction CSVs stay in local dataset directories.
"""

import argparse
import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np

PROBE_CAMERAS = {"P2E_S1": "C3", "P2E_S2": "C2", "P2E_S3": "C1", "P2E_S4": "C2"}
CSV_COLUMNS = [
    "sequence", "session", "camera", "frame", "time_s", "box_x1", "box_y1",
    "box_x2", "box_y2", "box_area", "best_id", "best_score", "second_score",
    "matches_label",
]


def read_faces(model, path):
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise OSError(f"Cannot read frame {path}")
    return model.get(image, max_num=0)


def box_area(box):
    x1, y1, x2, y2 = box
    return float((x2 - x1) * (y2 - y1))


def contains_eyes(box, label):
    x1, y1, x2, y2 = box
    return (
        x1 <= label.left_eye_x <= x2 and x1 <= label.right_eye_x <= x2
        and y1 <= label.left_eye_y <= y2 and y1 <= label.right_eye_y <= y2
    )


def unit(vector):
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Expected a finite, nonzero face embedding")
    return vector / norm


def build_gallery(model, labels, folder):
    """Average and re-normalize the matched embeddings for each enrolled person."""
    gallery = {}
    for person_id, person_labels in labels.groupby("person_id", sort=True):
        embeddings = []
        for label in person_labels.itertuples(index=False):
            path = folder / f"{label.frame:08d}.jpg"
            matches = [
                face for face in read_faces(model, path)
                if contains_eyes(face.bbox, label)
            ]
            if matches:
                selected = max(matches, key=lambda face: box_area(face.bbox))
                embeddings.append(unit(selected.normed_embedding))
        if not embeddings:
            raise ValueError(f"No matched enrolment faces for subject {person_id}")
        gallery[person_id] = unit(np.mean(embeddings, axis=0))
    return gallery


def predict_sequence(model, gallery, labels, folder, sequence, camera, output):
    """Predict every image and return frame-level detection/identification counts."""
    person_ids = sorted(gallery)
    if len(person_ids) < 2:
        raise ValueError("At least two gallery subjects are required")
    vectors = np.stack([gallery[person] for person in person_ids])
    counts = {
        "frames": 0, "detected_faces": 0, "labelled_faces": len(labels),
        "matched_labels": 0, "eligible_identifications": 0, "correct_identifications": 0,
    }
    if labels.empty:
        raise ValueError(f"No ground-truth labels for {sequence}/{camera}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", newline="", dir=output.parent, encoding="utf-8", delete=False) as temporary:
        partial = Path(temporary.name)
        try:
            writer = csv.writer(temporary)
            writer.writerow(CSV_COLUMNS)
            for session, session_labels in labels.groupby("session", sort=True):
                day = session.removeprefix(sequence)
                images = sorted((folder / f"{sequence}_{camera}{day}").glob("*.jpg"))
                by_frame = {row.frame: row for row in session_labels.itertuples(index=False)}
                if len(by_frame) != len(session_labels) or not set(by_frame).issubset(
                    int(image.stem) for image in images
                ):
                    raise ValueError(f"Duplicate labels or missing labelled images in {session}/{camera}")
                for path in images:
                    frame = int(path.stem)
                    label = by_frame.get(frame)
                    matches = []
                    counts["frames"] += 1
                    for face in read_faces(model, path):
                        coordinates = tuple(map(float, face.bbox))
                        area = box_area(coordinates)
                        scores = vectors @ unit(face.normed_embedding)
                        first, second = np.argsort(-scores, kind="stable")[:2]
                        best_id = person_ids[first]
                        matched = label is not None and contains_eyes(coordinates, label)
                        if matched:
                            matches.append((area, best_id))
                        writer.writerow((
                            sequence, session, camera, frame, frame / 30,
                            *coordinates, area, best_id, float(scores[first]), float(scores[second]),
                            matched,
                        ))
                        counts["detected_faces"] += 1
                    if matches:
                        counts["matched_labels"] += 1
                        if label.person_id in gallery:
                            counts["eligible_identifications"] += 1
                            # Match the largest-box convention used for downstream records.
                            selected_id = max(matches, key=lambda match: match[0])[1]
                            counts["correct_identifications"] += int(selected_id == label.person_id)
                print(f"Completed {session}/{camera}: {len(images):,} frames", flush=True)
            temporary.close()
            partial.replace(output)
        finally:
            temporary.close()
            partial.unlink(missing_ok=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/chokepoint"))
    parser.add_argument("--ground-truth", type=Path, help="Override the default dataset-dir/groundtruth")
    parser.add_argument("--enrol-dir", type=Path, help="Override the default dataset-dir/P1E_S1_C1")
    parser.add_argument("--frames-dir", type=Path, help="Override the directory containing Portal 2 camera folders")
    parser.add_argument("--model-root", type=Path, default=Path.home() / ".insightface")
    parser.add_argument("--sequences", nargs="+", choices=PROBE_CAMERAS, default=["P2E_S1"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    import cv2
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    from chokepoint_data import load_ground_truth

    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    model = FaceAnalysis(
        name="buffalo_l", root=str(args.model_root),
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"], sess_options=options,
    )
    model.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
    frames, _, _ = load_ground_truth(args.ground_truth or args.dataset_dir / "groundtruth")
    enrolment = frames.query('sequence == "P1E_S1" and camera == "C1"')
    started = perf_counter()
    gallery = build_gallery(model, enrolment, args.enrol_dir or args.dataset_dir / "P1E_S1_C1")
    if len(gallery) != 25:
        raise ValueError(f"Expected 25 enrolled subjects, found {len(gallery)}")
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.dataset_dir / "gallery.npz",
        person_ids=np.array(list(gallery)),
        embeddings=np.stack(list(gallery.values())),
    )
    print(json.dumps({"gallery_size": len(gallery), "gallery_seconds": perf_counter() - started}), flush=True)

    for sequence in args.sequences:
        camera = PROBE_CAMERAS[sequence]
        labels = frames.query("sequence == @sequence and camera == @camera")
        output = args.dataset_dir / "predictions" / f"{sequence}_{camera}.csv"
        started = perf_counter()
        counts = predict_sequence(
            model, gallery, labels, args.frames_dir or args.dataset_dir, sequence, camera, output
        )
        print(json.dumps({"sequence": sequence, "camera": camera, **counts,
                          "prediction_seconds": perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
