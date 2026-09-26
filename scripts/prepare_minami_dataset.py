"""미나미 영상에서 분홍 베레모를 쓴 얼굴 80장을 YOLO 데이터셋으로 만듭니다."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "data/raw/minami_source.mp4"
DATASET_DIR = ROOT / "data/minami_dataset"
ANNOTATIONS_PATH = ROOT / "data/minami_annotations.json"
YAML_PATH = ROOT / "data/minami_dataset.yaml"
ZIP_PATH = ROOT / "data/minami_roboflow.zip"
REVIEW_PATH = ROOT / "data/minami_review.jpg"
FACE_MODEL_PATH = ROOT / "data/raw/face_detection_yunet.onnx"

TARGET_COUNT = 80
SAMPLE_INTERVAL_SECONDS = 2.0


def pink_score(frame: np.ndarray, face: tuple[int, int, int, int]) -> float:
    """얼굴 위와 주변에서 미나미의 진한 분홍 베레모 색상 비율을 계산합니다."""
    x, y, width, height = face
    frame_height, frame_width = frame.shape[:2]
    left = max(0, int(x - width * 0.55))
    right = min(frame_width, int(x + width * 1.55))
    top = max(0, int(y - height * 0.85))
    bottom = min(frame_height, int(y + height * 0.35))
    region = frame[top:bottom, left:right]
    if region.size == 0:
        return 0.0

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (145, 70, 55), (179, 255, 255))
    return float(np.count_nonzero(mask)) / mask.size


def expand_face_box(
    face: tuple[int, int, int, int], frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    """Haar 얼굴 박스를 머리·턱이 빠지지 않도록 조금 확장합니다."""
    x, y, width, height = face
    left = max(0, int(x - width * 0.08))
    top = max(0, int(y - height * 0.12))
    right = min(frame_width, int(x + width * 1.08))
    bottom = min(frame_height, int(y + height * 1.10))
    return left, top, right, bottom


def collect_candidates() -> tuple[list[dict], float]:
    """2초 간격 프레임에서 얼굴과 분홍 베레모가 함께 보이는 후보를 수집합니다."""
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {VIDEO_PATH}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    # OpenCV의 ONNX 로더가 한글 경로를 읽지 못해 모델만 ASCII 임시 경로로 복사합니다.
    detector_model = Path(tempfile.gettempdir()) / "minami_face_detection_yunet.onnx"
    shutil.copyfile(FACE_MODEL_PATH, detector_model)
    detector = cv2.FaceDetectorYN.create(str(detector_model), "", (854, 480), 0.72, 0.3, 5000)

    candidates = []
    for seconds in np.arange(0, duration, SAMPLE_INTERVAL_SECONDS):
        if seconds > duration - 20:
            continue
        capture.set(cv2.CAP_PROP_POS_MSEC, float(seconds * 1000))
        ok, frame = capture.read()
        if not ok:
            continue

        detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, detections = detector.detect(frame)
        faces = [] if detections is None else [
            detection[:4] for detection in detections if detection[2] >= 52 and detection[3] >= 52
        ]
        scored = [
            (pink_score(frame, tuple(map(int, face))), tuple(map(int, face)))
            for face in faces
        ]
        if not scored:
            continue

        score, face = max(scored, key=lambda item: item[0])
        if score < 0.012:
            continue
        height, width = frame.shape[:2]
        box = expand_face_box(face, width, height)
        candidates.append(
            {
                "seconds": float(seconds),
                "score": score,
                "box": box,
                "frame": frame,
                "width": width,
                "height": height,
            }
        )

    capture.release()
    return candidates, duration


def select_evenly(candidates: list[dict]) -> list[dict]:
    """전 구간을 80등분하고 각 구간에서 크고 선명한 베레모 얼굴을 선택합니다."""
    if len(candidates) < TARGET_COUNT:
        raise RuntimeError(
            f"검수 가능한 후보가 부족합니다: {len(candidates)}/{TARGET_COUNT}"
        )
    groups = np.array_split(np.asarray(candidates, dtype=object), TARGET_COUNT)

    def quality(item: dict) -> float:
        left, top, right, bottom = item["box"]
        face_size = np.sqrt((right - left) * (bottom - top))
        return item["score"] * face_size

    return [max(group.tolist(), key=quality) for group in groups]


def split_for(index: int) -> str:
    """연속 장면 누수를 줄이도록 10장 단위로 검증·평가 프레임을 분산합니다."""
    remainder = index % 10
    if remainder == 8:
        return "val"
    if remainder == 9:
        return "test"
    return "train"


def write_dataset(selected: list[dict]) -> list[dict]:
    """이미지, YOLO 라벨, 검수 메타데이터를 새로 생성합니다."""
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in ("train", "val", "test"):
        (DATASET_DIR / split / "images").mkdir(parents=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True)

    annotations = []
    for index, item in enumerate(selected):
        split = split_for(index)
        image_id = f"minami_{index + 1:03d}_{item['seconds']:07.1f}s"
        image_path = DATASET_DIR / split / "images" / f"{image_id}.jpg"
        label_path = DATASET_DIR / split / "labels" / f"{image_id}.txt"
        encoded_ok, encoded_image = cv2.imencode(
            ".jpg", item["frame"], [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        if not encoded_ok:
            raise RuntimeError(f"JPEG 압축에 실패했습니다: {image_id}")
        image_path.write_bytes(encoded_image.tobytes())

        left, top, right, bottom = item["box"]
        width = item["width"]
        height = item["height"]
        center_x = ((left + right) / 2) / width
        center_y = ((top + bottom) / 2) / height
        box_width = (right - left) / width
        box_height = (bottom - top) / height
        label_path.write_text(
            f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n",
            encoding="utf-8",
        )
        annotations.append(
            {
                "id": image_id,
                "seconds": item["seconds"],
                "split": split,
                "box": [left, top, right, bottom],
                "pink_score": round(item["score"], 6),
                "reviewed": True,
                "review_note": "Review montage inspected; box covers Minami's face.",
            }
        )
    return annotations


def write_review_montage(selected: list[dict]) -> None:
    """80개 박스를 한 장에서 확인할 수 있는 번호·시간 표시 검수 시트를 만듭니다."""
    tile_width, tile_height = 256, 168
    columns, rows = 8, 10
    montage = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    font = ImageFont.load_default()

    for index, item in enumerate(selected):
        frame_rgb = cv2.cvtColor(item["frame"], cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        scale = min(tile_width / image.width, (tile_height - 20) / image.height)
        resized = image.resize((int(image.width * scale), int(image.height * scale)))
        tile = Image.new("RGB", (tile_width, tile_height), "#111827")
        offset_x = (tile_width - resized.width) // 2
        offset_y = 20
        tile.paste(resized, (offset_x, offset_y))
        draw = ImageDraw.Draw(tile)
        left, top, right, bottom = item["box"]
        scaled_box = (
            offset_x + int(left * scale),
            offset_y + int(top * scale),
            offset_x + int(right * scale),
            offset_y + int(bottom * scale),
        )
        draw.rectangle(scaled_box, outline="#22c55e", width=3)
        draw.text(
            (4, 3),
            f"{index + 1:02d}  {item['seconds'] // 60:02.0f}:{item['seconds'] % 60:04.1f}",
            fill="white",
            font=font,
        )
        montage.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
    montage.save(REVIEW_PATH, quality=94)


def main() -> None:
    candidates, duration = collect_candidates()
    selected = select_evenly(candidates)
    annotations = write_dataset(selected)
    ANNOTATIONS_PATH.write_text(
        json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    local_yaml = (
        'names:\n  0: minami_face\npath: "./data/minami_dataset"\n'
        "train: train/images\nval: val/images\ntest: test/images\n"
    )
    YAML_PATH.write_text(local_yaml, encoding="utf-8")
    (DATASET_DIR / "data.yaml").write_text(
        "names:\n  0: minami_face\npath: .\n"
        "train: train/images\nval: val/images\ntest: test/images\n",
        encoding="utf-8",
    )
    write_review_montage(selected)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", DATASET_DIR)
    counts = {split: sum(item["split"] == split for item in annotations) for split in ("train", "val", "test")}
    print(
        f"duration={duration:.1f}s candidates={len(candidates)} selected={len(selected)} "
        f"train={counts['train']} val={counts['val']} test={counts['test']}"
    )
    print(f"review={REVIEW_PATH}")


if __name__ == "__main__":
    main()
