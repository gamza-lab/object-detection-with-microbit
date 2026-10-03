from base64 import b64encode
from pathlib import Path
from threading import Lock, Thread
import logging
import os
import time
import winsound

import cv2
import gradio as gr
import numpy as np
import requests
import serial

# 프로젝트 안의 데이터·로그·임시 파일 경로는 실행 위치와 무관하게 찾도록 한다.
ROOT = Path(__file__).resolve().parent
os.environ["GRADIO_TEMP_DIR"] = str(ROOT / "outputs/tmp")
API_URL = "https://serverless.roboflow.com"
TEST_VIDEO = ROOT / "data/test/woni_10s_10fps.mp4"
SMILE_SOUND = ROOT / "data/audio/geoje_yaho.wav"

# API 오류를 화면에 모두 노출하지 않고 상세 내용은 로그 파일에 남긴다.
(ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(filename=ROOT / "logs/app.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("face-detection")

# Gradio 이벤트가 동시에 직렬 포트를 열거나 쓰지 않도록 하나의 연결과 잠금을 공유한다.
board = None
board_lock = Lock()


class SmileSoundPlayer:
    """무표정에서 웃음으로 바뀔 때만 소리를 재생합니다."""

    def __init__(self):
        self.was_smiling = False
        self.playing = False
        self.lock = Lock()

    def update(self, smiling):
        with self.lock:
            # 웃는 상태가 유지되는 동안에는 매 프레임마다 소리가 반복되지 않게 한다.
            changed_to_smile = smiling and not self.was_smiling
            self.was_smiling = smiling
            if not changed_to_smile or self.playing:
                return
            self.playing = True
        Thread(target=self._play, daemon=True).start()

    def _play(self):
        try:
            winsound.PlaySound(str(SMILE_SOUND), winsound.SND_FILENAME)
        finally:
            with self.lock:
                self.playing = False


def connect_microbit(port):
    """입력한 COM 포트로 마이크로비트를 연결합니다."""
    global board
    with board_lock:
        if board:
            board.close()
            board = None
        if not port.strip():
            return "연결 해제"
        try:
            board = serial.Serial(port.strip(), 115200, write_timeout=0.5)
            return f"{port} 연결됨"
        except serial.SerialException as error:
            return f"연결 실패: {error}"


def send_face(smiling):
    """웃음은 1, 무표정은 0으로 전송합니다."""
    with board_lock:
        if not board:
            return "마이크로비트 미연결"
        board.write(b"1\n" if smiling else b"0\n")
        return "마이크로비트 전송 완료"


def make_face_image(smiling):
    """마이크로비트의 5×5 LED 표정을 화면에도 그립니다."""
    happy = ("00000", "09090", "00000", "90009", "09990")
    neutral = ("00000", "99099", "00000", "00000", "09990")
    image = np.full((300, 300, 3), (39, 50, 68), dtype=np.uint8)
    for row, line in enumerate(happy if smiling else neutral):
        for column, value in enumerate(line):
            color = (21, 204, 250) if value == "9" else (17, 24, 39)
            start = (25 + column * 55, 25 + row * 55)
            cv2.rectangle(image, start, (start[0] + 38, start[1] + 38), color, -1)
    return image[:, :, ::-1]


def read_video(video_path):
    """모든 프레임을 API 전송용 JPEG로 변환합니다."""
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    images, scales = [], []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        # 업로드 크기와 처리 시간을 줄이되 원본 종횡비는 유지한다.
        sent_width = min(width, 640)
        sent_height = round(height * sent_width / width)
        resized = cv2.resize(frame, (sent_width, sent_height))
        ok, jpeg = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ok:
            raise RuntimeError("프레임을 JPEG로 변환하지 못했습니다.")
        images.append({"type": "base64", "value": b64encode(jpeg).decode()})
        scales.append((width / sent_width, height / sent_height))
    capture.release()
    return fps, images, scales


def request_predictions(images, workspace, workflow, api_key):
    """모든 프레임을 Roboflow Workflow에 한 번만 요청합니다."""
    endpoint = f"{API_URL}/{workspace.strip()}/workflows/{workflow.strip()}"
    # 프레임마다 요청하지 않고 배열 하나로 보내 네트워크 왕복을 한 번으로 줄인다.
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key.strip()}"},
        json={"inputs": {"image": images}}, timeout=600,
    )
    logger.info("Roboflow status=%s frames=%s", response.status_code, len(images))
    if not response.ok:
        logger.error("Roboflow error=%s", response.text[:1000])
        raise RuntimeError(f"Roboflow HTTP {response.status_code}: {response.text[:300]}")
    outputs = response.json()["outputs"]
    return [output["predictions"]["predictions"] for output in outputs]


def prepare_frames(video_path, predictions, scales):
    """추론 상자를 그린 뒤 반복 재생용 JPEG로 저장합니다."""
    if len(predictions) != len(scales):
        raise RuntimeError("영상 프레임 수와 API 결과 수가 다릅니다.")
    capture = cv2.VideoCapture(video_path)
    prepared = []
    for frame_predictions, (scale_x, scale_y) in zip(predictions, scales):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("영상 프레임을 읽지 못했습니다.")
        for item in frame_predictions:
            # API 좌표는 축소 영상 기준이므로 원본 영상 크기로 다시 환산한다.
            x, y = item["x"] * scale_x, item["y"] * scale_y
            width, height = item["width"] * scale_x, item["height"] * scale_y
            left, top = int(x - width / 2), int(y - height / 2)
            right, bottom = int(x + width / 2), int(y + height / 2)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 220, 0), 4)
            label = f'{item["class"]} {item["confidence"]:.0%}'
            cv2.putText(frame, label, (left, max(25, top)), 0, 0.7, (0, 220, 0), 2)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        # 이 모델은 웃는 얼굴만 탐지하므로 탐지 결과의 존재 여부가 웃음 상태가 된다.
        prepared.append((jpeg.tobytes(), bool(frame_predictions), len(frame_predictions)))
    capture.release()
    return prepared


def stream_video(video_path, workspace, workflow, api_key):
    """추론을 모두 마친 뒤 준비된 프레임을 무한 반복합니다."""
    try:
        yield None, make_face_image(False), "영상 프레임 준비 중"
        fps, images, scales = read_video(video_path)
        yield None, make_face_image(False), f"{len(images)}프레임을 한 번에 추론 중"
        predictions = request_predictions(images, workspace, workflow, api_key)
        frames = prepare_frames(video_path, predictions, scales)
        sound = SmileSoundPlayer()

        # API는 최초 한 번만 호출하고, 가공된 프레임은 사용자가 중지할 때까지 반복한다.
        while True:
            started = time.perf_counter()
            for index, (jpeg, smiling, count) in enumerate(frames, start=1):
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                sound.update(smiling)
                microbit_status = send_face(smiling)
                status = f"프레임 {index}/{len(frames)} · 탐지 {count}개 · {microbit_status}"
                yield frame[:, :, ::-1], make_face_image(smiling), status
                # 인코딩·출력에 걸린 시간을 빼서 원본 영상의 FPS에 가깝게 재생한다.
                time.sleep(max(0, index / fps - (time.perf_counter() - started)))
    except Exception as error:
        logger.exception("video failed")
        yield None, make_face_image(False), f"오류: {error} · logs/app.log 확인"
    finally:
        send_face(False)


# 화면 구성과 버튼 이벤트 연결
with gr.Blocks(title="얼굴 탐지") as app:
    gr.Markdown("# 얼굴 탐지\n전체 영상을 한 번 추론한 뒤 결과를 반복 재생합니다.")
    workspace = gr.Textbox(label="Roboflow Workspace ID")
    workflow = gr.Textbox(label="Roboflow Workflow ID")
    api_key = gr.Textbox(label="Roboflow API 키", type="password")
    video = gr.Video(sources=["upload"], label="분석할 영상")
    gr.Examples([[str(TEST_VIDEO)]], inputs=video, label="10초 테스트 영상")
    with gr.Row():
        result = gr.Image(label="추론 결과", streaming=True)
        face = gr.Image(value=make_face_image(False), label="마이크로비트 표정", streaming=True)
    status = gr.Textbox(label="상태", interactive=False)
    with gr.Row():
        start = gr.Button("영상 추론 시작", variant="primary")
        stop = gr.Button("중지", variant="stop")
    event = start.click(stream_video, [video, workspace, workflow, api_key], [result, face, status])
    stop.click(fn=None, cancels=[event])
    port = gr.Textbox(label="마이크로비트 COM 포트", placeholder="예: COM3")
    connection = gr.Textbox(label="연결 상태", interactive=False)
    gr.Button("연결 / 해제").click(connect_microbit, port, connection)


if __name__ == "__main__":
    # 한 번에 영상 하나만 처리해 메모리 사용량과 직렬 포트 충돌을 제한한다.
    app.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7862)
