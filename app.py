from base64 import b64encode
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock, Thread
import logging
import os
import time
import winsound

# Gradio의 임시 파일과 앱 로그를 프로젝트 안에 모읍니다.
ROOT = Path(__file__).resolve().parent
os.environ['GRADIO_TEMP_DIR'] = str(ROOT / 'outputs/tmp')
LOG_DIR = ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

import cv2
import gradio as gr
import numpy as np
import requests
import serial
from PIL import Image, ImageDraw

ROBOFLOW_API_URL = 'https://serverless.roboflow.com'
TEST_VIDEO_PATH = ROOT / 'data/test/woni_10s_10fps.mp4'
SMILE_SOUND_PATH = ROOT / 'data/audio/geoje_yaho.wav'
MAX_WORKFLOW_REQUEST_BYTES = 4_500_000

logger = logging.getLogger('woni-face')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler(
        LOG_DIR / 'app.log', maxBytes=2_000_000, backupCount=3, encoding='utf-8'
    )
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)

board = None
board_lock = Lock()


class RoboflowAPIError(RuntimeError):
    """API 키를 포함하지 않는 사용자 표시용 Roboflow 오류입니다."""


class SmileSoundPlayer:
    """무표정에서 smile로 바뀔 때, 재생 중이 아니라면 효과음을 한 번 냅니다."""

    def __init__(self, sound_path=SMILE_SOUND_PATH, play_sound=None):
        self.sound_path = Path(sound_path)
        self.play_sound = play_sound or self._play_wav
        self.previous_detected = False
        self.playing = False
        self.lock = Lock()
        self.missing_sound_logged = False

    @staticmethod
    def _play_wav(sound_path):
        winsound.PlaySound(str(sound_path), winsound.SND_FILENAME)

    def update(self, detected):
        """상승 전이만 소비하며, 재생 중 발생한 전이는 나중에 다시 실행하지 않습니다."""
        with self.lock:
            rising_edge = detected and not self.previous_detected
            self.previous_detected = detected
            if not rising_edge or self.playing:
                return False
            if not self.sound_path.is_file():
                if not self.missing_sound_logged:
                    logger.warning('smile_sound_missing path=%s', self.sound_path)
                    self.missing_sound_logged = True
                return False
            self.playing = True

        Thread(target=self._play, daemon=True, name='smile-sound').start()
        return True

    def _play(self):
        try:
            logger.info('smile_sound_started path=%s', self.sound_path)
            self.play_sound(self.sound_path)
        except RuntimeError:
            logger.exception('smile_sound_failed')
        finally:
            with self.lock:
                self.playing = False
            logger.info('smile_sound_finished')


def connect(port):
    """마이크로비트 연결을 교체하거나 입력이 비어 있으면 해제합니다."""
    global board
    with board_lock:
        if board:
            board.close()
            board = None
        if not port.strip():
            logger.info('microbit_disconnected')
            return '연결 해제'
        try:
            board = serial.Serial(port.strip(), 115200, write_timeout=0.5)
            board.write(b'0\n')
            logger.info('microbit_connected port=%s', port.strip())
            return f'{port} 연결됨'
        except serial.SerialException as error:
            if board:
                board.close()
            board = None
            logger.warning('microbit_connection_failed error=%s', type(error).__name__)
            return f'연결 실패: {error}'


def send_face_command(detected):
    """탐지 여부를 실제 마이크로비트에 보내고 통신 상태를 반환합니다."""
    with board_lock:
        if not board:
            return '마이크로비트 미연결'
        try:
            board.write(b'1\n' if detected else b'0\n')
            return '마이크로비트 전송 완료'
        except serial.SerialException as error:
            logger.warning('microbit_write_failed error=%s', type(error).__name__)
            return '마이크로비트 전송 실패'


def render_microbit_face(detected):
    """실제 보드와 동시에 표시할 5×5 LED 표정을 이미지로 만듭니다."""
    happy = ('00000', '09090', '00000', '90009', '09990')
    asleep = ('00000', '99099', '00000', '00000', '09990')
    pattern = happy if detected else asleep
    canvas = Image.new('RGB', (300, 300), '#111827')
    draw = ImageDraw.Draw(canvas)
    for row, values in enumerate(pattern):
        for column, value in enumerate(values):
            color = '#facc15' if value == '9' else '#263244'
            left = 25 + column * 55
            top = 25 + row * 55
            draw.rounded_rectangle((left, top, left + 38, top + 38), radius=8, fill=color)
    return np.asarray(canvas)


def encode_image(image, max_encoded_chars):
    """배치 요청 예산에 맞춰 RGB 프레임을 JPEG base64로 압축합니다."""
    source = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert('RGB')
    original_width, original_height = source.size
    for max_dimension in (960, 800, 640, 480, 320):
        resized = source.copy()
        resized.thumbnail((max_dimension, max_dimension))
        for quality in (75, 65, 55, 45, 35):
            buffer = BytesIO()
            resized.save(buffer, format='JPEG', quality=quality, optimize=True)
            encoded = b64encode(buffer.getvalue()).decode('ascii')
            if len(encoded) <= max_encoded_chars:
                sent_width, sent_height = resized.size
                return encoded, original_width / sent_width, original_height / sent_height
    raise RoboflowAPIError('전체 영상을 한 요청에 담기에는 프레임 수가 너무 많습니다.')


def draw_predictions(image, predictions, scale_x=1.0, scale_y=1.0):
    """Roboflow 중심점 좌표를 원본 프레임 위의 상자와 라벨로 표시합니다."""
    output = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(output)
    for prediction in predictions:
        x = float(prediction['x']) * scale_x
        y = float(prediction['y']) * scale_y
        width = float(prediction['width']) * scale_x
        height = float(prediction['height']) * scale_y
        box = (x - width / 2, y - height / 2, x + width / 2, y + height / 2)
        confidence = float(prediction.get('confidence', 0))
        label = f"{prediction.get('class', 'object')} {confidence:.0%}"
        draw.rectangle(box, outline='#22c55e', width=4)
        text_box = draw.textbbox((box[0], box[1]), label)
        draw.rectangle(text_box, fill='#22c55e')
        draw.text((box[0], box[1]), label, fill='black')
    return np.asarray(output)


def compress_display_frame(image):
    """완성된 시각화 프레임을 반복 재생용 JPEG 바이트로 보관합니다."""
    ok, encoded = cv2.imencode(
        '.jpg', image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88]
    )
    if not ok:
        raise ValueError('시각화 프레임을 압축할 수 없습니다.')
    return encoded.tobytes()


def restore_display_frame(encoded):
    """반복 재생 캐시의 JPEG 바이트를 RGB 프레임으로 복원합니다."""
    bgr_frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr_frame is None:
        raise ValueError('시각화 프레임 캐시를 읽을 수 없습니다.')
    return bgr_frame[:, :, ::-1]


def find_predictions(result):
    """Workflow 출력 안에서 객체 탐지 prediction 목록을 찾습니다."""
    if isinstance(result, dict):
        predictions = result.get('predictions')
        if isinstance(predictions, list):
            if not predictions or all(
                isinstance(item, dict) and {'x', 'y', 'width', 'height'} <= item.keys()
                for item in predictions
            ):
                return predictions
        for value in result.values():
            found = find_predictions(value)
            if found is not None:
                return found
    elif isinstance(result, list):
        for item in result:
            found = find_predictions(item)
            if found is not None:
                return found
    return None


def infer_video_batch(session, encoded_frames, workspace_name, workflow_id, api_key):
    """영상의 모든 프레임을 단 한 번의 Workflow 요청으로 추론합니다."""
    workspace_name = workspace_name.strip().strip('/')
    workflow_id = workflow_id.strip().strip('/')
    if not workspace_name or '/' in workspace_name:
        raise RoboflowAPIError('Workspace ID를 한 항목으로 입력하세요.')
    if not workflow_id or '/' in workflow_id:
        raise RoboflowAPIError('Workflow ID를 한 항목으로 입력하세요.')

    endpoint = f'{ROBOFLOW_API_URL}/{workspace_name}/workflows/{workflow_id}'
    payload = {
        'inputs': {
            'image': [
                {'type': 'base64', 'value': encoded_frame} for encoded_frame in encoded_frames
            ]
        },
        'use_cache': True,
    }
    headers = {'Authorization': f'Bearer {api_key.strip()}'}
    started = time.perf_counter()
    logger.info(
        'batch_inference_request frames=%d workspace=%s workflow=%s encoded_chars=%d endpoint=%s',
        len(encoded_frames),
        workspace_name,
        workflow_id,
        sum(map(len, encoded_frames)),
        endpoint,
    )
    try:
        response = session.post(endpoint, json=payload, headers=headers, timeout=600)
    except requests.RequestException as error:
        logger.exception('batch_inference_network_error error=%s', type(error).__name__)
        raise RoboflowAPIError(f'네트워크 오류: {type(error).__name__}') from error

    elapsed = time.perf_counter() - started
    logger.info(
        'batch_inference_response status=%d elapsed=%.3f response_bytes=%d',
        response.status_code,
        elapsed,
        len(response.content),
    )
    if not response.ok:
        safe_body = response.text.replace(api_key.strip(), '[REDACTED]')[:1000]
        logger.error('batch_inference_http_error status=%d body=%s', response.status_code, safe_body)
        raise RoboflowAPIError(f'Roboflow HTTP {response.status_code}: {safe_body}')

    try:
        result = response.json()
    except requests.JSONDecodeError as error:
        logger.error('batch_inference_invalid_json')
        raise RoboflowAPIError('Roboflow 응답을 JSON으로 읽을 수 없습니다.') from error
    outputs = result.get('outputs') if isinstance(result, dict) else None
    if not isinstance(outputs, list):
        logger.error('workflow_outputs_missing result_type=%s', type(result).__name__)
        raise RoboflowAPIError('Workflow 응답에서 outputs 목록을 찾을 수 없습니다.')
    if len(outputs) != len(encoded_frames):
        logger.error('workflow_output_count_mismatch expected=%d actual=%d', len(encoded_frames), len(outputs))
        raise RoboflowAPIError(
            f'Workflow 결과 수가 프레임 수와 다릅니다: {len(outputs)}/{len(encoded_frames)}'
        )

    prediction_batches = []
    for frame_number, output in enumerate(outputs, start=1):
        predictions = find_predictions(output)
        if predictions is None:
            logger.error('workflow_predictions_missing frame=%d', frame_number)
            raise RoboflowAPIError(f'{frame_number}번 프레임의 탐지 결과를 찾을 수 없습니다.')
        prediction_batches.append(predictions)
    return prediction_batches, elapsed


def stream_video(video_path, workspace_name, workflow_id, api_key):
    """전체 영상을 한 번 추론한 뒤 저장된 결과와 LED 표정을 스트리밍합니다."""
    if not video_path:
        yield None, render_microbit_face(False), '영상을 업로드하세요.'
        return
    if not workspace_name.strip():
        yield None, render_microbit_face(False), 'Roboflow Workspace ID를 입력하세요.'
        return
    if not workflow_id.strip():
        yield None, render_microbit_face(False), 'Roboflow Workflow ID를 입력하세요.'
        return
    if not api_key.strip():
        yield None, render_microbit_face(False), 'Roboflow API 키를 입력하세요.'
        return

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        logger.error('video_open_failed path=%s', video_path)
        yield None, render_microbit_face(False), '영상을 열 수 없습니다.'
        return

    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        yield None, render_microbit_face(False), '영상의 프레임 수를 확인할 수 없습니다.'
        return

    logger.info('video_started path=%s fps=%.3f frames=%d', video_path, fps, total)
    yield None, render_microbit_face(False), f'{total}개 프레임을 배치 요청으로 준비하는 중…'

    encoded_budget = (MAX_WORKFLOW_REQUEST_BYTES - 20_000) // total
    encoded_frames = []
    scales = []
    frame_number = 0
    try:
        while True:
            ok, bgr_frame = capture.read()
            if not ok:
                break
            frame_number += 1
            encoded, scale_x, scale_y = encode_image(
                bgr_frame[:, :, ::-1], encoded_budget
            )
            encoded_frames.append(encoded)
            scales.append((scale_x, scale_y))
        capture.release()
        capture = None

        if not encoded_frames:
            yield None, render_microbit_face(False), '영상에 처리할 프레임이 없습니다.'
            return
        if sum(map(len, encoded_frames)) + len(encoded_frames) * 40 > MAX_WORKFLOW_REQUEST_BYTES:
            yield None, render_microbit_face(False), '배치 요청 크기 제한을 초과했습니다.'
            return

        yield None, render_microbit_face(False), (
            f'{len(encoded_frames)}개 프레임을 Roboflow에 한 번 요청하는 중…'
        )
        with requests.Session() as session:
            try:
                prediction_batches, latency = infer_video_batch(
                    session, encoded_frames, workspace_name, workflow_id, api_key
                )
            except (RoboflowAPIError, ValueError) as error:
                logger.error('video_stopped reason=%s', error)
                yield None, render_microbit_face(False), f'{error} · logs/app.log 확인'
                return

        # API 결과를 모두 받은 다음 모든 시각화 프레임을 먼저 완성합니다.
        del encoded_frames
        capture = cv2.VideoCapture(video_path)
        prepared_frames = []
        frame_summaries = []
        frame_number = 0
        while True:
            ok, bgr_frame = capture.read()
            if not ok or frame_number >= len(prediction_batches):
                break
            rgb_frame = bgr_frame[:, :, ::-1]
            predictions = prediction_batches[frame_number]
            scale_x, scale_y = scales[frame_number]
            annotated = draw_predictions(rgb_frame, predictions, scale_x, scale_y)
            frame_number += 1
            prepared_frames.append(compress_display_frame(annotated))
            frame_summaries.append((bool(predictions), len(predictions)))

        capture.release()
        capture = None
        if len(prepared_frames) != len(prediction_batches):
            raise RoboflowAPIError(
                f'시각화 프레임 수가 결과 수와 다릅니다: '
                f'{len(prepared_frames)}/{len(prediction_batches)}'
            )

        # 여기부터는 추가 추론 없이 완성된 프레임 캐시만 무한 반복합니다.
        playback_count = 0
        smile_sound = SmileSoundPlayer()
        while True:
            playback_count += 1
            playback_started = time.perf_counter()
            for index, (encoded_frame, summary) in enumerate(
                zip(prepared_frames, frame_summaries), start=1
            ):
                detected, detection_count = summary
                annotated = restore_display_frame(encoded_frame)
                smile_sound.update(detected)
                microbit_status = send_face_command(detected)
                status = (
                    f'캐시 재생 {playback_count}회차 · 프레임 {index}/{len(prepared_frames)} · '
                    f'후보 {detection_count}개 · 배치 API 1회 {latency:.2f}초 · '
                    f'{microbit_status}'
                )
                yield annotated, render_microbit_face(detected), status

                target_elapsed = index / fps
                delay = target_elapsed - (time.perf_counter() - playback_started)
                if delay > 0:
                    time.sleep(delay)
    finally:
        if capture is not None:
            capture.release()
        send_face_command(False)
        logger.info('video_finished processed_frames=%d total_frames=%d', frame_number, total)


# 영상 입력, API 인증, 추론 결과와 마이크로비트 표정을 한 화면에 구성합니다.
with gr.Blocks(title='원이 얼굴 탐지') as app:
    gr.Markdown(
        '# 원이 얼굴 탐지\n'
        '전체 프레임을 한 번에 추론한 뒤 결과와 마이크로비트 표정을 함께 재생합니다.'
    )
    with gr.Accordion('Roboflow API 설정', open=True):
        workspace_name = gr.Textbox(
            label='Roboflow Workspace ID', placeholder='Deploy Workflow 코드의 workspace_name'
        )
        workflow_id = gr.Textbox(
            label='Roboflow Workflow ID', placeholder='Deploy Workflow 코드의 workflow_id'
        )
        api_key = gr.Textbox(
            label='Roboflow API 키',
            type='password',
            placeholder='Workspace Settings에서 발급한 Private API Key',
        )

    video = gr.Video(sources=['upload'], label='분석할 영상')
    if TEST_VIDEO_PATH.is_file():
        gr.Examples([[str(TEST_VIDEO_PATH)]], inputs=video, label='10초 · 10fps 테스트 영상')

    with gr.Row():
        inference_output = gr.Image(label='현재 프레임 추론 결과', streaming=True)
        face_output = gr.Image(
            value=render_microbit_face(False), label='현재 마이크로비트 표정', streaming=True
        )
    status = gr.Textbox(label='처리 상태', interactive=False)
    with gr.Row():
        start = gr.Button('영상 추론 시작', variant='primary')
        stop = gr.Button('중지', variant='stop')
    stream_event = start.click(
        stream_video,
        [video, workspace_name, workflow_id, api_key],
        [inference_output, face_output, status],
    )
    stop.click(fn=None, cancels=[stream_event])

    gr.Markdown('### 실제 마이크로비트 연결\nCOM 번호를 입력하세요. 빈칸으로 연결하면 해제됩니다.')
    port = gr.Textbox(label='COM 포트', placeholder='예: COM3')
    connection = gr.Textbox(label='연결 상태', interactive=False)
    gr.Button('연결 / 해제').click(connect, port, connection)
    gr.Markdown('오류 상세는 API 키가 제거된 `logs/app.log`에 기록됩니다.')

if __name__ == '__main__':
    logger.info('app_started api_url=%s', ROBOFLOW_API_URL)
    app.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1', server_port=7862)
