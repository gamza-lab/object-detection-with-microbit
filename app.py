from pathlib import Path
from threading import Lock
import os

ROOT = Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.yolo')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(ROOT / '.torch-cache')
os.environ['GRADIO_TEMP_DIR'] = str(ROOT / 'outputs/tmp')

import gradio as gr
import serial
from ultralytics import YOLO

MODEL_PATH = ROOT / 'models/woni_face.pt'
if not MODEL_PATH.is_file():
    raise FileNotFoundError('Colab에서 best.pt를 내려받아 models/woni_face.pt로 넣어 주세요.')
model = YOLO(str(MODEL_PATH))
board = None
lock = Lock()


def connect(port):
    """COM 번호를 입력하면 연결, 비우면 해제합니다."""
    global board
    with lock:
        if board:
            board.close()
            board = None
        if not port.strip():
            return '연결 해제'
        try:
            board = serial.Serial(port.strip(), 115200, write_timeout=0.5)
            board.write(b'0\n')
            return f'{port} 연결됨'
        except serial.SerialException as error:
            if board:
                board.close()
            board = None
            return f'연결 실패: {error}'


def detect(image, confidence):
    if image is None:
        return None, '사진을 넣거나 웹캠을 시작하세요.'
    with lock:
        # Gradio는 RGB, YOLO의 배열 입력은 BGR입니다.
        result = model.predict(image[:, :, ::-1], conf=confidence, imgsz=416, verbose=False)[0]
        count = len(result.boxes)
        status = f'원이 얼굴 후보 {count}개'
        if board:
            try:
                board.write(b'1\n' if count else b'0\n')
            except serial.SerialException:
                status += ' · 마이크로비트를 다시 연결하세요.'
        return result.plot()[:, :, ::-1], status


with gr.Blocks(title='원이 얼굴 탐지') as app:
    gr.Markdown('# 원이 얼굴 탐지\n한 직캠으로 학습한 시제품입니다. 다른 영상에서는 오탐할 수 있습니다.')
    confidence = gr.Slider(0.1, 0.95, value=0.5, step=0.05, label='탐지 기준값')
    with gr.Row():
        with gr.Tabs():
            with gr.Tab('사진'):
                photo = gr.Image(sources=['upload'], type='numpy', label='사진')
            with gr.Tab('웹캠'):
                camera = gr.Image(sources=['webcam'], type='numpy', streaming=True, label='웹캠')
        output = gr.Image(label='얼굴 탐지 결과')
    status = gr.Textbox(label='상태', interactive=False)
    gr.Button('탐지하기', variant='primary').click(detect, [photo, confidence], [output, status])
    camera.stream(detect, [camera, confidence], [output, status], stream_every=0.3)
    gr.Examples([[str(p)] for p in sorted((ROOT / 'data/face_dataset/images/val').glob('*.jpg'))[:3]],
                inputs=photo, label='예제 사진')
    gr.Markdown('### 마이크로비트\nCOM 번호를 입력하세요. 빈칸으로 연결하면 해제됩니다.')
    port = gr.Textbox(label='COM 포트', placeholder='예: COM3')
    connection = gr.Textbox(label='연결 상태', interactive=False)
    gr.Button('연결 / 해제').click(connect, port, connection)

if __name__ == '__main__':
    app.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1', server_port=7862)

