# 원이 얼굴 탐지

학습은 **Colab GPU**, 사진·웹캠 탐지와 마이크로비트 연결은 **PC의 Gradio 앱**에서 진행합니다.

## 1. 데이터 준비

[Roboflow 프로젝트](https://app.roboflow.com/gangjun-jo/woni-face-keris/1)에서 얼굴을 `woni_face`로 라벨링하고 데이터셋 버전을 생성합니다. **Download Dataset → YOLOv8 → Download zip to computer**로 ZIP을 받습니다. Roboflow의 Train이나 모델 가중치 다운로드는 사용하지 않습니다.

원본 이미지 비율을 유지하면 됩니다. 정사각형 Resize는 필수가 아닙니다. 이미 생성한 v1에는 640×640 검은 여백 전처리가 들어 있으므로, 원본 크기로 내보내려면 Resize 없이 새 버전을 만드세요.

빠른 실습에는 `data/woni_roboflow.zip`을 사용합니다. 이 파일은 원본 비율의 로컬 준비 데이터로, Roboflow v1 전처리 결과와는 다릅니다. 얼굴 라벨 36장과 Null 1장, 학습/검증/평가 26/5/6장입니다.

## 2. Colab에서 학습과 다운로드

로컬 `notebooks/train.ipynb`를 [Colab](https://colab.research.google.com/)의 **노트 업로드**로 여세요. 이 파일이 최신 간단 버전입니다.

1. **런타임 → 런타임 유형 변경 → T4 GPU → 저장**을 선택합니다.
2. 1번 설치 셀을 실행합니다.
3. PC에서 YOLOv8 ZIP 이름을 `woni.zip`으로 바꾸고 2번 셀을 실행해 업로드합니다. ZIP에는 `train`, `valid`, `test` 폴더가 있어야 합니다. 원이 얼굴 한 클래스와 고정 경로를 사용합니다.
4. 3번 학습 셀을 실행합니다. IU 예제와 같은 **YOLO11n / 50 epochs / imgsz 640 / batch 16 / workers 0**을 사용합니다. `device=0`으로 Colab GPU를 지정합니다.
5. 학습이 끝나면 4번 다운로드 셀을 실행해 **best.pt**를 PC에 저장합니다.
6. 다운로드가 완료되면 **런타임 → 런타임 연결 해제 및 삭제**로 종료합니다. 임시 데이터와 학습 결과는 삭제되므로 다운로드를 먼저 확인하세요.

같은 런타임에서 재학습하면 `/content/runs/woni` 결과를 덮어씁니다. 다른 데이터셋은 새 런타임에서 시작하세요.

`imgsz=640`은 모델 입력 설정이며 원본 사진을 정사각형으로 저장하라는 뜻이 아닙니다. 무료 GPU 배정은 보장되지 않습니다. 현재 앱의 모델은 기존 PC 학습 결과이며, Colab 재학습 결과를 적용하려면 다음 단계를 진행하세요.

## 3. PC 앱에 모델 적용

실행 중인 앱을 종료하고 `best.pt`를 `woni_face.pt`로 이름을 바꿔 `models/`에 넣습니다. 기존 모델을 보존하려면 먼저 다른 이름으로 보관하세요.

처음 사용하는 PC는 Python 3.12 설치 후 프로젝트 폴더에서 실행합니다.

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -B app.py
```

기존 환경은 마지막 명령만 실행합니다. http://127.0.0.1:7862 에서 사진 또는 웹캠으로 탐지합니다. PC에서는 학습 명령을 실행하지 않습니다.

## 4. 마이크로비트 연결

[MakeCode](https://makecode.microbit.org/)에서 새 프로젝트를 만들고 **Python**을 선택해 `microbit/main.py`를 붙여 넣고 다운로드합니다. MakeCode 장치 연결을 해제한 뒤 PC 탐지 웹페이지에서 COM 포트를 연결합니다.

MakeCode Python 코드이며 PC Python 또는 MicroPython 편집기용이 아닙니다. USB로 `1`을 받으면 웃고, `0`을 받거나 2초간 명령이 없으면 기본 표정으로 돌아갑니다.

## 파일

- `notebooks/train.ipynb`: 웹 GPU 학습과 best.pt 다운로드
- `app.py`: Gradio 탐지 화면과 USB 연결
- `microbit/main.py`: USB 수신과 LED 표정
- `models/woni_face.pt`: 앱에서 읽는 모델
- `data/raw/`: 로컬 원본 영상과 메타데이터(Git 제외)
- `_보관/`: 이전 코드와 학습 기록

한 직캠으로 만든 시제품입니다. 인접 프레임 평가 점수는 다른 영상에서의 성능을 뜻하지 않습니다. 실물 보드와 웹캠 실시간 입력은 아직 미검증입니다.

출처: [원본 영상](https://www.youtube.com/watch?v=VINDNiicjb4), [IU-detection](https://github.com/whyz-dev/IU-detection), [Colab FAQ](https://research.google.com/colaboratory/faq.html)
