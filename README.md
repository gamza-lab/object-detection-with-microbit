# 원이 얼굴 탐지

업로드한 영상의 모든 프레임을 **Roboflow Serverless API에 한 번의 배치 요청으로 전송**합니다. 전체 추론이 끝나면 모든 시각화 프레임을 미리 생성한 뒤, 탐지 영상과 마이크로비트 5×5 LED 표정을 중지할 때까지 무한 반복 재생합니다. 로컬 모델, 사진 입력, 웹캠 입력은 사용하지 않습니다.

## 실행

Python 3.12에서 프로젝트 전용 가상환경을 새로 만듭니다. 기존 `.venv`가 Python 3.14로 만들어졌다면 지우고 아래 명령으로 다시 만드는 편이 안전합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B app.py
```

브라우저에서 <http://127.0.0.1:7862>를 열고 다음 순서로 실행합니다.

1. Deploy Workflow 코드에 표시된 Workspace ID를 입력합니다.
2. 같은 코드에 표시된 Workflow ID를 입력합니다.
3. 해당 Workspace의 Private API Key를 비밀번호 입력칸에 직접 입력합니다.
4. 영상을 업로드하고 **영상 추론 시작**을 누릅니다.
5. 전체 추론과 시각화 준비가 끝난 뒤 탐지 영상과 마이크로비트 표정이 무한 반복되는지 확인합니다.

Workspace ID, Workflow ID와 API 키는 기본값으로 넣지 않으며 API 키는 소스 코드나 Git에도 저장하지 않습니다. HTTP 요청은 영상당 한 번이지만 과금은 요청 수가 아니라 처리한 이미지 수를 기준으로 하므로, 긴 영상은 여전히 크레딧과 처리 시간이 크게 늘어날 수 있습니다.

## Roboflow API 키와 ID 확인 방법

### API 키

1. [Roboflow](https://app.roboflow.com/)에 로그인합니다.
2. 왼쪽 위에서 사용할 모델이 들어 있는 Workspace를 선택합니다.
3. 왼쪽 메뉴의 **Settings**를 누릅니다.
4. **API Keys** 메뉴를 엽니다.
5. Serverless Cloud API를 사용할 수 있는 **Private API Key**를 복사합니다.
6. 앱의 **Roboflow API 키** 입력칸에 직접 붙여 넣습니다.

Publishable Key나 다른 Workspace의 키를 넣으면 `401 Unauthorized`가 발생하거나 비공개 모델과 Workflow를 찾지 못할 수 있습니다. API 키는 README, 소스 코드, 로그 또는 Git 커밋에 넣지 않습니다.

### 모델 ID

1. Roboflow 왼쪽 메뉴에서 **Models**를 엽니다.
2. 사용할 학습 모델을 선택합니다.
3. 모델 상세 화면의 **Model URL** 옆 복사 버튼을 누릅니다.

이 프로젝트의 테스트 모델 ID는 다음과 같습니다.

```text
object-detection-workspace-38jwa/woni-face-1-rfdetr-small-t1
```

모델 ID는 모델 자체를 식별합니다. 현재 앱은 모델을 직접 호출하지 않고 배포된 Workflow를 호출하므로, 이 값을 앱의 Workflow ID 입력칸에 넣으면 안 됩니다.

### Workspace ID와 Workflow ID

1. 모델 상세 화면 오른쪽 위의 **Deploy Model**을 누릅니다.
2. Workflow를 만든 경우 왼쪽 메뉴의 **Deployments**를 엽니다.
3. 해당 Workflow 행의 **Deploy Workflow**를 누릅니다.
4. **Cloud API → Images**를 선택합니다.
5. Python 코드의 `workspace_name`과 `workflow_id`를 확인합니다.
6. 또는 **HTTP / cURL**로 전환해 요청 URL과 같은 값을 확인합니다.

예시:

```python
result = client.run_workflow(
    workspace_name="object-detection-workspace-38jwa",
    workflow_id="woni-face-vwoni-face-1-rfdetr-small-t1-logic",
    images={"image": "YOUR_IMAGE.jpg"},
)
```

현재 테스트 값은 다음과 같습니다.

```text
Workspace ID: object-detection-workspace-38jwa
Workflow ID: woni-face-vwoni-face-1-rfdetr-small-t1-logic
```

앱에는 모델 상세 화면의 이름이나 Model URL이 아니라, Deploy Workflow 코드에 표시된 `workspace_name`과 `workflow_id`를 각각 입력합니다. 화면 표시 이름과 실제 Workflow ID가 다를 수 있으므로 직접 타이핑하지 말고 코드에서 복사하는 것이 안전합니다.

## API와 오류 로그

앱은 Roboflow의 Deploy Workflow 화면에 표시되는 `POST /{workspace}/workflows/{workflow}` 경로를 사용합니다. API 키는 `Authorization: Bearer ...` 헤더로 보내고, 영상의 모든 프레임은 `inputs.image`의 base64 이미지 배열로 한 번에 전송합니다. 전송할 프레임은 최대 너비 640px, JPEG 품질 60으로 변환하며 결과 좌표는 원본 영상 크기로 복원합니다.

처리 순서는 `프레임 추출 → API 한 번 호출 → 전체 outputs 수신 → 모든 시각화 프레임 생성 → 원본 FPS로 캐시 무한 반복`입니다. API 응답과 시각화 준비가 모두 끝나기 전에는 추론 영상이 재생되지 않으며, 반복 중에는 API를 다시 호출하지 않습니다.

HTTP 상태와 오류 본문은 `logs/app.log`에 기록됩니다. API 키는 로그에 기록하지 않으며 로그 폴더는 Git에서 제외됩니다. 문제가 생기면 앱을 한 번 실행한 뒤 아래 명령으로 마지막 로그를 확인합니다.

```powershell
Get-Content .\logs\app.log -Tail 50
```

이전 구현은 단일 모델용 `/infer/object_detection` 경로를 사용해 Workflow로 배포된 RF-DETR 모델을 찾지 못하고 404를 반환했습니다. 현재 구현은 Deploy Workflow 화면에서 Roboflow이 제공한 요청 형식으로 바꾸고 HTTP 상태와 응답 본문을 안전하게 남깁니다.

## 로컬 테스트 영상

원본 `data/raw/woni_VINDNiicjb4.mp4`는 로컬에 그대로 보존하며 Git에서는 제외합니다. 원본 앞 10초를 10fps로 변환한 `data/test/woni_10s_10fps.mp4`는 총 100프레임이며, 앱 화면의 예제로 선택할 수 있도록 Git에 포함합니다.

테스트할 때만 아래 값을 직접 입력합니다. 앱 기본값은 비어 있습니다.

```text
Workspace ID: object-detection-workspace-38jwa
Workflow ID: woni-face-vwoni-face-1-rfdetr-small-t1-logic
```

배포 화면에 표시된 **0.1875크레딧/1,000장** 기준으로 100프레임 전체 추론은 약 **0.01875크레딧**입니다. 실제 단가는 모델과 현재 Roboflow 정책에 따라 달라질 수 있으므로 배포 화면의 표시를 우선합니다.

## 마이크로비트 연결

[MakeCode](https://makecode.microbit.org/)에서 새 프로젝트를 만들고 Python을 선택해 `microbit/main.py`를 붙여 넣어 다운로드합니다. MakeCode 장치 연결을 해제한 뒤 앱에 COM 포트를 입력합니다.

각 프레임에서 객체가 하나 이상 탐지되면 앱이 `1`을 보내 웃는 표정을, 탐지되지 않으면 `0`을 보내 기본 표정을 표시합니다. 보드가 연결되지 않아도 화면 속 5×5 LED 표정은 동일하게 갱신됩니다.

## smile 효과음

탐지 상태가 무표정에서 smile로 바뀌면 `data/audio/geoje_yaho.wav`의 “거제 야호~” 효과음을 한 번 재생합니다. smile 상태가 이어지는 동안에는 반복하지 않습니다. 소리가 재생되는 도중 무표정→smile 전이가 다시 발생해도 무시하며, 현재 소리가 끝난 뒤 새롭게 무표정→smile로 바뀔 때만 다시 재생합니다.

효과음은 [미나미 영상](https://www.youtube.com/watch?v=heifaIjlSUc)의 10분 48.5초부터 10분 51초까지를 추출했습니다. Windows 기본 WAV 재생 기능을 사용하므로 별도 오디오 패키지는 필요하지 않습니다.

## 파일

- `app.py`: 영상 프레임 API 추론, 결과 스트리밍, USB 통신
- `microbit/main.py`: USB 명령 수신과 LED 표정 표시
- `data/face_dataset/`: Roboflow 업로드용 이미지와 라벨
- `data/woni_roboflow.zip`: Roboflow 업로드용 데이터셋 ZIP
- `data/raw/`: 로컬 원본 영상(Git 제외)
- `data/test/`: 로컬 테스트 영상(Git 제외)
- `data/audio/geoje_yaho.wav`: smile 전환 시 재생하는 효과음

출처: [원본 영상](https://www.youtube.com/watch?v=VINDNiicjb4), [IU-detection](https://github.com/whyz-dev/IU-detection), [Roboflow Inference](https://inference.roboflow.com/quickstart/roboflow_ecosystem/)
