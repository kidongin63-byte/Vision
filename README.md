# VisionWell AI - 관상 & 수상 분석기

AI를 활용한 실시간 안면(관상) 및 손금(수상) 분석 애플리케이션입니다.
본 프로젝트는 데스크톱용 OpenCV 버전과 웹 배포용 Streamlit 버전을 모두 지원합니다.

## 🚀 주요 기능

- **관상학 분석 (Physiognomy)**: 안면 랜드마크 분석을 통한 초년, 중년, 말년운 분석.
- **수상학 분석 (Palmistry)**: 주요 손금 라인(감정선, 두뇌선, 생명선, 운명선)의 형태 및 의미 분석.
- **DeepFace 감정/나이 분석**: 실시간 표정을 분석하여 감정 상태와 추정 나이 제공.
- **다국어 지원**: 분석 결과를 한글로 표시하며 직관적인 UI 제공.

## 🛠️ 설치 및 설정

### 1. 전제 조건
- Python 3.9 이상 권장
- 웹캠 (실시간 분석용)

### 2. 가상환경 구축 및 의존성 설치
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # (Windows: .\venv\Scripts\activate)

# 의존성 설치
pip install -r requirements.txt
```

### 3. 필수 모델 파일
다음 파일들이 프로젝트 루트에 위치해야 합니다:
- `face_landmarker.task`
- `hand_landmarker.task`

## 🏃 실행 방법

### A. 데스크톱 앱 (OpenCV 기반)
로컬 환경에서 실시간 웹캠 창을 띄워 사용합니다.
```bash
python runner.py
```
*(또는 `python app.py` 직접 실행)*

### B. 웹 대시보드 (Streamlit 기반)
배포 및 웹 브라우저 기반 사용 시 권장됩니다.
```bash
streamlit run streamlit_app.py
```

## 🌐 배포 (Deployment)

### Streamlit Cloud 배포 가이드
1. GitHub 저장소에 코드를 업로드합니다.
2. `packages.txt`에 포함된 시스템 라이브러리(`libgl1` 등)가 있는지 확인합니다.
3. Streamlit Cloud 워크스페이스에서 `streamlit_app.py`를 메인 파일로 지정하여 배포합니다.
4. 배포 시 `face_landmarker.task`, `hand_landmarker.task` 모델 파일이 포함되어 있는지 확인하십시오.

## ⚠️ 주의사항
- 본 소프트웨어는 엔터테인먼트 및 웰니스 참고용이며, 의학적 진단이나 법적 효력을 갖지 않습니다.
- DeepFace 분석 시 초기 모델 로딩에 수 초가 소요될 수 있습니다.

---
© 2026 VisionWell Project. All rights reserved.
