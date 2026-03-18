import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image
import os

# --- 설정 및 초기화 ---
st.set_page_config(page_title="VisionWell AI", layout="wide")
st.title("🔮 VisionWell: AI 관상 및 수상 분석")

# MediaPipe 모델 경로 설정 (최상위 루트에 파일이 있어야 함)
MODEL_PATH = 'face_landmarker.task'

# 폰트 설정 (리눅스 서버용 기본 폰트 활용)
def get_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size)
    except:
        return ImageFont.load_default()

emotion_ko = {
    'angry': '분노', 'disgust': '혐오', 'fear': '공포', 
    'happy': '행복', 'sad': '슬픔', 'surprise': '놀람', 
    'neutral': '무표정'
}

# --- AI 엔진 클래스 ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        # 1. Hands(solutions) 대신 HandLandmarker(tasks) 사용 준비
        # 단, 실시간 처리를 위해 여기서는 최소한의 설정만 합니다.
        self.model_path = 'hand_landmarker.task' # 이 파일이 저장소에 있어야 합니다.
        
        # 만약 .task 파일이 없다면 우선 빈 값으로 둡니다.
        self.hand_landmarker = None 

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # [임시 조치] solutions 에러를 피하기 위해 
        # 일단은 화면만 출력하고, 에러가 나지 않는지 확인합니다.
        # 나중에 여기에 최신 Tasks API 로직을 추가할 예정입니다.
        
        return img


# --- UI 레이아웃 ---
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("⚙️ 분석 설정")
    mode = st.radio("모드 선택", ["Face (관상)", "Hand (수상)"])
    st.session_state['mode'] = "Face" if "Face" in mode else "Hand"
    
    st.write("---")
    st.info("실시간 분석을 시작하려면 Start 버튼을 누르세요.")
    
with col1:
    # WebRTC 스트리머 실행
    ctx = webrtc_streamer(
    key="visionwell",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True, # 비동기 처리 활성화로 에러 방지
)

st.warning("[주의] 본 결과는 참고용이며 의료적 판단을 대체할 수 없습니다.")
