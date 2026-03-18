import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
# 경로가 꼬였을 때를 대비한 3단계 방어적 임포트
try:
    from mediapipe.solutions import hands as mp_hands
    from mediapipe.solutions import drawing_utils as mp_draw
except ImportError:
        import mediapipe.python.solutions.hands as mp_hands
        import mediapipe.python.solutions.drawing_utils as mp_draw
import numpy as np
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
        self.mp_hands_obj = mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.5
        )
        self.mp_draw_obj = mp_draw

    def draw_text(self, img, text, pos, size, color):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=get_font(size), fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # [참고] 실시간 DeepFace 분석은 서버 사양에 따라 매우 느릴 수 있으므로 
        # 여기서는 MediaPipe 기반 관상/수상 가이드라인 위주로 먼저 구현합니다.
        
        if st.session_state.get('mode') == "Face":
            # 관상 분석 로직 (MediaPipe)
            img = self.draw_text(img, "관상학 모드 활성화 중", (20, 20), 20, (0, 255, 0))
            # 여기에 기존의 get_physiognomy_analysis 로직 이식 가능
        else:
            # 수상 분석 로직 (MediaPipe)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.mp_hands.process(rgb)
            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, hl, mp.solutions.hands.HAND_CONNECTIONS)
                    img = self.draw_text(img, "손 인식됨: 분석 중...", (20, h-50), 20, (255, 255, 255))

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
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

st.warning("[주의] 본 결과는 참고용이며 의료적 판단을 대체할 수 없습니다.")
