import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import threading
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image
import os
import time

# 1. 페이지 설정
st.set_page_config(page_title="VisionWell (Bandi) - AI 관상 & 수상", layout="wide")

# 2. 모델 경로 및 라이브러리 초기화
@st.cache_resource
def load_models():
    # 1. MediaPipe Tasks 초기화
    from mediapipe.tasks import python as mp_python
    BaseOptions = mp_python.BaseOptions
    FaceLandmarker = mp_python.vision.FaceLandmarker
    FaceLandmarkerOptions = mp_python.vision.FaceLandmarkerOptions
    
    model_path = 'face_landmarker.task'
    base_options = BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, num_faces=1)
    detector = FaceLandmarker.create_from_options(options)
    
    from mediapipe.solutions import hands as mp_hands
    hand_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    
    return detector, hand_detector, mp_hands

detector, hand_detector, mp_hands = load_models()

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
if not os.path.exists(font_path):
    font_path = "arial.ttf"

emotion_ko = {
    'angry': '분노', 'disgust': '혐오', 'fear': '공포', 
    'happy': '행복', 'sad': '슬픔', 'surprise': '놀람', 
    'neutral': '무표정'
}

# 3. 분석 함수들
def draw_text_ko(img_pil, text, position, font_size, color):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return img_pil

def get_physiognomy_analysis(landmarks, selected_physiognomy):
    if not landmarks: return ""
    y_top = landmarks[10].y
    y_brow = (landmarks[105].y + landmarks[334].y) / 2
    y_nose = landmarks[1].y
    y_chin = landmarks[152].y
    h_total = y_chin - y_top
    r_early = (y_brow - y_top) / h_total
    r_mid = (y_nose - y_brow) / h_total
    r_late = (y_chin - y_nose) / h_total
    
    if selected_physiognomy == "초년":
        res = "[초년운 - 상정]\n"
        if r_early > 0.35: res += "이마가 넓어 초년운이 강하고 부모의 덕이 큽니다."
        else: res += "자수성가형으로 스스로 기회를 만들어가는 타입입니다."
    elif selected_physiognomy == "중년":
        res = "[중년운 - 중정]\n"
        if r_mid > 0.35: res += "코와 광대의 에너지가 좋아 경제적 성취감이 높습니다."
        else: res += "부드럽고 원만한 관계를 통해 안정을 추구하는 타이밍입니다."
    elif selected_physiognomy == "말년":
        res = "[말년운 - 하정]\n"
        if r_late > 0.35: res += "하관의 기운이 듬직하여 노후가 평안하고 자식복이 있습니다."
        else: res += "자기만의 전문성이나 취미로 즐거운 말년을 보낼 관상입니다."
    else:
        res = "상세 분석 버튼을 눌러주세요."
    return res

# 4. Streamlit UI
st.title("💡 VisionWell (Bandi) - AI 관상 & 수상 분석")
st.sidebar.markdown("### ⚙️ 설정")
mode = st.sidebar.radio("모드 선택", ["관상 (Face)", "수상 (Hand)"])

# 카메라 입력
img_file = st.camera_input("분석할 사진을 찍어주세요!")

if img_file:
    # 이미지 로드
    img = Image.open(img_file)
    frame = np.array(img)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("### 📸 분석 영상")
        processed_img_pil = img.copy()
        
        if mode == "관상 (Face)":
            # MediaPipe 분석
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            res = detector.detect(mp_image)
            
            if res.face_landmarks:
                for fl in res.face_landmarks:
                    # 랜드마크 그리기 (PIL 기반)
                    draw = ImageDraw.Draw(processed_img_pil)
                    w, h = img.size
                    for lm in fl:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        draw.ellipse([cx-1, cy-1, cx+1, cy+1], fill=(0, 255, 0))
                    
                    # DeepFace 분석 (메인 스레드에서 즉시 수행)
                    try:
                        results = DeepFace.analyze(rgb_frame, actions=['emotion', 'age'], enforce_detection=False, silent=True)
                        if results:
                            emotion = results[0]['dominant_emotion']
                            age = int(results[0]['age'])
                            st.sidebar.success(f"😊 감정: {emotion_ko.get(emotion, emotion)}")
                            st.sidebar.info(f"🎂 추정 나이: {age}세")
                    except:
                        st.sidebar.warning("감정 분석 실패")

                # 관상 분석 상세 버튼
                sub_col1, sub_col2, sub_col3 = st.columns(3)
                with sub_col1:
                    if st.button("🌱 초년운"): st.session_state['phys'] = get_physiognomy_analysis(res.face_landmarks[0], "초년")
                with sub_col2:
                    if st.button("🌳 중년운"): st.session_state['phys'] = get_physiognomy_analysis(res.face_landmarks[0], "중년")
                with sub_col3:
                    if st.button("🏛️ 말년운"): st.session_state['phys'] = get_physiognomy_analysis(res.face_landmarks[0], "말년")
            
            st.image(processed_img_pil, use_container_width=True)
            
        else: # 수상 모드
            results = hand_detector.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    # 손금 분석 (간단 버전)
                    palm_h = abs(hl.landmark[0].y - hl.landmark[9].y)
                    finger_h = abs(hl.landmark[9].y - hl.landmark[12].y)
                    ratio = finger_h / (palm_h + 0.0001)
                    if ratio > 1.1:
                        st.session_state['phys'] = "[수상학] 손가락이 길어 사색적이고 예술적 기질이 풍부합니다."
                    else:
                        st.session_state['phys'] = "[수상학] 손바닥이 두툼하고 현실적인 실행력이 뛰어납니다."
            else:
                st.session_state['phys'] = "손바닥이 잘 보이게 찍어주세요."
            
            st.image(img, caption="입력 이미지", use_container_width=True)

    with col2:
        st.write("### 📜 분석 결과")
        if 'phys' in st.session_state:
            st.info(st.session_state['phys'])
        else:
            st.write("위 버튼을 눌러 분석을 시작하세요.")
        
        st.divider()
        st.caption("주의: 본 분석 결과는 오락 및 웰니스 참고용입니다.")

else:
    st.info("카메라를 켜서 사진을 찍으면 분석이 시작됩니다.")

# 5. 실행 가이드 (사이드바)
st.sidebar.divider()
st.sidebar.markdown("""
**사용 방법:**
1. 왼쪽 사이드바에서 모드를 선택합니다.
2. 아래 카메라 화면에서 사진을 촬영합니다.
3. 분석 결과를 확인합니다.
""")
