import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image
import os

# 1. 페이지 설정
st.set_page_config(page_title="VisionWell (Bandi) - AI 관상 & 수상", layout="wide")

# 2. 모델 경로 및 라이브러리 초기화
@st.cache_resource
def load_models():
    # MediaPipe Tasks 초기화
    from mediapipe.tasks import python as mp_python
    BaseOptions = mp_python.BaseOptions
    FaceLandmarker = mp_python.vision.FaceLandmarker
    FaceLandmarkerOptions = mp_python.vision.FaceLandmarkerOptions
    HandLandmarker = mp_python.vision.HandLandmarker
    HandLandmarkerOptions = mp_python.vision.HandLandmarkerOptions

    # 안면 인식기 초기화
    face_model_path = 'face_landmarker.task'
    face_base_options = BaseOptions(model_asset_path=face_model_path)
    face_options = FaceLandmarkerOptions(base_options=face_base_options, output_face_blendshapes=True, num_faces=1)
    face_detector = FaceLandmarker.create_from_options(face_options)
    
    # 손 인식기 초기화
    hand_model_path = 'hand_landmarker.task'
    hand_base_options = BaseOptions(model_asset_path=hand_model_path)
    hand_options = HandLandmarkerOptions(base_options=hand_base_options, num_hands=1)
    hand_detector = HandLandmarker.create_from_options(hand_options)
    
    return face_detector, hand_detector

detector, hand_detector = load_models()

# 한글 폰트 설정 (Windows/Linux 범용)
font_paths = [
    "C:/Windows/Fonts/malgun.ttf", # Windows
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", # Linux
    "arial.ttf" # Fallback
]
font_path = "arial.ttf"
for f in font_paths:
    if os.path.exists(f):
        font_path = f
        break

emotion_ko = {
    'angry': '분노', 'disgust': '혐오', 'fear': '공포', 
    'happy': '행복', 'sad': '슬픔', 'surprise': '놀람', 
    'neutral': '무표정'
}

PALMISTRY_LINE_MEANINGS = {
    'Heart': '감정·연애를 나타내는 라인. 길고 곧으면 감정이 풍부하고 안정적.',
    'Head': '지성·학습·사고 방식을 나타내는 라인. 길고 깊으면 논리적·창의적.',
    'Life': '생명·체력·활동성을 나타내는 라인. 굵고 곡선이면 활력과 건강이 좋음.',
    'Fate': '운명·외부 영향·직업 경로를 나타내는 라인. 존재하면 인생 방향성이 뚜렷함.'
}

# 3. 분석 함수들
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
    return res

def analyze_palmistry_detail(hand_landmarks):
    """Deep palmistry analysis logic from app.py"""
    pts = {
        'wrist': np.array([hand_landmarks[0].x, hand_landmarks[0].y]),
        'thumb_cmc': np.array([hand_landmarks[1].x, hand_landmarks[1].y]),
        'index_mcp': np.array([hand_landmarks[5].x, hand_landmarks[5].y]),
        'middle_mcp': np.array([hand_landmarks[9].x, hand_landmarks[9].y]),
        'ring_mcp': np.array([hand_landmarks[13].x, hand_landmarks[13].y]),
        'pinky_mcp': np.array([hand_landmarks[17].x, hand_landmarks[17].y]),
    }
    heart_len = np.linalg.norm(pts['index_mcp'] - pts['middle_mcp'])
    head_len  = np.linalg.norm(pts['ring_mcp'] - pts['pinky_mcp'])
    life_len  = np.linalg.norm(pts['wrist'] - pts['pinky_mcp'])
    fate_len  = np.linalg.norm(pts['thumb_cmc'] - pts['wrist'])

    def describe(name, length):
        if length > 0.12: return f"✅ {name}: 길고 뚜렷함 ({PALMISTRY_LINE_MEANINGS[name]})"
        elif length > 0.07: return f"〽️ {name}: 중간 길이 ({PALMISTRY_LINE_MEANINGS[name]})"
        else: return f"❓ {name}: 짧거나 흐릿함"

    lines = [describe('Heart', heart_len), describe('Head', head_len), describe('Life', life_len)]
    if fate_len > 0.05: lines.append(describe('Fate', fate_len))
    return "\n\n".join(lines)

# 4. Streamlit UI
st.title("💡 VisionWell AI - 관상 & 수상 분석")
st.sidebar.markdown("### ⚙️ 설정")
mode = st.sidebar.radio("모드 선택", ["관상 (Face)", "수상 (Hand)"])

# 카메라 입력
img_file = st.camera_input("분석할 사진을 찍어주세요!")

if img_file:
    img = Image.open(img_file)
    frame = np.array(img)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("### 📸 분석 영상")
        processed_img_pil = img.copy()
        draw = ImageDraw.Draw(processed_img_pil)
        w, h = img.size
        
        if mode == "관상 (Face)":
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            res = detector.detect(mp_image)
            
            if res.face_landmarks:
                for fl in res.face_landmarks:
                    for lm in fl:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        draw.ellipse([cx-1, cy-1, cx+1, cy+1], fill=(0, 255, 0))
                    
                    try:
                        # DeepFace 분석
                        results = DeepFace.analyze(rgb_frame, actions=['emotion', 'age'], enforce_detection=False, silent=True)
                        if results:
                            emotion = results[0]['dominant_emotion']
                            age = int(results[0]['age'])
                            st.sidebar.success(f"😊 감정: {emotion_ko.get(emotion, emotion)}")
                            st.sidebar.info(f"🎂 추정 나이: {age}세")
                    except:
                        st.sidebar.warning("감정 분석 실패 (얼굴을 더 가까이 해주세요)")

                sub_col1, sub_col2, sub_col3 = st.columns(3)
                with sub_col1:
                    if st.button("🌱 초년운"): st.session_state['phys'] = get_physiognomy_analysis(res.face_landmarks[0], "초년")
                with sub_col2:
                    if st.button("🌳 중년운"): st.session_state['phys'] = get_physiognomy_analysis(res.face_landmarks[0], "중년")
                with sub_col3:
                    if st.button("🏛️ 말년운"): st.session_state['phys'] = get_physiognomy_analysis(res.face_landmarks[0], "말년")
            
            st.image(processed_img_pil, use_container_width=True)
            
        else: # 수상 모드
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            res = hand_detector.detect(mp_image)
            if res.hand_landmarks:
                for hl in res.hand_landmarks:
                    # 랜드마크 그리기
                    for lm in hl:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill=(255, 0, 0))
                    
                    st.session_state['phys'] = analyze_palmistry_detail(hl)
            else:
                st.session_state['phys'] = "손바닥이 잘 보이게 찍어주세요."
            
            st.image(processed_img_pil, caption="수상 분석 중", use_container_width=True)

    with col2:
        st.write("### 📜 분석 결과")
        if 'phys' in st.session_state:
            st.info(st.session_state['phys'])
        else:
            st.write("카메라 촬영 후 버튼을 눌러주세요.")
        
        st.divider()
        st.caption("주의: 본 분석 결과는 오락 및 웰니스 참고용입니다.")

else:
    st.info("카메라를 켜서 사진을 찍으면 분석이 시작됩니다.")

st.sidebar.divider()
st.sidebar.markdown("""
**사용 방법:**
1. 모드를 선택하고 사진 촬영
2. 관상 모드 시 상단 버튼 클릭
3. 수상 모드 시 결과 즉시 확인
""")

