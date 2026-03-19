import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image
import os

# --- 웹 페이지 기본 설정 ---
st.set_page_config(page_title="VisionWell MVP", layout="wide")
st.title("VisionWell MVP - 관상 & 수상 분석기")
st.caption("[주의] 본 소프트웨어는 웰니스 분석용이며 의료 진단용이 아닙니다.")

# --- 1. MediaPipe 및 모델 초기화 ---
try:
    from mediapipe.solutions import drawing_utils as mp_draw
    from mediapipe.solutions import hands as mp_hands
except:
    try:
        import mediapipe.python.solutions.drawing_utils as mp_draw
        import mediapipe.python.solutions.hands as mp_hands
    except:
        mp_draw = None
        mp_hands = None

from mediapipe.tasks import python as mp_python
BaseOptions = mp_python.BaseOptions
FaceLandmarker = mp_python.vision.FaceLandmarker
FaceLandmarkerOptions = mp_python.vision.FaceLandmarkerOptions
HandLandmarker = mp_python.vision.HandLandmarker
HandLandmarkerOptions = mp_python.vision.HandLandmarkerOptions

face_model_path = 'face_landmarker.task'
hand_model_path = 'hand_landmarker.task'

for path in [face_model_path, hand_model_path]:
    if not os.path.exists(path):
        st.error(f"Error: '{path}' 모델 파일을 찾을 수 없습니다.")
        st.stop()

# 폰트 설정 (Streamlit Cloud 우분투 환경을 위해 나눔고딕 우선 고려)
font_path = "arial.ttf"
standard_fonts = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", # Ubuntu (Streamlit Cloud)
    "C:/Windows/Fonts/malgun.ttf", # Windows
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf" # macOS
]
for f in standard_fonts:
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

# 모델 로드 (캐싱을 통해 매번 다시 로드하지 않도록 설정)
@st.cache_resource
def load_models():
    f_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model_path), 
        output_face_blendshapes=True, num_faces=1)
    f_detector = FaceLandmarker.create_from_options(f_options)
    
    h_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path), 
        num_hands=1)
    h_detector = HandLandmarker.create_from_options(h_options)
    
    return f_detector, h_detector

detector, hand_detector = load_models()

# --- 2. 유틸리티 함수 (기존 로직 유지) ---
def draw_text_ko(img, text, position, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype(font_path, font_size)
    except: font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_hand_marks(img, landmarks):
    h, w, _ = img.shape
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20), (0, 17), (5, 9), (9, 13), (13, 17)
    ]
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
    for start_idx, end_idx in connections:
        start_pos = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end_pos = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(img, start_pos, end_pos, (255, 255, 255), 2)

def analyze_face_deepface(bgr_frame):
    try:
        results = DeepFace.analyze(
            bgr_frame, actions=['emotion', 'age'],
            enforce_detection=False, detector_backend='opencv',
            align=False, silent=True
        )
        if results:
            emotion = results[0]['dominant_emotion'].strip().lower()
            age = int(results[0]['age'])
            return emotion, age
    except:
        pass
    return None, None

def get_physiognomy_analysis(landmarks, selected_part):
    if not landmarks or selected_part == "None": return "원하시는 분석 부위를 선택해주세요."
    y_top, y_brow, y_nose, y_chin = landmarks[10].y, (landmarks[105].y + landmarks[334].y)/2, landmarks[1].y, landmarks[152].y
    h_total = y_chin - y_top
    r_early, r_mid, r_late = (y_brow - y_top)/h_total, (y_nose - y_brow)/h_total, (y_chin - y_nose)/h_total
    
    if selected_part == "Early":
        if r_early > 0.35: return "[초년운 - 상정]\n이마가 넓어 초년운이 강하고 부모의 덕이 큽니다.\n일찍이 학문이나 관운에서 두각을 나타낼 수 있습니다."
        else: return "[초년운 - 상정]\n자수성가형으로 스스로 기회를 만들어가는 타입입니다.\n끈기 있게 초반의 어려움을 이겨내는 기운이 있습니다."
    elif selected_part == "Mid":
        if r_mid > 0.35: return "[중년운 - 중정]\n코와 광대의 에너지가 좋아 경제적 성취감이 높습니다.\n인생의 황금기를 주체적으로 이끌어갈 저력이 있습니다."
        else: return "[중년운 - 중정]\n부드럽고 원만한 관계를 통해 안정을 추구하는 타이밍입니다.\n내실을 기하며 실속을 챙기는 중년이 기대됩니다."
    elif selected_part == "Late":
        if r_late > 0.35: return "[말년운 - 하정]\n하관의 기운이 듬직하여 노후가 평안하고 자식복이 있습니다.\n주변에 사람이 모여들고 존경받는 말년입니다."
        else: return "[말년운 - 하정]\n자기만의 전문성이나 취미로 즐거운 말년을 보낼 관상입니다."
    elif selected_part == "Eyes":
        ratio = (landmarks[133].x - landmarks[33].x) / (landmarks[145].y - landmarks[159].y + 0.0001)
        if ratio > 3.0: return "[눈 관상]\n지혜롭고 통찰력이 뛰어난 '봉황안'의 특징이 보입니다.\n학문적 성취나 명예운이 남달리 강합니다."
        else: return "[눈 관상]\n솔직하고 열정적이며 감정 표현이 풍부한 눈매입니다.\n예술적 감각이 좋고 대인관계가 매우 활발합니다."
    elif selected_part == "Nose":
        if (landmarks[294].x - landmarks[64].x) > 0.15: return "[코 관상]\n재물 창고인 콧망울이 넉넉하여 경제력이 탄탄합니다.\n배짱이 두둑하고 사업적 수완이 뛰어난 관상입니다."
        else: return "[코 관상]\n자기 주관이 뚜렷하고 분석력이 좋아 전문직에서 성공할 확률이 높습니다."
    elif selected_part == "Mouth":
        if (landmarks[61].y + landmarks[291].y) / 2 < landmarks[0].y: return "[입 관상]\n입꼬리가 올라가 복이 들어오는 상입니다.\n긍정적인 에너지로 주변을 밝히며 인복이 많습니다."
        else: return "[입 관상]\n의지가 강하고 맡은 바 책임을 다하는 신중한 상입니다."
    return ""

def get_facial_comment(face_landmarks, img_w, img_h):
    import numpy as np
    xs = np.array([lm.x for lm in face_landmarks])
    ys = np.array([lm.y for lm in face_landmarks])
    ratio = ((xs.max() - xs.min()) * img_w) / ((ys.max() - ys.min()) * img_h + 0.0001)
    shape = '넓은 얼굴' if ratio > 1.2 else ('길쭉한 얼굴' if ratio < 0.8 else '균형 잡힌 얼굴')
    smile = (face_landmarks[291].y - face_landmarks[61].y) > 0.02
    brow_up = (face_landmarks[70].y + face_landmarks[300].y) / 2 < 0.3
    mood = '밝고 자신감 있는' if smile and brow_up else ('조용하고 침착한' if not smile and not brow_up else '중립적인')
    tone = '밝은 피부톤' if ys.mean() < 0.5 else '어두운 피부톤'
    return f"{shape}·{mood}·{tone} 얼굴입니다."

def analyze_palmistry(hand_landmarks):
    import numpy as np
    pts = {k: np.array([hand_landmarks[v].x, hand_landmarks[v].y]) for k, v in {'wrist':0, 'thumb_cmc':1, 'index_mcp':5, 'middle_mcp':9, 'ring_mcp':13, 'pinky_mcp':17}.items()}
    lines = []
    def desc(name, l):
        return f"{name} 라인: {'길고 뚜렷함' if l > 0.12 else '중간 길이' if l > 0.07 else '짧고 흐릿함'} ({PALMISTRY_LINE_MEANINGS.get(name, '')})"
    lines.extend([desc('Heart', np.linalg.norm(pts['index_mcp'] - pts['middle_mcp'])), desc('Head', np.linalg.norm(pts['ring_mcp'] - pts['pinky_mcp'])), desc('Life', np.linalg.norm(pts['wrist'] - pts['pinky_mcp']))])
    lines.append(desc('Fate', np.linalg.norm(pts['thumb_cmc'] - pts['wrist'])) if np.linalg.norm(pts['thumb_cmc'] - pts['wrist']) > 0.05 else 'Fate 라인: 미검출')
    return "\n\n".join(lines)

# --- 3. Streamlit 웹 UI 및 메인 로직 ---

# 기존의 버튼/마우스 클릭 대신 Streamlit 라디오 버튼과 셀렉트 박스 사용
col1, col2 = st.columns(2)
with col1:
    current_mode = st.radio("분석 모드를 선택하세요", ["관상학 (Face)", "수상학 (Hand)"])
with col2:
    selected_physiognomy = "None"
    if current_mode == "관상학 (Face)":
        selected_physiognomy = st.selectbox("상세 관상 부위 선택", ["None", "Early (초년운)", "Mid (중년운)", "Late (말년운)", "Eyes (눈)", "Nose (코)", "Mouth (입)"])
        # 영문 키워드만 추출
        selected_physiognomy = selected_physiognomy.split(" ")[0]

st.write("카메라 켜기 버튼을 눌러 사진을 촬영해주세요. (스마트폰의 경우 전면 카메라 권장)")

# 기존 cv2.VideoCapture(0) 대신 웹캠 사진 입력 사용
img_buffer = st.camera_input("촬영")

if img_buffer is not None:
    # 1. 이미지 읽기 및 변환
    bytes_data = img_buffer.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    output_frame = frame.copy()
    result_text = ""

    # 2. 얼굴 모드 분석
    if current_mode == "관상학 (Face)":
        with st.spinner('얼굴과 감정을 분석 중입니다...'):
            res = detector.detect(mp_image)
            emotion, age = analyze_face_deepface(frame)
            
            if res.face_landmarks:
                for fl in res.face_landmarks:
                    for lm in fl: cv2.circle(output_frame, (int(lm.x*w), int(lm.y*h)), 1, (0, 255, 0), -1)
                    facial_comment = get_facial_comment(fl, w, h)
                    result_text = get_physiognomy_analysis(fl, selected_physiognomy)
                    output_frame = draw_text_ko(output_frame, facial_comment, (20, 20), 22, (200, 200, 0))
            else:
                st.warning("얼굴이 명확히 인식되지 않았습니다. 밝은 곳에서 다시 촬영해주세요.")
                
            if emotion and age:
                emo_text = f"감정: {emotion_ko.get(emotion, emotion)} | 추정 나이: {age}세"
                output_frame = draw_text_ko(output_frame, emo_text, (20, 60), 25, (0, 255, 255))

    # 3. 손 모드 분석
    else:
        with st.spinner('손금을 분석 중입니다...'):
            res = hand_detector.detect(mp_image)
            if res.hand_landmarks:
                for hl in res.hand_landmarks:
                    draw_hand_marks(output_frame, hl)
                    result_text = analyze_palmistry(hl)
            else:
                st.warning("손바닥이 명확히 인식되지 않았습니다. 손바닥을 펼쳐 카메라에 잘 보이게 촬영해주세요.")

    # 4. 최종 결과 출력 (cv2.imshow 대체)
    st.image(output_frame, channels="BGR", caption="분석 완료된 이미지", use_column_width=True)
    
    if result_text:
        st.info("💡 **분석 결과**")
        st.write(result_text)
