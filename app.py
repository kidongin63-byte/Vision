import cv2
import mediapipe as mp
import numpy as np
import threading
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image
import os

# 1. MediaPipe 초기화 (부위별 유틸리티)
# 현재 환경에서 solutions가 누락된 경우를 대비한 방어적 프로그래밍
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

# 1. MediaPipe Tasks 초기화
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
        print(f"Error: '{path}' not found.")
        exit(1)

# 한글 폰트 설정 (Windows/Linux/Mac 공통 지원을 위한 시도)
standard_fonts = [
    "C:/Windows/Fonts/malgun.ttf", # Windows (맑은 고딕)
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", # Ubuntu (나눔 고딕)
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf", # macOS
    "arial.ttf" # Fallback
]
font_path = "arial.ttf"
for f in standard_fonts:
    if os.path.exists(f):
        font_path = f
        break

emotion_ko = {
    'angry': '분노', 'disgust': '혐오', 'fear': '공포', 
    'happy': '행복', 'sad': '슬픔', 'surprise': '놀람', 
    'neutral': '무표정', 'Analyzing...': '분석 중...'
}

# 손금(수상학) 라인 의미 사전 – 위키피디아·전문 자료 요약
PALMISTRY_LINE_MEANINGS = {
    'Heart': '감정·연애를 나타내는 라인. 길고 곧으면 감정이 풍부하고 안정적.',
    'Head': '지성·학습·사고 방식을 나타내는 라인. 길고 깊으면 논리적·창의적.',
    'Life': '생명·체력·활동성을 나타내는 라인. 굵고 곡선이면 활력과 건강이 좋음.',
    'Fate': '운명·외부 영향·직업 경로를 나타내는 라인. 존재하면 인생 방향성이 뚜렷함.',
    'Mole': '작은 점·특정 사건·특징을 암시하는 보조 라인.'
}


# 안면 인식기 초기화
face_base_options = BaseOptions(model_asset_path=face_model_path)
face_options = FaceLandmarkerOptions(base_options=face_base_options, output_face_blendshapes=True, num_faces=1)
detector = FaceLandmarker.create_from_options(face_options)

# Hand detector is initialized above using MediaPipe HandLandmarker (no legacy mp_hands needed).
hand_base_options = BaseOptions(model_asset_path=hand_model_path)
hand_options = HandLandmarkerOptions(base_options=hand_base_options, num_hands=1)
hand_detector = HandLandmarker.create_from_options(hand_options)

# 2. 전역 변수 및 유틸리티
camera = cv2.VideoCapture(0)
current_emotion = "Analyzing..."
current_age = "Analyzing..."
frame_count = 0
is_analyzing = False
selected_physiognomy = "None"
current_mode = "Face"

def draw_text_ko(img, text, position, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 커스텀 손 관절 그리기 함수 (mp_draw 부재 대비)
def draw_hand_marks(img, landmarks):
    h, w, _ = img.shape
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), # 엄지
        (0, 5), (5, 6), (6, 7), (7, 8), # 검지
        (9, 10), (10, 11), (11, 12),    # 중지
        (13, 14), (14, 15), (15, 16),   # 약지
        (17, 18), (18, 19), (19, 20),   # 소지
        (0, 17), (5, 9), (9, 13), (13, 17) # 손바닥
    ]
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
    for start_idx, end_idx in connections:
        start_lm = landmarks[start_idx]
        end_lm = landmarks[end_idx]
        start_pos = (int(start_lm.x * w), int(start_lm.y * h))
        end_pos = (int(end_lm.x * w), int(end_lm.y * h))
        cv2.line(img, start_pos, end_pos, (255, 255, 255), 2)

def get_palmistry_analysis(hand_landmarks):
    """Simple palmistry analysis to detect "몰수" (absence of clear palm lines).
    Returns a Korean string indicating whether palm lines are detected.
    """
    import numpy as np
    # hand_landmarks: list of 21 Landmark objects with .x, .y (normalized 0~1)
    # Compute average distance between palm center (landmark 0) and fingertip landmarks.
    palm_center = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
    tip_indices = [4, 8, 12, 16, 20]
    distances = []
    for idx in tip_indices:
        tip = np.array([hand_landmarks[idx].x, hand_landmarks[idx].y])
        distances.append(np.linalg.norm(tip - palm_center))
    avg_dist = np.mean(distances)
    # Threshold chosen empirically: if average distance is very small, fingers are close to palm -> likely no clear lines (몰수).
    if avg_dist < 0.07:
        return "몰수 (손바닥에 뚜렷한 선이 없습니다.)"
    else:
        return "손금이 존재합니다."

# Placeholder for normalize_emotion if it's not defined elsewhere
# If this function is meant to be added, please provide its definition.
def normalize_emotion(emotion_str):
    """DeepFace 감정 문자열을 소문자·공백 제거 후 반환합니다."""
    return emotion_str.strip().lower()

def analyze_face(face_frame):
    """Analyze face frame with DeepFace using only OpenCV backend.
    This avoids the broken `mediapipe.solutions` import.
    """
    global current_emotion, current_age, is_analyzing
    try:
        # Convert RGB (MediaPipe) to BGR for DeepFace
        bgr_frame = cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR)
        # Use OpenCV detector and disable alignment to prevent mediapipe usage
        results = DeepFace.analyze(
            bgr_frame,
            actions=['emotion', 'age'],
            enforce_detection=False,
            detector_backend='opencv',
            align=False,
            silent=True,
        )
        print('[DeepFace] 결과:', results)
        if results:
            raw_emotion = results[0]['dominant_emotion']
            current_emotion = normalize_emotion(raw_emotion)
            current_age = int(results[0]['age'])
            print(f'[DeepFace] 업데이트 – 감정: {current_emotion}, 나이: {current_age}')
    except Exception as e:
        print('[DeepFace] 오류:', e)
    finally:
        is_analyzing = False

def get_physiognomy_analysis(landmarks):
    if not landmarks: return ""
    y_top = landmarks[10].y
    y_brow = (landmarks[105].y + landmarks[334].y) / 2
    y_nose = landmarks[1].y
    y_chin = landmarks[152].y
    h_total = y_chin - y_top
    r_early = (y_brow - y_top) / h_total
    r_mid = (y_nose - y_brow) / h_total
    r_late = (y_chin - y_nose) / h_total
    if selected_physiognomy == "Early":
        res = "[초년운 - 상정]\n"
        if r_early > 0.35: res += "이마가 넓어 초년운이 강하고 부모의 덕이 큽니다.\n일찍이 학문이나 관운에서 두각을 나타낼 수 있습니다."
        else: res += "자수성가형으로 스스로 기회를 만들어가는 타입입니다.\n끈기 있게 초반의 어려움을 이겨내는 기운이 있습니다."
    elif selected_physiognomy == "Mid":
        res = "[중년운 - 중정]\n"
        if r_mid > 0.35: res += "코와 광대의 에너지가 좋아 경제적 성취감이 높습니다.\n인생의 황금기를 주체적으로 이끌어갈 저력이 있습니다."
        else: res += "부드럽고 원만한 관계를 통해 안정을 추구하는 타이밍입니다.\n내실을 기하며 실속을 챙기는 중년이 기대됩니다."
    elif selected_physiognomy == "Late":
        res = "[말년운 - 하정]\n"
        if r_late > 0.35: res += "하관의 기운이 듬직하여 노후가 평안하고 자식복이 있습니다.\n주변에 사람이 모여들고 존경받는 말년입니다."
        else: res += "자기만의 전문성이나 취미로 즐거운 말년을 보낼 관상입니다."
    elif selected_physiognomy == "Eyes":
        res = "[눈 관상 - 마음의 창]\n"
        iw = landmarks[133].x - landmarks[33].x
        ih = landmarks[145].y - landmarks[159].y
        ratio = iw / (ih + 0.0001)
        if ratio > 3.0: res += "지혜롭고 통찰력이 뛰어난 '봉황안'의 특징이 보입니다.\n학문적 성취나 명예운이 남달리 강합니다."
        else: res += "솔직하고 열정적이며 감정 표현이 풍부한 눈매입니다.\n예술적 감각이 좋고 대인관계가 매우 활발합니다."
    elif selected_physiognomy == "Nose":
        res = "[코 관상 - 재물과 자아]\n"
        nw = landmarks[294].x - landmarks[64].x
        if nw > 0.15: res += "재물 창고인 콧망울이 넉넉하여 경제력이 탄탄합니다.\n배짱이 두둑하고 사업적 수완이 뛰어난 관상입니다."
        else: res += "자기 주관이 뚜렷하고 분석력이 좋아 전문직에서 성공할 확률이 높습니다."
    elif selected_physiognomy == "Mouth":
        res = "[입 관상 - 복과 그릇]\n"
        up_corner = (landmarks[61].y + landmarks[291].y) / 2 < landmarks[0].y
        if up_corner: res += "입꼬리가 올라가 복이 들어오는 상입니다.\n긍정적인 에너지로 주변을 밝히며 인복이 많습니다."
        else: res += "의지가 강하고 맡은 바 책임을 다하는 신중한 상입니다."
    else:
        res = "상단의 버튼들을 클릭하여 상세 관상을 확인하세요."
    return res
def get_facial_comment(face_landmarks, img_w, img_h):
    """얼굴 비율·표정·피부톤을 기반으로 한 한줄 코멘트 반환."""
    import numpy as np
    xs = np.array([lm.x for lm in face_landmarks])
    ys = np.array([lm.y for lm in face_landmarks])
    width = (xs.max() - xs.min()) * img_w
    height = (ys.max() - ys.min()) * img_h
    ratio = width / height if height != 0 else 1
    if ratio > 1.2:
        shape = '넓은 얼굴'
    elif ratio < 0.8:
        shape = '길쭉한 얼굴'
    else:
        shape = '균형 잡힌 얼굴'
    # 표정 판단 – 눈썹과 입술 위치
    left_eyebrow = face_landmarks[70]
    right_eyebrow = face_landmarks[300]
    mouth_left = face_landmarks[61]
    mouth_right = face_landmarks[291]
    smile = (mouth_right.y - mouth_left.y) > 0.02
    brow_up = (left_eyebrow.y + right_eyebrow.y) / 2 < 0.3
    if smile and brow_up:
        mood = '밝고 자신감 있는'
    elif not smile and not brow_up:
        mood = '조용하고 침착한'
    else:
        mood = '중립적인'
    avg_y = ys.mean()
    tone = '밝은 피부톤' if avg_y < 0.5 else '어두운 피부톤'
    return f"{shape}·{mood}·{tone} 얼굴입니다."
def get_palmistry_analysis(hand_landmarks):
    if not hand_landmarks: return "손바닥을 카메라에 정면으로 비춰주세요."
    # hand_landmarks는 [Landmark, ...] 형태
    palm_height = abs(hand_landmarks[0].y - hand_landmarks[9].y)
    finger_height = abs(hand_landmarks[9].y - hand_landmarks[12].y)
    ratio = finger_height / (palm_height + 0.0001)
    res = "[수상학 분석 결과]\n"
    if ratio > 1.1:
        res += "손가락이 긴 편입니다. 사색적이고 예술적 기질이 풍부하며,\n세밀하고 꼼꼼한 성격으로 전문 분야에서 성취가 클 상입니다."
    else:
        res += "손바닥이 두툼하고 에너지가 넘치는 상입니다.\n실행력이 뛰어난 현실적인 타입으로 재물운을 잘 잡는 손입니다."
    return res
def analyze_palmistry(hand_landmarks):
    """간단한 손금 라인 추정 및 의미 반환.
    - Heart 라인: index MCP(5)와 middle MCP(9) 사이 거리
    - Head 라인: ring MCP(13)와 pinky MCP(17) 사이 거리
    - Life 라인: wrist(0)와 pinky base(17) 사이 거리
    - Fate 라인: thumb CMC(1)와 wrist(0) 사이 거리 (존재 여부 판단)
    """
    import numpy as np
    idx = {
        'wrist': 0,
        'thumb_cmc': 1,
        'index_mcp': 5,
        'middle_mcp': 9,
        'ring_mcp': 13,
        'pinky_mcp': 17,
    }
    pts = {k: np.array([hand_landmarks[v].x, hand_landmarks[v].y]) for k, v in idx.items()}
    heart_len = np.linalg.norm(pts['index_mcp'] - pts['middle_mcp'])
    head_len  = np.linalg.norm(pts['ring_mcp'] - pts['pinky_mcp'])
    life_len  = np.linalg.norm(pts['wrist'] - pts['pinky_mcp'])
    fate_len  = np.linalg.norm(pts['thumb_cmc'] - pts['wrist'])
    def describe(name, length):
        if length > 0.12:
            return f"{name} 라인: 길고 뚜렷함. {PALMISTRY_LINE_MEANINGS.get(name, '')}"
        elif length > 0.07:
            return f"{name} 라인: 중간 길이. {PALMISTRY_LINE_MEANINGS.get(name, '')}"
        else:
            return f"{name} 라인: 짧고 흐릿함. {PALMISTRY_LINE_MEANINGS.get(name, '')}"
    lines = [describe('Heart', heart_len), describe('Head', head_len), describe('Life', life_len)]
    if fate_len > 0.05:
        lines.append(describe('Fate', fate_len))
    else:
        lines.append('Fate 라인: 미검출')
    return " | ".join(lines)
def on_mouse_click(event, x, y, flags, param):
    global selected_physiognomy, current_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        if 20 <= y <= 60 and 20 <= x <= 120:
            current_mode = "Hand" if current_mode == "Face" else "Face"
            selected_physiognomy = "None"
        elif 5 <= y <= 45:
            if 350 <= x <= 430: selected_physiognomy = "Early"
            elif 440 <= x <= 520: selected_physiognomy = "Mid"
            elif 530 <= x <= 610: selected_physiognomy = "Late"
        elif 50 <= y <= 90:
            if 350 <= x <= 430: selected_physiognomy = "Eyes"
            elif 440 <= x <= 520: selected_physiognomy = "Nose"
            elif 530 <= x <= 610: selected_physiognomy = "Mouth"

cv2.namedWindow('VisionWell MVP')
cv2.setMouseCallback('VisionWell MVP', on_mouse_click)

while True:
    ret, frame = camera.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output_frame = frame.copy()
    phys_text = ""
    btn_color = (0, 150, 200) if current_mode == "Face" else (200, 100, 0)
    cv2.rectangle(output_frame, (20, 20), (120, 60), btn_color, -1)
    output_frame = draw_text_ko(output_frame, "관상학" if current_mode == "Face" else "수상학", (35, 25), 20, (255, 255, 255))
    if current_mode == "Face":
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = detector.detect(mp_image)
        colors = {"Early": (0, 180, 0), "Mid": (0, 0, 180), "Late": (180, 0, 0), "Eyes": (150, 0, 150), "Nose": (150, 150, 0), "Mouth": (0, 150, 150)}
        cv2.rectangle(output_frame, (350, 5), (430, 45), colors["Early"], -1)
        cv2.rectangle(output_frame, (440, 5), (520, 45), colors["Mid"], -1)
        cv2.rectangle(output_frame, (530, 5), (610, 45), colors["Late"], -1)
        output_frame = draw_text_ko(output_frame, "초년", (370, 10), 20, (255, 255, 255))
        output_frame = draw_text_ko(output_frame, "중년", (460, 10), 20, (255, 255, 255))
        output_frame = draw_text_ko(output_frame, "말년", (550, 10), 20, (255, 255, 255))
        cv2.rectangle(output_frame, (350, 50), (430, 90), colors["Eyes"], -1)
        cv2.rectangle(output_frame, (440, 50), (520, 90), colors["Nose"], -1)
        cv2.rectangle(output_frame, (530, 50), (610, 90), colors["Mouth"], -1)
        output_frame = draw_text_ko(output_frame, "눈", (380, 55), 20, (255, 255, 255))
        output_frame = draw_text_ko(output_frame, "코", (470, 55), 20, (255, 255, 255))
        output_frame = draw_text_ko(output_frame, "입", (560, 55), 20, (255, 255, 255))
        if res.face_landmarks:
            for fl in res.face_landmarks:
                phys_text = get_physiognomy_analysis(fl)
                for lm in fl: cv2.circle(output_frame, (int(lm.x*w), int(lm.y*h)), 1, (0, 255, 0), -1)
                # 얼굴 전체 코멘트 추가
                output_frame = draw_text_ko(output_frame, get_facial_comment(fl, w, h), (20, 150), 22, (200,200,0))
        frame_count += 1
        if frame_count % 30 == 0 and not is_analyzing:
            is_analyzing = True
            threading.Thread(target=analyze_face, args=(rgb_frame.copy(),), daemon=True).start()
        output_frame = draw_text_ko(output_frame, f'감정: {emotion_ko.get(current_emotion, current_emotion)}', (20, 80), 25, (255, 255, 0))
        output_frame = draw_text_ko(output_frame, f'추정 나이: {current_age}', (20, 115), 25, (0, 255, 255))
    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = hand_detector.detect(mp_image)
        if res.hand_landmarks:
            for hl in res.hand_landmarks:
                draw_hand_marks(output_frame, hl)
                phys_text = analyze_palmistry(hl)
        else: phys_text = "손바닥을 카메라에 비춰주세요."
    cv2.rectangle(output_frame, (10, h-105), (w-10, h-5), (40, 40, 40), -1)
    output_frame = draw_text_ko(output_frame, phys_text, (20, h-95), 18, (240, 240, 240))
    disclaimer = "[주의] 본 소프트웨어는 웰니스 분석용이며 의료 진단용이 아닙니다."
    output_frame = draw_text_ko(output_frame, disclaimer, (w - 430, 5), 13, (50, 50, 255))
    cv2.imshow('VisionWell MVP', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

camera.release()
hand_detector.close()
cv2.destroyAllWindows()
