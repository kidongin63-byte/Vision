import cv2
import mediapipe as mp
import numpy as np
import threading
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image
import os

# 1. MediaPipe 초기화 (부위별 유틸리티 - 명시적 임포트)
from mediapipe.solutions import drawing_utils as mp_draw
from mediapipe.solutions import hands as mp_hands

# 1. MediaPipe Tasks 초기화
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print(f"Error: '{model_path}' not found.")
    exit(1)

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
if not os.path.exists(font_path):
    font_path = "arial.ttf"

emotion_ko = {
    'angry': '분노', 'disgust': '혐오', 'fear': '공포', 
    'happy': '행복', 'sad': '슬픔', 'surprise': '놀람', 
    'neutral': '무표정', 'Analyzing...': '분석 중...'
}

# 안면 인식기 (정밀 분석용)
from mediapipe.tasks import python as mp_python
BaseOptions = mp_python.BaseOptions
FaceLandmarker = mp_python.vision.FaceLandmarker
FaceLandmarkerOptions = mp_python.vision.FaceLandmarkerOptions

base_options = BaseOptions(model_asset_path=model_path)
options = FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, num_faces=1)
detector = FaceLandmarker.create_from_options(options)

# 손 인식기
hand_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

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

def analyze_face(face_frame):
    global current_emotion, current_age, is_analyzing
    try:
        results = DeepFace.analyze(face_frame, actions=['emotion', 'age'], enforce_detection=False, detector_backend='mediapipe', silent=True)
        if results:
            current_emotion = results[0]['dominant_emotion']
            current_age = int(results[0]['age'])
    except:
        pass
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

def get_palmistry_analysis(hand_landmarks):
    if not hand_landmarks: return "손바닥을 카메라에 정면으로 비춰주세요."
    palm_height = abs(hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y)
    finger_height = abs(hand_landmarks.landmark[9].y - hand_landmarks.landmark[12].y)
    ratio = finger_height / (palm_height + 0.0001)
    res = "[수상학 분석 결과]\n"
    if ratio > 1.1:
        res += "손가락이 긴 편입니다. 사색적이고 예술적 기질이 풍부하며,\n세밀하고 꼼꼼한 성격으로 전문 분야에서 성취가 클 상입니다."
    else:
        res += "손바닥이 두툼하고 에너지가 넘치는 상입니다.\n실행력이 뛰어난 현실적인 타입으로 재물운을 잘 잡는 손입니다."
    return res

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
    output_frame = draw_text_ko(output_frame, "수상학" if current_mode == "Face" else "관상학", (35, 25), 20, (255, 255, 255))
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
        frame_count += 1
        if frame_count % 30 == 0 and not is_analyzing:
            is_analyzing = True
            threading.Thread(target=analyze_face, args=(rgb_frame.copy(),), daemon=True).start()
        output_frame = draw_text_ko(output_frame, f'감정: {emotion_ko.get(current_emotion, current_emotion)}', (20, 80), 25, (255, 255, 0))
        output_frame = draw_text_ko(output_frame, f'추정 나이: {current_age}', (20, 115), 25, (0, 255, 255))
    else:
        res = hand_detector.process(rgb_frame)
        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(output_frame, hl, mp_hands.HAND_CONNECTIONS)
                phys_text = get_palmistry_analysis(hl)
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
