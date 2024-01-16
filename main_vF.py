
# 캠/스켈레톤/각도
import cv2

# FPS for checking real-time
import time

# object - skeleton / detection
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import math

# 음성 출력
from gtts import gTTS
from IPython.display import Audio, display

# skeleton 파일 실행
import subprocess

#### ----------------------------------------------------------------------------

# FPS 초기화
prev_time = 0

# pose 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 캡쳐 초기화 for object detection
model = YOLO("yolo8n.pt")
prev_frame_time = 0 # 직전 프레임
image_save_time = 5 # 
image_captured = False

#### ----------------------------------------------------------------------------

# 각도 계산 함수
def calculate_angle_norm(a, b, c):
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

def calculate_angle_spec(a, b, c):
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(90 - radians * 180.0 / np.pi)

    return angle

# 현재 파일 캠 지속 시간 (연속 재생 시 필요 없음, runtime 이후 종료 위해 사용)
runtime = 15
start_time = time.time()

# 다른 파일 실행
def execute_file(file_path, delay):
    time.sleep(delay)
    subprocess.run(['python', file_path])

#### ----------------------------------------------------------------------------

## 웹캠 생성
cap = cv2.VideoCapture(0) # 노트북 웹캠_안쪽 (기본)
# cap = cv2.VideoCapture(1) # 노트북 웹캠_바깥쪽 (기본)
# cap = cv2.VideoCapture(cv2.CAP_DSHOW+2) # 외부 캠

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

#### ----------------------------------------------------------------------------

## skeleton : Mediapipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 레터박스
        cv2.rectangle(image, (0,0), (675, 1000), (0,0,0), -1 ) # -1: 사각형 채우기
        cv2.line(image, (150,0), (150,1000), (255,255,255), 2)

        # Mediapipe에 입력
        results = pose.process(image)

        # 좌표 정보를 화면에 표시
        if results.pose_landmarks is not None:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), # blue
                                      mp_drawing.DrawingSpec(color=(204, 0, 153), thickness=2, circle_radius=2)) # pink

            # 특정 키포인트들의 각도 계산 및 출력
            landmarks = results.pose_landmarks.landmark

            # [오른손잡이] 전경각: 머리 - 
            head = np.array([landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y])
            right_heel = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y])
            left_heel = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y])
            
            # [오른손잡이] 팔꿈치: (우)어깨 - (우)팔꿈치 - (우)손목 각도
            right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
            right_elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
            right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])

            # [오른손잡이] 무릎: (좌)엉덩이 - (좌)무릎 - (좌)발목 각도
            left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
            left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
            left_ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])

            angle_back = calculate_angle_spec(head, right_heel, left_heel)
            angle_elbow = calculate_angle_norm(right_shoulder, right_elbow, right_wrist)
            angle_knee = calculate_angle_norm(left_hip, left_knee, left_ankle)

            # coaching
            coaching_back = ''; coaching_elbow = ''; coaching_knee = ''

            ### back angle (33.9, 48.86): 전경각
            if 33.9 <= round(angle_back, 1) <= 48.86:
                coaching_back = 'BACK  >> Nice!'
            elif round(angle_back, 1) < 33.9:
                coaching_back = 'BACK  >> Fold a little bit'
            elif round(angle_back, 1) > 48.86:
                coaching_back = 'BACK  >> Stretch more'

            ### elbow angle (158.26, 164.4)
            if 158.26 <= round(angle_elbow, 1) <= 164.4:
                coaching_elbow = 'ELBOW >> Nice!'
            elif round(angle_elbow, 1) < 158.26:
                coaching_elbow = 'ELBOW >> Stretch more'
            elif round(angle_elbow, 1) > 164.4:
                coaching_elbow = 'ELBOW >> Fold a little bit'

            ### knee angle (109.61, 116.45)
            if 109.61 <= round(angle_knee, 1) <= 116.45:
                coaching_knee = 'KNEE  >> Nice!'
            elif round(angle_knee, 1) < 109.61:
                coaching_knee = 'KNEE  >> Stretch more'
            elif round(angle_knee, 1) > 116.45:
                coaching_knee = 'KNEE  >> Fold a little bit'

            similarity_back = round((angle_back / 41.38) * 100) # mean 41.38
            similarity_elbow = round((angle_elbow / 161.33) * 100) # mean 161.33
            similarity_knee = round((angle_knee / 113.03) * 100) # mean 113.03
                    
            if similarity_back > 100:
                similarity_back = 100
                
            if similarity_elbow > 100:
                similarity_elbow = 100
                
            if similarity_knee > 100:
                similarity_knee = 100
                

            ## FPS
            # 현재 시간 구하기
            current_time = time.time()
            # 경과 시간 계산
            fps = 1.0 / (current_time - prev_time)
            # 경과 시간 업데이트
            prev_time = current_time
            
            # 코칭/각도 화면 문구 출력
            ## FPS
            cv2.putText(image, f'FPS', (45, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)    
            cv2.putText(image, f'{fps:.2f}', (35, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA) 

            ## Coaching
            cv2.putText(image, coaching_back, (190, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, coaching_elbow, (190, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, coaching_knee, (190, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            ## Similarity
            cv2.putText(image, f'(Similarity {similarity_back} %)', (360, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'(Similarity {similarity_elbow} %)', (360, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'(Similarity {similarity_knee} %)', (360, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

### ----------------------------------------------------------------------------

            # ball 인식 시점 캡쳐

            balls = model.predict(frame)

            for i in balls:
                uniq, cnt = np.unique(i.boxes.cls.cpu().numpy(), return_counts=True)
                uniq_cnt_dict = dict(zip(uniq, cnt))

                for c in i.boxes.cls:
                    class_num = int(c)
                    class_name = model.names[class_num]

                    if class_name in ["bowling-ball"]:
                        # 현재 시간과 이전 프레임 시간 비교
                        current_time = time.time()
                        elapsed_time = current_time - prev_frame_time

                        if elapsed_time >= image_save_time and not image_captured:
                            # 5초 이상 경과한 경우 이미지 캡처
                            cv2.imwrite("captured_frame.jpg", frame)
                            print("the release time pic")
                            image_captured = True

                        # 이전 프레임 시간 갱신
                        prev_frame_time = current_time

                    print('class num =', class_num, 'class name =', class_name)
                    
#### ----------------------------------------------------------------------------

        # 화면에 출력
        cv2.imshow('Bowling Coaching', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


        # 경과 시간이 runtime을 초과하면 종료
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time >= runtime:
            break

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
           
    # 객체 해제
    cap.release()
    cv2.destroyAllWindows()

#### ----------------------------------------------------------------------------

# skeleton 파일 실행

    file_path = 'output_vF.py'
    delay = 5

    execute_file(file_path, delay)

