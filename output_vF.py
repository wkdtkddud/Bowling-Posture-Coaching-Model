import cv2
import math
import numpy as np
import mediapipe as mp

# 음성 출력
import os
from gtts import gTTS
from IPython.display import Audio, display
from playsound import playsound

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

def draw_skeleton(image_path, output_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 레터박스
    cv2.rectangle(img, (0,0), (675, 1000), (0,0,0), -1 ) # -1: 사각형 채우기
    cv2.line(img, (150,0), (150,1000), (255,255,255), 2)

    # 코칭/각도 화면 문구 출력
    cv2.putText(img, "Let's", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)    
    cv2.putText(img, 'Bowling', (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # pose 초기화
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Mediapipe에 입력
    pose = mp_pose.Pose()
    bone = pose.process(img)

    if bone.pose_landmarks is not None:
        mp_drawing.draw_landmarks(img, bone.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), # blue
                                    mp_drawing.DrawingSpec(color=(204, 0, 153), thickness=2, circle_radius=2)) # pink


        # 특정 키포인트들의 각도 계산 및 출력
        landmarks = bone.pose_landmarks.landmark

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
        coaching_back = ''
        coaching_elbow = ''
        coaching_knee = ''

        ### back angle (33.9, 48.86): 전경각
        if 33.9 <= round(angle_back, 1) <= 48.86:
            coaching_back = 'BACK  >> Nice!'
            coaching_back_tts = 'Your back posture is nice!'
            coaching_back_tts_ko = '상체는 그대로도 좋습니다!'
        elif round(angle_back, 1) < 33.9:
            coaching_back = 'BACK  >> Fold a little bit'
            coaching_back_tts = 'Fold your back posture a litte bit'
            coaching_back_tts_ko = '상체를 조금만 더 숙여보세요!'
        elif round(angle_back, 1) > 48.86:
            coaching_back = 'BACK  >> Stretch more'
            coaching_back_tts = 'Stretch your back posture more'
            coaching_back_tts_ko = '상체를 조금만 더 들어보세요!'

        ### elbow angle (158.26, 164.4)
        if 158.26 <= round(angle_elbow, 1) <= 164.4:
            coaching_elbow = 'ELBOW >> Nice!'
            coaching_elbow_tts = 'Your arm posture is nice!'
            coaching_elbow_tts_ko = '투구하는 팔 자세가 좋습니다!'
        elif round(angle_elbow, 1) < 158.26:
            coaching_elbow = 'ELBOW >> Stretch more'
            coaching_elbow_tts = 'Stretch your forearm more'
            coaching_elbow_tts_ko = '투구하는 팔을 조금 덜 접어보세요'
        elif round(angle_elbow, 1) > 164.4:
            coaching_elbow = 'ELBOW >> Fold a little bit'
            coaching_elbow_tts = 'Fold your forearm a litte bit'
            coaching_elbow_tts_ko = '투구하는 팔을 조금 덜 접어보세요!'

        ### knee angle (109.61, 116.45)
        if 109.61 <= round(angle_knee, 1) <= 116.45:
            coaching_knee = 'KNEE  >> Nice!'
            coaching_knee_tts = 'Your leg posture is nice!'
            coaching_knee_tts_ko = '다리 자세가 좋습니다!'
        elif round(angle_knee, 1) < 109.61:
            coaching_knee = 'KNEE  >> Stretch more'
            coaching_knee_tts = 'Stretch your leg more'
            coaching_knee_tts_ko = '왼쪽 다리를 조금 덜 굽혀보세요'
        elif round(angle_knee, 1) > 116.45:
            coaching_knee = 'KNEE  >> Fold a little bit'
            coaching_knee_tts = 'Fold your knee a litte bit'
            coaching_knee_tts_ko = '왼쪽 다리를 조금 더 굽혀보세요'

        similarity_back = round((angle_back / 41.38) * 100) # mean 41.38
        similarity_elbow = round((angle_elbow / 161.33) * 100) # mean 161.33
        similarity_knee = round((angle_knee / 113.03) * 100) # mean 113.03
                
        if similarity_back > 100:
            similarity_back = 100
            
        if similarity_elbow > 100:
            similarity_elbow = 100
            
        if similarity_knee > 100:
            similarity_knee = 100


        ## Coaching
        cv2.putText(img, coaching_back, (190, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, coaching_elbow, (190, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, coaching_knee, (190, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        ## Angle
        cv2.putText(img, f'(Similarity {similarity_back} %)', (360, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f'(Similarity {similarity_elbow} %)', (360, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f'(Similarity {similarity_knee} %)', (360, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path, img)

        coaching_texts = [coaching_back_tts + 'and' + coaching_elbow_tts + 'and' + coaching_knee_tts]
        file_names = ['coaching.mp3']

        for text, file_name in zip(coaching_texts, file_names):
            tts = gTTS(text, lang='en')
            tts.save(file_name)
            audio = Audio(file_name, autoplay=True)
            display(audio)
            # playsound('coaching.mp3')

        coaching_texts_ko = [coaching_back_tts_ko + '그리고' + coaching_elbow_tts_ko + '그리고' + coaching_knee_tts_ko]
        file_names_ko = ['coaching_ko.mp3']

        for text_ko, file_name_ko in zip(coaching_texts_ko, file_names_ko):
            tts_ko = gTTS(text_ko, lang='ko')
            tts_ko.save(file_name_ko)
            audio_ko = Audio(file_name_ko, autoplay = True)
            display(audio_ko)
            playsound('coaching_ko.mp3')


# 스켈레톤 그리기
draw_skeleton('captured_frame.jpg', 'captured_frame_skeleton.jpg')

        