
import cv2
from pathlib import Path
import time

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    
# 각 파일 path
BASE_DIR=Path(__file__).resolve().parent
protoFile = str(BASE_DIR)+"/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = str(BASE_DIR)+"/models/pose/mpi/pose_iter_160000.caffemodel"
 
# 위의 path에 있는 network 모델 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

capture = cv2.VideoCapture(1) # 카메라 정보 받아옴
fps = 0.0
start_time = time.time()

# 반복문을 통해 카메라에서 프레임을 지속적으로 받아옴
while cv2.waitKey(1) < 0:  # 아무 키나 누르면 끝난다.
    # 웹캠으로부터 영상 가져옴
    hasFrame, frame = capture.read()
    
    # 영상이 커서 느리면 사이즈를 줄이자
    # frame=cv2.resize(frame,dsize=(320,240),interpolation=cv2.INTER_AREA)
    
    # 웹캠으로부터 영상을 가져올 수 없으면 웹캠 중지
    if not hasFrame:
        cv2.waitKey()
        break
    
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0/255, (320, 240), (0, 0, 0), swapRB=False, crop=False)
    
    imgb = cv2.dnn.imagesFromBlob(inpBlob)

    net.setInput(inpBlob)
    output = net.forward()
    
    # 키포인트 검출시 이미지에 그려줌
    points = []
    for i in range(0, 15):
        probMap = output[0, i, :, :]
        
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        x = (frameWidth * point[0]) / output.shape[3]
        y = (frameHeight * point[1]) / output.shape[2]
        
        if prob > 0.1:    
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else:
            points.append(None)
    
    for pair in POSE_PAIRS:
        partA = pair[0]
        partA = BODY_PARTS[partA]
        partB = pair[1]
        partB = BODY_PARTS[partB]
        
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)
    
    current_time = time.time()
    fps = 1.0 / (current_time - start_time)
    start_time = current_time
    
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Output-Keypoints", frame)
 
capture.release()
cv2.destroyAllWindows()
