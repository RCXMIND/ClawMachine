import cv2
import torch
import time

# YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 웹캠 설정 (HD 해상도로 설정)
cap = cv2.VideoCapture(0)  # 0번 카메라 (OBS 가상 카메라)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 보조 모니터에 전체화면으로 창 설정
window_name = 'YOLOv5 Object Tracking'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name, 1920, 0)  # 1번 보조 모니터 위치에 따라 조정
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 영상을 읽는 데 실패했습니다.")
        break

    # YOLO 감지
    results = model(frame)

    # FPS 계산
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # 추적 및 표시
    count = 0
    for *xyxy, conf, cls in results.xyxy[0]:
        if results.names[int(cls)] != 'person':
            continue

        label = f'{results.names[int(cls)]} {conf:.2f}'
        color = (0, 255, 0)  # 녹색

        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        count += 1

    # 카운트 및 FPS 표시
    cv2.putText(frame, f'Tracked: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'FPS: {fps:.2f}', (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 화면에 출력
    cv2.imshow(window_name, frame)

    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
