import cvzone
import cv2
from cvzone.PoseModule import PoseDetector
from ultralytics import YOLO
import math
from twilio.rest import Client
import time

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

cap = cv2.VideoCapture('CCTV footage.mp4')
detector = PoseDetector(staticMode=False, detectionCon=0.6)

model = YOLO('yolov8n.pt')

time_interval = 120  # 2 minutes
last_time = int(time.time()) - time_interval  # Ensure the first message is sent immediately

while True:

    success, img = cap.read()

    results = model(img, stream=True)

    # Find the human pose in the frame
    img = detector.findPose(img)

    personCount = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            bbox = int(x1), int(y1), int(w), int(h)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == 'person' and conf > 0.4:
                personCount += 1
                cvzone.cornerRect(img, (x1, y1, w, h), rt=2)
                cvzone.putTextRect(img, f'{currentClass}', (max(0, x1), max(35, y1 - 20)), offset=3,scale=2)
            
            if currentClass == 'car' or currentClass=='motorbike':
                cvzone.cornerRect(img, (x1, y1, w, h), rt=2)
                cvzone.putTextRect(img, f'{currentClass}', (max(0, x1), max(35, y1 - 20)), offset=3,scale=2)

    current_time = time.time()
    if personCount > 0 and (current_time - last_time) >= time_interval:
        if personCount > 1:
            x = 'are'
            y = 'people'
        else:
            x = 'is'
            y = 'person'
                
        message_body = f'There {x} {personCount} {y} at your doorstep'
        print(message_body)

        # Your Account SID from twilio.com/console
        account_sid = "XXXXXXXXXXXXXXX"
        # Your Auth Token from twilio.com/console
        auth_token = "XXXXXXXXXXXXXX"
        client = Client(account_sid, auth_token)
        message = client.messages.create(to="+XXXXXXXXXX", from_="+XXXXXXXXX", body=message_body)

        last_time = current_time

    cv2.imshow("Img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
