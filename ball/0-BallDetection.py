from ultralytics import YOLO
import cv2
import cvzone
import math

# cam = cv2.VideoCapture("../1-videos-test/Real Madrid 3-2 AC Milan TEST VIDEO.mp4")
cam = cv2.VideoCapture("../1-videos-test/Real Madrid vs Bayern Munich.mp4")

model = YOLO('../0-YOLO-Weights/Ball-4.pt')

model.to(device='mps')

classNames = ['Ball']

while True:
    success, img = cam.read()
    results = model(img)
    # for frames
    for r in results:
        boxes = r.boxes
        # for informations
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=5)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])                                             # x1              y1
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



