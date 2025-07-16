import cv2
import math
import cvzone
import concurrent.futures
import matplotlib.pyplot as plt
import cvlib as cv
from ultralytics import YOLO

# url = 'http://192.168.120.83/cam-hi.jpg'
url = 0
im = None


def run2():
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    # cam = cv2.VideoCapture(url)
    model = YOLO('../0-YOLO-Weights/best-852.pt')
    model.to(device='mps')

    while True:
        model.to(device='mps')

        cam = cv2.VideoCapture(url)

        classNames = ['carrot', 'Nescafe', 'tomato', 'pringles', 'Pepsi', 'potato', 'Heets', 'Chipsy', 'object',
                      'object', 'Water']
        ret, frame = cam.read()

        results = model(frame, stream=True)

        global detected_classes
        detected_classes = []

        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                class_id = int(class_id)
                detections.append([x1, y1, x2, y2])

                boxes = result.boxes

                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # import Class Name
                    cls = int(box.cls[0])
                    CurrentClass = classNames[cls]

                    if CurrentClass != "carrot" and CurrentClass != "tomato" and CurrentClass != "potato" and conf > 0.79:
                        cvzone.cornerRect(frame, (x1, y1, w, h), l=15)
                        cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1.5,
                                           thickness=1)

                # detected_classes.append(class_id)
                detected_classes.clear()
                detected_classes.append(class_id)

        cv2.imshow('detection', frame)

        # for second of frames and end button
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        ret, frame = cam.read()

    cv2.destroyAllWindows()


def dataa():
    unique_numbers = []
    global detected_classes2
    detected_classes2 = list(set(detected_classes))

    for num in detected_classes2:
        if num not in unique_numbers:
            unique_numbers.append(num)
    # return detected_classes2
    print(detected_classes2)


# run2()
# dataa()

# if __name__ == '__main__':
#     print("started")
#     with concurrent.futures.ProcessPoolExecutor() as executer:
#         f2 = executer.submit(run2)
