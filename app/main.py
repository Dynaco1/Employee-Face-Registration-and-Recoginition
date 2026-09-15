import cv2
from insightface.app import FaceAnalysis

HIEGHT = 640
WIDTH = 640

app = FaceAnalysis (
    name="buffalo_l",
    root="employee face detection/app/models",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(HIEGHT, WIDTH)
)



# img = cv2.imread("employee face detection/val/download (1).jpg")

# faces = app.get(img)

# for face in faces:
#     print("Bounding box:", face.bbox)
#     print("Detection score:", face.det_score)

camera = cv2.VideoCapture(0)


while True:
    ret, frame = camera.read()

    if not ret:
        break

    frame = cv2.resize(frame, (HIEGHT, WIDTH))

    faces = app.get(frame)

    for face in faces:
        bbox = face.bbox
        conf = face.det_score

        x1, y1, x2, y2 = bbox.astype(int).tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), thickness=2)

        label = f"Person {conf}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), thickness=2)

    cv2.imshow("Employee view test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()