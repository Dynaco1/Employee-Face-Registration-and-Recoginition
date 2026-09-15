import cv2
import numpy as np
from face_database import FaceDatabase
from insightface.app import FaceAnalysis
from streamlit_webrtc import VideoProcessorBase


def normalize_embedding(embedding):

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


# --------------------------------------------------
# Recognition Processor
# --------------------------------------------------

class RecognitionProcessor(VideoProcessorBase):

    def __init__(self, app, face_db):

        self.name = None
        self.similarity = 0.0
        self.app: FaceAnalysis = app
        self.face_db: FaceDatabase = face_db

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        faces = self.app.get(img)

        for face in faces:
            embedding = face.embedding
            name, similarity = self.face_db.search(
                embedding,
                threshold=0.6
            )

            x1, y1, x2, y2 = map(
                int,
                face.bbox
            )

            if name:
                label = (
                    f"{name} "
                    f"{similarity:.2f}"
                )

            else:
                label = (
                    f"Unknown "
                    f"{similarity:.2f}"
                )

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return frame.from_ndarray(
            img,
            format="bgr24"
        )
