import cv2
from constants.settings import HEIGHT, WIDTH
from insightface.app import FaceAnalysis
from streamlit_webrtc import VideoProcessorBase

# --------------------------------------------------
# Registration Processor
# --------------------------------------------------

class RegistrationProcessor(VideoProcessorBase):

    def __init__(self, app):

        self.embeddings = []
        self.app: FaceAnalysis = app

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        faces = self.app.get(img)
        
        if len(faces) > 1:
            text = "Only One person Should be available for the registration."
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            (text_width, text_height), _ = cv2.getTextSize(
                text,
                font,
                font_scale,
                thickness
            )
            
            x = (WIDTH - text_width) // 2
            y = (HEIGHT + text_height) // 2
            cv2.putText(
                img,
                "Only One person Should be available for the registration.",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        elif len(faces) == 1:

            face = faces[0]

            embedding = face.embedding

            self.embeddings.append(
                embedding
            )

            # Draw bounding box
            x1, y1, x2, y2 = map(
                int,
                face.bbox
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
                f"Samples: {len(self.embeddings)}",
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