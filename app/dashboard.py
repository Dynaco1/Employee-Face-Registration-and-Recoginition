from functools import partial

import numpy as np
import streamlit as st
from constants.settings import HEIGHT, WIDTH
from face_database import FaceDatabase
from insightface.app import FaceAnalysis
from stream import RecognitionProcessor, RegistrationProcessor
from streamlit_webrtc import webrtc_streamer
from collections.abc import Callable

# --------------------------------------------------
# InsightFace
# --------------------------------------------------

@st.cache_resource
def load_face_model():

    app = FaceAnalysis(
        name="buffalo_l",
        root="app/models/",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(WIDTH, HEIGHT)
    )

    return app


app = load_face_model()


# --------------------------------------------------
# FAISS database
# --------------------------------------------------

@st.cache_resource
def load_database():

    return FaceDatabase(
        index_path="app/vector_store/faces.index",
        people_path="app/vector_store/people.json",
        embedding_dim=512
    )


face_db = load_database()

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("Face Recognition System")

tab1, tab2 = st.tabs(
    ["Registration", "Recognition"]
)


# ==================================================
# REGISTRATION
# ==================================================

with tab1:

    st.header("Register Person")

    name = st.text_input(
        "Person name"
    )
    registration_factory: Callable[[], RegistrationProcessor] = partial(RegistrationProcessor, app)

    ctx = webrtc_streamer(
        key="registration",
        video_processor_factory=registration_factory,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
    )

    if st.button("Register"):

        if not name:
            st.error(
                "Please enter a name."
            )

        elif ctx.video_processor is None:
            st.error(
                "Camera is not running."
            )

        else:

            embeddings = (
                ctx.video_processor.embeddings
            )

            if len(embeddings) < 10:

                st.error(
                    "Collect at least 10 face samples."
                )

            else:

                # ----------------------------------
                # Average embeddings
                # ----------------------------------

                final_embedding = np.mean(
                    embeddings,
                    axis=0
                )

                # ----------------------------------
                # Normalize
                # ----------------------------------

                final_embedding /= np.linalg.norm(
                    final_embedding
                )

                # ----------------------------------
                # Add to FAISS
                # ----------------------------------

                person_id = face_db.add_person(
                    name,
                    final_embedding
                )

                st.success(
                    f"{name} registered successfully!"
                )

                st.write(
                    f"Person ID: {person_id}"
                )

                st.write(
                    f"Samples used: {len(embeddings)}"
                )

with tab2:
    recognition_factory: Callable[[], RecognitionProcessor] = partial(
        RecognitionProcessor,
        app,
        face_db,
    )

    st.header("Recognize Person")
    ctx = webrtc_streamer(
        key="recognition",
        video_processor_factory=recognition_factory, # type: ignore[arg-type]

        media_stream_constraints={
            "video": True,
            "audio": False
        },
    )