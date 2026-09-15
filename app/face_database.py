import json
import os

import faiss
import numpy as np


class FaceDatabase:

    def __init__(
        self,
        index_path="./vector_store/faces.index",
        people_path="./vector_store/people.json",
        embedding_dim=512
    ):
        self.index_path = index_path
        self.people_path = people_path
        self.embedding_dim = embedding_dim

        # Cosine similarity using normalized vectors
        self.index = faiss.IndexFlatIP(embedding_dim)

        self.people = {}

        self.load()

    def add_person(self, name, embedding):
        """
        Add one person's final embedding to FAISS.
        """

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).reshape(1, -1)

        # Current FAISS index position
        person_id = self.index.ntotal

        # Add embedding
        self.index.add(embedding)

        # Store metadata
        self.people[person_id] = name

        self.save()

        return person_id

    def search(self, embedding, threshold=0.5):
        """
        Search for the closest registered person.
        """

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).reshape(1, -1)

        # Normalize query embedding
        embedding /= np.linalg.norm(
            embedding,
            axis=1,
            keepdims=True
        )

        if self.index.ntotal == 0:
            return None, 0.0

        scores, indices = self.index.search(
            embedding,
            k=1
        )

        similarity = float(scores[0][0])
        person_id = int(indices[0][0])

        if similarity < threshold:
            return None, similarity

        name = self.people.get(person_id)

        return name, similarity

    def save(self):
        faiss.write_index(
            self.index,
            self.index_path
        )

        with open(self.people_path, "w") as f:
            json.dump(self.people, f)

    def load(self):

        if os.path.exists(self.index_path):

            self.index = faiss.read_index(
                self.index_path
            )

        if os.path.exists(self.people_path):

            with open(self.people_path, "r") as f:
                data = json.load(f)

            # JSON converts integer keys to strings
            self.people = {
                int(k): v
                for k, v in data.items()
            }