from typing import Dict, List

import numpy as np
import face_recognition

from .abstract.face_encode import IFaceEncoder


class FaceEncoderDlib(IFaceEncoder):
    def __init__(self):
        super().__init__()

        self.face_encodings = face_recognition.face_encodings
        self.face_distance = face_recognition.face_distance

    def encode(self, image, face_locations: List[Dict], swapRB=True):
        (h, w) = image.shape[:2]
        bboxes = [(loc["top"], loc["right"], loc["bottom"], loc["left"]) for loc in face_locations]
        bboxes = [(box * np.array([h, w, h, w])).astype("int") for box in bboxes]
        return self.face_encodings(image[:, :, ::-1] if swapRB else image, bboxes)

    def encoding_length(self):
        return 128

    def verify(self, faces_from_db: List[np.array], face_to_verify: np.array):
        return self.face_distance(faces_from_db, face_to_verify)

    def verify_threshold(self):
        return 0.6
