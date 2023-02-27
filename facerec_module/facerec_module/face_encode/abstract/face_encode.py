from abc import abstractmethod
from typing import Dict, List

import numpy as np


class IFaceEncoder():
    @abstractmethod
    def encode(self, image, face_locations: List[Dict], swapRB=True):
        """
        Given an image and postions of faces, return the N-dimension face encoding for each face in the image.

        :param image: An image that contains one or more faces, set ``swapRB=False`` for RGB image
        :param face_locations: List of the bounding boxes of each face in the image,
                               each bounding box is a dict with ``left``, ``top``, ``right``, ``bottom``
                               keys containing relative coordinates respect to the image sizes.
        :return: A list of N-dimensional face encodings (one for each face in the image)
        """
        raise NotImplementedError

    @abstractmethod
    def encoding_length(self):
        """
        :return: A length of face enconding vector
        """
        raise NotImplementedError

    @abstractmethod
    def verify(self, faces_from_db: List[np.array], face_to_verify: np.array):
        """
        Given a list of some known face encodings, compare them to an unknown face to verify it.
        The distance tells you how similar the faces are.

        :param faces_from_db: List of known face encodings
        :param face_to_verify: A face encoding to compare against for verification
        :return: A numpy ndarray with distances meaning how similar an unknown face to each known face encoding
        """
        raise NotImplementedError

    @abstractmethod
    def verify_threshold(self):
        """
        :return: A distance threshold for verification
        """
        raise NotImplementedError
