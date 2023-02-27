from os import path
import cv2
import numpy as np


here = path.abspath(path.dirname(__file__))

detector_model = path.join(here, "assets/opencv_face_detector_uint8.pb")
detector_config = path.join(here, "assets/opencv_face_detector.pbtxt")


class FaceDetector():
    def __init__(self,
                 face_confidence=0.97,
                 detection_size=(227, 227)):
        self.face_detector = cv2.dnn.readNetFromTensorflow(
            detector_model,
            detector_config)
        self.face_confidence = face_confidence
        self.detection_size = detection_size

    def detect(self, image, swapRB=False):
        """
        Returns a list of bounding boxes of faces in the image

        :param image: An image (as a numpy array), set ``swapRB=True`` for RGB image
        :param swapRB: A flag to change the order of R and B channels in the image.
        :return: A list of the bounding boxes of each face in the image,
                 each bounding box is a dict with ``left``, ``top``, ``right``, ``bottom``
                 keys containing relative coordinates respect to the image sizes.
        """
        face_locations = []
        if image is not None:
            (h, w) = image.shape[:2]

            blob = cv2.dnn.blobFromImage(
                cv2.resize(image[:, :, ::-1] if swapRB else image,
                           self.detection_size),
                scalefactor=1.0,
                size=self.detection_size,
                mean=(104.0, 177.0, 123.0))
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()

            found_face = None
            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > self.face_confidence:
                    # print('confidence:', confidence)

                    # compute the (x, y)-coordinates of the bounding box
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

                    # make bbox shape similar to dlib
                    x, y = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
                    size = 1.2 * (box[2] - box[0])

                    face_locations.append({
                        "left": (x - 0.5 * size) / w,
                        "top": (y - 0.4 * size) / h,
                        "right": (x + 0.5 * size) / w,
                        "bottom": (y + 0.6 * size) / h
                    })

        return face_locations
