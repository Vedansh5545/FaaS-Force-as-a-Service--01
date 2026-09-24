import cv2 as cv
import numpy as np

"""
idea: Cut the frame into 12 squares like a chocolate bar. In each square, count how much red, green, blue, and so on. Write those count on a row and that row is the fingerprint of the frame

"""


class ColorFingerprint:
    def __init__(self, image1, image2):
        frame1 = cv.imread(image1)
        frame2 = cv.imread(image2)
        if not frame1 or not frame2:
            raise FileNotFoundError(
                f"could not read image: {image1 if not frame1 else image2}"
            )
        self.frame1 = self.convertHsv(self.resize(frame1))
        self.frame2 = self.convertHsv(self.resize(frame2))
        self.fingerprint = []

    def resize(self, frame):
        # resize image to 160 x 90; INTER_AREA averages pixels, best for shrinking
        return cv.resize(frame, (160, 90), interpolation=cv.INTER_AREA)

    def convertHsv(self, frame):
        return cv.cvtColor(frame, cv.COLOR_BGR2HSV)
