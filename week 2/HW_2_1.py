import cv2
import numpy as np
import argparse
import imutils
from scipy.interpolate import splprep, splev

'''
Please make a python script which will do the following steps for each frame of the video from the home task 1 and save 
the result as a new video file.

- put text description e.g. type of a figure (circle, triangle or rectangle) near (or over) each found figure
- visualize contours of found figures by a type specific color (R,G,B, for example)
- make sure your script does correct handling of figures cropped by a frame. (if there is class uncertainty then ignore)

You definitely will need: 
findContours
'''


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return the name of the shape
        return shape


def run_video_manipulation_samples():
    video_name = 'input_video.avi'
    cap = cv2.VideoCapture(video_name)
    ret,frm = cap.read()
    frm_count = 0
    key = None

    # Setting video format. Google for "fourcc"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # Setting up new video writer
    frames_per_second = 30
    image_size = (frm.shape[1], frm.shape[0])
    writer = cv2.VideoWriter('sample.avi', fourcc, frames_per_second, image_size)

    while ret:

        # convert to hsv
        frm_hsv = cv2.cvtColor(frm, cv2.COLOR_BGR2HSV)
        frm_split_ch = np.zeros((frm_hsv.shape[0], frm_hsv.shape[1] * frm_hsv.shape[2]), dtype=np.uint8)
        # filling in the empty image by channels
        for i in range(frm_hsv.shape[2]):
            frm_split_ch[:, frm_hsv.shape[1] * i:frm_hsv.shape[1] * (i + 1)] = frm_hsv[:, :, i]

        # mask for green and yellow
        mask_gy = cv2.inRange(frm_hsv, np.array([20, 40, 100]), np.array([80, 100, 255]))

        # mask for pink
        mask_p = cv2.inRange(frm_hsv, np.array([140, 50, 0]), np.array([179, 150, 255]))

        # mask for black
        mask_b = cv2.inRange(frm_hsv, np.array([0, 0, 0]), np.array([179, 255, 40]))

        threshed = mask_gy + mask_p + mask_b
        threshed = cv2.dilate(threshed, np.ones((3, 3)))  # удаляем все мелкое
        # threshed = cv2.erode(threshed, np.ones((11, 11)))  # делаем жирным все, что выжило

        threshed = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)

        edged = cv2.Canny(threshed[:, :, 0], 33, 150)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        cnts = imutils.grab_contours(cnts)

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:8]

        sd = ShapeDetector()

        # loop over the contours
        for c in cnts:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)

            if M["m00"] != 0:
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))
            else:
                cX = 0
                cY = 0

            shape = sd.detect(c)

            c = c.astype("int")

            if shape == 'triangle':
                cv2.drawContours(threshed, [c], -1, (0, 255, 0), 2)
                cv2.putText(threshed, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)

            if shape == 'square':
                cv2.drawContours(threshed, [c], -1, (255, 0, 0), 2)
                cv2.putText(threshed, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 3)

            if shape == 'rectangle':
                cv2.drawContours(threshed, [c], -1, (0, 0, 255), 2)
                cv2.putText(threshed, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)

            if shape == 'circle':
                cv2.drawContours(threshed, [c], -1, (255, 0, 127), 2)
                cv2.putText(threshed, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 127), 3)

        writer.write(threshed)
        cv2.imshow('Video frame', threshed)

        # Pause on pressing of space.
        if key == ord(' '):
            wait_period = 0
        else:
            wait_period = 30

        # drawing, waiting, getting key, reading another frame

        key = cv2.waitKey(wait_period)
        ret, frm = cap.read()
        frm_count += 1
    cap.release()
    writer.release()

    return 0


if __name__ == '__main__':
    run_video_manipulation_samples()



