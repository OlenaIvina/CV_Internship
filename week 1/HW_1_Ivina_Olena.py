import cv2
import numpy as np

'''
To make a function that creates a binary mask with foreground objects like triangles, circles, rectangles.
The mask should be as clean as possible (minimum of noisy pixels and with edges as smooth as possible).
Input: video. Output: video (actual mask).

You will probably need:
color space transform "cv2.cvtColor",
threshold "cv2.threshold",
morphology:
Histogram equalization
Adaptive threshold
Blur
Median
cv2.filter2D
'''


def run_video_manipulation_samples():
    video_name = 'input_video.avi'
    cap = cv2.VideoCapture(video_name)
    ret,frm = cap.read()
    frm_count = 0
    key = None
    kernel = np.ones((5, 5), np.uint8)

    # Setting video format. Google for "fourcc"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # Setting up new video writer
    frames_per_second = 30
    image_size = (frm.shape[1], frm.shape[0])
    writer = cv2.VideoWriter('sample.avi', fourcc, frames_per_second, image_size, 0)

    while ret:

        # First mask

        lab_image = cv2.cvtColor(frm, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        l_channel = cv2.Canny(l_channel, 33, 150)
        l_channel = cv2.adaptiveThreshold(l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh, l_channel = cv2.threshold(l_channel, 140, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        l_channel = cv2.equalizeHist(l_channel)
        l_channel = cv2.GaussianBlur(l_channel, (5, 5), 0)
        l_channel = cv2.morphologyEx(l_channel, cv2.MORPH_TOPHAT, kernel=kernel)
        l_channel = cv2.erode(l_channel, kernel=np.ones((1, 1), dtype=np.uint8))
        a_channel = cv2.threshold(a_channel, 140, 255, cv2.THRESH_BINARY)[1] # круто выделяет 2 объекта
        b_channel = cv2.threshold(b_channel, 140, 255, cv2.THRESH_BINARY)[1]  # круто выделяет 2 объекта
        lab_image = cv2.merge([l_channel, a_channel, b_channel])
        lab_brg = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
        brg_gray = cv2.cvtColor(lab_brg, cv2.COLOR_BGR2GRAY)
        not_brg_gray = cv2.bitwise_not(brg_gray)
        thresh, not_brg_gray = cv2.threshold(not_brg_gray, 100, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY) #  140, 255

        # Second mask

        brg_gray_2 = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        ret, brg_gray_2 = cv2.threshold(brg_gray_2, 127, 255, cv2.THRESH_BINARY)
        brg_gray_2 = cv2.bitwise_not(brg_gray_2)

        # Mask combination

        frm = not_brg_gray | brg_gray_2

        writer.write(frm)
        cv2.imshow('Video frame', frm)

        # cv2.waitKey(10)

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
