import numpy as np
import cv2.cv2

"""
The task is to track the object (marker.jpg) on the video (find_chocolate.mp4) using two approaches:
1) tracking by detection using orb features. You need to find homography and use it to draw a plane rectangle on
the top of the marker so that it follows the orientation of the marker.

2) tracking using optical flow (Lucas-Kanade). In this case you initialize the tracker using the ORB
features like in the above solution, but after that you update the positioning between the frames using optical flow.
The output is to be drawn in the same way as in the approach 1. But it will have different behavior.

As an output you should provide two videos obtained using those two approaches.
"""


def tracking_orb():

    cap = cv2.VideoCapture('find_chocolate.mp4')

    ret, frm = cap.read()
    img_chocolate = cv2.imread('marker.jpg')

    frm_count = 0
    key = None

    # Setting video format. Google for "fourcc"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Setting up new video writer
    image_size = (frm.shape[1], frm.shape[0])
    # writer = cv2.VideoWriter('sample_tracking_orb.avi', fourcc, frames_per_second, image_size)
    out = cv2.VideoWriter('sample_tracking_orb.mp4', fourcc, 30.0, image_size)

    while ret:

        ## Create ORB object and BF object(using HAMMING)
        orb = cv2.ORB_create()

        gray2 = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img_chocolate, cv2.COLOR_BGR2GRAY)

        # gray2 = cv2.equalizeHist(gray2)
        # gray1 = cv2.equalizeHist(gray1)

        ## Find the keypoints and descriptors with ORB
        kpts1, descs1 = orb.detectAndCompute(gray1, None)
        kpts2, descs2 = orb.detectAndCompute(gray2, None)

        # create BFMatcher object
        ## match descriptors and sort them in the order of their distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

        # Match descriptors.
        matches = bf.match(descs1, descs2)

        # Sort them in the order of their distance.
        dmatches = sorted(matches, key=lambda x: x.distance)

        ## extract the matched keypoints
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1, 1, 2)

        ## find homography matrix and do perspective transform
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = img_chocolate.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        ## draw found regions
        frm = cv2.polylines(frm, [np.int32(dst)], True, (0, 0, 255), 1, cv2.LINE_AA)

        ## draw match lines
        res = cv2.drawMatches(img_chocolate, kpts1, frm, kpts2, dmatches[:8], None, flags=2)

        # writer.write(res)
        cv2.namedWindow('orb_match', cv2.WINDOW_NORMAL)
        # cv2.imshow("orb_match", frm)
        out.write(frm)
        cv2.imshow("orb_match", res)

        # Pause on pressing of space.
        if key == ord(' '):
            wait_period = 0
        else:
            wait_period = 30

        key = cv2.waitKey(wait_period)
        ret, frm = cap.read()
        frm_count += 1


    cv2.destroyAllWindows()
    cap.release()
    out.release()

    return 0


def tracking_lucas_kanade():

    cap = cv2.VideoCapture('find_chocolate.mp4')

    img_chocolate = cv2.imread('marker.jpg')
    gray_chocolate = cv2.cvtColor(img_chocolate, cv2.COLOR_BGR2GRAY)

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.2,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(35, 35),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (1000, 3))

    # Take first frame and find corners in it

    ret, old_frame = cap.read()

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(1000, 1.1, 13)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    kpts1, descs1 = orb.detectAndCompute(gray_chocolate, None)

    # Setting video format. Google for "fourcc"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Setting up new video writer
    image_size = (old_frame.shape[1], old_frame.shape[0])
    # writer = cv2.VideoWriter('sample_tracking_orb.avi', fourcc, frames_per_second, image_size)
    out = cv2.VideoWriter('sample_tracking_lucas_kanade.avi', fourcc, 30.0, image_size)

    frno = 0
    restart = False
    while (1):
        frno += 1
        ret, frame = cap.read()
        if ret:

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if restart:
                orb = cv2.ORB_create(1000, 1.1, 13)
                kpts2, descs2 = orb.detectAndCompute(frame_gray, None)
                restart = False

            kpts2, descs2 = orb.detectAndCompute(frame_gray, None)

            matches = bf.match(descs1, descs2)
            # Sort them in the order of their distance.
            dmatches = sorted(matches, key=lambda x: x.distance)

            ## extract the matched keypoints
            src_pts = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1, 1, 2)

            ## find homography matrix and do perspective transform
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = img_chocolate.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            ## draw found regions
            frm = cv2.polylines(frame, [np.int32(dst)], True, (0, 0, 255), 1, cv2.LINE_AA)

            # ## draw match lines
            # res = cv2.drawMatches(img_chocolate, kpts1, frm, kpts2, dmatches[:8], None, flags=2)

            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, dst_pts, None, **lk_params)
            successful = (st == 1)
            if np.sum(successful) == 0:
                restart = True
            # Select good points
            good_new = p1[successful]
            good_old = dst_pts[successful]

            # draw the tracks
            count_of_moved = 0
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                velocity = np.sqrt((a - c) ** 2 + (b - d) ** 2)
                if velocity > 1:
                    mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 4, color[i].tolist(), -1)
                    count_of_moved += 1

            # res = cv2.drawMatches(img_chocolate, kpts1, frm, kpts2, dmatches, None, flags=2) #[:8]
            out.write(frame)

            cv2.namedWindow('orb_match', cv2.WINDOW_NORMAL)

            cv2.imshow('orb_match', frame)


            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        else:
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()


if __name__ == '__main__':
    # tracking_orb()
    tracking_lucas_kanade()


