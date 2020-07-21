import cv2
import numpy as np
import os,glob
import re

'''
Please make a python script that will find symbols on the plan. Better if you manage to find separately symbols with different inner parts.
Save your result as a new image where each found symbol will be highlighted by a bounding rect with a text label in a corner.
Ideally a font or bounding rect color should be a color code of a symbol type.
Bonus task: if you want to get familiar with JSON, please save location and type information of each found symbol in a *.json file.

You probably will need:
findContours
'''


def run_template_match_demo():
    '''
    Playing with template matcher
    :return: No return
    '''

    # Opening of an image
    image_filename = 'plan.png'
    img_loaded = cv2.imread(image_filename)
    img_grey = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_grey, (5, 5), 0)
    img_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    colors_list = [(135, 131, 250), (64, 75, 190), (114, 166, 255), (154, 251, 253), (143, 206, 199), (122, 183, 136),
                   (226, 0, 255), (196, 56, 255), (131, 77, 245), (145, 106, 0), (157, 146, 0), (143, 221, 107),
                   (93, 154, 61), (217,242, 177), (181, 187, 222), (217, 194, 255)]

# open templates

    folder_path = r'C:\Users\Admin\PycharmProjects\IT_Jim_2\W2\symbols'
    color_num = 0
    for filename in glob.glob(os.path.join(folder_path, '*.png')):
        color = colors_list[color_num]

        # template_filename = r'symbols/001.png'
        tm_loaded = cv2.imread(filename)

        tm_label = re.findall(r'\d+', filename)

        filename = tm_label[2]

        tm_grey = cv2.cvtColor(tm_loaded, cv2.COLOR_BGR2GRAY)
        tm_blur = cv2.GaussianBlur(tm_grey, (5, 5), 0)
        tm_thresh_angle0 = cv2.threshold(tm_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # template rotation

        (h, w) = tm_thresh_angle0.shape[:2]
        # calculate the center of the image
        center = (w / 2, h / 2)

        angle90 = 90
        angle180 = 180
        angle270 = 270

        scale = 1.0

        # Perform the counter clockwise rotation holding at the center
        # 90 degrees
        M = cv2.getRotationMatrix2D(center, angle90, scale)
        rotated90 = cv2.warpAffine(tm_thresh_angle0, M, (h, w))

        # 180 degrees
        M = cv2.getRotationMatrix2D(center, angle180, scale)
        rotated180 = cv2.warpAffine(tm_thresh_angle0, M, (w, h))

        # 270 degrees
        M = cv2.getRotationMatrix2D(center, angle270, scale)
        rotated270 = cv2.warpAffine(tm_thresh_angle0, M, (h, w))

        tm_list = [tm_thresh_angle0, rotated90, rotated180, rotated270]


    # matching

        for tm_image in tm_list:
            matching = cv2.matchTemplate(img_thresh, tm_image, cv2.TM_CCOEFF_NORMED)
            matching = cv2.normalize(matching, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            thresh = cv2.threshold(matching, np.max(matching) * 0.97, 255, cv2.THRESH_BINARY)[1]  # изменять порог 0.9 | Подаем бинарную картинку
            not_thresh = cv2.bitwise_not(thresh)
            not_thresh = cv2.erode(not_thresh, np.ones((25, 25)))
            edged = cv2.Canny(not_thresh, 50, 200)
            conts, hier = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # какие-то контуры, родители, дети, вложенные контуры О_о

            # print found contours


            for cnt in conts:
                # clr = np.random.randint(100,200,(3,),dtype=np.uint8)

                x, y, w, h = cv2.boundingRect(cnt)
                x += 15
                y += 15
                cv2.rectangle(img_loaded, (x, y), (x + w, y + h), color, 4)
                cv2.putText(img_loaded, filename, (x-5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3)

        color_num += 1

# find contours coordinates

    # path = 'C:/Users/Desktop/'
    # contour_num = 0
    # for i in range(0, len(conts)):
    #     coords = conts[i]
    #     # coords = np.array2string(conts[i])
    #     contour_num += 1
    #     # open(path + 'contour_%d.txt' % contour_num, "w").write(coords)
    #     print(contour_num, coords)

    # Filename
    result_image_filename = 'result_image.jpg'

    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(result_image_filename, img_loaded)

    # Showing images
    img_loaded_resized = cv2.resize(img_loaded, (img_loaded.shape[1] // 3, img_loaded.shape[0] // 3))
    cv2.imshow('Window with example', img_loaded_resized)
    cv2.waitKey(0)  # won't draw anything without this function. Argument - time in milliseconds, 0 - until key pressed


if __name__ == '__main__':
    run_template_match_demo()

