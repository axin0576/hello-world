import cv2 as cv
import numpy as np

#排序算法
def sort_boxes(contours):
    for i in range(len(contours)):
        for j in range(i, len(contours), 1):
            x, y, w, h = cv.boundingRect(contours[j])
            if x < cv.boundingRect(contours[i])[0]:
                contours0 = np.copy(contours[i])
                contours[i] = np.copy(contours[j])
                contours[j] = np.copy(contours0)
    return contours

def mode(src):
    image1 = cv.GaussianBlur(src, (3, 3), 0) # 1、GaussianBlur去噪声
    blurred = cv.pyrMeanShiftFiltering(image1, 10, 100) # 1、pyrMeanShiftFiltering去噪声
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY) # 2、灰度图像
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) # 3、#二值化THRESH_BINARY与自动阀值算法THRESH_OTSU
    se = cv.getStructuringElement(cv.MORPH_RECT, (2, 4), (-1, -1))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, se)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, se)
    # cv.imshow("mode", binary)

    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 轮廓发现
    print(len(contours))
    contours = sort_boxes(contours)
    # for c in range(len(contours)):
    #     x, y, w, h = cv.boundingRect(contours[c]) #x, y, w, h
    #     cv.rectangle(src, (x-5, y-5), (x+w+10, y+h+10), (0, 0, 255), 1, cv.LINE_8, 0)
    #     cv.imshow("mode", src)
    return contours

def num(src):
    image1 = cv.GaussianBlur(src, (3, 3), 0) # 1、GaussianBlur去噪声
    blurred = cv.pyrMeanShiftFiltering(image1, 10, 100) # 1、pyrMeanShiftFiltering去噪声
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY) # 2、灰度图像
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) # 3、#二值化THRESH_BINARY与自动阀值算法THRESH_OTSU
    se = cv.getStructuringElement(cv.MORPH_RECT, (2, 4), (-1, -1))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, se)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, se)
    # cv.imshow("num", binary)

    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 轮廓发现
    print(len(contours))
    contours = sort_boxes(contours)
    # for c in range(len(contours)):
    #     x, y, w, h = cv.boundingRect(contours[c]) #x, y, w, h
    #     cv.rectangle(src, (x-5, y-5), (x+w+10, y+h+10), (0, 0, 255), 1, cv.LINE_8, 0)
    #     cv.imshow("num", src)
    return contours


# mode_src = cv.imread("opencv_tutorial\\data\\images\\0123456789.jpg")
mode_src = cv.imread("0123456789.jpg")
# cv.imshow("mode_src", mode_src)
mode_contours = mode(mode_src)


num_src = cv.imread("num.jpg")
# cv.imshow("num_src", num_src)
num_contours = num(num_src)


# 几何矩计算与hu矩计算
# mm2 = cv.moments(mode_contours[5])
# hum2 = cv.HuMoments(mm2)
for i in range(len(num_contours)):
    mm = cv.moments(num_contours[i])
    hum = cv.HuMoments(mm)
    dist_min = 100
    c_min = -1
    for c in range(len(mode_contours)):
        mm2 = cv.moments(mode_contours[c])
        hum2 = cv.HuMoments(mm2)
        dist = cv.matchShapes(hum, hum2, cv.CONTOURS_MATCH_I1, 0)
        # print(dist)
        if dist < dist_min:
            dist_min = dist
            c_min = c
    else:
        print(c_min)
        # cv.drawContours(num_src, num_contours, i, (0, 0, 255), 2, 8)
    x, y, w, h = cv.boundingRect(num_contours[i]) #x, y, w, h
    cv.rectangle(num_src, (x-5, y-5), (x+w+10, y+h+10), (0, 0, 255), 1, cv.LINE_8, 0)
    cv.putText(num_src, str(c_min), ((x+w+x)//2-10, y-10), cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 2)
    cv.imshow("num_src", num_src)
    # cv.waitKey(0)

            # cv.drawContours(num_src, num_contours, i, (0, 0, 255), 2, 8)
        # print(dist)



cv.imshow("num_src", num_src)
cv.waitKey(0)
cv.destroyAllWindows()








