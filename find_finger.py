#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : ManWingloeng


import cv2
import matplotlib.pyplot as plt
# from imutils.video.webcamvideostream import WebcamVideoStream
import numpy as np
import imutils

def test_HSV(frame):
    HSV_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    (H, S, V) = cv2.split(HSV_frame)
    HH = cv2.equalizeHist(H)
    SH = cv2.equalizeHist(S)
    VH = cv2.equalizeHist(V)
    HSV_H = cv2.merge((HH, SH, VH))
    ret1, SH = cv2.threshold(SH, 0, 255, type=cv2.THRESH_OTSU)
    ret2, VH = cv2.threshold(VH, 0, 255, type=cv2.THRESH_OTSU)
    HSV_mask = cv2.bitwise_and(SH, VH)
    cv2.imshow("HSVbin", HSV_mask)



def find_hand_old(frame):
    img = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    YCrCb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # YCrCb_frame = cv2.GaussianBlur(YCrCb_frame, (9, 9), 0)
    YCrCb_frame = cv2.GaussianBlur(YCrCb_frame, (3, 3), 0)
    # YCrCb_frame = cv2.GaussianBlur(YCrCb_frame, (1, 1), 0)
    # cv2.imshow("YCrCb_frame_old", YCrCb_frame)
    # print(frame.shape[:2])
    # mask = cv2.inRange(YCrCb_frame, np.array([0, 135, 97]), np.array([255, 177, 127]))#140 170 100 120
    # mask = cv2.inRange(YCrCb_frame, np.array([0, 133, 77]), np.array([255, 173, 127])) # best enough
    mask = cv2.inRange(YCrCb_frame, np.array([0, 127, 75]), np.array([255, 177, 130]))
    bin_mask = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.dilate(bin_mask, kernel, iterations=5)
    res = cv2.bitwise_and(frame, frame, mask=bin_mask)

    cv2.imshow("res_old", res)

    return img, bin_mask, res



