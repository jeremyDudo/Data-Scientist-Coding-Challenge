"""File for analyzing a sickle cell microscope image."""
import numpy as np
import cv2
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt


def image_solution():
    """Image analysis on the testSCD.png image found in this file.
    
    This code will produce an image of the outlined sickle-predicted cells 
    The prediction is based on an area threshold to exclude small noise, 
    and an aspect ratio threshold to distinguish between sickle and non sickle cells.
    """
    # step 1: identify all of the cells (exclude the background) 
    # step 1.1 load the image & find cells
    image = cv2.imread('testSCD.png')
    # print(image.shape)  # (w, h, (bgr))
    # convert down to (w, h, 0-255)
    gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # convert down to (w, h, 0-1)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 8)
    # there are still smaller particulate that have not been cleared

    # combat the overlap/touching cells
    # room for improvement
    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    binary = cv2.dilate(binary, kernel, iterations=2)
    binary = cv2.dilate(binary, kernel, iterations=2)

    # canny edge detection -- thresholds relative to the binary range 
    edges = cv2.Canny(binary, threshold1=0.5, threshold2=1)

    # take edges -> find contours 
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # track aspect ratios
    tracking_aspect_ratio = []
    tracking_area = []

    # per contour measurement approximates the cell statistics
    for i, contour in tqdm(enumerate(contours)):
        # check if nested contours
        if hierarchy[0][i][3] == -1:
            # calculate the bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            # area 
            area = cv2.contourArea(contour)
            tracking_area.append(area)

            area_thresh = 32.5
            if area > area_thresh:
                
                # psuedo-aspect raio 
                lower = w 
                higher = h
                if h < w:
                    lower = h 
                    higher = w 
                aspect_ratio = lower/higher 
                tracking_aspect_ratio.append(aspect_ratio)

                if aspect_ratio < 0.75:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)

    sickle = [x for x in tracking_aspect_ratio if x < 0.75]
    non = [x for x in tracking_aspect_ratio if x > 0.75]
    print(f'Percent Est: {len(sickle)/len(non):.2f}%')
    
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    sys.exit()


if __name__ == '__main__':
    image_solution()
