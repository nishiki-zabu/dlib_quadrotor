#!/usr/bin/env python
# coding:utf-8
import cv2
import numpy as np

def find_rect_of_target_color(image):
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
  h = hsv[:, :, 0]
  s = hsv[:, :, 1]
  mask = np.zeros(h.shape, dtype=np.uint8)
  mask[((h < 20) | (h > 200)) & (s > 128)] = 255

  mask,contours,_= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  rects = []
  for contour in contours:
    approx = cv2.convexHull(contour)
    rect = cv2.boundingRect(approx)
    rects.append(np.array(rect))
  return rects

if __name__ == "__main__":
  cap = cv2.VideoCapture(0)
  while cv2.waitKey(30) < 0:
    _,frame = cap.read()
    rects = find_rect_of_target_color(frame)
    for rect in rects:
				cv2.rectangle(frame,tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]),(0,0,255),thickness=2)
    cv2.imshow('red', frame)
  cap.release()
  cv2.destroyAllWindows()
