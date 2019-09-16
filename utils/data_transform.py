import cv2
import sys
from math import *
import numpy as np
class DataTransform(object):
    def __init__(self):
        pass
    def RotateImg(self,in_mat,labels,degree,borderValue=(128, 128, 128)):
        height, width = in_mat.shape[:2]
        nheight = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        nwidth = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        transform_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        transform_matrix[0, 2] += (nwidth - width) / 2
        transform_matrix[1, 2] += (nheight - height) / 2

        rotated_mat = cv2.warpAffine(in_mat, transform_matrix, (nwidth, nheight), borderValue=borderValue)

        x1 = (labels[:, 1] - labels[:, 3] / 2.)*width
        x2 = (labels[:, 1] + labels[:, 3] / 2.)*width
        y1 = (labels[:, 2] - labels[:, 4] / 2.)*height
        y2 = (labels[:, 2] - labels[:, 4] / 2.)*height

        nx1=transform_matrix[0,0]*x1+transform_matrix[0,1]*y1
        nx2=
        ny1=
        ny2=


        cv2.imshow("img", in_mat)
        cv2.imshow("imgRotation", rotated_mat)
        cv2.waitKey(-1)

if __name__=='__main__':
    data_transform =DataTransform()
    in_mat=cv2.imread(sys.argv[1])
    data_transform.RotateImg(in_mat,None,90)