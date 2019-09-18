import cv2
import sys
from math import *
import numpy as np
import random
class DataTransform(object):
    def __init__(self):
        pass

    def DrawLabel(self,in_mat,labels):
        '''
        将标注坐标以框的形式在图上画出
        :param in_mat: 输入图像
        :param labels: 输入标签,bounding box的归一化坐标表示,n*5 [[class,x,y,w,h],[class,x,y,w,h]]
        :return: 画好标注的图片
        '''
        out_mat=in_mat.copy()
        height, width = in_mat.shape[:2]
        x1 = (labels[:, 1] - labels[:, 3] / 2.) * width
        x2 = (labels[:, 1] + labels[:, 3] / 2.) * width
        y1 = (labels[:, 2] - labels[:, 4] / 2.) * height
        y2 = (labels[:, 2] + labels[:, 4] / 2.) * height
        for n in range(x1.shape[0]):
            cv2.rectangle(out_mat,(int(x1[n]),int(y1[n])),( int(x2[n]),int(y2[n]) ),(0, 0, 255), 2)
        return out_mat

    def RotateImg(self,in_mat,labels,degree,borderValue=(128, 128, 128)):
        '''
        旋转图像并修正对应的bounding box的坐标
        :param in_mat: 输入图像
        :param labels: 输入标签,bounding box的归一化坐标表示,n*5 [[class,x,y,w,h],[class,x,y,w,h]]
        :param degree: 旋转角度
        :param borderValue: 填充背景色
        :return:旋转后的图片以及修正后的标签
        '''

        height, width = in_mat.shape[:2]
        nheight = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        nwidth = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        transform_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        transform_matrix[0, 2] += (nwidth - width) / 2
        transform_matrix[1, 2] += (nheight - height) / 2

        rotated_mat = cv2.warpAffine(in_mat, transform_matrix, (nwidth, nheight), borderValue=borderValue)

        nlabels =labels.copy()
        nlabels[:,4]=(labels[:,3] * width* fabs(sin(radians(degree))) + labels[:,4] * height* fabs(cos(radians(degree))))/nheight
        nlabels[:,3]=(labels[:,4] * height* fabs(sin(radians(degree))) + labels[:,3] * width* fabs(cos(radians(degree))))/nwidth
        nlabels[:,1]=(transform_matrix[0,0]*labels[:,1]*width + transform_matrix[0,1]*labels[:,2]*height+transform_matrix[0,2])/nwidth
        nlabels[:,2]=(transform_matrix[1,0]*labels[:,1]*width + transform_matrix[1,1]*labels[:,2]*height+transform_matrix[1,2])/nheight
        return rotated_mat, nlabels

    def RandomCrop(self, in_mat, labels, hkmin_ratio=0.64,wkmin_ratio=0.64, borderValue=(128, 128, 128)):
        '''
        :param in_mat: 输入图像
        :param labels: 输入标签,归一化坐标表示,n*5 [[class,x,y,w,h],[class,x,y,w,h]]
        :param hkmin_ratio: 高度方向裁减后保留的原图最小比列
        :param wkmin_ratio: 宽度方向裁减后保留的原图最小比列
        :param borderValue: 填充的背景颜色
        :return: 裁减填充后的图片以及修正后的坐标
        '''
        height, width = in_mat.shape[:2]
        xmin=(labels[:,1]-labels[:,3]/2)
        ymin=(labels[:,2]-labels[:,4]/2)
        xmax=(labels[:,1]+labels[:,3]/2)
        ymax=(labels[:,2]+labels[:,4]/2)
        xmin=np.min(xmin)
        ymin=np.min(ymin)
        xmax=np.max(xmax)
        ymax=np.max(ymax)

        board=np.ones_like(in_mat)
        board[:,:,0]=borderValue[0]
        board[:,:,1]=borderValue[1]
        board[:,:,2]=borderValue[2]

        print(xmin,xmax,ymin,ymax)
        wkmin_ratio=max(wkmin_ratio,xmax-xmin)
        hkmin_ratio=max(hkmin_ratio,ymax-ymin)
        print(wkmin_ratio,hkmin_ratio)
        x_min_offset=xmax-wkmin_ratio if xmax-wkmin_ratio >0 else 0
        x_max_offset= 1-wkmin_ratio if 1-wkmin_ratio< xmin else xmin
        y_min_offset = ymax - hkmin_ratio if ymax - hkmin_ratio > 0 else 0
        y_max_offset = 1-hkmin_ratio if 1-hkmin_ratio< ymin else ymin
        print(x_min_offset,x_max_offset,y_min_offset,y_max_offset)
        x_offset=random.uniform(x_min_offset,x_max_offset)
        y_offset=random.uniform(y_min_offset,y_max_offset)

        print(x_offset,y_offset)

        patch=in_mat[ int(y_offset*height):int((y_offset+hkmin_ratio)*height),int(x_offset*width):int((x_offset+wkmin_ratio)*width),:]

        deltax=random.randint(0,width-patch.shape[1])
        deltay=random.randint(0,height-patch.shape[0])
        board[deltay:deltay+patch.shape[0],deltax:deltax+patch.shape[1],:]=patch

        nlabels=labels.copy()
        nlabels[:, 1] += deltax/width-x_offset
        nlabels[:, 2] += deltay / height-y_offset
        return board,nlabels

if __name__=='__main__':
    data_transform =DataTransform()
    in_mat=cv2.imread(sys.argv[1])
    cv2.imshow('in', in_mat)
    with open(sys.argv[2], 'r') as f:
        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)

    data_transform.DrawLabel(in_mat, l)

    out_mat,labels =data_transform.RandomCrop(in_mat,l)
    print(labels)
    out_mat = data_transform.DrawLabel(out_mat, labels)
    cv2.imshow('out', out_mat)
    cv2.waitKey(-1)