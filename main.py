import os
import numpy as np
from datetime import datetime
from detect_box_yolo import detect_box
import time
from threading import Thread
import cv2

def convert(box,size = [1050,1400]):
    #size =  [width,height] of the frame
    #box = [xmin,ymin,xmax,ymax]----------------->
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[2])/2.0 - 1
    y = (box[1] + box[3])/2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)



def is_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print('Creating {}...'.format(path))

def autoanno(rois, filename):

    # print(rois)
    

    f = open(filename + ".txt", 'w+')

    for roi in rois:

        try:
            
            class_corr = int(roi[4])
            """
            x_corr = '{0:6f}'.format(roi[1])
            y_corr = '{0:6f}'.format(roi[0])
            w_corr = '{0:6f}'.format(roi[3])
            h_corr = '{0:6f}'.format(roi[2])
            """
            box = [roi[0],roi[1],roi[2],roi[3]]
            x_corr,y_corr,w_corr,h_corr = convert(box)
            text = f'{class_corr} {x_corr} {y_corr} {w_corr} {h_corr}\n'
            f.write(text)


        except:
            pass

    f.close()

def createvideo():

    cap = cv2.VideoCapture("wash.mp4")

    count = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            rois = detect_box(frame,0.40)
            autoanno(rois, str('annotations/{}'.format(count)))
            cv2.imwrite("annotations/{}.jpg".format(count), frame)
            print('Frame No: ',count)
            # cap.release()
            count = count + 1
            # cv2.imwrite("{}.jpg".format(count), frame)


        else:
            cap.release()
            


if __name__ == "__main__":
    
    createvideo()
