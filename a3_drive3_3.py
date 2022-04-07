#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################
# 프로그램명 : hough_drive.py
# 작 성 자 : 자이트론
# 생 성 일 : 2020년 08월 12일
# 수 정 일 : 2021년 03월 16일
# 검 수 인 : 조 이현
# 본 프로그램은 상업 라이센스에 의해 제공되므로 무단 배포 및 상업적 이용을 금합니다.
####################################################################

import rospy, rospkg
import numpy as np
import cv2, random, math
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
from sensor_msgs.msg import Image
#from tkinter import *
from pid import PID
import sys
import os
import signal

all_lines = []
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
frame_size = (640, 480)
out_bin = cv2.VideoWriter('/home/nvidia/xycar_ws/src/a3_drive/src/a3_bin.mp4', fourcc, fps, frame_size)
out_edge = cv2.VideoWriter('/home/nvidia/xycar_ws/src/a3_drive/src/a3_edge.mp4', fourcc, fps, frame_size)
out_track = cv2.VideoWriter('/home/nvidia/xycar_ws/src/a3_drive/src/a3_track.mp4', fourcc, fps, frame_size)

l_ = 1
r_ = 1

pid = PID(0.45, 0.00075, 0.065)

def signal_handler(sig, frame):
    os.system('killall -9 python rosout')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

image = np.empty(shape=[0])
bridge = CvBridge()
pub = None
Width = 640
Height = 480
Offset = 365
Gap = 40

def img_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

# publish xycar_motor msg
def drive(Angle, Speed):
    global pub

    msg = xycar_motor()
    msg.angle = Angle
    msg.speed = Speed

    pub.publish(msg)

# draw lines
def draw_lines(img, lines):
    global Offset
    for line in lines:
        x1, y1, x2, y2 = line[0]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = cv2.line(img, (x1, y1+Offset), (x2, y2+Offset), color, 2)
    return img

# draw rectangle
def draw_rectangle(img, lpos, rpos, offset=0):
    center = (lpos + rpos) / 2

    cv2.rectangle(img, (lpos - 5, 15 + offset),
                       (lpos + 5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (rpos - 5, 15 + offset),
                       (rpos + 5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (center-5, 15 + offset),
                       (center+5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (315, 15 + offset),
                       (325, 25 + offset),
                       (0, 0, 255), 2)
    return img

# left lines, right lines
def divide_left_right(lines): #왼, 오 선분 나누는 함수
    global Width
    low_slope_threshold = 0
    high_slope_threshold = 10

    # calculate slope & filtering with threshold
    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0: # 수직선일 경우 기울기 0
            slope = 0
        else:
            slope = float(y2-y1) / float(x2-x1)

        if abs(slope) > low_slope_threshold and abs(
            slope) < high_slope_threshold:
            slopes.append(slope)
            new_lines.append(line[0])

    # divide lines left to right
    left_lines = []
    right_lines = []

    for j in range(len(slopes)):
        Line = new_lines[j]
        slope = slopes[j]

        x1, y1, x2, y2 = Line
        
        
        if (slope < 0) and (x2 < Width/2 - 45):
            left_lines.append([Line.tolist()])
        elif (slope > 0) and (x1 > Width/2 + 45):
            right_lines.append([Line. tolist()])

    return left_lines, right_lines

# get average m, b of lines
def get_line_params(lines):
    
    # sum of x, y, m
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    size = len(lines)
    if size == 0:
        return 0, 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_sum += x1 + x2
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)

    x_avg = x_sum / (size * 2)
    y_avg = y_sum / (size * 2)
    m = m_sum / size
    b = y_avg - m * x_avg
    return m, b

# get lpos, rpos
def get_line_pos(img, lines, left=False, right=False):
    global Width, Height
    global Offset, Gap
    global l_,r_


    m, b = get_line_params(lines)
    if m == 0 and b == 0:
        if left:
            pos = 0
            l_ = 0
        if right:
            pos = Width
            r_ = 0
    else:
        if left:
            l_ = 1
        if right:
            r_ = 1
        y = Gap / 2
        pos = (y - b) / m

        b += Offset
        x1 = (Height - b) / float(m)
        x2 = ((Height/2) - b) / float(m)

        cv2.line(img, (int(x1), Height), (int(x2), (Height/2)), (255, 0,0), 3)

    return img, int(pos)

# show image and return lpos, rpos
def process_image(frame):
    global Width
    global Offset, Gap
    global l_,r_
    global out_track, out_edge, out_bin
    global all_lines
    global l_slope,r_slope
    # gray
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # blur
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

    _,binary_gray=cv2.threshold(blur_gray, 100, 255, cv2.THRESH_BINARY)

    # canny edge
    low_threshold = 170
    high_threshold = 200
    edge_img = cv2.Canny(np.uint8(binary_gray), low_threshold, high_threshold)


    # HoughLinesP
    roi = edge_img[Offset : Offset+Gap, 0 : Width]
    all_lines = cv2.HoughLinesP(roi,1,math.pi/180,20,30,5)

    # divide left, right lines
    if all_lines is None:
        r_ = 0
        l_ = 0
        return 0, 640
    left_lines, right_lines = divide_left_right(all_lines)

    # get center of lines
    frame, lpos = get_line_pos(frame, left_lines, left=True)
    frame, rpos = get_line_pos(frame, right_lines, right=True)

    # draw lines
    frame = draw_lines(frame, left_lines)
    frame = draw_lines(frame, right_lines)
    frame = cv2.line(frame, (230, 235), (410, 235), (255,255,255), 2)

    # draw rectangle
    frame = draw_rectangle(frame, lpos, rpos, offset=Offset)
    #roi2 = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    #roi2 = draw_rectangle(roi2, lpos, rpos)

    # show image
    cv2.imshow("binary_gray", binary_gray)
    binary_gray = cv2.cvtColor(binary_gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(binary_gray, (0, Offset), (640, Offset+ Gap), (0, 255, 255), 2)
    out_bin.write(binary_gray)

    cv2.imshow("edge_img",edge_img)
    edge_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(edge_img, (0, Offset), (640, Offset+ Gap), (0, 255, 255), 2)
    out_edge.write(edge_img)

    cv2.imshow('calibration', frame)
    cv2.rectangle(frame, (0, Offset), (640, Offset+ Gap), (0, 255, 255), 2)
    out_track.write(frame)
    tmp,tmp2 = 0,0
    for i in len(left_lines):
        tmp += (left_lines[1]-left_lines[3])/(left_lines[i][0]-left_lines[i][2])
    l_slope = tmp/len(left_lines)
     
    for i in len(right_lines):
        tmp2 += (right_lines[1]-right_lines[3])/(right_lines[i][0]-right_lines[i][2])
    r_slope = tmp2/len(right_lines)
    return lpos, rpos

def start():
    angle = 0
    global pub
    global image
    global cap
    global Width, Height
    global m_
    global l_,r_
    global pid
    global all_lines
    kg = 0
    count = 0
    _count = 10
    global speed, speed_tmp
    speed = 20
    speed1 = 20
    speed2 = 9
    speed3 = 30
    l_slope = 0
    r_slope = 0
    
    rospy.init_node('auto_drive')
    pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)

    image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, img_callback)
    print "---------- Xycar A2 v1.0 ---------"
    rospy.sleep(1)
    
    while True:
        # trackbar1 = cv2.getTrackbarPos('speed', 'TrackBar Test')
        while not image.size == (640*480*3):
            continue
        
        lpos, rpos = process_image(image)

        #cv2.rectangle(image, (0, Offset), (640, Offset+Gap), (0, 255, 255))
        #cv2.line(image, (320-90, Offset), (320-90, Offset+Gap), (0, 255, 255))
        #cv2.line(image, (320+90, Offset), (320+90, Offset+Gap), (0, 255, 255))
        if kg == 2:
           speed = speed3
           center = (lpos + rpos) / 2
           error = (center -Width/2)
           angle = pid.pid_control(error)
           drive(angle,speed)

        elif kg == 1:
            Offset = 440
            count +=1
            speed = speed2
            center = (lpos + rpos) / 2
            #angle = -(Width/2 - center)
            error = (center - Width/2)
            if angle > 0:
                angle = 50
            elif angle < 0:
                angle = -50
                
            drive(angle,speed)
            if count > _count:
                kg = 0
                count = 0
                speed = speed1
            if l_ == 1 and r_ == 1:
                kg = 0
                speed = speed1

        else:
            if l_ == 0 and r_ == 0:
                center = (lpos + rpos) / 2
                #angle = -(Width/2 - center)
                error = (center - Width/2)
                drive(angle,speed)
                kg = 1

            else:
                if abs(l_slope + r_slope) < 0.1 :
                    kg = 3
                print("else")
                Offset = 365
                speed = speed1
                center = (lpos + rpos) / 2
                #angle = -(Width/2 - center)
                error = (center - Width/2)


                angle = pid.pid_control(error)
                # if abs(angle) >30:
                #     speed = 10
                drive(angle,speed)


        ### angle, kg, speed
        kg_text = "KG: " + str(kg)
        cv2.putText(image, kg_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

        angle_text = "Angle: " + str(angle)
        cv2.putText(image, angle_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

        speed_text = "speed: " + str(speed)
        cv2.putText(image, speed_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255)) #lines

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out_bin.release()
    out_edge.release()
    out_track.release()
    cv2.destroyAllWindows()
    sys.exit(0)
    #rospy.spin()

if __name__ == '__main__':
    start()