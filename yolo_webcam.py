# -*- coding: utf-8 -*-
"""
Reference:
https://qiita.com/yoyoyo_/items/10d550b03b4b9c175d9c
https://madeinpc.blog.fc2.com/?no=1364
"""

import sys
import argparse
from yolo import YOLO

def detect_cam(yolo, cam_number, video_directory_path, video_filename):
    import numpy as np
    from PIL import Image
    import cv2
    import datetime
    from timeit import default_timer as timer

    delay=1
    window_name='Press q to quit'
    camera_scale = 1.

    cap = cv2.VideoCapture(cam_number)

    if not cap.isOpened():
        print('No Camera')
        return
 
    camera_fps = cap.get(cv2.CAP_PROP_FPS)

    isOutput = True if video_directory_path != '' else False
    if isOutput:
        now = datetime.datetime.now()
        if video_filename == '':
            video_filename = 'out_' + now.strftime('%Y%m%d_%H%M%S') + '.mp4' 
        video_filename = video_directory_path + '/' + video_filename
        video_fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_size = 640, 480

        out = cv2.VideoWriter(video_filename, video_fourcc, camera_fps, video_size)

    # Init FPS variable
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    if isOutput:
        # Init CBR variable
        t = 0
        bt = 0
        frameTime = 1 / camera_fps

    while True:
        if isOutput:
            beforeloop_time = timer()
        
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break

        ret, frame = cap.read()

        # Resize
        h, w = frame.shape[:2]
        rh = int(h * camera_scale)
        rw = int(w * camera_scale)
        image = cv2.resize(frame, (rw, rh))

        # BGR to RGB
        image = image[:,:,(2,1,0)]

        # Detect
        image = Image.fromarray(image)
        r_image = yolo.detect_image(image)
        result = np.asarray(r_image)

        # Write FPS
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps) + " Cam FPS: " + str(camera_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)

        # RGB to BGR
        result = result[:,:,(2,1,0)]

        cv2.imshow(window_name, result)

        if isOutput:
            # Write Video CBR
            afterloop_time = timer()
            t = afterloop_time - beforeloop_time

            n = 0
            tt = t + bt
            i = 0
            while True:
                if i >= tt:
                    break
                i += frameTime

                out.write(result)
                n += 1
            bt = tt - frameTime * n
 
    cv2.destroyWindow(window_name)
    yolo.close_session()

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--cam_number', type=int,
        help='VideoCapture Camera Number, default 0',
        default=0
    )

    parser.add_argument(
        '--video_directory_path', type=str,
        help='If you want to output video file, specify directory',
        default= ''
    )

    parser.add_argument(
        '--video_filename', type=str,
        help='If you want to fix the video file name , specify file name',
        default= ''
    )

    FLAGS = parser.parse_args()

    detect_cam(YOLO(**vars(FLAGS)), FLAGS.cam_number,
               FLAGS.video_directory_path, FLAGS.video_filename)
