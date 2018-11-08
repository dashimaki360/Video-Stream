import os
import datetime
import numpy as np
import cv2
from base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0
    dump_count = 0
    face_cascade = cv2.CascadeClassifier(os.path.join('haar_dicts', 'haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(os.path.join('haar_dicts', 'haarcascade_eye.xml'))

    pre_frame = None

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()
            img, faces, delta_sum = Camera.process(img)

            Camera.dumpImg(img, faces, delta_sum)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
    
    @staticmethod
    def process(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # move detect
        # move_area = np.zeros(gray.shape)
        if Camera.pre_frame is None:
            Camera.pre_frame = gray
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(Camera.pre_frame)) / 255
        frameDelta[frameDelta < 0.3] = 0.
        # diff = gray - Camera.pre_frame
        Camera.pre_frame = gray

        delta_sum = int(frameDelta.sum())
        print(delta_sum)

        # face detect
        faces = Camera.face_cascade.detectMultiScale(gray, 1.3, 5)

        # make img
        gray = (gray * (frameDelta + 0.2) * 1.3).astype(np.uint8)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for (x, y, w, h) in faces:
            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cvt face area gray to color
            gray_3ch[y:y+h, x:x+w] = img[y:y+h, x:x+w]
        gray_3ch = cv2.resize(gray_3ch, (320, 240))

        #return gray_3ch, faces
        return gray_3ch, faces, delta_sum
    
    @staticmethod
    def dumpImg(img, faces, delta_sum):
        if len(faces) == 0:
            Camera.dump_count += delta_sum
        else:
            Camera.dump_count += 3*delta_sum
        
        if Camera.dump_count > 30000:
            # save img
            img_name = "dump_{0:%Y%m%d%I%M%S}.jpg".format(datetime.datetime.now())
            img_path = os.path.join("dump", img_name)
            if not os.path.exists(os.path.dirname(img_path)):
                os.mkdir(os.path.dirname(img_path))
            cv2.imwrite(img_path, img)
            print("save img {}".format(img_path))
            
            Camera.dump_count = 0

