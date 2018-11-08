import os
import datetime
import cv2
from base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0
    dump_count = 0
    face_cascade = cv2.CascadeClassifier(os.path.join('haar_dicts', 'haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(os.path.join('haar_dicts', 'haarcascade_eye.xml'))

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
            img, faces = Camera.process(img)

            Camera.dumpImg(img, faces)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
    
    @staticmethod
    def process(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = Camera.face_cascade.detectMultiScale(gray, 1.3, 5)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cvt face area gray to color
            gray_3ch[y:y+h, x:x+w] = img[y:y+h, x:x+w]
        gray_3ch = cv2.resize(gray_3ch, (320, 240))
        return gray_3ch, faces
    
    @staticmethod
    def dumpImg(img, faces):
        if len(faces) == 0:
            Camera.dump_count += 1
        else:
            Camera.dump_count += 10
        
        if Camera.dump_count > 10000:
            # save img
            img_name = "dump_{0:%Y%m%d%I%M%S}.jpg".format(datetime.datetime.now())
            img_path = os.path.join("dump", img_name)
            if not os.path.exists(os.path.dirname(img_path)):
                os.mkdir(os.path.dirname(img_path))
            cv2.imwrite(img_path, img)
            print("save img {}".format(img_path))
            
            Camera.dump_count = 0

