#!/usr/bin/env python
from flask import Flask, render_template, Response

# import camera driver
from camera_opencv import Camera

app = Flask(__name__)
cam = Camera()


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/good_btn')
def good_btn():
    return render_template('thanks.html')


if __name__ == '__main__':
    debug = False 
    app.run(host="0.0.0.0", debug=debug)

    print("del camera thred")
    cam.deleteThred()
    print("quiet server")
