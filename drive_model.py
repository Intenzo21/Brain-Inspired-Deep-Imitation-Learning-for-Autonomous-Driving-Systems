"""
Drive a pretrained model.

Script employed in driving the Udacity simulator car in autonomous
mode by loading a pretrained model.
"""

import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import cv2
import math

from keras.models import load_model
from kerasncp.tf import LTCCell
import h5py
import tensorflow as tf
from keras import __version__ as keras_version
from constants import SPEED, KP, KI
from utils import preprocess

sio = socketio.Server()  # Create the server
app = Flask(__name__)
model = None
prev_image_array = None

# Get the list of GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
# To turn on memory growth for a specific GPU, we use the following code prior to allocating
# any tensors or executing any ops.
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class SimplePIController:
    """
    Class for creating autonomous car controller instances.
    """

    def __init__(self, Kp, Ki):
        """
        Set the instance attributes.

        :param Kp: proportional gain
        :param Ki: integral gain
        """
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.  # Holds the desired speed
        self.error = 0.  # Holds the proportional error
        self.integral = 0.  # Holds the integral error

    def set_desired(self, desired):
        """
        Set the desired autonomous car driving speed.

        :param desired: the desired speed value of the car
        :return: None
        """

        self.set_point = desired

    def update(self, measurement, angle):
        """
        Update the car throttle.

        :param measurement: current speed measurement
        :param angle: current steering angle
        :return: calculated new throttle value
        """

        # Proportional error
        self.error = self.set_point - measurement

        # Integral error
        self.integral += self.error

        # Adjustment applied to the throttle taking into consideration the current steering angle.
        # This is to slow down the car when turning and speeding up when driving straight.
        # Only partially implemented and needs further testing.
        angle_adjustment = (abs(angle) * max((1.5 - (1 / self.set_point)), 1))

        return (self.Kp * self.error + self.Ki * self.integral) - angle_adjustment


@sio.on('telemetry')
def telemetry(sid, data):
    """
    Update controller and communicate the new controls to the car.

    :param sid: session ID
    :param data: current car control data
    :return: None
    """

    if data:

        # The current steering angle of the car
        steering_angle = data["steering_angle"]

        # The current throttle of the car
        throttle = data["throttle"]

        # The current speed of the car
        speed = data["speed"]

        # The current image from the center camera of the car
        img_string = data["image"]
        image = Image.open(BytesIO(base64.b64decode(img_string)))
        image_array = np.asarray(image)
        image_array = preprocess(image_array)  # Preprocess the image

        # Get the steering angle prediction
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        throttle = controller.update(float(speed), steering_angle)

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # Save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    """ Connect to a session given its ID.

    :param sid: session ID
    :param environ: environment
    :return: None
    """

    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    """ Send the steering angle and throttle values to the car.

    :param steering_angle: current steering angle of the car
    :param throttle: current throttle value of the car
    :return: None
    """

    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':

    # Instantiate the car controller and set the desired speed
    controller = SimplePIController(KP, KI)
    controller.set_desired(SPEED)

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Remote driving program.')

    # Fill the ArgumentParser object with information about program arguments.
    parser.add_argument(
        'model',
        type=str,
        help='path to the model HDF5 file'
    )

    parser.add_argument(
        'speed',
        type=int,
        nargs='?',
        default=SPEED,
        help='speed of the autonomous driving car.'
    )

    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='path to image folder where the images from the run will be saved'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model, custom_objects={'CustomLayer': LTCCell})

    if args.speed != '':
        desired_speed = args.speed
    else:
        desired_speed = SPEED
    controller.set_desired(desired_speed)

    # Record the run if an image folder argument is provided
    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # Wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # Deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
