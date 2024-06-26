import os
import time
import cv2
import threading

from openpilot.common.realtime import set_realtime_priority
from openpilot.common.basedir import BASEDIR
from openpilot.system.pycamerad.camera_util import CameraUtil
from openpilot.system.pycamerad.camera_config import CameraConf
from openpilot.common.realtime import Ratekeeper

class Cameras:
  def __init__(self) -> None:
    self.road_camera_width = CameraConf.ROAD_CAMERA_WIDTH
    self.road_camera_height = CameraConf.ROAD_CAMERA_HEIGHT
    self.driver_camera_width = CameraConf.DRIVER_CAMERA_WIDTH
    self.driver_camera_height = CameraConf.DRIVER_CAMERA_HEIGHT
    self.camera_util = CameraUtil(self.road_camera_width, self.road_camera_height, self.driver_camera_width, self.driver_camera_height)
    self.roadcamid = os.getenv("ROADCAM_ID")
    self.drivercamid = os.getenv("DRIVERCAM_ID")
    self.widecamid = os.getenv("WIDECAM_ID")
    self.debugcam = os.getenv("DEBUG_CAM") is not None

  def send_camera_images(self, rgbimg):
    # yuv = self.camera_util.rgb_to_yuv(rgbimg, CameraConf.ROAD)
    yuv = self.camera_util.cv_rgb2nv12(rgbimg)
    self.camera_util.cam_send_yuv_road(yuv)

  def send_wide_camera_images(self, rgbimg):
    yuv = self.camera_util.rgb_to_yuv(rgbimg, CameraConf.WIDE)
    self.camera_util.cam_send_yuv_wide_road(yuv)

  def send_driver_images(self, rgbimg):
    yuv = self.camera_util.rgb_to_yuv(rgbimg, CameraConf.DRIVER)
    # yuv = self.camera_util.cv_rgb2nv12(rgbimg)
    self.camera_util.cam_send_yuv_driver(yuv)

  def road_cam(self, exit_event: threading.Event):
    cap = cv2.VideoCapture("rtsp://192.168.3.2:8554/live.stream", cv2.CAP_GSTREAMER ) if self.debugcam else \
     cv2.VideoCapture("gst-launch-1.0 v4l2src device=/dev/video" + str(self.roadcamid) + " io-mode=2 ! image/jpeg, width=" + \
                      str(self.road_camera_width) + ", height=" + str(self.road_camera_height) + \
                        ", framerate=30/1, format=MJPG ! mppjpegdec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
    if cap.isOpened():
      while not exit_event.is_set():
        ret_val, img = cap.read()
        if ret_val:
          self.send_camera_images(img)
      cap.release()

  
  def wide_cam(self, exit_event: threading.Event):
    rk = Ratekeeper(20, None)
    cap = cv2.VideoCapture("%s/selfdrive/assets/videos/driving.hevc" % (BASEDIR,)) if self.debugcam else \
      cv2.VideoCapture("gst-launch-1.0 v4l2src device=/dev/video" + str(self.widecamid) + " io-mode=2 ! image/jpeg, width=1920, height=1080, framerate=30/1, format=MJPG ! mppjpegdec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
    if cap.isOpened():
      while not exit_event.is_set():
        # read next frame to wide came
        ret_val, img = cap.read()
        if ret_val:
          self.send_wide_camera_images(img)
        rk.keep_time()
      cap.release()

  def driver_cam(self, exit_event: threading.Event):
    #rk = Ratekeeper(20, None)
    cap = cv2.VideoCapture("%s/selfdrive/assets/videos/driver.hevc" % (BASEDIR,)) if self.debugcam else \
      cv2.VideoCapture("gst-launch-1.0 v4l2src device=/dev/video" + str(self.drivercamid) + " io-mode=2 ! image/jpeg, width=1920, height=1080, framerate=30/1, format=MJPG ! mppjpegdec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
    if cap.isOpened():
      while not exit_event.is_set():
        ret_val, img = cap.read()
        if ret_val:
          self.send_driver_images(img)
        #rk.keep_time()
      cap.release()
  
  def starts(self):
    threads = []
    exit_event = threading.Event()
    if self.drivercamid is not None:
      threads.append(threading.Thread(target=self.driver_cam, args=(exit_event,)))
    threads.append(threading.Thread(target=self.road_cam, args=(exit_event,)))
    # threads.append(threading.Thread(target=self.wide_cam, args=(exit_event,)))
    for t in threads:
      t.start()
    for t in reversed(threads):
      t.join()
