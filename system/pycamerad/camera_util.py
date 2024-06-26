import numpy as np
import os
import pyopencl as cl
import pyopencl.array as cl_array
import cv2

from msgq.visionipc import VisionIpcServer, VisionStreamType
from cereal import messaging

from openpilot.common.basedir import BASEDIR
from openpilot.system.pycamerad.camera_config import CameraConf

class CameraUtil:
  """Simulates the camerad daemon"""
  def __init__(self, road_camera_width, road_camera_height, driver_camera_width, driver_camera_height):
    # self.topics = ['driverCameraState', 'roadCameraState']
    self.topics = ['roadCameraState']
    self.pm = messaging.PubMaster(self.topics)
    self.road_camera_width = road_camera_width
    self.road_camera_height = road_camera_height
    self.driver_camera_width = driver_camera_width
    self.driver_camera_height = driver_camera_height
    self.frame_road_id = 0
    self.frame_wide_id = 0
    self.frame_driver_id = 0
    self.vipc_server = VisionIpcServer("camerad")

    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 5, False, self.road_camera_width, self.road_camera_height)
    # self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 5, False, self.road_camera_width, self.road_camera_height)
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_DRIVER, 5, False, self.driver_camera_width, self.driver_camera_height)
    self.vipc_server.start_listener()

    # set up for pyopencl rgb to yuv conversion
    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue(self.ctx)

    ## cl_arg = f" -DHEIGHT={H} -DWIDTH={W} -DRGB_STRIDE={W * 3} -DUV_WIDTH={W // 2} -DUV_HEIGHT={H // 2} -DRGB_SIZE={W * H} -DCL_DEBUG "
    cl_driver_arg = f""" -DHEIGHT={self.driver_camera_height} -DWIDTH={self.driver_camera_width} -DRGB_STRIDE={self.driver_camera_width * 3} -DUV_WIDTH={self.driver_camera_width // 2}  -DUV_HEIGHT={self.driver_camera_height // 2} -DRGB_SIZE={self.driver_camera_width * self.driver_camera_height} -DCL_DEBUG """
    cl_road_arg = f""" -DHEIGHT={self.road_camera_height} -DWIDTH={self.road_camera_width} -DRGB_STRIDE={self.road_camera_width * 3} -DUV_WIDTH={self.road_camera_width // 2}  -DUV_HEIGHT={self.road_camera_height // 2} -DRGB_SIZE={self.road_camera_width * self.road_camera_height} -DCL_DEBUG """

    kernel_fn = os.path.join(BASEDIR, "system/pycamerad/rgb_to_nv12.cl")
    with open(kernel_fn) as f:
      kernel_info = f.read()
      prg_driver = cl.Program(self.ctx, kernel_info).build(cl_driver_arg)
      self.krnl_driver = prg_driver.rgb_to_nv12
      prg_road = cl.Program(self.ctx, kernel_info).build(cl_road_arg)
      prg_wide = cl.Program(self.ctx, kernel_info).build(cl_road_arg)
      self.krnl_road = prg_road.rgb_to_nv12
      self.krnl_wide = prg_wide.rgb_to_nv12
    

    ## self.Wdiv4 = W // 4 if (W % 4 == 0) else (W + (4 - W % 4)) // 4
    self.Wdiv4_driver = self.driver_camera_width // 4 if (self.driver_camera_width % 4 == 0) else (self.driver_camera_width + (4 - self.driver_camera_width % 4)) // 4
    self.Hdiv4_driver = self.driver_camera_height // 4 if (self.driver_camera_height % 4 == 0) else (self.driver_camera_height + (4 - self.driver_camera_height % 4)) // 4

    self.Wdiv4_road = self.road_camera_width // 4 if (self.road_camera_width % 4 == 0) else (self.road_camera_width + (4 - self.road_camera_width % 4)) // 4
    self.Hdiv4_road = self.road_camera_height // 4 if (self.road_camera_height % 4 == 0) else (self.road_camera_height + (4 - self.road_camera_height % 4)) // 4

  def cam_send_yuv_road(self, yuv):
    self._send_yuv(yuv, self.frame_road_id, 'roadCameraState', VisionStreamType.VISION_STREAM_ROAD)
    self.frame_road_id += 1

  def cam_send_yuv_wide_road(self, yuv):
    self._send_yuv(yuv, self.frame_wide_id, 'wideRoadCameraState', VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.frame_wide_id += 1

  def cam_send_yuv_driver(self, yuv):
    self._send_yuv(yuv, self.frame_driver_id, 'driverCameraState', VisionStreamType.VISION_STREAM_DRIVER)
    self.frame_driver_id += 1

  # convert to yuv use opencl
  def rgb_to_yuv(self, rgb, cam_type):
    if cam_type in (CameraConf.ROAD, CameraConf.WIDE):
      assert rgb.shape == ( self.road_camera_height, self.road_camera_width, 3), f"{rgb.shape}"
    elif cam_type == CameraConf.DRIVER:
      assert rgb.shape == (self.driver_camera_height, self.driver_camera_width, 3), f"{rgb.shape}"
    else:
      pass
    assert rgb.dtype == np.uint8

    rgb_cl = cl_array.to_device(self.queue, rgb)
    yuv_cl = cl_array.empty_like(rgb_cl)

    if cam_type == CameraConf.ROAD:
      self.krnl_road(self.queue, (self.Wdiv4_road, self.Hdiv4_road), None, rgb_cl.data, yuv_cl.data).wait()
    elif cam_type == CameraConf.WIDE:
      self.krnl_wide(self.queue, (self.Wdiv4_road, self.Hdiv4_road), None, rgb_cl.data, yuv_cl.data).wait()
    elif cam_type == CameraConf.DRIVER:
      self.krnl_driver(self.queue, (self.Wdiv4_driver, self.Hdiv4_driver), None, rgb_cl.data, yuv_cl.data).wait()
    else:
      pass

    yuv = np.resize(yuv_cl.get(), rgb.size // 2)
    return yuv.data.tobytes()

  # convert to yuv use opencv
  def cv_rgb2nv12(self, rgb):
    yuv = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV_I420)
    uv_row_cnt = yuv.shape[0] // 3
    uv_plane = np.transpose(yuv[uv_row_cnt * 2:].reshape(2, -1), [1, 0])
    yuv[uv_row_cnt * 2:] = uv_plane.reshape(uv_row_cnt, -1)
    return yuv.tobytes()
  
  def _send_yuv(self, yuv, frame_id, pub_type, yuv_type):
    eof = int(frame_id * 0.05 * 1e9)
    self.vipc_server.send(yuv_type, yuv, frame_id, eof, eof)

    dat = messaging.new_message(pub_type)
    msg = {
      "frameId": frame_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat, pub_type, msg)
    self.pm.send(pub_type, dat)
