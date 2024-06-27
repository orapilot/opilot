import os,time
import sys
import numpy as np
from typing import Tuple, Dict, Union, Any

from openpilot.selfdrive.modeld.runners.runmodel_pyx import RunModel
from openpilot.system.swaglog import cloudlog
from rknnlite.api import RKNNLite


def create_rknn_session(path):
  os.environ["OMP_NUM_THREADS"] = "4"
  os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

  rknn_lite = RKNNLite()
  ret = rknn_lite.load_rknn(path)
  assert ret == 0, "load rknn model failed"
  ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL)
  assert ret == 0, "initial rknn runtime failed"
  return rknn_lite

model_inputs = {
  "road": {
    "input_names": ["input_imgs","big_input_imgs", "desire",
        "traffic_convention","lateral_control_params",
        "prev_desired_curv", "features_buffer"
        ],
    "input_shapes":  {
        "input_imgs": [1,12,128,256],
        "big_input_imgs": [1,12,128,256],
        "desire": [1,100,8],
        "traffic_convention": [1,2],
        "lateral_control_params": [1,2],
        "prev_desired_curv": [1,100,1],
        'features_buffer': [1, 99,512]
    },
    "input_dtypes": {
      'input_imgs': np.float16,
      'big_input_imgs': np.float16,
      'desire': np.float16,
      'traffic_convention': np.float16,
      'lateral_control_params': np.float16,
      'prev_desired_curv': np.float16,
      'features_buffer': np.float16
    }
  },
  "dmonitor":{
    "input_names": ['input_img', 'calib'],
    "input_shapes": {
      "input_img":[1,1382400],
      "calib": [1,3]
    },
    "input_dtypes": {
      "input_img":np.float16,
      "calib": np.float16
    }
  }
}


class RKNNModel(RunModel):
  def __init__(self, path, output, runtime, use_tf8, cl_context):
    self.inputs = {}
    self.output = output
    self.use_tf8 = use_tf8
    self.model_path = path
    self.session = create_rknn_session(path)
    self.search_key = ""
    if "nav" in path:
      self.search_key = "nav"
    elif "dmonitor" in path:
      self.search_key = "dmonitor"
    elif "supercombo" in path:
      self.search_key = "road"
    self.input_names = model_inputs.get(self.search_key).get("input_names")
    self.input_shapes = model_inputs.get(self.search_key).get("input_shapes")
    self.input_dtypes = model_inputs.get(self.search_key).get("input_dtypes")
    print("ready to run rknn model", self.input_shapes, file=sys.stderr)

  def addInput(self, name, buffer):
    assert name in self.input_names, "name: %s not in inputs" % (name,)
    self.inputs[name] = buffer

  def setInputBuffer(self, name, buffer):
    assert name in self.inputs, "name: %s not in inputs" % (name,)
    self.inputs[name] = buffer

  def getCLBuffer(self, name):
    return None

  def execute(self):
    inputs = {k: (v.view(np.uint8) / 255. if self.use_tf8 and k == 'input_img' else v) for k,v in self.inputs.items()}
    inputs = {k: v.reshape(self.input_shapes[k]).astype(self.input_dtypes[k]) for k,v in inputs.items()}
    '''
    self.inputs = {
      'input_imgs': ,
      'big_input_imgs': ,
      'desire': np.zeros(DESIRE_LEN * (HISTORY_BUFFER_LEN+1), dtype=np.float16),
      'traffic_convention': np.zeros(TRAFFIC_CONVENTION_LEN, dtype=np.float16),
      'nav_features': np.zeros(NAV_FEATURE_LEN, dtype=np.float16),
      'nav_instructions': np.zeros(NAV_INSTRUCTION_LEN, dtype=np.float16),
      'features_buffer': np.zeros(HISTORY_BUFFER_LEN * FEATURE_LEN, dtype=np.float16),
    }
    '''
    rknn_inputs = []
    if "supercombo" in self.model_path:
      rknn_inputs = [
        inputs.get("input_imgs").transpose(0, 2, 3, 1),
        inputs.get("big_input_imgs").transpose(0, 2, 3, 1),
        inputs.get("desire"),
        inputs.get("traffic_convention"),
        inputs.get("lateral_control_params"),
        inputs.get("prev_desired_curv"),
        inputs.get("features_buffer"),
      ]
    elif "dmonitoring" in self.model_path:
      rknn_inputs = [
        inputs.get("input_img"), #why can't use transpose
        inputs.get("calib")
      ]



    elif "nav" in self.model_path:
      rknn_inputs = [
        inputs.get("input_img").transpose(0, 2, 3, 1)
      ]
    else:
      print("this model is not supported yet")
      return
    outputs = self.session.inference(inputs=rknn_inputs)
    assert len(outputs) == 1, "Only single model outputs are supported"
    self.output[:] = outputs[0]
    return self.output
