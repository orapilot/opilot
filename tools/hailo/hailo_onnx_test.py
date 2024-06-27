import os,time
import sys
import numpy as np
import itertools
from typing import Tuple, Dict, Union, Any
import onnx


ORT_TYPES_TO_NP_TYPES = {'tensor(float16)': np.float16, 'tensor(float)': np.float32, 'tensor(uint8)': np.uint8}

def attributeproto_fp16_to_fp32(attr):
  float32_list = np.frombuffer(attr.raw_data, dtype=np.float16)
  attr.data_type = 1
  attr.raw_data = float32_list.astype(np.float32).tobytes()

def convert_fp16_to_fp32(path):
  model = onnx.load(path)
  for i in model.graph.initializer:
    if i.data_type == 10:
      attributeproto_fp16_to_fp32(i)
  for i in itertools.chain(model.graph.input, model.graph.output):
    if i.type.tensor_type.elem_type == 10:
      i.type.tensor_type.elem_type = 1
  for i in model.graph.node:
    for a in i.attribute:
      if hasattr(a, 't'):
        if a.t.data_type == 10:
          attributeproto_fp16_to_fp32(a.t)
  return model.SerializeToString()


def create_ort_session(path):
  os.environ["OMP_NUM_THREADS"] = "4"
  os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

  import onnxruntime as ort
  print("Onnx available providers: ", ort.get_available_providers(), file=sys.stderr)
  options = ort.SessionOptions()
  #options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

  options.intra_op_num_threads = 4
  options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  provider = 'HailoExecutionProvider'
  #provider = 'CPUExecutionProvider'

  # print("Onnx selected provider: ", [provider], file=sys.stderr)
  ort_session = ort.InferenceSession(path, options, providers=[provider])
  # print("Onnx using ", ort_session.get_providers(), file=sys.stderr)
  return ort_session

class RKNNModel():
  def __init__(self, path):
    self.inputs = {}
    self.model_path = path
    self.use_tf8 = False
    self.session = create_ort_session(convert_fp16_to_fp32(path.replace("rknn", "onnx")))
    self.input_names = [x.name for x in self.session.get_inputs()]
    self.input_shapes = {x.name: [1, *x.shape[1:]] for x in self.session.get_inputs()}
    self.input_dtypes = {x.name: ORT_TYPES_TO_NP_TYPES[x.type] for x in self.session.get_inputs()}
    self.inited = False

  def execute(self):
  
    input_imgs = np.random.randn(1, 12, 128, 256).astype('f')
    big_input_imgs = np.random.randn(1, 12, 128, 256).astype('f')
    desire = np.random.randn(1, 100, 8).astype('f')
    traffic_convention = np.random.randn(1, 2).astype('f')
    lateral_control_params = np.random.randn(1, 2).astype('f')
    prev_desired_curv = np.random.randn(1, 100, 1).astype('f')
    features_buffer = np.random.randn(1, 99, 512).astype('f')


    try:
      onnx_inputs = {
        "input_imgs": input_imgs,
        "big_input_imgs": big_input_imgs,
        "desire": desire,
        "traffic_convention": traffic_convention,
        "lateral_control_params": lateral_control_params,
        "prev_desired_curv": prev_desired_curv,
        "features_buffer": features_buffer
      }
      onnx_inputs = {k: (v.view(np.uint8) / 255. if self.use_tf8 and k == 'input_img' else v) for k,v in onnx_inputs.items()}
      onnx_inputs = {k: v.reshape(self.input_shapes[k]).astype(self.input_dtypes[k]) for k,v in onnx_inputs.items()}
      for i in range(100):
        onnx_output = self.session.run(None, onnx_inputs)[0]
        # assert len(onnx_output) == 1, "Only single model outputs are supported"
        #print("ONNX:", onnx_output, onnx_output.shape)

    except Exception as e:
      print(str(e))

if __name__ == "__main__":
    myrknn = RKNNModel("/data/openpilot/selfdrive/modeld/models/supercombo.rknn")
    myrknn.execute()
