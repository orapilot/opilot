#pragma once

#include <string>

#include "selfdrive/modeld/runners/runmodel.h"
#include "rknn_api.h"
#include <map>

class RKNNModel : public RunModel {
public:
  RKNNModel(const std::string path, float *_output, size_t _output_size, int runtime, bool use_tf8 = false, cl_context context = NULL);
  void addInput(const std::string name, float *buffer, int size);
  void setInputBuffer(const std::string name, float *buffer, int size);
  void* getCLBuffer(const std::string name);
  void execute();
  unsigned char *load_model(const char *filename, int *model_size);
  unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);

private:
  unsigned char *model_data;
  rknn_context ctx;
  rknn_input_output_num io_num;
  rknn_input rk_inputs[7];
  rknn_tensor_attr input_attrs[7];
  std::map<std::string, int> rkMap;
  std::map<std::string, int> rkMapSize;
  int            input_size[7];
  float *output;
  size_t output_size;
};
