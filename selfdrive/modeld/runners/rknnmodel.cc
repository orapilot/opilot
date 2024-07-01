#include "selfdrive/modeld/runners/rknnmodel.h"
#include "rknn_api.h"

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip> 

#include "common/util.h"
#include "common/timing.h"

// 函数用于将格式从NCHW改为NHWC
// static void convertNCHWtoNHWC(float* data, float* nhwcData, int channels, int height, int width) {
//     int nchwIndex, nhwcIndex = 0;
//     for (int h = 0; h < height; ++h) {
//         for (int w = 0; w < width; ++w) {
//             for (int c = 0; c < channels; ++c) {
//                 nchwIndex = c * height * width + h * width + w;
//                 nhwcIndex = h * width * channels + w * channels + c;
//                 nhwcData[nhwcIndex] = data[nchwIndex];
//             }
//         }
//     }
// }

RKNNModel::RKNNModel(const std::string path, float *_output, size_t _output_size, int runtime, bool use_tf8, cl_context context){
    // std::cout << "------------rknn init " << path.c_str() << "-----------\n" << std::endl;
    int model_data_size = 0;
    output = _output;
    output_size = _output_size;
    model_data = load_model(path.c_str(), &model_data_size);
    int ret = rknn_init(&ctx, model_data, model_data_size, RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU, NULL);
    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
    ret = rknn_set_core_mask(ctx, core_mask);
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            exit(-1);
        }
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs) );
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    // one output
    output_size = output_attrs[0].size;

    int            input_type[io_num.n_input];
    int            input_layout[io_num.n_input];
    
    for (int i = 0; i < io_num.n_input; i++) {
        input_type[i]   = RKNN_TENSOR_FLOAT32;
        if(i == 1 || i == 0){
            input_layout[i] = RKNN_TENSOR_NHWC;
        }else{
            input_layout[i] = RKNN_TENSOR_UNDEFINED;
        }
        input_size[i]   = input_attrs[i].n_elems * sizeof(float);
        rkMap[input_attrs[i].name] = i;
        rkMapSize[input_attrs[i].name] = input_size[i];
        
    }
    memset(rk_inputs, 0, sizeof(rk_inputs));
    for (int i = 0; i < io_num.n_input; i++) {
        rk_inputs[i].index        = i;
        rk_inputs[i].pass_through = 0;
        rk_inputs[i].type         = (rknn_tensor_type)input_type[i];
        rk_inputs[i].fmt          = (rknn_tensor_format)input_layout[i];
        rk_inputs[i].size         = input_size[i];
        
    }

}

void RKNNModel::addInput(const std::string name, float *buffer, int size) {
    if (buffer != nullptr) {
        rk_inputs[rkMap[name]].buf = (void*)buffer;
    }else{
        rk_inputs[rkMap[name]].buf = NULL;
    }
}

void* RKNNModel::getCLBuffer(const std::string name) {
    return nullptr; 
}

void RKNNModel::setInputBuffer(const std::string name, float *buffer, int size){
    // std::cout << "------------rknn setinputbuffer for " << name << " -----------\n" << std::endl;

    if (buffer != nullptr) {
        // int channels = 12;
        // int height = 128;
        // int width = 256;
        // int dataSize = channels * height * width;
        // float* nhwcData = new float[dataSize];
        // convertNCHWtoNHWC(buffer, nhwcData, channels, height, width);
        rk_inputs[rkMap[name]].buf = (void*)buffer;
    }else{
        rk_inputs[rkMap[name]].buf = NULL;
    }

    


}


void RKNNModel::execute(){
    rknn_inputs_set( ctx,  io_num.n_input,  rk_inputs);
    // std::cout << "------------rknn execute -----------\n" << std::endl;
    int ret = rknn_run(ctx, NULL);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
    }
    rknn_output outputs[1]; // 只有一个输出
    memset(outputs, 0, sizeof(outputs));
    outputs[0].is_prealloc = 0;
    outputs[0].want_float = 1;
    outputs[0].index = 0;
    ret = rknn_outputs_get( ctx,  1, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
    }
    float* outBlob = (float*)outputs[0].buf;
    // std::cout << "[PYX] Pointer values:" << std::endl;
    // for (size_t i = 0; i < 10; ++i) {
    //     std::cout << outBlob[i] << " ";
    // }
    // std::cout << std::endl;
    memcpy(output, outBlob, output_size);
    ret = rknn_outputs_release( ctx,  1, outputs);
}

unsigned char *RKNNModel::load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

unsigned char *RKNNModel::load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}
