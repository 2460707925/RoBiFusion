#include <torch/extension.h>

void launch_generate_depth(const float *position_x,
                           const float *position_y,
                           const float *sampling_depth_features,
                           float *updated_depth_features,
                           int32_t *sampling_indexes,
                           float *sampling_confs,
                           const int32_t sampling_num,
                           const int32_t height,
                           const int32_t width,
                           const int32_t depth_dim);

void launch_backward_generate_depth(const float *grad_output,
                                    float *grad_sampling_depth_features,
                                    const int32_t *sampling_indexes,
                                    const float *sampling_confs,
                                    const int32_t sampling_num,
                                    const int32_t depth_dim);

void torch_launch_generate_depth(torch::Tensor &position_x,
                                 torch::Tensor &position_y,
                                 torch::Tensor &sampling_depth_features,
                                 torch::Tensor &updated_depth_features,
                                 torch::Tensor &sampling_indexes,
                                 torch::Tensor &sampling_confs,
                                 torch::Tensor &shape_value)
{
    const int sampling_num=shape_value[0].item<int>();
    const int height=shape_value[1].item<int>();
    const int width=shape_value[2].item<int>();
    const int depth_dim=shape_value[3].item<int>();
    launch_generate_depth((float*)position_x.data_ptr(),(float*)position_y.data_ptr(),
                          (float*)sampling_depth_features.data_ptr(),(float*)updated_depth_features.data_ptr(),
                          (int32_t*)sampling_indexes.data_ptr(),(float*)sampling_confs.data_ptr(),sampling_num,height,width,depth_dim);
    
}

void torch_backward_generate_depth(torch::Tensor &grad_output,
                                   torch::Tensor &grad_sampling_depth_features,
                                   torch::Tensor &sampling_indexes,
                                   torch::Tensor &sampling_confs,
                                   torch::Tensor &shape_value)
{
    const int sampling_num=shape_value[0].item<int>();
    const int depth_dim=shape_value[3].item<int>();
    launch_backward_generate_depth((float*)grad_output.data_ptr(),(float*)grad_sampling_depth_features.data_ptr(),
                                   (int32_t*)sampling_indexes.data_ptr(),(float*)sampling_confs.data_ptr(),sampling_num,depth_dim);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_generate_depth",
          &torch_launch_generate_depth,
          "launch generate depth");
    m.def("torch_backward_generate_depth",
         &torch_backward_generate_depth,
         "backward generate depth");
}


