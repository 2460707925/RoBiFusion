#define THREAD_NUM_PER_BLOCK 1024

__global__ void generate_depth_kernel(const float *sampling_positions_x,
                                      const float *sampling_positions_y,
                                      const float *sampling_depth_features,
                                      float *updated_depth_features,
                                      int32_t *sampling_indexes,
                                      float *sampling_confs,
                                      const int32_t sampling_num,
                                      const int32_t height,
                                      const int32_t width,
                                      const int32_t depth_dim)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sampling_num)
    {
        // 获取索引值的坐标和置信度
        float position_x = (sampling_positions_x[idx] + 1.0) / 2.0 * (float)(width-1);
        float position_y = (sampling_positions_y[idx] + 1.0) / 2.0 * (float)(height-1);

        int32_t lx = __float2int_rd(position_x);
        int32_t ly = __float2int_rd(position_y);
        int32_t rx = __float2int_ru(position_x);
        int32_t ry = __float2int_ru(position_y);

        // 更新索引值
        int32_t lu_index = ly * width + lx;
        int32_t ru_index = ly * width + rx;
        int32_t ld_index = ry * width + lx;
        int32_t rd_index = ry * width + rx;

        int32_t *sampling_index=sampling_indexes+4*idx;
        sampling_index[0]=lu_index;
        sampling_index[1]=ru_index;
        sampling_index[2]=ld_index;
        sampling_index[3]=rd_index;

        // 更新置信度
        float lu_conf = ((float)rx - position_x) * ((float)ry - position_y);
        float ru_conf = (position_x - (float)lx) * ((float)ry - position_y);
        float ld_conf = ((float)rx - position_x) * (position_y - (float)ly);
        float rd_conf = (position_x - (float)lx) * (position_y - (float)ly);

        float *sampling_conf=sampling_confs+4*idx;
        sampling_conf[0]=lu_conf;
        sampling_conf[1]=ru_conf;
        sampling_conf[2]=ld_conf;
        sampling_conf[3]=rd_conf;

        const float *sampling_depth_feature = sampling_depth_features + idx * depth_dim;
        float *lu_updated_depth_feature = updated_depth_features + lu_index*depth_dim;
        float *ru_updated_depth_feature = updated_depth_features + ru_index*depth_dim;
        float *ld_updated_depth_feature = updated_depth_features + ld_index*depth_dim;
        float *rd_updated_depth_feature = updated_depth_features + rd_index*depth_dim;

        for (int i = 0; i < depth_dim; i++)
        {
            float feature_element = *(sampling_depth_feature+i);
            
            // 赋值左上角
            atomicAdd(lu_updated_depth_feature+i, lu_conf * feature_element);
            // 赋值右上角
            atomicAdd(ru_updated_depth_feature+i, ru_conf * feature_element);
            // 赋值左下角
            atomicAdd(ld_updated_depth_feature+i, ld_conf * feature_element);
            // 赋值右下角
            atomicAdd(rd_updated_depth_feature+i, rd_conf * feature_element);

        }

    }
}


__global__ void backward_generate_depth_kernel(const float *grad_output,
                                               float *grad_sampling_depth_features,
                                               const int32_t *sampling_indexes,
                                               const float *sampling_confs,
                                               const int32_t sampling_num,
                                               const int32_t depth_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sampling_num){

        const int32_t *sampling_index=sampling_indexes+4*idx;
        int lu_index=sampling_index[0];
        int ru_index=sampling_index[1];
        int ld_index=sampling_index[2];
        int rd_index=sampling_index[3];


        const float *sampling_conf=sampling_confs+4*idx;
        float lu_conf=sampling_conf[0];
        float ru_conf=sampling_conf[1];
        float ld_conf=sampling_conf[2];
        float rd_conf=sampling_conf[3];

        const float *lu_grad_output = grad_output + lu_index*depth_dim;
        const float *ru_grad_output= grad_output + ru_index*depth_dim;
        const float *ld_grad_output = grad_output + ld_index*depth_dim;
        const float *rd_grad_output = grad_output + rd_index*depth_dim;

        float *grad_sampling_depth_feature=grad_sampling_depth_features+idx*depth_dim;

        for(int i=0;i<depth_dim;i++){
            grad_sampling_depth_feature[i]=lu_conf*lu_grad_output[i]+ru_conf*ru_grad_output[i]+ld_conf*ld_grad_output[i]+rd_conf*rd_grad_output[i];
        }
    }
}

void launch_generate_depth(const float *position_x,
                           const float *position_y,
                           const float *sampling_depth_features,
                           float *updated_depth_features,
                           int32_t *sampling_indexes,
                           float *sampling_confs,
                           const int32_t sampling_num,
                           const int32_t height,
                           const int32_t width,
                           const int32_t depth_dim)
{
    dim3 grid((sampling_num + THREAD_NUM_PER_BLOCK - 1) / THREAD_NUM_PER_BLOCK);
    dim3 block(THREAD_NUM_PER_BLOCK);
    generate_depth_kernel<<<grid,block>>>(position_x,position_y,sampling_depth_features,updated_depth_features,sampling_indexes,sampling_confs,sampling_num,height,width,depth_dim);
    cudaDeviceSynchronize();
}

void launch_backward_generate_depth(const float *grad_output,
                                    float *grad_sampling_depth_features,
                                    const int32_t *sampling_indexes,
                                    const float *sampling_confs,
                                    const int32_t sampling_num,
                                    const int32_t depth_dim)
{
    dim3 grid((sampling_num + THREAD_NUM_PER_BLOCK - 1) / THREAD_NUM_PER_BLOCK);
    dim3 block(THREAD_NUM_PER_BLOCK);
    backward_generate_depth_kernel<<<grid,block>>>(grad_output,grad_sampling_depth_features,sampling_indexes,sampling_confs,sampling_num,depth_dim);
    cudaDeviceSynchronize();
}