from setuptools import find_packages, setup

import os
import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)




if __name__ == '__main__':
    setup(
        name='mmdet3d',
        packages=find_packages(),
        include_package_data=True,
        package_data={"mmdet3d.ops": ["*/*.so"]},
        classifiers=[
            'Programming Language :: Python :: 3',
            'Operating System :: OS Independent',
        ],
        install_requires=[],  
        ext_modules=[
            make_cuda_ext(
                name='sparse_conv_ext',
                module='mmdet3d.ops.spconv',
                extra_include_path=[
                    # PyTorch 1.5 uses ninjia, which requires absolute path
                    # of included files, relative path will cause failure.
                    os.path.abspath(
                        os.path.join(*'mmdet3d.ops.spconv'.split('.'),
                                     'include/'))
                ],
                sources=[
                    'src/all.cc',
                    'src/reordering.cc',
                    'src/reordering_cuda.cu',
                    'src/indice.cc',
                    'src/indice_cuda.cu',
                    'src/maxpool.cc',
                    'src/maxpool_cuda.cu',
                ],
                extra_args=['-w', '-std=c++14']),
            make_cuda_ext(
                name='iou3d_cuda',
                module='mmdet3d.ops.iou3d',
                sources=[
                    'src/iou3d.cpp',
                    'src/iou3d_kernel.cu',
                ]),
            make_cuda_ext(
                name='voxel_layer',
                module='mmdet3d.ops.voxel',
                sources=[
                    'src/voxelization.cpp',
                    'src/scatter_points_cpu.cpp',
                    'src/scatter_points_cuda.cu',
                    'src/voxelization_cpu.cpp',
                    'src/voxelization_cuda.cu',
                ]),
            make_cuda_ext(
                name='roiaware_pool3d_ext',
                module='mmdet3d.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/points_in_boxes_cpu.cpp',
                ],
                sources_cuda=[
                    'src/roiaware_pool3d_kernel.cu',
                    'src/points_in_boxes_cuda.cu',
                ]),
            make_cuda_ext(
                name='ball_query_ext',
                module='mmdet3d.ops.ball_query',
                sources=['src/ball_query.cpp'],
                sources_cuda=['src/ball_query_cuda.cu']),
            make_cuda_ext(
                name='knn_ext',
                module='mmdet3d.ops.knn',
                sources=['src/knn.cpp'],
                sources_cuda=['src/knn_cuda.cu']),
            make_cuda_ext(
                name='group_points_ext',
                module='mmdet3d.ops.group_points',
                sources=['src/group_points.cpp'],
                sources_cuda=['src/group_points_cuda.cu']),
            make_cuda_ext(
                name='interpolate_ext',
                module='mmdet3d.ops.interpolate',
                sources=['src/interpolate.cpp'],
                sources_cuda=[
                    'src/three_interpolate_cuda.cu', 'src/three_nn_cuda.cu'
                ]),
            make_cuda_ext(
                name='furthest_point_sample_ext',
                module='mmdet3d.ops.furthest_point_sample',
                sources=['src/furthest_point_sample.cpp'],
                sources_cuda=['src/furthest_point_sample_cuda.cu']),
            make_cuda_ext(
                name='gather_points_ext',
                module='mmdet3d.ops.gather_points',
                sources=['src/gather_points.cpp'],
                sources_cuda=['src/gather_points_cuda.cu']),
            make_cuda_ext(
                name="bev_pool_ext",
                module="mmdet3d.ops.bev_pool",
                sources=[
                    "src/bev_pool.cpp",
                    "src/bev_pool_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name='depth_generate_ext',
                module='mmdet3d.ops.depth_generate',
                sources=['src/generate_depth_pts.cpp'],
                sources_cuda=['src/generate_depth_pts_cuda.cu'],
            ),
            make_cuda_ext(
                name="iou3d_ext",
                module="mmdet3d.ops.iou3d",
                sources=['src/iou3d.cpp'],
                sources_cuda=['src/iou3d_kernel.cu'],
            ),
            make_cuda_ext(
                name="voxel_pooling_inference_ext",
                module="mmdet3d.ops.voxel_pooling_inference",
                sources=['src/voxel_pooling_inference_forward.cpp'],
                sources_cuda=['src/voxel_pooling_inference_forward_cuda.cu']
            ),
            make_cuda_ext(
                name="voxel_pooling_train_ext",
                module="mmdet3d.ops.voxel_pooling_train",
                sources=["src/voxel_pooling_train_forward.cpp"],
                sources_cuda=["src/voxel_pooling_train_forward_cuda.cu"]
            )
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
