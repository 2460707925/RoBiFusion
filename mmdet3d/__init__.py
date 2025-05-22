import mmcv

import mmdet
from .version import __version__, short_version


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


mmcv_minimum_version = '1.2.4'
mmcv_maximum_version = '1.4.0'
mmcv_version = digit_version(mmcv.__version__)



mmdet_minimum_version = '2.5.0'
mmdet_maximum_version = '3.0.0'
mmdet_version = digit_version(mmdet.__version__)

__all__ = ['__version__', 'short_version']
