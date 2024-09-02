"""
-------------------------------------------------
   File Name:    Blocks.py
   Date:         2019/10/17
   Description:  Copy from: https://github.com/lernapparat/lernapparat
-------------------------------------------------
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from .CustomLayers import EqualizedLinear, EqualizedConv1d, BlurLayer1d, View, StddevLayer1d

class DiscriminatorTop(nn.Sequential):
    def __init__(self,
                 mbstd_group_size,
                 mbstd_num_features,
                 in_channels,
                 intermediate_channels,
                 gain, use_wscale,
                 activation_layer,
                 resolution=4,
                 in_channels2=None,
                 output_features=1,
                 last_gain=1):
        """
        :param mbstd_group_size:
        :param mbstd_num_features:
        :param in_channels:
        :param intermediate_channels:
        :param gain:
        :param use_wscale:
        :param activation_layer:
        :param resolution:
        :param in_channels2:
        :param output_features:
        :param last_gain:
        """

        layers = []
        if mbstd_group_size > 1:
            layers.append(('stddev_layer', StddevLayer1d(mbstd_group_size, mbstd_num_features)))

        if in_channels2 is None:
            in_channels2 = in_channels
        layers.append(('conv', EqualizedConv1d(in_channels + mbstd_num_features, in_channels2, kernel_size=3,
                                               gain=gain, use_wscale=use_wscale)))
        layers.append(('act0', activation_layer))
        layers.append(('view', View(-1)))
        layers.append(('dense0', EqualizedLinear(in_channels2 * resolution, intermediate_channels,
                                                 gain=gain, use_wscale=use_wscale)))
        layers.append(('act1', activation_layer))
        layers.append(('dense1', EqualizedLinear(intermediate_channels, output_features,
                                                 gain=last_gain, use_wscale=use_wscale)))

        super().__init__(OrderedDict(layers))


class DiscriminatorBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, gain, use_wscale, activation_layer, blur_kernel):
        super().__init__(OrderedDict([
            ('conv0', EqualizedConv1d(in_channels, in_channels, kernel_size=3, gain=gain, use_wscale=use_wscale)),
            # out channels nf(res-1)
            ('act0', activation_layer),
            ('blur', BlurLayer1d(kernel=blur_kernel)),
            ('conv1_down', EqualizedConv1d(in_channels, out_channels, kernel_size=3,
                                           gain=gain, use_wscale=use_wscale, downscale=True)),
            ('act1', activation_layer)]))


if __name__ == '__main__':
    # discriminator = DiscriminatorTop()
    print('Done.')