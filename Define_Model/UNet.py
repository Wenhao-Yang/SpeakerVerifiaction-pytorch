#!/usr/bin/env python
# encoding: utf-8
'''
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: VS Code
@File: UNet.py
@Time: 2023/09/20 17:33
@Overview: 
'''

import torch
import torch.nn as nn


class DoubleConv2(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, 
                 kernel_size=3,
                 activation='relu', short_connection=False, 
                **kwargs):
        super(DoubleConv2, self).__init__()

        activation_type = {'relu':  nn.ReLU,
                           'leaky': nn.LeakyReLU}
        padding = (kernel_size-1) // 2
        activation_func = activation_type[activation]

        self.short_connection = short_connection

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, stride=1, padding=padding, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            activation_func(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=1, padding=padding, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            activation_func()
        )

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
                nn.BatchNorm2d(out_channels),      
            )
        else:
            self.downsample = None


        # self.last_activation = last_activation
        # self.last_activation_func = 

    def forward(self, x):
        identity = x

        x_1 = self.layer1(x)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        if self.short_connection:
            x_1 = x_1 + identity

        # if self.last_activation:
        #     x_1 = self.last_activation_func(x_1)

        return x_1
    

class TripleConv2(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, 
                 kernel_size=3,
                 activation='relu', short_connection=False, 
                **kwargs):
        super(TripleConv2, self).__init__()

        activation_type = {'relu':  nn.ReLU,
                           'leaky': nn.LeakyReLU}
        padding = (kernel_size-1) // 2
        activation_func = activation_type[activation]
        self.short_connection = short_connection

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, stride=1, padding=padding, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            activation_func(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            activation_func(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=1, padding=padding, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            activation_func()
        )

        if in_channels != out_channels and short_connection:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),      
            )
        else:
            self.downsample = None

        # self.last_activation = last_activation
        # self.last_activation_func = 

    def forward(self, x):
        identity = x

        x_1 = self.layer1(x)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        if self.short_connection:
            x_1 = x_1 + identity

        # if self.last_activation:
        #     x_1 = self.last_activation_func(x_1)

        return x_1


class UNet(nn.Module):
    def __init__(self, channels=16, activation='relu',
                 short_connection=False, block_type='double', 
                 depth=3, **kwargs):
        
        super(UNet, self).__init__()
        self.depth = depth
        # channels = [32, 64, 128, 256]
        channels_type = {
            16: [16, 32, 64, 128, 256],
            32: [32, 64, 128, 256, 512],
            64: [64, 128, 256, 512, 1024]
        }

        activation_type = {'relu':  nn.ReLU,
                           'leaky': nn.LeakyReLU}
        
        activation_func = activation_type[activation]

        channels = channels_type[channels]
        short_connection = short_connection

        if block_type == 'double':
            block = DoubleConv2

        elif block_type == 'triple':
            block = TripleConv2
        
        self.layer1 = block(in_channels=1, out_channels=channels[0],
                                  activation=activation, short_connection=short_connection,
                                  )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            block(in_channels=channels[0], out_channels=channels[1],
                                  activation=activation, short_connection=short_connection,
                                  )
        )

        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            block(in_channels=channels[1], out_channels=channels[2],
                                  activation=activation, short_connection=short_connection,
                                  )
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            block(in_channels=channels[2], out_channels=channels[3],
                                  activation=activation, short_connection=short_connection,
                                  )
        )

        if self.depth > 3:
            self.layer4_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            block(in_channels=channels[3], out_channels=channels[4],
                                  activation=activation, short_connection=short_connection,
                                  )
            )

            self.layer5_0 = nn.Sequential(
            block(in_channels=channels[4], out_channels=channels[4],
                                  activation=activation, short_connection=short_connection,
                                  ),
            nn.ConvTranspose2d(in_channels=channels[4], out_channels=channels[3],
                               kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(channels[3]),
            activation_func()
            )

            self.layer5 = nn.Sequential(
            block(in_channels=channels[4], out_channels=channels[3],
                                  activation=activation, short_connection=short_connection,
                                  ),
            nn.ConvTranspose2d(in_channels=channels[3], out_channels=channels[2],
                               kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(channels[2]),
            activation_func()
            )
        else:
            self.layer5 = nn.Sequential(
                block(in_channels=channels[3], out_channels=channels[3],
                                    activation=activation, short_connection=short_connection,
                                    ),
                nn.ConvTranspose2d(in_channels=channels[3], out_channels=channels[2],
                                kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(channels[2]),
                activation_func()
            )

        self.layer6 = nn.Sequential(
            block(in_channels=channels[3], out_channels=channels[2],
                                  activation=activation, short_connection=short_connection,
                                  ),
            nn.ConvTranspose2d(in_channels=channels[2], out_channels=channels[1],
                               kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(channels[1]),
            activation_func(),
        )

        self.layer7 = nn.Sequential(
            block(in_channels=channels[2], out_channels=channels[1],
                                  activation=activation, short_connection=short_connection,
                                  ),
            nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[0],
                               kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(channels[0]),
            activation_func(),
        )

        self.layer8 = nn.Sequential(
            block(in_channels=channels[1], out_channels=channels[0],
                                  activation=activation, short_connection=short_connection,
                                  ),
            nn.Conv2d(in_channels=channels[0], out_channels=1,
                      kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            # nn.ReLU(),
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         if activation == 'relu':
        #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         else:
        #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                # nn.init.normal_(m.weight, mean=0., std=1.)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        
        x_1 = self.layer1(x)
        # print(x_1.shape)

        x_2 = self.layer2(x_1)
        # print(x_2.shape)

        x_3 = self.layer3(x_2)
        # print(x_3.shape)

        x_4 = self.layer4(x_3)
        # print(x_4.shape)

        if self.depth > 3:
            x_41 = self.layer4_2(x_4)
            x_50 = self.layer5_0(x_41)

            x_5 = self.layer5(torch.cat([x_50, x_4], dim=1))

        else:
            x_5 = self.layer5(x_4)
        # print(x_5.shape)
        
        
        x_6 = self.layer6(torch.cat([x_5, x_3], dim=1))
        # print(x_6.shape)

        x_7 = self.layer7(torch.cat([x_6, x_2], dim=1))
        # print(x_7.shape)

        x_8 = self.layer8(torch.cat([x_7, x_1], dim=1))
        # print(x_8.shape)
        
        
        return x_8
