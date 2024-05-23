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
import numpy as np
import torch.nn.functional as F

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


class DownSample(nn.Module):
    def __init__(self, type='max', **kwargs):
        super(DownSample, self).__init__()

        self.type = type

        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpooling = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        max_x = self.maxpooling(x)

        if self.type == 'both':
            max_x = 0.5*max_x + 0.5*self.avgpooling(x)
        elif self.type == 'avg':
            max_x = self.avgpooling(x)

        return max_x


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class TransformerEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec=512, n_layers=2, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=2048, dropout=0.1, n_position=624, scale_emb=False):

        super().__init__()

        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        if n_position > 0:
            self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        else:
            self.position_enc = lambda x: x
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        # enc_output = self.src_word_emb(src_seq)
        enc_output = src_seq
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        
        return enc_output


class AttentionEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, in_channels=512, n_layers=2, n_head=8, d_k=64, d_v=64,
            out_channels=512, d_inner=2048, dropout=0.1, n_position=624, scale_emb=False):

        super(AttentionEncoder, self).__init__()
        self.encoder = TransformerEncoder(d_word_vec=in_channels, d_model=out_channels,
                                          n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                                          d_inner=d_inner, dropout=dropout, n_position=n_position,
                                          scale_emb=scale_emb)
        
    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(x_shape[0], x_shape[1], -1)

        len_s = x.shape[-1]  # length at bottleneck
        attn_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=x.device), diagonal=1)).bool()

        x = x.permute(0, 2, 1)
        x = self.encoder(x, attn_mask)
        x = x.permute(0, 2, 1)

        x = x.reshape(x_shape[0], x_shape[1], x_shape[2], x_shape[3])

        return x


class UNet(nn.Module):
    def __init__(self, channels=16, activation='relu',
                 short_connection=False, block_type='double', 
                 downsample='max', attention=True,
                 depth=3, **kwargs):
        
        super(UNet, self).__init__()
        self.depth = depth
        self.downsample = downsample
        self.attention = attention
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
            DownSample(type=downsample),
            block(in_channels=channels[0], out_channels=channels[1],
                                  activation=activation, short_connection=short_connection,
                                  )
        )

        self.layer3 = nn.Sequential(
            DownSample(type=downsample),
            block(in_channels=channels[1], out_channels=channels[2],
                                  activation=activation, short_connection=short_connection,
                                  )
        )

        self.layer4 = nn.Sequential(
            DownSample(type=downsample),
            block(in_channels=channels[2], out_channels=channels[3],
                                  activation=activation, short_connection=short_connection,
                                  )
        )

        if self.depth > 3:
            self.layer4_2 = nn.Sequential(
            DownSample(type=downsample),
            block(in_channels=channels[3], out_channels=channels[4],
                                  activation=activation, short_connection=short_connection,
                                  )
            )
            hidden_size = channels[4]

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
            hidden_size = channels[3]

            self.layer5 = nn.Sequential(
                block(in_channels=channels[3], out_channels=channels[3],
                                    activation=activation, short_connection=short_connection,
                                    ),
                nn.ConvTranspose2d(in_channels=channels[3], out_channels=channels[2],
                                kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(channels[2]),
                activation_func()
            )

        if self.attention:
            self.attention_block = AttentionEncoder(in_channels=hidden_size, out_channels=hidden_size)
        else:
            self.attention_block = None

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
            if self.attention_block != None:
                x_41 = self.attention_block(x_41)

            x_50 = self.layer5_0(x_41)

            x_5 = self.layer5(torch.cat([x_50, x_4], dim=1))
        else:
            if self.attention_block != None:
                x_4 = self.attention_block(x_4)

            x_5 = self.layer5(x_4)
        # print(x_5.shape)
        
        
        x_6 = self.layer6(torch.cat([x_5, x_3], dim=1))
        # print(x_6.shape)

        x_7 = self.layer7(torch.cat([x_6, x_2], dim=1))
        # print(x_7.shape)

        x_8 = self.layer8(torch.cat([x_7, x_1], dim=1))
        # print(x_8.shape)
        
        
        return x_8
