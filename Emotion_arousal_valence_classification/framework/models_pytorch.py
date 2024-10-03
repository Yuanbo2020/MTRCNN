import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import framework.config as config


def move_data_to_gpu(x, cuda, using_float=False):
    if using_float:
        x = torch.Tensor(x)
    else:
        if 'float' in str(x.dtype):
            x = torch.Tensor(x)

        elif 'int' in str(x.dtype):
            x = torch.LongTensor(x)

        else:
            raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


# ----------------------------------------------------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_single_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):

        super(ConvBlock_single_layer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_dilation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dilation=(2,2), padding=(1, 1)):

        super(ConvBlock_dilation, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_dilation_single_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dilation=(2,2), padding=(1, 1)):

        super(ConvBlock_dilation_single_layer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


############################################################################
from framework.Yamnet_params import YAMNetParams

class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF Slim
    """
    def __init__(self, *args, **kwargs):
        # remove padding argument to avoid conflict
        padding = kwargs.pop("padding", "SAME")
        # initialize nn.Conv2d
        super().__init__(*args, **kwargs)
        self.padding = padding
        assert self.padding == "SAME"
        self.num_kernel_dims = 2
        self.forward_func = lambda input, padding: F.conv2d(
            input, self.weight, self.bias, self.stride,
            padding=padding, dilation=self.dilation, groups=self.groups,
        )

    def tf_SAME_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.kernel_size[dim]

        dilate = self.dilation
        dilate = dilate if isinstance(dilate, int) else dilate[dim]
        stride = self.stride
        stride = stride if isinstance(stride, int) else stride[dim]

        effective_kernel_size = (filter_size - 1) * dilate + 1
        out_size = (input_size + stride - 1) // stride
        total_padding = max(
            0, (out_size - 1) * stride + effective_kernel_size - input_size
        )
        total_odd = int(total_padding % 2 != 0)
        return total_odd, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return self.forward_func(input, padding=0)
        odd_1, padding_1 = self.tf_SAME_padding(input, dim=0)
        odd_2, padding_2 = self.tf_SAME_padding(input, dim=1)
        if odd_1 or odd_2:
            # NOTE: F.pad argument goes from last to first dim
            input = F.pad(input, [0, odd_2, 0, odd_1])

        return self.forward_func(
            input, padding=[ padding_1 // 2, padding_2 // 2 ]
        )


class CONV_BN_RELU(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm2d(
            conv.out_channels, eps=YAMNetParams.BATCHNORM_EPSILON
        )  # NOTE: yamnet uses an eps of 1e-4. This causes a huge difference
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv(nn.Module):
    def __init__(self, kernel, stride, input_dim, output_dim):
        super().__init__()
        self.fused = CONV_BN_RELU(
            Conv2d_tf(
                in_channels=input_dim, out_channels=output_dim,
                kernel_size=kernel, stride=stride,
                padding='SAME', bias=False
            )
        )

    def forward(self, x):
        return self.fused(x)


class SeparableConv(nn.Module):
    def __init__(self, kernel, stride, input_dim, output_dim):
        super().__init__()
        self.depthwise_conv = CONV_BN_RELU(
            Conv2d_tf(
                in_channels=input_dim, out_channels=input_dim, groups=input_dim,
                kernel_size=kernel, stride=stride,
                padding='SAME', bias=False,
            ),
        )
        self.pointwise_conv = CONV_BN_RELU(
            Conv2d_tf(
                in_channels=input_dim, out_channels=output_dim,
                kernel_size=1, stride=1,
                padding='SAME', bias=False,
            ),
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class YAMNet(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        net_configs = [
            # (layer_function, kernel, stride, num_filters)
            (Conv, [3, 3], 2, 32),
            (SeparableConv, [3, 3], 1, 64),
            (SeparableConv, [3, 3], 2, 128),
            (SeparableConv, [3, 3], 1, 128),
            (SeparableConv, [3, 3], 2, 256),
            (SeparableConv, [3, 3], 1, 256),
            (SeparableConv, [3, 3], 2, 512),
            (SeparableConv, [3, 3], 1, 512),
            (SeparableConv, [3, 3], 1, 512),
            (SeparableConv, [3, 3], 1, 512),
            (SeparableConv, [3, 3], 1, 512),
            (SeparableConv, [3, 3], 1, 512),
            (SeparableConv, [3, 3], 2, 1024),
            (SeparableConv, [3, 3], 1, 1024)
        ]

        input_dim = 1
        self.layer_names = []
        for (i, (layer_mod, kernel, stride, output_dim)) in enumerate(net_configs):
            name = 'layer{}'.format(i + 1)
            self.add_module(name, layer_mod(kernel, stride, input_dim, output_dim))
            input_dim = output_dim
            self.layer_names.append(name)

        self.bn0 = nn.BatchNorm2d(config.mel_bins)

        last_units = 1024
        self.fc_final = nn.Linear(last_units, class_num, bias=True)

    def forward(self, x):
        x = x.unsqueeze(1)  # torch.Size([64, 1, 480, 64])

        for name in self.layer_names:
            mod = getattr(self, name)
            x = mod(x)
        x = F.adaptive_avg_pool2d(x, 1)
        # print(x.size())
        x = x.reshape(x.shape[0], -1)
        # print(x.size())

        t = self.fc_final(x)

        return t


class MTRCNN(nn.Module):
    def __init__(self, class_num_total, batchnormal=True):

        super(MTRCNN, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)

        frequency_num = 6
        frequency_emb_dim = 1
        # --------------------------------------------------------------------------------------------------------
        self.conv_block1 = ConvBlock_single_layer(in_channels=1, out_channels=16)
        self.conv_block2 = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32, padding=(0,0), dilation=(2, 1))
        self.conv_block3 = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64, padding=(0,0), dilation=(3, 1))
        self.k_3_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 5
        kernel_size = (5, 5)
        self.conv_block1_kernel_5 = ConvBlock_single_layer(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=(0,2))
        self.conv_block2_kernel_5 = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 1), dilation=(2, 1))
        self.conv_block3_kernel_5 = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 1), dilation=(3, 1))
        self.k_5_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 7
        kernel_size = (7, 7)
        self.conv_block1_kernel_7 = ConvBlock_single_layer(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=(0, 3))
        self.conv_block2_kernel_7 = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 2), dilation=(2, 1))
        self.conv_block3_kernel_7 = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 2), dilation=(3, 1))
        self.k_7_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)


        scene_event_embedding_dim = 128
        # embedding layers
        self.fc_embedding_event = nn.Linear(64*3, scene_event_embedding_dim, bias=True)


        self.fc_total = nn.Linear(scene_event_embedding_dim, class_num_total, bias=True)

        ##############################################################################################################

        self.init_weight()

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)

        init_layer(self.fc_embedding_event)

    def mean_max(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        return x

    def forward(self, input):

        # torch.Size([32, 3001, 64])
        (_, seq_len, mel_bins) = input.shape
        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''


        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        batch_x = x

        # print(x.size())  # torch.Size([32, 1, 1001, 64])  (batch, channels, frames, freqs.)
        x_k_3 = self.conv_block1(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)

        x_k_3 = self.conv_block2(x_k_3, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)

        x_k_3 = self.conv_block3(x_k_3, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)

        x_k_3 = self.mean_max(x_k_3)
        x_k_3_mel = F.relu_(self.k_3_freq_to_1(x_k_3))[:, :, 0]
        # print('x_k_3_mel: ', x_k_3_mel.size())  # x_k_3_mel:  torch.Size([32, 64])

        # kernel 5 -----------------------------------------------------------------------------------------------------
        x_k_5 = self.conv_block1_kernel_5(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([8, 64, 1496, 64])

        x_k_5 = self.conv_block2_kernel_5(x_k_5, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([8, 128, 740, 52])

        x_k_5 = self.conv_block3_kernel_5(x_k_5, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size(), '\n')  # torch.Size([8, 256, 358, 32])

        x_k_5 = self.mean_max(x_k_5)  # torch.Size([8, 256, 5])
        x_k_5_mel = F.relu_(self.k_5_freq_to_1(x_k_5))[:, :, 0]
        # print('x_k_5_mel: ', x_k_5_mel.size())  torch.Size([32, 64])

        # kernel 7 -----------------------------------------------------------------------------------------------------
        x_k_7 = self.conv_block1_kernel_7(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([8, 64, 1494, 64])

        x_k_7 = self.conv_block2_kernel_7(x_k_7, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([8, 128, 735, 48])

        x_k_7 = self.conv_block3_kernel_7(x_k_7, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size(), '\n')  # torch.Size([8, 256, 349, 20])

        x_k_7 = self.mean_max(x_k_7)  # torch.Size([8, 256, 5])
        x_k_7_mel = F.relu_(self.k_7_freq_to_1(x_k_7))[:, :, 0]

        event_embs_log_mel = torch.cat([x_k_3_mel, x_k_5_mel,
                                        x_k_7_mel], dim=-1)
        # print(event_embs_log_mel.size())  # torch.Size([32, 64*4])  (node_num, batch, edge_dim)

        # -------------------------------------------------------------------------------------------------------------
        event_embeddings = F.gelu(self.fc_embedding_event(event_embs_log_mel))
        # -------------------------------------------------------------------------------------------------------------

        total = self.fc_total(event_embeddings)

        return total


#################################### mha ########################################################
import numpy as np
# transformer
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_heads = 8  # number of heads in Multi-Head Attention

class ScaledDotProductAttention_nomask(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention_nomask, self).__init__()

    def forward(self, Q, K, V, d_k=d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention_nomask(nn.Module):
    def __init__(self, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads,
                 output_dim=d_model):
        super(MultiHeadAttention_nomask, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.layernorm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_heads * d_v, output_dim)

    def forward(self, Q, K, V, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        context, attn = ScaledDotProductAttention_nomask()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        x = self.layernorm(output + residual)
        return x, attn


class EncoderLayer(nn.Module):
    def __init__(self, output_dim=d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention_nomask(output_dim=output_dim)

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, input_dim, n_layers, output_dim=d_model):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(output_dim) for _ in range(n_layers)])
        self.mel_projection = nn.Linear(input_dim, d_model)

    def forward(self, enc_inputs):
        # print(enc_inputs.size())  # torch.Size([64, 54, 8, 8])
        size = enc_inputs.size()
        enc_inputs = enc_inputs.reshape(size[0], size[1], -1)
        enc_outputs = self.mel_projection(enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k=d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
#################################################################################################


class CNN_Transformer(nn.Module):
    def __init__(self, class_num):

        super(CNN_Transformer, self).__init__()

        out_channels = 64
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 128
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 256
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        d_model = 512
        self.mha = Encoder(input_dim=256, n_layers=1, output_dim=d_model)

        last_units = 5120
        self.fc_final = nn.Linear(last_units, class_num, bias=True)


    def forward(self, input):
        # print(input.shape)
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (16, 481, 64)

        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        # print(x.size())  # torch.Size([64, 1, 480, 64])
        x = F.relu_(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(4, 3))
        # print(x.size())  # torch.Size([64, 32, 95, 12])

        x = F.relu_(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(4, 3))
        # print(x.size())  # torch.Size([64, 64, 18, 1])

        x = F.relu_(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=(4, 3))
        # print(x.size())  # torch.Size([64, 256, 6, 1])

        x = x.transpose(1, 2)  # torch.Size([64, 6, 256, 1])
        x, x_self_attns = self.mha(x)  # already have reshape
        # print(x_event.size())  # torch.Size([64, 6, 512])

        x = torch.flatten(x, start_dim=1)

        t = self.fc_final(x)

        return t



################################################################################
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            _layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            self.conv = _layers
        else:
            _layers = [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[1])
            init_layer(_layers[3])
            init_bn(_layers[5])
            init_layer(_layers[7])
            init_bn(_layers[8])
            self.conv = _layers

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, class_num):

        super(MobileNetV2, self).__init__()


        width_mult = 1.
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 2],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers

        def conv_1x1_bn(inp, oup):
            _layers = nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
            init_layer(_layers[0])
            init_bn(_layers[1])
            return _layers

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        d_embeddings = 1280
        self.fc_final = nn.Linear(d_embeddings, class_num, bias=True)


    def forward(self, input):
        """
        Input: (batch_size, data_length)"""
        x = input.unsqueeze(1)  # torch.Size([64, 1, 480, 64])


        x = self.features(x)

        x = torch.mean(x, dim=3)

        x = torch.mean(x, dim=2)

        t = self.fc_final(x)

        return t


class PANN(nn.Module):
    def __init__(self, class_num, batchnormal=False):

        super(PANN, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)
            # self.bn0_loudness = nn.BatchNorm2d(1)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        # self.fc1 = nn.Linear(2048, 2048, bias=True)

        last_units = 2048
        self.fc1 = nn.Linear(last_units, last_units, bias=True)

        # # ------------------- classification layer -----------------------------------------------------------------
        self.fc_final_event = nn.Linear(last_units, class_num, bias=True)


        # ##############################################################################################################

    def mean_max(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        return x

    def forward(self, input):

        # torch.Size([32, 3001, 64])
        (_, seq_len, mel_bins) = input.shape
        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        # -------------------------------------------------------------------------------------------------------------
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())
        # x.size(): torch.Size([64, 64, 1500, 32])
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size()) # x.size(): torch.Size([64, 128, 750, 16])
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([64, 256, 375, 8])
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([64, 512, 187, 4])
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([64, 1024, 93, 2])
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([64, 2048, 93, 2])
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)

        # print('6x.size():', x.size())  torch.Size([64, 2048, 93])

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2  # torch.Size([64, 2048])

        common_embeddings = F.relu_(self.fc1(x))

        event = self.fc_final_event(common_embeddings)

        return event



