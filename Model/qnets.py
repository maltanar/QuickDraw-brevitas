import torch
import torch.nn as nn
import brevitas.nn as qnn

class QuantTinyCNN(nn.Module):
    def __init__(self, num_classes, weight_bit_width, act_bit_width, per_channel_scaling=False, quantize_input=False, narrow_range=True, use_bias=False, channel_scale=1.0):
        super(QuantTinyCNN, self).__init__()

        if channel_scale <= 0:
            raise ValueError("channel_scale must be > 0")

        def scaled_channels(channels):
            return max(1, int(round(channels * channel_scale)))

        c1 = scaled_channels(16)
        c2 = scaled_channels(32)
        c3 = scaled_channels(32)
        fc_hidden = scaled_channels(64)
        
        # Optional input quantization (binarization if bit_width is low)
        if quantize_input:
            self.input_quant = qnn.QuantIdentity(bit_width=act_bit_width, narrow_range=narrow_range, return_quant_tensor=True)
        else:
            self.input_quant = nn.Identity()

        # Microcontroller suitable architecture: small number of channels, aggressive pooling
        self.layer1 = nn.Sequential(
            qnn.QuantConv2d(1, c1, kernel_size=3, stride=1, padding=1, 
                            weight_bit_width=weight_bit_width, 
                            weight_scaling_per_output_channel=per_channel_scaling,
                            weight_narrow_range=narrow_range,
                            bias=use_bias),
            nn.MaxPool2d(kernel_size=2),
            qnn.QuantReLU(bit_width=act_bit_width, narrow_range=narrow_range),
        )
        self.layer2 = nn.Sequential(
            qnn.QuantConv2d(c1, c2, kernel_size=3, stride=1, padding=1, 
                            weight_bit_width=weight_bit_width,
                            weight_scaling_per_output_channel=per_channel_scaling,
                            weight_narrow_range=narrow_range,
                            bias=use_bias),
            nn.MaxPool2d(kernel_size=2),
            qnn.QuantReLU(bit_width=act_bit_width, narrow_range=narrow_range),
        )
        self.layer3 = nn.Sequential(
            qnn.QuantConv2d(c2, c3, kernel_size=3, stride=1, padding=1, 
                            weight_bit_width=weight_bit_width,
                            weight_scaling_per_output_channel=per_channel_scaling,
                            weight_narrow_range=narrow_range,
                            bias=use_bias),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            qnn.QuantReLU(bit_width=act_bit_width, narrow_range=narrow_range),
        )
        
        # 28x28 -> 14x14 -> 7x7 -> 3x3
        self.fc = nn.Sequential(
            qnn.QuantLinear(c3 * 3 * 3, fc_hidden, weight_bit_width=weight_bit_width,
                            weight_scaling_per_output_channel=per_channel_scaling,
                            weight_narrow_range=narrow_range,
                            bias=use_bias),
            qnn.QuantReLU(bit_width=act_bit_width, narrow_range=narrow_range),
            qnn.QuantLinear(fc_hidden, num_classes, weight_bit_width=weight_bit_width,
                            weight_scaling_per_output_channel=per_channel_scaling,
                            weight_narrow_range=narrow_range,
                            bias=use_bias)
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x

def qtinycnn(num_classes, weight_bit_width, act_bit_width, per_channel_scaling=False, quantize_input=False, narrow_range=True, use_bias=False, channel_scale=1.0):
    return QuantTinyCNN(num_classes, weight_bit_width, act_bit_width, per_channel_scaling, quantize_input, narrow_range, use_bias, channel_scale)
