import torch
import torch.nn as nn
import brevitas.nn as qnn
from dependencies import value
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType, RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.stats import StatsOp
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import WeightQuantSolver


class _CommonQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT


class _WeightQuant(_CommonQuant, WeightQuantSolver):
    scaling_per_output_channel = False  # overridden per instance
    scaling_stats_op = StatsOp.AVE  # mean(|W|) per channel — XNOR-net style

    @value
    def scaling_impl_type(quant_type):
        # Binary weights: XNOR-net per-channel scale = mean(|W_c|)
        # INT weights: learned per-channel or per-tensor scale
        if quant_type == QuantType.BINARY:
            return ScalingImplType.STATS
        else:
            return ScalingImplType.PARAMETER

    scaling_const = 1.0  # unused, kept for safety



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

        # Build per-instance weight quantizer class with the requested bit_width,
        # narrow_range and per-channel scaling baked in. quant_type is
        # resolved automatically: BINARY at 1 bit, INT otherwise.
        weight_quant = type(
            '_WeightQuantInst',
            (_WeightQuant,),
            dict(
                bit_width=weight_bit_width,
                narrow_range=narrow_range,
                scaling_per_output_channel=per_channel_scaling,
            ))

        # Optional input quantization
        if quantize_input:
            self.input_quant = qnn.QuantIdentity(
                bit_width=act_bit_width, narrow_range=narrow_range, return_quant_tensor=True)
        else:
            self.input_quant = nn.Identity()

        # Microcontroller suitable architecture: small number of channels, aggressive pooling
        self.layer1 = nn.Sequential(
            qnn.QuantConv2d(1, c1, kernel_size=3, stride=1, padding=1,
                            weight_quant=weight_quant,
                            bias=use_bias),
            nn.MaxPool2d(kernel_size=2),
            qnn.QuantReLU(bit_width=act_bit_width, narrow_range=narrow_range),
        )
        self.layer2 = nn.Sequential(
            qnn.QuantConv2d(c1, c2, kernel_size=3, stride=1, padding=1,
                            weight_quant=weight_quant,
                            bias=use_bias),
            nn.MaxPool2d(kernel_size=2),
            qnn.QuantReLU(bit_width=act_bit_width, narrow_range=narrow_range),
        )
        self.layer3 = nn.Sequential(
            qnn.QuantConv2d(c2, c3, kernel_size=3, stride=1, padding=1,
                            weight_quant=weight_quant,
                            bias=use_bias),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            qnn.QuantReLU(bit_width=act_bit_width, narrow_range=narrow_range),
        )

        # 28x28 -> 14x14 -> 7x7 -> 3x3
        self.fc = nn.Sequential(
            qnn.QuantLinear(c3 * 3 * 3, fc_hidden,
                            weight_quant=weight_quant,
                            bias=use_bias),
            qnn.QuantReLU(bit_width=act_bit_width, narrow_range=narrow_range),
            qnn.QuantLinear(fc_hidden, num_classes,
                            weight_quant=weight_quant,
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
