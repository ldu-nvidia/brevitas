# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.function_wrapper import FloatClamp
from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.quant.float import FloatQuant
from brevitas.core.scaling.float_scaling import FloatScaling
from brevitas.inject import ExtendedInjector
from brevitas.inject import value
from brevitas.proxy import ActFloatQuantProxyFromInjector
from brevitas.proxy import WeightFloatQuantProxyFromInjector
from brevitas.quant.solver import ActQuantSolver
from brevitas.quant.solver import WeightQuantSolver
from brevitas.quant.solver.common import SolveTensorQuantFloatToIntImplFromEnum


class FloatBase(SolveTensorQuantFloatToIntImplFromEnum):
    tensor_quant = FloatQuant
    signed = True
    float_to_int_impl_type = 'round'
    scaling_min_val = 1e-10
    float_clamp_impl = FloatClamp
    tensor_clamp_impl = TensorClamp

    @value
    def exponent_bias(exponent_bit_width):
        return 2 ** (exponent_bit_width - 1) - 1


class FloatWeightBase(FloatBase, WeightQuantSolver):
    proxy_class = WeightFloatQuantProxyFromInjector


class FloatActBase(FloatBase, ActQuantSolver):
    proxy_class = ActFloatQuantProxyFromInjector


class ScaledFloatWeightBase(FloatWeightBase, WeightQuantSolver):
    scaling_stats_op = 'max'
    scaling_impl_type = 'stats'
    restrict_scaling_type = 'fp'
    float_scaling_impl = FloatScaling


class ScaledFloatActBase(FloatActBase, ActQuantSolver):
    scaling_stats_op = 'max'
    scaling_impl_type = 'parameter_from_stats'
    restrict_scaling_type = 'fp'
    collect_stats_steps = 300
    float_scaling_impl = FloatScaling


class Fp8e4m3Mixin(ExtendedInjector):
    bit_width = 8
    exponent_bit_width = 4
    mantissa_bit_width = 3
    saturating = True


class Fp8e5m2Mixin(ExtendedInjector):
    bit_width = 8
    exponent_bit_width = 5
    mantissa_bit_width = 2
    saturating = True


class Fp6e3m2Mixin(ExtendedInjector):
    bit_width = 6
    exponent_bit_width = 3
    mantissa_bit_width = 2
    saturating = True


class Fp6e2m3Mixin(ExtendedInjector):
    bit_width = 6
    exponent_bit_width = 2
    mantissa_bit_width = 3
    saturating = True


class Fp4e2m1Mixin(ExtendedInjector):
    bit_width = 4
    exponent_bit_width = 2
    mantissa_bit_width = 1
    saturating = True

# ---------------------------------------------------------------------
#  FP-format wrappers not shipped by default
# ---------------------------------------------------------------------
#
# The mix-ins above encode the numerical format (bit-width, exponent,
# mantissa) while the *Base* classes implement scale learning, STE, etc.
# Combining the two yields ready-to-use quantisers that can be passed to
# QuantConv2d / QuantIdentity.
#
# FP4  (e2m1)  – 4-bit float
# FP6a (e3m2)  – 6-bit float
# FP6b (e2m3)  – 6-bit float
# ---------------------------------------------------------------------

class Fp4WeightPerTensorFloat(Fp4e2m1Mixin, ScaledFloatWeightBase):
    """Per-tensor 4-bit (e2m1) floating-point **weight** quantiser."""
    pass


class Fp4ActPerTensorFloat(Fp4e2m1Mixin, ScaledFloatActBase):
    """Per-tensor 4-bit (e2m1) floating-point **activation** quantiser."""
    pass


class Fp6e3m2WeightPerTensorFloat(Fp6e3m2Mixin, ScaledFloatWeightBase):
    """Per-tensor 6-bit (e3m2) floating-point weight quantiser."""
    pass


class Fp6e3m2ActPerTensorFloat(Fp6e3m2Mixin, ScaledFloatActBase):
    """Per-tensor 6-bit (e3m2) floating-point activation quantiser."""
    pass


class Fp6e2m3WeightPerTensorFloat(Fp6e2m3Mixin, ScaledFloatWeightBase):
    """Per-tensor 6-bit (e2m3) floating-point weight quantiser."""
    pass


class Fp6e2m3ActPerTensorFloat(Fp6e2m3Mixin, ScaledFloatActBase):
    """Per-tensor 6-bit (e2m3) floating-point activation quantiser."""
    pass
