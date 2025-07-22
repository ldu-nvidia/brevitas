# brevitas/quant/experimental/fp4.py
from brevitas.quant.experimental.float_base import (
    Fp4e2m1Mixin,
    ScaledFloatWeightBase,
    ScaledFloatActBase,
)

class Fp4WeightPerTensorFloat(Fp4e2m1Mixin, ScaledFloatWeightBase):
    """Per-tensor 4-bit floating-point weight quantiser (E2-M1)."""
    pass

class Fp4ActPerTensorFloat(Fp4e2m1Mixin, ScaledFloatActBase):
    """Per-tensor 4-bit floating-point activation quantiser (E2-M1)."""
    pass