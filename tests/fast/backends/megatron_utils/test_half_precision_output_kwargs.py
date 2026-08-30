"""``half_precision_output_kwargs`` has to find Float16Module through the wrapper chain.

The log-prob forward is handed a DDP-wrapped model, so the flag only reaches the module that
consumes it if the helper walks ``.module`` down to it. Passing it to a model that is not wrapped
would be a TypeError, so the helper must also stay silent in that case.
"""

import torch

from miles.backends.megatron_utils.model import half_precision_output_kwargs


class _Leaf(torch.nn.Module):
    def forward(self, x):
        return x


class _Float16ModuleStub(torch.nn.Module):
    """Stands in for megatron.core.transformer.module.Float16Module."""

    def __init__(self, module):
        super().__init__()
        self.module = module


class _Wrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


def test_unwrapped_model_gets_no_kwarg():
    assert half_precision_output_kwargs(_Leaf()) == {}


def test_wrapped_but_not_half_precision_gets_no_kwarg():
    assert half_precision_output_kwargs(_Wrapper(_Wrapper(_Leaf()))) == {}


def test_float16_module_is_found_through_the_wrapper_chain(monkeypatch):
    monkeypatch.setattr(
        "miles.backends.megatron_utils.model.Float16Module", _Float16ModuleStub, raising=True
    )

    direct = _Float16ModuleStub(_Leaf())
    nested = _Wrapper(_Float16ModuleStub(_Leaf()))

    assert half_precision_output_kwargs(direct) == {"fp32_output": False}
    assert half_precision_output_kwargs(nested) == {"fp32_output": False}
