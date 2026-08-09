from __future__ import annotations

import inspect

from xenolect.compiler.install import install_real_endpoint, install_target
from xenolect.compiler.xpt_real import compile_real_endpoint
from xenolect.xpt.frontier import CERTIFICATION_GENERATION_UPPER_BOUND
from xenolect.xpt.session import Budget


def _default(function, parameter: str):
    return inspect.signature(function).parameters[parameter].default


def test_default_online_budget_and_certification_reserve_remain_frozen() -> None:
    budget = Budget()
    assert budget.max_generations == 12
    assert budget.certification_reserve == 3
    assert budget.certification_reserve == CERTIFICATION_GENERATION_UPPER_BOUND
    assert budget.max_generations - budget.certification_reserve == 9


def test_public_compile_and_install_defaults_preserve_bounded_contract() -> None:
    for function in (compile_real_endpoint, install_target, install_real_endpoint):
        assert _default(function, "deadline_s") == 300.0
        assert _default(function, "max_generations") == 12
