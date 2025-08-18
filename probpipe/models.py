from typing import Callable, List, TypeVar, Mapping, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
import pymc as pm
import scipy.stats as sp
from scipy.special import logsumexp
from prefect import flow, task, unmapped
from numpy.typing import NDArray
from .distributions import Distribution, NormalDistribution, BootstrapDistribution
import pytensor.tensor as pt

T=TypeVar("T")



class InputSpec:
    def __init__(self, name: str, dtype: type, ndim: Optional[int], constraints: Optional[tuple]):
        self.name = name
        self.dtype = dtype
        self.ndim = ndim
        self.constraints = constraints
        
        self.var = f"{name}_placeholder"

    def __str__(self):
        return f"<InputSpec: {self.name} (dtype={self.dtype.__name__}, ndim={self.ndim})>"
    
    def __call__(self, *args, **kwargs):
        return self.var




class Workflow:
    def __init__(self):
        self.inputs: Dict[str, InputSpec] = {}
        self._defaults: Dict[str, Any] = {}

        self._run_func: Optional[Callable] = None

        self._prefect_task: Optional[Callable] = None
        self._prefect_flow: Optional[Callable] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def Input(self, name: str, dtype: type, ndim: Optional[int] = None, constraints: Optional[tuple] = None):
        if name is None:
            raise ValueError("Input name cannot be None")
        if name in self.inputs:
            raise ValueError(f"Input '{name}' already defined")
        inp = InputSpec(name, dtype, ndim, constraints)
        self.inputs[name] = inp
        return inp.var

    def instantiate(self, **input_values):
        self._defaults.update(input_values)

    def run_decorator(self, func: Callable = None, *, as_task: bool = False):
        """
        Decorate user function as a Prefect task or flow.
        - as_task=True: decorate as Prefect task
        - as_task=False: decorate as Prefect flow (default)
        """
        def decorator(f):
            if self._run_func is not None:
                raise RuntimeError("Run function already set")
            self._run_func = f
            if as_task:
                self._prefect_task = task(f)
            else:
                self._prefect_flow = flow(f)
            return f

        if func is None:
            return decorator
        else:
            return decorator(func)

    @flow
    def _task_runner_flow(self, **runtime_inputs):
        assert self._prefect_task is not None
        return self._prefect_task(**runtime_inputs)

    def run(self, **runtime_inputs):
        inputs = {**self._defaults, **runtime_inputs}
        missing = [k for k in self.inputs if k not in inputs]
        if missing:
            raise ValueError(f"Missing inputs for run: {missing}")

        if self._prefect_flow is not None:
            return self._prefect_flow(**inputs)

        elif self._prefect_task is not None:
            return self._task_runner_flow(**inputs)

        else:
            raise RuntimeError("No run function registered")

    def show_inputs(self):
        for name, inp in self.inputs.items():
            print(inp)






















