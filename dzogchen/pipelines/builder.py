"""Declarative, composable async processing pipelines."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    step_name: str
    status: StepStatus
    input: Any
    output: Any
    error: Optional[str]
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineStep:
    """A single executable step within a pipeline."""

    def __init__(
        self,
        name: str,
        fn: Callable,
        condition: Optional[Callable[[Any], bool]] = None,
        retry: int = 0,
        timeout: Optional[float] = None,
        catch_errors: bool = False,
    ) -> None:
        self.name = name
        self.fn = fn
        self.condition = condition
        self.retry = max(0, retry)
        self.timeout = timeout
        self.catch_errors = catch_errors

    async def execute(self, input_data: Any) -> StepResult:
        """Execute the step, honouring condition, retry, and timeout settings."""
        start = time.perf_counter()

        # Check condition guard
        if self.condition is not None:
            try:
                cond_result = self.condition(input_data)
                if asyncio.iscoroutine(cond_result):
                    cond_result = await cond_result
                if not cond_result:
                    return StepResult(
                        step_name=self.name,
                        status=StepStatus.SKIPPED,
                        input=input_data,
                        output=input_data,  # pass-through when skipped
                        error=None,
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )
            except Exception as exc:
                latency = (time.perf_counter() - start) * 1000
                if self.catch_errors:
                    return StepResult(
                        step_name=self.name,
                        status=StepStatus.FAILED,
                        input=input_data,
                        output=None,
                        error=f"Condition evaluation error: {exc}",
                        latency_ms=latency,
                    )
                raise

        attempts = self.retry + 1
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                if inspect.iscoroutinefunction(self.fn):
                    coro = self.fn(input_data)
                else:
                    loop = asyncio.get_event_loop()
                    coro = loop.run_in_executor(None, self.fn, input_data)

                if self.timeout is not None:
                    output = await asyncio.wait_for(
                        coro if asyncio.iscoroutine(coro) else asyncio.ensure_future(coro),
                        timeout=self.timeout,
                    )
                else:
                    output = await (coro if asyncio.iscoroutine(coro) else coro)

                latency = (time.perf_counter() - start) * 1000
                return StepResult(
                    step_name=self.name,
                    status=StepStatus.SUCCESS,
                    input=input_data,
                    output=output,
                    error=None,
                    latency_ms=latency,
                    metadata={"attempt": attempt + 1},
                )

            except Exception as exc:
                last_error = exc
                if attempt < attempts - 1:
                    backoff = 0.1 * (2 ** attempt)
                    await asyncio.sleep(backoff)

        latency = (time.perf_counter() - start) * 1000
        error_msg = str(last_error)

        if self.catch_errors:
            return StepResult(
                step_name=self.name,
                status=StepStatus.FAILED,
                input=input_data,
                output=None,
                error=error_msg,
                latency_ms=latency,
                metadata={"attempts": attempts},
            )

        raise RuntimeError(
            f"Step '{self.name}' failed after {attempts} attempt(s): {error_msg}"
        ) from last_error


# Internal structure used to represent pipeline nodes

_NODE_SEQUENTIAL = "sequential"
_NODE_PARALLEL = "parallel"
_NODE_BRANCH = "branch"


class Pipeline:
    """Composable async pipeline that chains PipelineStep objects."""

    def __init__(self, name: str = "pipeline", telemetry=None) -> None:
        self.name = name
        self.telemetry = telemetry
        # Each element is one of:
        #   ("sequential", PipelineStep)
        #   ("parallel",   List[PipelineStep])
        #   ("branch",     condition_fn, true_pipeline, false_pipeline)
        self._nodes: List[Tuple] = []

    # ---- builder API ---------------------------------------------------------

    def step(
        self,
        name: str,
        fn: Callable,
        condition: Optional[Callable] = None,
        retry: int = 0,
        timeout: Optional[float] = None,
        catch_errors: bool = False,
    ) -> "Pipeline":
        """Append a sequential step and return self for chaining."""
        ps = PipelineStep(
            name=name,
            fn=fn,
            condition=condition,
            retry=retry,
            timeout=timeout,
            catch_errors=catch_errors,
        )
        self._nodes.append((_NODE_SEQUENTIAL, ps))
        return self

    def parallel(self, *step_defs: Union[PipelineStep, Tuple]) -> "Pipeline":
        """
        Append a parallel group.

        Accepts either PipelineStep objects or (name, fn, **kwargs) tuples.
        All steps in the group execute concurrently; their outputs are collected
        as a list that becomes the input to the next step.
        """
        steps: List[PipelineStep] = []
        for item in step_defs:
            if isinstance(item, PipelineStep):
                steps.append(item)
            elif isinstance(item, tuple):
                name, fn = item[0], item[1]
                kwargs = item[2] if len(item) > 2 else {}
                steps.append(PipelineStep(name=name, fn=fn, **kwargs))
            else:
                raise TypeError(f"Unsupported parallel step type: {type(item)}")
        self._nodes.append((_NODE_PARALLEL, steps))
        return self

    def branch(
        self,
        condition_fn: Callable[[Any], bool],
        true_pipeline: "Pipeline",
        false_pipeline: "Pipeline",
    ) -> "Pipeline":
        """Append a conditional branch node."""
        self._nodes.append((_NODE_BRANCH, condition_fn, true_pipeline, false_pipeline))
        return self

    # ---- execution -----------------------------------------------------------

    async def run(self, input_data: Any = None) -> Dict[str, Any]:
        """
        Execute the pipeline sequentially, passing each step's output to the next.

        Returns a result dict with keys:
            success          bool
            output           Any  (final step's output)
            steps            List[StepResult]
            total_latency_ms float
            error            Optional[str]
        """
        all_results: List[StepResult] = []
        current = input_data
        pipeline_start = time.perf_counter()

        span_ctx = None
        if self.telemetry is not None:
            span_ctx = self.telemetry.span(f"pipeline.{self.name}")
            await span_ctx.__aenter__()

        try:
            for node in self._nodes:
                node_type = node[0]

                if node_type == _NODE_SEQUENTIAL:
                    ps: PipelineStep = node[1]
                    result = await ps.execute(current)
                    all_results.append(result)
                    if result.status == StepStatus.FAILED:
                        total_ms = (time.perf_counter() - pipeline_start) * 1000
                        if span_ctx:
                            span_ctx.set_status("ERROR", result.error or "")
                            await span_ctx.__aexit__(None, None, None)
                        return {
                            "success": False,
                            "output": None,
                            "steps": all_results,
                            "total_latency_ms": total_ms,
                            "error": result.error,
                        }
                    if result.status != StepStatus.SKIPPED:
                        current = result.output

                elif node_type == _NODE_PARALLEL:
                    parallel_steps: List[PipelineStep] = node[1]
                    tasks = [ps.execute(current) for ps in parallel_steps]
                    gathered: List[StepResult] = await asyncio.gather(*tasks, return_exceptions=False)
                    all_results.extend(gathered)
                    failed = [r for r in gathered if r.status == StepStatus.FAILED]
                    if failed:
                        total_ms = (time.perf_counter() - pipeline_start) * 1000
                        error_msg = "; ".join(r.error or "" for r in failed)
                        if span_ctx:
                            span_ctx.set_status("ERROR", error_msg)
                            await span_ctx.__aexit__(None, None, None)
                        return {
                            "success": False,
                            "output": None,
                            "steps": all_results,
                            "total_latency_ms": total_ms,
                            "error": error_msg,
                        }
                    # Pass list of parallel outputs to the next step
                    current = [r.output for r in gathered]

                elif node_type == _NODE_BRANCH:
                    _, condition_fn, true_pipeline, false_pipeline = node
                    try:
                        condition_result = condition_fn(current)
                        if asyncio.iscoroutine(condition_result):
                            condition_result = await condition_result
                    except Exception as exc:
                        total_ms = (time.perf_counter() - pipeline_start) * 1000
                        if span_ctx:
                            span_ctx.set_status("ERROR", str(exc))
                            await span_ctx.__aexit__(None, None, None)
                        return {
                            "success": False,
                            "output": None,
                            "steps": all_results,
                            "total_latency_ms": total_ms,
                            "error": f"Branch condition error: {exc}",
                        }

                    chosen_pipeline = true_pipeline if condition_result else false_pipeline
                    branch_result = await chosen_pipeline.run(current)
                    all_results.extend(branch_result.get("steps", []))
                    if not branch_result.get("success", False):
                        total_ms = (time.perf_counter() - pipeline_start) * 1000
                        if span_ctx:
                            span_ctx.set_status("ERROR", branch_result.get("error", ""))
                            await span_ctx.__aexit__(None, None, None)
                        return {
                            "success": False,
                            "output": None,
                            "steps": all_results,
                            "total_latency_ms": total_ms,
                            "error": branch_result.get("error"),
                        }
                    current = branch_result.get("output", current)

        except Exception as exc:
            total_ms = (time.perf_counter() - pipeline_start) * 1000
            if span_ctx:
                span_ctx.set_status("ERROR", str(exc))
                await span_ctx.__aexit__(None, None, None)
            return {
                "success": False,
                "output": None,
                "steps": all_results,
                "total_latency_ms": total_ms,
                "error": str(exc),
            }

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        if span_ctx:
            span_ctx.set_status("OK")
            await span_ctx.__aexit__(None, None, None)

        return {
            "success": True,
            "output": current,
            "steps": all_results,
            "total_latency_ms": total_ms,
            "error": None,
        }

    # ---- composition operators -----------------------------------------------

    def __or__(self, other: "Pipeline") -> "Pipeline":
        """Chain two pipelines: pipeline1 | pipeline2."""
        combined = Pipeline(name=f"{self.name}|{other.name}", telemetry=self.telemetry)
        combined._nodes = list(self._nodes) + list(other._nodes)
        return combined

    def __add__(self, step_obj: Union[PipelineStep, "Pipeline"]) -> "Pipeline":
        """Append a PipelineStep or merge another Pipeline: pipeline + step."""
        new_pipeline = Pipeline(name=self.name, telemetry=self.telemetry)
        new_pipeline._nodes = list(self._nodes)
        if isinstance(step_obj, PipelineStep):
            new_pipeline._nodes.append((_NODE_SEQUENTIAL, step_obj))
        elif isinstance(step_obj, Pipeline):
            new_pipeline._nodes.extend(step_obj._nodes)
        else:
            raise TypeError(f"Cannot add type {type(step_obj)} to Pipeline.")
        return new_pipeline

    # ---- visualization -------------------------------------------------------

    def visualize(self) -> str:
        """Return an ASCII representation of the pipeline."""
        lines: List[str] = [f"Pipeline: {self.name}", "=" * (len(self.name) + 10)]
        for i, node in enumerate(self._nodes):
            prefix = f"[{i + 1}]"
            node_type = node[0]

            if node_type == _NODE_SEQUENTIAL:
                ps: PipelineStep = node[1]
                retry_info = f" (retry={ps.retry})" if ps.retry else ""
                timeout_info = f" (timeout={ps.timeout}s)" if ps.timeout else ""
                cond_info = " [conditional]" if ps.condition else ""
                catch_info = " [catch_errors]" if ps.catch_errors else ""
                lines.append(f"  {prefix} --> {ps.name}{retry_info}{timeout_info}{cond_info}{catch_info}")

            elif node_type == _NODE_PARALLEL:
                parallel_steps: List[PipelineStep] = node[1]
                names = ", ".join(ps.name for ps in parallel_steps)
                lines.append(f"  {prefix} ==> [PARALLEL] {names}")
                for ps in parallel_steps:
                    lines.append(f"           |-- {ps.name}")

            elif node_type == _NODE_BRANCH:
                _, condition_fn, true_pipeline, false_pipeline = node
                cond_name = getattr(condition_fn, "__name__", "condition")
                lines.append(f"  {prefix} <IF {cond_name}>")
                lines.append(f"           TRUE  --> {true_pipeline.name}")
                lines.append(f"           FALSE --> {false_pipeline.name}")

        lines.append("=" * (len(self.name) + 10))
        return "\n".join(lines)


# ---- convenience constructor -------------------------------------------------

def pipe(*steps: Union[PipelineStep, Tuple, Callable]) -> Pipeline:
    """
    Build a Pipeline from a sequence of steps.

    Each element may be:
      - a PipelineStep instance
      - a callable (used as fn with its __name__ as the step name)
      - a (name, fn) tuple
      - a (name, fn, dict_of_kwargs) tuple
    """
    pipeline = Pipeline()
    for item in steps:
        if isinstance(item, PipelineStep):
            pipeline._nodes.append((_NODE_SEQUENTIAL, item))
        elif callable(item):
            name = getattr(item, "__name__", "step")
            pipeline._nodes.append((_NODE_SEQUENTIAL, PipelineStep(name=name, fn=item)))
        elif isinstance(item, tuple):
            if len(item) >= 2:
                name, fn = item[0], item[1]
                kwargs = item[2] if len(item) > 2 else {}
                pipeline._nodes.append(
                    (_NODE_SEQUENTIAL, PipelineStep(name=name, fn=fn, **kwargs))
                )
            else:
                raise ValueError(f"Tuple step must have at least (name, fn), got: {item}")
        else:
            raise TypeError(f"Unsupported step type: {type(item)}")
    return pipeline
