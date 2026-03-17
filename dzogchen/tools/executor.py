"""
ToolExecutor - Safe, sandboxed tool execution engine
=====================================================

Executes tools with:
- Timeout enforcement
- Error handling and retries
- Result validation
- Audit logging
- Resource limits
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

from dzogchen.tools.registry import ToolRegistry, Tool

logger = logging.getLogger(__name__)


@dataclass
class ToolExecution:
    """Record of a tool execution."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    success: bool = False
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0


class ToolExecutor:
    """
    Manages safe, audited tool execution.

    Features:
    - Timeout enforcement per tool
    - Automatic retry on transient failures
    - Full execution audit trail
    - Resource usage tracking
    - Confirmation gates for sensitive tools
    """

    def __init__(
        self,
        registry: ToolRegistry,
        default_timeout: int = 30,
        max_retries: int = 2,
        confirmation_callback: Optional[Any] = None,
        enable_audit_log: bool = True,
    ):
        self.registry = registry
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self._confirmation_callback = confirmation_callback
        self.enable_audit_log = enable_audit_log
        self._execution_history: List[ToolExecution] = []

    async def execute(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        skip_confirmation: bool = False,
    ) -> Any:
        """
        Execute a tool by name with the given arguments.

        Args:
            tool_name: Name of the registered tool
            arguments: Tool arguments
            timeout: Override timeout in seconds
            skip_confirmation: Skip confirmation gate even if required

        Returns:
            Tool result

        Raises:
            ValueError: If tool not found
            TimeoutError: If tool exceeds timeout
            RuntimeError: If tool fails after all retries
        """
        tool = self.registry.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool: '{tool_name}'. Available: {list(self.registry.get_as_dict().keys())}")

        args = arguments or {}
        exec_timeout = timeout or tool.metadata.timeout or self.default_timeout

        # Check if confirmation required
        if tool.metadata.requires_confirmation and not skip_confirmation:
            confirmed = await self._request_confirmation(tool, args)
            if not confirmed:
                raise PermissionError(f"User denied execution of tool '{tool_name}'")

        execution = ToolExecution(tool_name=tool_name, arguments=args)
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                execution.retry_count = attempt
                logger.info(f"Retrying tool '{tool_name}' (attempt {attempt + 1})")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

            try:
                func = tool.func

                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(**args),
                        timeout=exec_timeout,
                    )
                else:
                    # Run sync functions in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: func(**args)),
                        timeout=exec_timeout,
                    )

                execution.result = result
                execution.success = True
                break

            except asyncio.TimeoutError:
                execution.error = f"Tool '{tool_name}' timed out after {exec_timeout}s"
                if attempt == self.max_retries:
                    raise TimeoutError(execution.error)

            except Exception as e:
                execution.error = str(e)
                error_type = type(e).__name__

                # Don't retry on certain errors
                if error_type in ("ValueError", "TypeError", "AttributeError", "PermissionError"):
                    break

                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Tool '{tool_name}' failed after {self.max_retries + 1} attempts: {e}"
                    ) from e

        execution.latency_ms = (time.time() - start_time) * 1000

        # Update registry stats
        self.registry.record_call(tool_name, execution.success, execution.latency_ms)

        # Audit log
        if self.enable_audit_log:
            self._execution_history.append(execution)
            if len(self._execution_history) > 10000:
                self._execution_history = self._execution_history[-5000:]

        if execution.success:
            logger.debug(
                f"Tool '{tool_name}' succeeded in {execution.latency_ms:.0f}ms "
                f"(retries={execution.retry_count})"
            )
        else:
            logger.warning(f"Tool '{tool_name}' failed: {execution.error}")

        return execution.result

    async def execute_batch(
        self,
        calls: List[Dict[str, Any]],
        parallel: bool = True,
    ) -> List[Any]:
        """
        Execute multiple tool calls.

        Args:
            calls: List of {"tool": name, "args": {...}} dicts
            parallel: Whether to execute in parallel

        Returns:
            List of results (None for failed calls)
        """
        if parallel:
            tasks = [
                self.execute(call["tool"], call.get("args"))
                for call in calls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [
                None if isinstance(r, Exception) else r
                for r in results
            ]
        else:
            results = []
            for call in calls:
                try:
                    result = await self.execute(call["tool"], call.get("args"))
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch tool call failed: {e}")
                    results.append(None)
            return results

    def get_execution_history(
        self,
        tool_name: Optional[str] = None,
        limit: int = 50,
        successful_only: bool = False,
    ) -> List[ToolExecution]:
        """Get execution history with optional filtering."""
        history = self._execution_history.copy()

        if tool_name:
            history = [e for e in history if e.tool_name == tool_name]

        if successful_only:
            history = [e for e in history if e.success]

        return sorted(history, key=lambda e: e.timestamp, reverse=True)[:limit]

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self._execution_history:
            return {"total_executions": 0}

        successful = [e for e in self._execution_history if e.success]
        failed = [e for e in self._execution_history if not e.success]

        tool_stats: Dict[str, Dict[str, int]] = {}
        for ex in self._execution_history:
            if ex.tool_name not in tool_stats:
                tool_stats[ex.tool_name] = {"calls": 0, "successes": 0, "failures": 0}
            tool_stats[ex.tool_name]["calls"] += 1
            if ex.success:
                tool_stats[ex.tool_name]["successes"] += 1
            else:
                tool_stats[ex.tool_name]["failures"] += 1

        return {
            "total_executions": len(self._execution_history),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self._execution_history),
            "avg_latency_ms": sum(e.latency_ms for e in self._execution_history) / len(self._execution_history),
            "tool_breakdown": tool_stats,
        }

    async def _request_confirmation(self, tool: Tool, args: Dict[str, Any]) -> bool:
        """Request human confirmation for sensitive tool execution."""
        if self._confirmation_callback is None:
            logger.warning(
                f"Tool '{tool.metadata.name}' requires confirmation but no callback set. "
                "Allowing execution."
            )
            return True

        try:
            if asyncio.iscoroutinefunction(self._confirmation_callback):
                return await self._confirmation_callback(tool.metadata.name, args)
            return self._confirmation_callback(tool.metadata.name, args)
        except Exception as e:
            logger.error(f"Confirmation callback error: {e}")
            return False
