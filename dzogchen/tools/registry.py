"""
ToolRegistry - Dynamic tool registration and discovery
======================================================

Central registry for all tools available to agents.
Supports dynamic tool registration, capability discovery,
and tool composition.
"""

from __future__ import annotations

import functools
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, get_type_hints
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    """Metadata about a registered tool."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    returns: str
    category: str = "general"
    is_async: bool = False
    requires_confirmation: bool = False
    rate_limit: Optional[int] = None  # calls per minute
    timeout: int = 30
    tags: List[str] = field(default_factory=list)


@dataclass
class Tool:
    """A registered tool with its metadata and implementation."""
    metadata: ToolMetadata
    func: Callable
    call_count: int = 0
    success_count: int = 0
    total_latency: float = 0.0
    last_called: Optional[float] = None

    @property
    def success_rate(self) -> float:
        if self.call_count == 0:
            return 1.0
        return self.success_count / self.call_count

    @property
    def avg_latency_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_latency / self.call_count

    def to_model_definition(self) -> Dict[str, Any]:
        """Convert to model tool definition format."""
        return {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "input_schema": self.metadata.parameters,
        }


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "general",
    requires_confirmation: bool = False,
    timeout: int = 30,
    tags: Optional[List[str]] = None,
):
    """
    Decorator to register a function as a tool.

    Usage:
        @tool(description="Search the web for information")
        async def web_search(query: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"

        # Auto-generate JSON Schema from type hints
        parameters = _build_json_schema(func)

        metadata = ToolMetadata(
            name=tool_name,
            description=tool_description.strip(),
            parameters=parameters,
            returns=_get_return_type(func),
            category=category,
            is_async=inspect.iscoroutinefunction(func),
            requires_confirmation=requires_confirmation,
            timeout=timeout,
            tags=tags or [],
        )

        # Store metadata on the function
        func._tool_metadata = metadata  # type: ignore

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs) if metadata.is_async else func(*args, **kwargs)

        wrapper._tool_metadata = metadata  # type: ignore
        return wrapper

    return decorator


class ToolRegistry:
    """
    Central registry for all agent tools.

    Handles registration, discovery, versioning, and
    performance tracking of tools.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}  # category -> [tool_names]

    def register(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: str = "general",
        parameters: Optional[Dict[str, Any]] = None,
        requires_confirmation: bool = False,
        timeout: int = 30,
        tags: Optional[List[str]] = None,
    ) -> Tool:
        """
        Register a callable as a tool.

        Args:
            func: The function to register
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            category: Tool category for organization
            parameters: JSON Schema for parameters (auto-generated if not provided)
            requires_confirmation: Whether to require human confirmation
            timeout: Execution timeout in seconds
            tags: Descriptive tags

        Returns:
            The registered Tool object
        """
        # Check if already has tool metadata from decorator
        if hasattr(func, "_tool_metadata"):
            metadata = func._tool_metadata
        else:
            tool_name = name or func.__name__
            tool_description = description or func.__doc__ or f"Tool: {tool_name}"
            tool_params = parameters or _build_json_schema(func)

            metadata = ToolMetadata(
                name=tool_name,
                description=tool_description.strip(),
                parameters=tool_params,
                returns=_get_return_type(func),
                category=category,
                is_async=inspect.iscoroutinefunction(func),
                requires_confirmation=requires_confirmation,
                timeout=timeout,
                tags=tags or [],
            )

        registered_tool = Tool(metadata=metadata, func=func)
        self._tools[metadata.name] = registered_tool

        if metadata.category not in self._categories:
            self._categories[metadata.category] = []
        if metadata.name not in self._categories[metadata.category]:
            self._categories[metadata.category].append(metadata.name)

        logger.debug(f"Registered tool: {metadata.name} ({metadata.category})")
        return registered_tool

    def register_class(self, cls: Any, category: Optional[str] = None) -> List[Tool]:
        """Register all tool methods from a class instance."""
        registered = []
        for method_name in dir(cls):
            method = getattr(cls, method_name)
            if callable(method) and hasattr(method, "_tool_metadata"):
                cat = category or getattr(cls, "category", "general")
                t = self.register(method, category=cat)
                registered.append(t)
        return registered

    def get(self, name: str) -> Optional[Tool]:
        """Get a registered tool by name."""
        return self._tools.get(name)

    def get_func(self, name: str) -> Optional[Callable]:
        """Get just the callable for a tool."""
        tool = self._tools.get(name)
        return tool.func if tool else None

    def list_tools(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Tool]:
        """List registered tools, optionally filtered."""
        tools = list(self._tools.values())

        if category:
            tool_names = set(self._categories.get(category, []))
            tools = [t for t in tools if t.metadata.name in tool_names]

        if tags:
            tools = [
                t for t in tools
                if any(tag in t.metadata.tags for tag in tags)
            ]

        return tools

    def to_model_definitions(
        self,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Export all tools as model-compatible definitions."""
        return [t.to_model_definition() for t in self.list_tools(category=category)]

    def get_as_dict(self) -> Dict[str, Callable]:
        """Get all tools as a name -> callable dictionary."""
        return {name: t.func for name, t in self._tools.items()}

    def record_call(self, name: str, success: bool, latency_ms: float) -> None:
        """Record a tool call result for tracking."""
        if name in self._tools:
            tool = self._tools[name]
            tool.call_count += 1
            tool.last_called = time.time()
            tool.total_latency += latency_ms
            if success:
                tool.success_count += 1

    @property
    def categories(self) -> List[str]:
        return list(self._categories.keys())

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_tools": len(self._tools),
            "categories": {cat: len(names) for cat, names in self._categories.items()},
            "most_used": sorted(
                [(name, t.call_count) for name, t in self._tools.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:5],
            "lowest_success_rate": sorted(
                [(name, t.success_rate) for name, t in self._tools.items() if t.call_count > 0],
                key=lambda x: x[1],
            )[:3],
        }


def _build_json_schema(func: Callable) -> Dict[str, Any]:
    """Build a JSON Schema from a function's type hints."""
    try:
        hints = get_type_hints(func)
        sig = inspect.signature(func)
    except Exception:
        return {"type": "object", "properties": {}, "required": []}

    properties = {}
    required = []

    type_map = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "List": "array",
        "Dict": "object",
    }

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "return"):
            continue

        type_hint = hints.get(param_name, Any)
        type_name = getattr(type_hint, "__name__", str(type_hint))
        json_type = type_map.get(type_name, "string")

        # Get description from docstring if available
        param_desc = ""
        if func.__doc__:
            doc_lines = func.__doc__.split("\n")
            for line in doc_lines:
                if param_name + ":" in line:
                    param_desc = line.split(":", 1)[-1].strip()

        properties[param_name] = {
            "type": json_type,
            "description": param_desc or f"Parameter: {param_name}",
        }

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _get_return_type(func: Callable) -> str:
    """Get the return type annotation as a string."""
    try:
        hints = get_type_hints(func)
        ret = hints.get("return", Any)
        return getattr(ret, "__name__", str(ret))
    except Exception:
        return "Any"
