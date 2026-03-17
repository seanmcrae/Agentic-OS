"""
Built-in tools for Dzogchen agents
====================================

A curated set of essential tools:
- Web search and content fetching
- File I/O
- Code execution
- Math and data processing
- Date/time utilities
- Text processing
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Web Tools
# ============================================================================

async def web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web for information.

    Args:
        query: Search query string
        num_results: Number of results to return (1-10)

    Returns:
        List of {title, url, snippet} dicts
    """
    # This would use a real search API in production
    logger.info(f"Web search: {query}")
    return [
        {
            "title": f"Search result for: {query}",
            "url": "https://example.com",
            "snippet": f"Simulated search result for query: {query}. In production, this would use a real search API.",
        }
    ]


async def fetch_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Fetch content from a URL.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Dict with {content, status_code, content_type}
    """
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                content = await response.text()
                return {
                    "content": content[:10000],  # Limit content size
                    "status_code": response.status,
                    "content_type": response.headers.get("content-type", "unknown"),
                    "url": str(response.url),
                }
    except ImportError:
        return {"error": "aiohttp not installed", "content": "", "status_code": 0}
    except Exception as e:
        return {"error": str(e), "content": "", "status_code": 0}


# ============================================================================
# File Tools
# ============================================================================

def read_file(path: str, encoding: str = "utf-8") -> str:
    """
    Read the contents of a file.

    Args:
        path: File path to read
        encoding: File encoding (default: utf-8)

    Returns:
        File contents as string
    """
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write_file(path: str, content: str, encoding: str = "utf-8") -> bool:
    """
    Write content to a file.

    Args:
        path: File path to write
        content: Content to write
        encoding: File encoding

    Returns:
        True if successful
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        f.write(content)
    return True


def list_directory(path: str = ".", pattern: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List files and directories.

    Args:
        path: Directory path to list
        pattern: Optional glob pattern filter

    Returns:
        List of {name, path, type, size} dicts
    """
    import glob as glob_module

    items = []
    search_path = os.path.join(path, pattern or "*")

    for item_path in glob_module.glob(search_path):
        stat = os.stat(item_path)
        items.append({
            "name": os.path.basename(item_path),
            "path": item_path,
            "type": "directory" if os.path.isdir(item_path) else "file",
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })

    return sorted(items, key=lambda x: (x["type"] == "file", x["name"]))


# ============================================================================
# Code Execution Tools
# ============================================================================

async def execute_python(code: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Execute Python code in a subprocess.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dict with {stdout, stderr, return_code}
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )

        return {
            "stdout": stdout.decode("utf-8", errors="replace")[:5000],
            "stderr": stderr.decode("utf-8", errors="replace")[:2000],
            "return_code": proc.returncode,
            "success": proc.returncode == 0,
        }

    except asyncio.TimeoutError:
        return {"error": f"Code execution timed out after {timeout}s", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


async def run_shell(command: str, timeout: int = 10, cwd: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute a shell command.

    Args:
        command: Shell command to run
        timeout: Maximum execution time
        cwd: Working directory

    Returns:
        Dict with {stdout, stderr, return_code}
    """
    # Safety check - block dangerous commands
    dangerous_patterns = [
        r"rm\s+-rf",
        r":(){ :|:& };:",  # Fork bomb
        r"dd\s+if=",
        r"mkfs",
        r"> /dev/",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, command):
            return {"error": f"Command blocked for safety: matches pattern '{pattern}'", "success": False}

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )

        return {
            "stdout": stdout.decode("utf-8", errors="replace")[:5000],
            "stderr": stderr.decode("utf-8", errors="replace")[:2000],
            "return_code": proc.returncode,
            "success": proc.returncode == 0,
        }

    except asyncio.TimeoutError:
        return {"error": f"Command timed out after {timeout}s", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


# ============================================================================
# Math Tools
# ============================================================================

def calculate(expression: str) -> Dict[str, Any]:
    """
    Evaluate a mathematical expression safely.

    Args:
        expression: Math expression to evaluate (e.g., "2 + 2", "sqrt(16)")

    Returns:
        Dict with {result, expression}
    """
    # Safe math environment
    safe_globals = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "ceil": math.ceil,
        "floor": math.floor,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "inf": math.inf,
    }

    try:
        # Basic sanitization
        if any(keyword in expression for keyword in ["import", "exec", "eval", "__"]):
            return {"error": "Unsafe expression", "result": None}

        result = eval(expression, safe_globals)  # noqa: S307
        return {"result": result, "expression": expression, "success": True}
    except Exception as e:
        return {"error": str(e), "result": None, "success": False}


# ============================================================================
# Data Processing Tools
# ============================================================================

def json_parse(text: str) -> Any:
    """
    Parse JSON text.

    Args:
        text: JSON string to parse

    Returns:
        Parsed JSON value
    """
    return json.loads(text)


def json_format(data: Any, indent: int = 2) -> str:
    """
    Format data as JSON.

    Args:
        data: Data to format
        indent: Indentation level

    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=indent, default=str)


def text_stats(text: str) -> Dict[str, Any]:
    """
    Compute statistics about a text.

    Args:
        text: Input text

    Returns:
        Dict with word count, sentence count, etc.
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    paragraphs = text.split('\n\n')

    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "paragraph_count": len([p for p in paragraphs if p.strip()]),
        "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
        "unique_words": len(set(w.lower() for w in words)),
    }


# ============================================================================
# Time & Date Tools
# ============================================================================

def get_current_time(timezone_name: str = "UTC") -> Dict[str, str]:
    """
    Get current date and time.

    Args:
        timezone_name: Timezone name (e.g., "UTC", "US/Eastern")

    Returns:
        Dict with {datetime, date, time, timezone, unix_timestamp}
    """
    now = datetime.now(timezone.utc)
    return {
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": "UTC",
        "unix_timestamp": now.timestamp(),
        "day_of_week": now.strftime("%A"),
    }


# ============================================================================
# Tool Collection
# ============================================================================

def get_builtin_tools() -> Dict[str, Any]:
    """
    Get all built-in tools as a name -> callable dictionary.

    Returns:
        Dict mapping tool names to callables
    """
    return {
        # Web
        "web_search": web_search,
        "fetch_url": fetch_url,
        # Files
        "read_file": read_file,
        "write_file": write_file,
        "list_directory": list_directory,
        # Code execution
        "execute_python": execute_python,
        "run_shell": run_shell,
        # Math
        "calculate": calculate,
        # Data
        "json_parse": json_parse,
        "json_format": json_format,
        "text_stats": text_stats,
        # Time
        "get_current_time": get_current_time,
    }


def get_safe_tools() -> Dict[str, Any]:
    """
    Get a safe subset of tools (no code execution or shell access).
    """
    return {
        "web_search": web_search,
        "calculate": calculate,
        "json_parse": json_parse,
        "json_format": json_format,
        "text_stats": text_stats,
        "get_current_time": get_current_time,
    }
