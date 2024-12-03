"""Network capabilities for Agentic OS
Provides HTTP, WebSocket, and general networking functionality."""
from typing import Dict, Any, Optional
import aiohttp
import asyncio
from ..core.capabilities import Capability, CapabilityMetadata
from ..utils.error_handling import CapabilityError
from ..utils.monitoring import monitor

class NetworkCapability(Capability):
    """Handles network operations"""
    
    def __init__(self):
        metadata = CapabilityMetadata(
            name="network",
            version="1.0.0",
            description="Network operations capability",
            requirements=["aiohttp"],
            parameters={
                "http_get": ["url", "headers"],
                "http_post": ["url", "data", "headers"],
                "websocket": ["url", "protocol"]
            }
        )
        super().__init__(metadata)
        self._session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Initialize network capability"""
        self._session = aiohttp.ClientSession()
        self.register_handler("http_get", self._handle_http_get)
        self.register_handler("http_post", self._handle_http_post)
        self.register_handler("websocket", self._handle_websocket)
        self._initialized = True

    async def shutdown(self) -> None:
        """Clean up network resources"""
        if self._session:
            await self._session.close()

    @monitor
    async def _handle_http_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle HTTP GET requests"""
        url = params.get("url")
        headers = params.get("headers", {})

        if not url:
            raise CapabilityError("URL required for HTTP GET", "PARAMETER_ERROR")

        try:
            async with self._session.get(url, headers=headers) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "content": await response.text()
                }
        except Exception as e:
            raise CapabilityError(f"HTTP GET failed: {str(e)}", "NETWORK_ERROR")

    @monitor
    async def _handle_http_post(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle HTTP POST requests"""
        url = params.get("url")
        data = params.get("data")
        headers = params.get("headers", {})

        if not url or data is None:
            raise CapabilityError("URL and data required for HTTP POST", "PARAMETER_ERROR")

        try:
            async with self._session.post(url, json=data, headers=headers) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "content": await response.text()
                }
        except Exception as e:
            raise CapabilityError(f"HTTP POST failed: {str(e)}", "NETWORK_ERROR")

    @monitor
    async def _handle_websocket(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle WebSocket connections"""
        url = params.get("url")
        protocol = params.get("protocol")

        if not url:
            raise CapabilityError("URL required for WebSocket", "PARAMETER_ERROR")

        try:
            async with self._session.ws_connect(url, protocols=[protocol] if protocol else None) as ws:
                return {
                    "status": "connected",
                    "socket_id": id(ws)
                }
        except Exception as e:
            raise CapabilityError(f"WebSocket connection failed: {str(e)}", "NETWORK_ERROR")"
