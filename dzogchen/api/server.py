"""
Dzogchen API Server
====================

FastAPI-based REST API and WebSocket server providing:
- Chat completions endpoint
- Agent task execution
- Swarm intelligence
- Memory operations
- System status
- Real-time streaming via WebSocket
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def create_app(dzogchen_instance=None):
    """
    Create and configure the FastAPI application.

    Args:
        dzogchen_instance: Optional pre-configured Dzogchen instance

    Returns:
        Configured FastAPI application
    """
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "FastAPI not found. Install with: pip install fastapi uvicorn"
        )

    from dzogchen.core import Dzogchen
    from dzogchen.config import DzogchenConfig

    app = FastAPI(
        title="Dzogchen AI",
        description="Super-Intelligent AI Framework - REST API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State
    app.state.dz = dzogchen_instance
    app.state.connections: Dict[str, WebSocket] = {}

    # =========================================================================
    # Request/Response Models
    # =========================================================================

    class ThinkRequest(BaseModel):
        query: str = Field(..., description="Question or problem to reason about")
        strategy: str = Field("cot", description="Reasoning strategy: cot|tot|self_consistency")
        reflect: bool = Field(True, description="Enable self-reflection")
        context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

    class ActRequest(BaseModel):
        task: str = Field(..., description="Task to accomplish")
        tools: Optional[List[str]] = Field(None, description="Tool names to use")
        context: Optional[Dict[str, Any]] = None

    class SwarmRequest(BaseModel):
        problem: str = Field(..., description="Problem for swarm to solve")
        swarm_size: int = Field(3, ge=1, le=10)
        iterations: int = Field(2, ge=1, le=5)
        tools: Optional[List[str]] = None

    class PlanRequest(BaseModel):
        goal: str = Field(..., description="Goal to plan for")
        execute: bool = Field(False, description="Execute the plan after creation")
        tools: Optional[List[str]] = None

    class LearnRequest(BaseModel):
        concept: str
        definition: str
        domain: str = "general"
        source: str = "user"

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        messages: List[ChatMessage]
        model: Optional[str] = None
        max_tokens: int = 4096
        temperature: float = 0.7
        stream: bool = False

    # =========================================================================
    # Dependencies
    # =========================================================================

    async def get_dz() -> Dzogchen:
        """Get or create the Dzogchen instance."""
        if app.state.dz is None:
            config = DzogchenConfig.from_env()
            app.state.dz = Dzogchen(config)
            await app.state.dz.start()
        return app.state.dz

    # =========================================================================
    # Lifecycle
    # =========================================================================

    @app.on_event("startup")
    async def startup():
        if app.state.dz is None:
            try:
                config = DzogchenConfig.from_env()
                app.state.dz = Dzogchen(config)
                await app.state.dz.start()
                logger.info("Dzogchen API server started")
            except Exception as e:
                logger.error(f"Failed to start Dzogchen: {e}")

    @app.on_event("shutdown")
    async def shutdown():
        if app.state.dz:
            await app.state.dz.stop()

    # =========================================================================
    # API Endpoints
    # =========================================================================

    @app.get("/")
    async def root():
        """API root - returns basic info."""
        return {
            "name": "Dzogchen AI",
            "version": "0.1.0",
            "description": "Super-Intelligent AI Framework",
            "endpoints": ["/think", "/act", "/swarm", "/plan", "/learn", "/memory", "/status"],
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": time.time()}

    @app.get("/status")
    async def status(dz: Dzogchen = Depends(get_dz)):
        """Get complete system status."""
        return dz.get_status()

    @app.post("/think")
    async def think(request: ThinkRequest, dz: Dzogchen = Depends(get_dz)):
        """Apply multi-step reasoning to answer a query."""
        try:
            response = await dz.think(
                query=request.query,
                strategy=request.strategy,
                context=request.context,
                reflect=request.reflect,
            )
            return {"response": response, "query": request.query, "strategy": request.strategy}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/think/stream")
    async def think_stream(request: ThinkRequest, dz: Dzogchen = Depends(get_dz)):
        """Stream reasoning steps."""
        async def generate():
            async for step in dz.stream_think(request.query):
                yield f"data: {json.dumps({'step': step})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/act")
    async def act(request: ActRequest, dz: Dzogchen = Depends(get_dz)):
        """Execute a task autonomously using agents."""
        try:
            result = await dz.act(
                task=request.task,
                tools=request.tools,
                context=request.context,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/swarm")
    async def swarm(request: SwarmRequest, dz: Dzogchen = Depends(get_dz)):
        """Apply swarm intelligence to a complex problem."""
        try:
            result = await dz.swarm(
                problem=request.problem,
                swarm_size=request.swarm_size,
                iterations=request.iterations,
                tools=request.tools,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/plan")
    async def plan(request: PlanRequest, dz: Dzogchen = Depends(get_dz)):
        """Create and optionally execute a multi-step plan."""
        try:
            result = await dz.plan(
                goal=request.goal,
                execute=request.execute,
                tools=request.tools,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/learn")
    async def learn(request: LearnRequest, dz: Dzogchen = Depends(get_dz)):
        """Teach Dzogchen a new fact."""
        await dz.learn(
            concept=request.concept,
            definition=request.definition,
            domain=request.domain,
            source=request.source,
        )
        return {"status": "learned", "concept": request.concept}

    @app.get("/memory")
    async def memory_recall(q: str, dz: Dzogchen = Depends(get_dz)):
        """Query the memory system."""
        return await dz.remember(q)

    @app.post("/chat")
    async def chat(request: ChatRequest, dz: Dzogchen = Depends(get_dz)):
        """
        OpenAI-compatible chat completions endpoint.
        Routes to the optimal model based on routing strategy.
        """
        try:
            from dzogchen.models.base import ModelMessage, MessageRole

            messages = [
                ModelMessage(
                    role=MessageRole(msg.role),
                    content=msg.content,
                )
                for msg in request.messages
            ]

            response = await dz.model_router.route(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            # OpenAI-compatible response format
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "model": response.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response.content},
                        "finish_reason": response.finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": response.input_tokens,
                    "completion_tokens": response.output_tokens,
                    "total_tokens": response.total_tokens,
                },
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/tools")
    async def list_tools(dz: Dzogchen = Depends(get_dz)):
        """List all available tools."""
        tools = dz.tool_registry.list_tools()
        return {
            "tools": [
                {
                    "name": t.metadata.name,
                    "description": t.metadata.description,
                    "category": t.metadata.category,
                    "call_count": t.call_count,
                    "success_rate": t.success_rate,
                }
                for t in tools
            ],
            "total": len(tools),
        }

    # =========================================================================
    # WebSocket
    # =========================================================================

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """
        WebSocket endpoint for real-time interaction.

        Send JSON messages with:
        {"type": "think", "query": "..."}
        {"type": "act", "task": "..."}
        {"type": "ping"}
        """
        await websocket.accept()
        app.state.connections[session_id] = websocket
        dz = await get_dz()

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                msg_type = message.get("type")

                try:
                    if msg_type == "think":
                        response = await dz.think(message.get("query", ""))
                        await websocket.send_json({"type": "response", "content": response})

                    elif msg_type == "act":
                        result = await dz.act(message.get("task", ""))
                        await websocket.send_json({"type": "result", **result})

                    elif msg_type == "stream_think":
                        await websocket.send_json({"type": "stream_start"})
                        async for step in dz.stream_think(message.get("query", "")):
                            await websocket.send_json({"type": "stream_chunk", "content": step})
                        await websocket.send_json({"type": "stream_end"})

                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong", "timestamp": time.time()})

                    else:
                        await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})

                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})

        except WebSocketDisconnect:
            del app.state.connections[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")

    return app
