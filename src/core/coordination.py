"""Team Coordination System for Agentic OS
Manages communication, task distribution, and collaboration between agents."""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime
import json

from .agent import Agent
from ..utils.error_handling import CoordinationError
from ..utils.monitoring import monitor

@dataclass
class Message:
    sender: str
    receiver: str
    content: Dict[str, Any]
    message_type: str
    timestamp: datetime
    priority: int = 1
    thread_id: Optional[str] = None

class CoordinationSystem:
    """Manages team coordination and communication"""
    
    def __init__(self):
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.active_threads: Dict[str, List[Message]] = {}
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.task_assignments: Dict[str, List[str]] = {}

    @monitor
    async def send_message(self, message: Message) -> None:
        """Send message between agents"""
        try:
            # Log message
            self._log_message(message)
            
            # Add to thread if exists
            if message.thread_id:
                self._add_to_thread(message)
            
            # Queue message for delivery
            await self.message_queue.put(message)
            
        except Exception as e:
            raise CoordinationError(f"Failed to send message: {str(e)}")

    async def process_messages(self) -> None:
        """Process queued messages"""
        while True:
            message = await self.message_queue.get()
            try:
                await self._deliver_message(message)
            finally:
                self.message_queue.task_done()

    async def create_task_group(self, task: Dict[str, Any], agents: List[str]) -> str:
        """Create a coordinated task group"""
        group_id = self._generate_group_id()
        
        # Initialize task assignments
        self.task_assignments[group_id] = agents
        
        # Create coordination thread
        thread_id = self._create_thread(group_id)
        
        # Notify all agents
        await self._notify_group_creation(group_id, task, agents, thread_id)
        
        return group_id

    async def monitor_progress(self, group_id: str) -> Dict[str, Any]:
        """Monitor task group progress"""
        if group_id not in self.task_assignments:
            raise CoordinationError(f"Unknown task group: {group_id}")
            
        agents = self.task_assignments[group_id]
        progress = {}
        
        for agent_id in agents:
            progress[agent_id] = self.agent_states.get(agent_id, {}).get("progress", 0)
            
        return {
            "group_id": group_id,
            "total_agents": len(agents),
            "progress": progress,
            "average_progress": sum(progress.values()) / len(agents) if agents else 0
        }

    def get_thread_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get communication history for a thread"""
        if thread_id not in self.active_threads:
            raise CoordinationError(f"Unknown thread: {thread_id}")
            
        return [
            {
                "sender": msg.sender,
                "receiver": msg.receiver,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "type": msg.message_type
            }
            for msg in self.active_threads[thread_id]
        ]

    async def _deliver_message(self, message: Message) -> None:
        """Deliver message to receiving agent"""
        try:
            # Update receiver state
            if message.receiver in self.agent_states:
                self.agent_states[message.receiver]["last_message"] = message.timestamp
                
            # Handle different message types
            if message.message_type == "task_assignment":
                await self._handle_task_assignment(message)
            elif message.message_type == "progress_update":
                await self._handle_progress_update(message)
            elif message.message_type == "coordination":
                await self._handle_coordination(message)
                
        except Exception as e:
            raise CoordinationError(f"Message delivery failed: {str(e)}")

    def _create_thread(self, group_id: str) -> str:
        """Create new communication thread"""
        thread_id = f"thread_{group_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        self.active_threads[thread_id] = []
        return thread_id

    def _add_to_thread(self, message: Message) -> None:
        """Add message to existing thread"""
        if message.thread_id in self.active_threads:
            self.active_threads[message.thread_id].append(message)

    async def _notify_group_creation(self, group_id: str, task: Dict[str, Any], agents: List[str], thread_id: str) -> None:
        """Notify all agents about new task group"""
        for agent_id in agents:
            message = Message(
                sender="coordinator",
                receiver=agent_id,
                content={
                    "group_id": group_id,
                    "task": task,
                    "team_members": agents,
                    "thread_id": thread_id
                },
                message_type="group_creation",
                timestamp=datetime.utcnow(),
                thread_id=thread_id
            )
            await self.send_message(message)

    def _log_message(self, message: Message) -> None:
        """Log message details"""
        log_entry = {
            "timestamp": message.timestamp,
            "sender": message.sender,
            "receiver": message.receiver,
            "type": message.message_type,
            "thread_id": message.thread_id
        }
        # Implement logging mechanism

    def _generate_group_id(self) -> str:
        """Generate unique group ID"""
        return f"group_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
