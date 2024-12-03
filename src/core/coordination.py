"""Enhanced Team Coordination System for Agentic OS
Provides advanced communication, task allocation, and team optimization."""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime
from enum import Enum

class MessageType(Enum):
    TASK_ASSIGNMENT = "task_assignment"
    STATUS_UPDATE = "status_update"
    KNOWLEDGE_SHARE = "knowledge_share"
    REQUEST_HELP = "request_help"
    PROVIDE_FEEDBACK = "provide_feedback"
    SYSTEM_ALERT = "system_alert"

@dataclass
class Message:
    id: str
    type: MessageType
    sender: str
    receiver: str
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 1
    requires_response: bool = False

class TeamCoordinator:
    """Manages team coordination and task optimization"""
    
    def __init__(self, team_id: str):
        self.team_id = team_id
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.agent_status: Dict[str, Dict[str, Any]] = {}
        self.task_assignments: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}

    async def broadcast_message(self, sender: str, message_type: MessageType, content: Dict[str, Any]):
        """Broadcast message to all team members"""
        message = Message(
            id=self._generate_message_id(),
            type=message_type,
            sender=sender,
            receiver="all",
            content=content,
            timestamp=datetime.utcnow()
        )
        await self.message_queue.put(message)

    async def optimize_task_allocation(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Optimize task allocation based on agent capabilities and workload"""
        allocations = {}
        agent_workloads = self._calculate_workloads()

        for task in tasks:
            best_agent = self._find_best_agent(task, agent_workloads)
            if best_agent:
                if best_agent not in allocations:
                    allocations[best_agent] = []
                allocations[best_agent].append(task["id"])
                agent_workloads[best_agent] += self._estimate_task_load(task)

        return allocations

    def _find_best_agent(self, task: Dict[str, Any], workloads: Dict[str, float]) -> Optional[str]:
        """Find the best agent for a task based on capabilities and current workload"""
        best_agent = None
        best_score = float("-inf")

        for agent_id, status in self.agent_status.items():
            if not self._can_handle_task(agent_id, task):
                continue

            # Calculate score based on capability match and workload
            capability_score = self._calculate_capability_match(agent_id, task)
            workload_score = 1.0 / (1.0 + workloads.get(agent_id, 0))
            performance_score = self._get_agent_performance_score(agent_id)

            # Combined score with weights
            score = (
                0.4 * capability_score +
                0.3 * workload_score +
                0.3 * performance_score
            )

            if score > best_score:
                best_score = score
                best_agent = agent_id

        return best_agent

    def _calculate_capability_match(self, agent_id: str, task: Dict[str, Any]) -> float:
        """Calculate how well agent capabilities match task requirements"""
        agent_capabilities = set(self.agent_status[agent_id].get("capabilities", []))
        task_requirements = set(task.get("required_capabilities", []))

        if not task_requirements:
            return 1.0

        return len(agent_capabilities & task_requirements) / len(task_requirements)

    def _get_agent_performance_score(self, agent_id: str) -> float:
        """Get agent's performance score based on historical metrics"""
        metrics = self.performance_metrics.get(agent_id, {})
        if not metrics:
            return 0.5  # Default score for new agents

        # Calculate weighted average of different performance metrics
        weights = {
            "task_completion_rate": 0.4,
            "quality_score": 0.3,
            "cooperation_score": 0.3
        }

        score = sum(
            metrics.get(metric, 0.5) * weight
            for metric, weight in weights.items()
        )

        return score

    def _calculate_workloads(self) -> Dict[str, float]:
        """Calculate current workload for each agent"""
        workloads = {}
        for agent_id in self.agent_status:
            assigned_tasks = []
            for task_id, assignment in self.task_assignments.items():
                if assignment["agent_id"] == agent_id:
                    assigned_tasks.append(assignment["task"])
            
            workload = sum(self._estimate_task_load(task) for task in assigned_tasks)
            workloads[agent_id] = workload

        return workloads

    def _estimate_task_load(self, task: Dict[str, Any]) -> float:
        """Estimate computational/time load of a task"""
        # Implement task load estimation logic
        return 1.0  # Placeholder

    def _can_handle_task(self, agent_id: str, task: Dict[str, Any]) -> bool:
        """Check if agent can handle the task based on requirements"""
        agent_status = self.agent_status.get(agent_id, {})
        if not agent_status:
            return False

        # Check agent availability
        if not agent_status.get("available", False):
            return False

        # Check capability requirements
        required_capabilities = set(task.get("required_capabilities", []))
        agent_capabilities = set(agent_status.get("capabilities", []))
        
        return required_capabilities.issubset(agent_capabilities)

    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        return f"msg_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
