"""Multi-Agent Team System for Agentic OS
Manages teams of agents working on collaborative tasks."""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime

from .agent import Agent, AgentConfig
from ..utils.error_handling import TeamError
from ..utils.monitoring import monitor

@dataclass
class TeamConfig:
    name: str
    coordinator_agent: str
    member_agents: List[str]
    max_concurrent_tasks: int = 5
    collaboration_mode: str = "parallel"  # or "sequential"

class AgentTeam:
    """Manages a team of collaborative agents"""
    
    def __init__(self, config: TeamConfig):
        self.config = config
        self.agents: Dict[str, Agent] = {}
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self._initialize()

    def _initialize(self):
        """Initialize team structure"""
        # Create coordinator agent
        coordinator_config = AgentConfig(
            name=f"{self.config.name}_coordinator",
            capabilities=["coordination", "planning"]
        )
        self.coordinator = Agent(coordinator_config)
        
        # Create member agents
        for agent_id in self.config.member_agents:
            agent_config = AgentConfig(
                name=f"{self.config.name}_{agent_id}",
                capabilities=["execution", "collaboration"]
            )
            self.agents[agent_id] = Agent(agent_config)

    @monitor
    async def assign_task(self, task: Dict[str, Any]) -> str:
        """Assign new task to team"""
        try:
            # Generate task ID
            task_id = self._generate_task_id()
            
            # Create task context
            task_context = {
                "id": task_id,
                "status": "pending",
                "task": task,
                "assigned_agents": [],
                "timestamp": datetime.utcnow()
            }
            
            # Plan task execution
            execution_plan = await self._plan_execution(task_context)
            task_context["execution_plan"] = execution_plan
            
            # Add to task queue
            await self.task_queue.put(task_context)
            self.active_tasks[task_id] = task_context
            
            return task_id
            
        except Exception as e:
            raise TeamError(f"Failed to assign task: {str(e)}")

    async def _plan_execution(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan task execution using coordinator agent"""
        planning_input = {
            "task": "plan_execution",
            "parameters": {
                "task_context": task_context,
                "available_agents": list(self.agents.keys()),
                "collaboration_mode": self.config.collaboration_mode
            }
        }
        
        result = await self.coordinator.process(planning_input)
        return result.get("execution_plan")

    async def execute_tasks(self):
        """Execute tasks in the queue"""
        while True:
            # Get next task
            task_context = await self.task_queue.get()
            
            try:
                if self.config.collaboration_mode == "parallel":
                    await self._execute_parallel(task_context)
                else:
                    await self._execute_sequential(task_context)
                    
            except Exception as e:
                task_context["status"] = "failed"
                task_context["error"] = str(e)
            
            finally:
                self.task_queue.task_done()

    async def _execute_parallel(self, task_context: Dict[str, Any]):
        """Execute task with parallel agent collaboration"""
        execution_plan = task_context["execution_plan"]
        subtasks = execution_plan["subtasks"]
        
        # Create tasks for each agent
        tasks = [
            self._execute_subtask(subtask, agent_id)
            for subtask, agent_id in zip(subtasks, self.agents.keys())
        ]
        
        # Execute in parallel
        results = await asyncio.gather(*tasks)
        
        # Combine results
        task_context["results"] = results
        task_context["status"] = "completed"

    async def _execute_sequential(self, task_context: Dict[str, Any]):
        """Execute task with sequential agent collaboration"""
        execution_plan = task_context["execution_plan"]
        subtasks = execution_plan["subtasks"]
        
        results = []
        for subtask in subtasks:
            # Select agent for subtask
            agent_id = self._select_agent_for_subtask(subtask)
            
            # Execute subtask
            result = await self._execute_subtask(subtask, agent_id)
            results.append(result)
        
        task_context["results"] = results
        task_context["status"] = "completed"

    async def _execute_subtask(self, subtask: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        """Execute single subtask using specified agent"""
        agent = self.agents[agent_id]
        return await agent.process(subtask)

    def _select_agent_for_subtask(self, subtask: Dict[str, Any]) -> str:
        """Select appropriate agent for subtask based on capabilities"""
        # Implement agent selection logic
        return list(self.agents.keys())[0]  # Placeholder