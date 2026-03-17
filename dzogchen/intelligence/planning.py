"""
PlanningEngine - Hierarchical task planning and execution
=========================================================

Implements STRIPS-style planning enhanced with LLM reasoning:
- Goal decomposition into subgoals
- Action selection and sequencing
- Constraint satisfaction
- Dynamic replanning on failure
- Progress monitoring
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PlanStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    REPLANNING = "replanning"


@dataclass
class Action:
    """A concrete action that can be executed."""
    name: str
    description: str
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    tool_name: Optional[str] = None
    tool_params: Dict[str, Any] = field(default_factory=dict)
    estimated_time: float = 1.0
    cost: float = 0.0
    reversible: bool = True


@dataclass
class PlanStep:
    """A step in an execution plan."""
    step_id: str
    goal: str
    actions: List[Action]
    dependencies: List[str] = field(default_factory=list)  # step IDs this depends on
    status: PlanStatus = PlanStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    subtasks: List["PlanStep"] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def can_execute(self, completed_step_ids: Set[str]) -> bool:
        """Check if this step's dependencies are satisfied."""
        return all(dep in completed_step_ids for dep in self.dependencies)


@dataclass
class ExecutionPlan:
    """A complete execution plan for achieving a goal."""
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    total_estimated_time: float = 0.0
    total_cost: float = 0.0
    replan_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_ready_steps(self, completed_ids: Optional[Set[str]] = None) -> List[PlanStep]:
        """Get steps that are ready to execute."""
        if completed_ids is None:
            completed_ids = {
                s.step_id for s in self.steps
                if s.status == PlanStatus.COMPLETED
            }
        return [
            s for s in self.steps
            if s.status == PlanStatus.PENDING and s.can_execute(completed_ids)
        ]

    def get_progress(self) -> float:
        """Get plan completion progress (0.0 - 1.0)."""
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == PlanStatus.COMPLETED)
        return completed / len(self.steps)

    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(s.status == PlanStatus.COMPLETED for s in self.steps)

    def get_failed_steps(self) -> List[PlanStep]:
        """Get all failed steps."""
        return [s for s in self.steps if s.status == PlanStatus.FAILED]


class PlanningEngine:
    """
    Hierarchical task planning engine with dynamic replanning.

    Creates multi-step plans for complex goals, manages dependencies,
    handles failures with graceful replanning, and monitors execution.
    """

    PLANNING_PROMPT = """You are an expert task planner. Create a detailed execution plan.

Goal: {goal}
Available Tools: {tools}
Current Context: {context}
Constraints: {constraints}

Create a plan with the following format:
STEP 1: [Goal]
  - Prerequisite: [step IDs this depends on, or "none"]
  - Actions: [list of concrete actions using available tools]
  - Success criteria: [how to verify completion]

STEP 2: ...

Make the plan:
- Minimal: only steps needed
- Ordered: respect dependencies
- Parallel: identify steps that can run concurrently
- Robust: include error handling steps for critical actions"""

    REPLAN_PROMPT = """The execution plan has encountered an issue. Replan.

Original Goal: {goal}
Failed Step: {failed_step}
Failure Reason: {reason}
Completed Steps: {completed}
Remaining Steps: {remaining}

Create an updated plan to recover and complete the goal.
Focus on working around the failure while achieving the original goal."""

    def __init__(
        self,
        model_caller: Optional[Callable] = None,
        max_steps: int = 50,
        max_replan_attempts: int = 3,
        enable_parallel_execution: bool = True,
    ):
        self._model_caller = model_caller
        self.max_steps = max_steps
        self.max_replan_attempts = max_replan_attempts
        self.enable_parallel_execution = enable_parallel_execution
        self._execution_history: List[ExecutionPlan] = []

    async def plan(
        self,
        goal: str,
        available_tools: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[str]] = None,
    ) -> ExecutionPlan:
        """
        Create an execution plan for the given goal.

        Args:
            goal: The high-level goal to achieve
            available_tools: Tools available for use in the plan
            context: Current context information
            constraints: Constraints the plan must satisfy

        Returns:
            A detailed ExecutionPlan
        """
        logger.info(f"Planning for goal: {goal}")

        plan = ExecutionPlan(goal=goal)

        if self._model_caller:
            steps = await self._llm_plan(goal, available_tools, context, constraints)
            plan.steps = steps
        else:
            # Fallback: simple decomposition
            plan.steps = self._simple_decompose(goal)

        # Calculate estimates
        plan.total_estimated_time = sum(
            sum(a.estimated_time for a in step.actions)
            for step in plan.steps
        )
        plan.total_cost = sum(
            sum(a.cost for a in step.actions)
            for step in plan.steps
        )

        logger.info(f"Plan created: {len(plan.steps)} steps, ~{plan.total_estimated_time:.1f}s estimated")
        return plan

    async def execute(
        self,
        plan: ExecutionPlan,
        action_executor: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
    ) -> ExecutionPlan:
        """
        Execute a plan, handling failures and replanning as needed.

        Args:
            plan: The plan to execute
            action_executor: Callable to execute individual actions
            progress_callback: Optional callback for progress updates

        Returns:
            Updated plan with execution results
        """
        plan.status = PlanStatus.IN_PROGRESS
        completed_ids: Set[str] = set()
        replan_attempts = 0

        while not plan.is_complete():
            ready_steps = plan.get_ready_steps(completed_ids)

            if not ready_steps:
                failed = plan.get_failed_steps()
                if failed:
                    # Try replanning
                    if replan_attempts < self.max_replan_attempts:
                        replan_attempts += 1
                        logger.warning(f"Steps failed, attempting replan ({replan_attempts}/{self.max_replan_attempts})")
                        plan = await self._replan(plan, failed[0])
                        plan.replan_count = replan_attempts
                        continue
                    else:
                        plan.status = PlanStatus.FAILED
                        break
                else:
                    # Deadlock or all complete
                    break

            if self.enable_parallel_execution and len(ready_steps) > 1:
                # Execute ready steps in parallel
                tasks = [
                    self._execute_step(step, action_executor)
                    for step in ready_steps
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Execute sequentially
                await self._execute_step(ready_steps[0], action_executor)

            # Update completed set
            newly_completed = [
                s.step_id for s in ready_steps
                if s.status == PlanStatus.COMPLETED
            ]
            completed_ids.update(newly_completed)

            if progress_callback:
                await self._notify_progress(plan, progress_callback)

        if plan.is_complete():
            plan.status = PlanStatus.COMPLETED
            plan.completed_at = time.time()

        self._execution_history.append(plan)
        return plan

    async def decompose_goal(self, goal: str, depth: int = 0) -> List[PlanStep]:
        """
        Recursively decompose a high-level goal into concrete steps.

        Args:
            goal: The goal to decompose
            depth: Current recursion depth

        Returns:
            List of PlanStep objects
        """
        if depth > 3:  # Max decomposition depth
            return [PlanStep(
                step_id=f"step_{depth}_0",
                goal=goal,
                actions=[Action(name="atomic_action", description=goal)],
            )]

        if self._model_caller:
            sub_goals = await self._extract_subgoals(goal)
        else:
            sub_goals = [goal]

        steps = []
        for i, sub_goal in enumerate(sub_goals):
            step = PlanStep(
                step_id=f"step_{depth}_{i}",
                goal=sub_goal,
                actions=[Action(name=f"action_{i}", description=sub_goal)],
                dependencies=[f"step_{depth}_{i-1}"] if i > 0 else [],
            )
            steps.append(step)

        return steps

    async def _llm_plan(
        self,
        goal: str,
        available_tools: Optional[List[str]],
        context: Optional[Dict[str, Any]],
        constraints: Optional[List[str]],
    ) -> List[PlanStep]:
        """Use LLM to create a detailed plan."""
        prompt = self.PLANNING_PROMPT.format(
            goal=goal,
            tools=", ".join(available_tools or ["none"]),
            context=str(context or {}),
            constraints="\n".join(constraints or ["none"]),
        )

        try:
            response = await self._model_caller(prompt)
            return self._parse_plan_response(response, goal)
        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return self._simple_decompose(goal)

    def _parse_plan_response(self, response: str, goal: str) -> List[PlanStep]:
        """Parse LLM plan response into PlanStep objects."""
        steps = []
        current_step_text = []
        step_num = 0

        for line in response.split("\n"):
            if line.startswith("STEP "):
                if current_step_text:
                    step = self._parse_step_block(current_step_text, step_num, goal)
                    steps.append(step)
                current_step_text = [line]
                step_num += 1
            elif current_step_text:
                current_step_text.append(line)

        if current_step_text:
            step = self._parse_step_block(current_step_text, step_num, goal)
            steps.append(step)

        # Fallback if parsing fails
        if not steps:
            return self._simple_decompose(goal)

        return steps

    def _parse_step_block(self, lines: List[str], step_num: int, parent_goal: str) -> PlanStep:
        """Parse a block of lines into a PlanStep."""
        goal_line = lines[0] if lines else f"Step {step_num}"
        goal = goal_line.split(":", 1)[-1].strip() if ":" in goal_line else goal_line.strip()

        deps = []
        actions = []

        for line in lines[1:]:
            line = line.strip()
            if "Prerequisite:" in line:
                prereq = line.split(":", 1)[-1].strip()
                if prereq.lower() not in ("none", ""):
                    deps = [d.strip() for d in prereq.split(",")]
            elif "Actions:" in line or (line.startswith("-") and not deps):
                action_text = line.lstrip("- ").replace("Actions:", "").strip()
                if action_text:
                    actions.append(Action(
                        name=f"action_{step_num}_{len(actions)}",
                        description=action_text,
                    ))

        if not actions:
            actions = [Action(name=f"action_{step_num}", description=goal)]

        return PlanStep(
            step_id=f"step_{step_num}",
            goal=goal,
            actions=actions,
            dependencies=deps,
        )

    def _simple_decompose(self, goal: str) -> List[PlanStep]:
        """Simple fallback decomposition without LLM."""
        return [
            PlanStep(
                step_id="step_0",
                goal=f"Analyze and understand: {goal}",
                actions=[Action(name="analyze", description="Analyze the goal")],
            ),
            PlanStep(
                step_id="step_1",
                goal=f"Execute: {goal}",
                actions=[Action(name="execute", description="Execute the plan")],
                dependencies=["step_0"],
            ),
            PlanStep(
                step_id="step_2",
                goal="Verify completion",
                actions=[Action(name="verify", description="Verify goal was achieved")],
                dependencies=["step_1"],
            ),
        ]

    async def _execute_step(
        self,
        step: PlanStep,
        action_executor: Optional[Callable],
    ) -> None:
        """Execute a single plan step."""
        step.status = PlanStatus.IN_PROGRESS
        step.start_time = time.time()

        try:
            results = []
            for action in step.actions:
                if action_executor:
                    result = await action_executor(action)
                    results.append(result)
                else:
                    results.append({"simulated": True, "action": action.name})

            step.result = results
            step.status = PlanStatus.COMPLETED
            logger.debug(f"Step {step.step_id} completed: {step.goal}")

        except Exception as e:
            step.error = str(e)
            step.status = PlanStatus.FAILED
            logger.error(f"Step {step.step_id} failed: {e}")

        finally:
            step.end_time = time.time()

    async def _replan(self, plan: ExecutionPlan, failed_step: PlanStep) -> ExecutionPlan:
        """Create a new plan to recover from a failure."""
        if self._model_caller is None:
            # Simple retry: reset failed step
            failed_step.status = PlanStatus.PENDING
            failed_step.error = None
            return plan

        completed = [s.step_id for s in plan.steps if s.status == PlanStatus.COMPLETED]
        remaining = [s.goal for s in plan.steps if s.status == PlanStatus.PENDING]

        prompt = self.REPLAN_PROMPT.format(
            goal=plan.goal,
            failed_step=f"{failed_step.step_id}: {failed_step.goal}",
            reason=failed_step.error or "Unknown error",
            completed=", ".join(completed),
            remaining=", ".join(remaining),
        )

        try:
            response = await self._model_caller(prompt)
            new_steps = self._parse_plan_response(response, plan.goal)

            # Keep completed steps, replace rest
            kept = [s for s in plan.steps if s.status == PlanStatus.COMPLETED]
            new_plan = ExecutionPlan(goal=plan.goal, steps=kept + new_steps)
            new_plan.status = PlanStatus.IN_PROGRESS
            return new_plan

        except Exception as e:
            logger.error(f"Replanning failed: {e}")
            # Reset the failed step and try again
            failed_step.status = PlanStatus.PENDING
            failed_step.error = None
            return plan

    async def _extract_subgoals(self, goal: str) -> List[str]:
        """Extract subgoals from a high-level goal using LLM."""
        prompt = f"""Break down this goal into 3-5 concrete, sequential subgoals:
Goal: {goal}

Reply with a numbered list of subgoals only."""

        try:
            response = await self._model_caller(prompt)
            subgoals = [
                line.strip().lstrip("0123456789. ")
                for line in response.split("\n")
                if line.strip() and line.strip()[0].isdigit()
            ]
            return subgoals[:5] or [goal]
        except Exception:
            return [goal]

    async def _notify_progress(self, plan: ExecutionPlan, callback: Callable) -> None:
        """Notify progress callback."""
        try:
            progress_info = {
                "goal": plan.goal,
                "progress": plan.get_progress(),
                "status": plan.status.value,
                "completed_steps": len([s for s in plan.steps if s.status == PlanStatus.COMPLETED]),
                "total_steps": len(plan.steps),
            }
            if asyncio.iscoroutinefunction(callback):
                await callback(progress_info)
            else:
                callback(progress_info)
        except Exception as e:
            logger.warning(f"Progress callback error: {e}")

    def get_planning_stats(self) -> Dict[str, Any]:
        """Get statistics about planning history."""
        if not self._execution_history:
            return {"total_plans": 0}

        completed = [p for p in self._execution_history if p.status == PlanStatus.COMPLETED]
        failed = [p for p in self._execution_history if p.status == PlanStatus.FAILED]

        return {
            "total_plans": len(self._execution_history),
            "completed": len(completed),
            "failed": len(failed),
            "success_rate": len(completed) / len(self._execution_history),
            "avg_steps": sum(len(p.steps) for p in self._execution_history) / len(self._execution_history),
            "avg_replan_count": sum(p.replan_count for p in self._execution_history) / len(self._execution_history),
        }
