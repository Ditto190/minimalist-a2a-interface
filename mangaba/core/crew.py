"""
Crew v3.0 — multi-agent orchestration with all process types.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from mangaba.core.agent import Agent
from mangaba.core.task import Task, TaskOutput
from mangaba.core.events import EventBus, Event, EventType, start_trace
from mangaba.core.exceptions import CrewError

log = logging.getLogger(__name__)


class Process(Enum):
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    PARALLEL = "parallel"
    CONSENSUAL = "consensual"


class CrewOutput:
    """Result of a crew execution, including metrics."""

    def __init__(
        self,
        tasks_outputs: List[TaskOutput],
        process: Process,
        duration: float,
        crew_id: str,
        token_usage: Optional[Dict[str, Any]] = None,
        usage_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tasks_outputs = tasks_outputs
        self.process = process
        self.duration = duration
        self.crew_id = crew_id
        self.token_usage: Dict[str, Any] = token_usage or {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.usage_metrics: Dict[str, Any] = usage_metrics or dict(self.token_usage)
        from datetime import datetime
        self.timestamp = datetime.now().isoformat()

    @property
    def final_output(self) -> str:
        if self.tasks_outputs:
            return self.tasks_outputs[-1].result
        return ""

    @property
    def raw(self) -> str:
        """CrewAI-compatible alias for the final string output."""
        return self.final_output

    @property
    def pydantic(self) -> Optional[Any]:
        if self.tasks_outputs:
            return getattr(self.tasks_outputs[-1], "pydantic", None)
        return None

    @property
    def json_dict(self) -> Optional[Dict[str, Any]]:
        if self.tasks_outputs:
            return getattr(self.tasks_outputs[-1], "json_dict", None)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crew_id": self.crew_id,
            "process": self.process.value,
            "duration": self.duration,
            "final_output": self.final_output,
            "tasks_outputs": [t.to_dict() if hasattr(t, "to_dict") else str(t) for t in self.tasks_outputs],
            "token_usage": self.token_usage,
            "usage_metrics": self.usage_metrics,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        return self.final_output


class Crew:
    """Orchestrate multiple agents working on multiple tasks.

    Example::

        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analyze_task, write_task],
            process=Process.SEQUENTIAL,
            verbose=True,
        )
        result = crew.kickoff(inputs={"topic": "AI trends"})

    Plan every task before starting, and put a dedicated manager in charge of
    delegation instead of promoting the first crew member::

        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analyze_task, write_task],
            process=Process.HIERARCHICAL,
            planning=True,
            manager_llm=create_llm_client(provider="openai", api_key=key),
            knowledge=shared_knowledge,
        )
    """

    def __init__(
        self,
        agents: List[Agent],
        tasks: List[Task],
        process: Process = Process.SEQUENTIAL,
        verbose: bool = False,
        max_rpm: Optional[int] = None,
        memory: Optional[Any] = None,
        crew_id: Optional[str] = None,
        planning: bool = False,
        planning_llm: Optional[Any] = None,
        manager_agent: Optional[Agent] = None,
        manager_llm: Optional[Any] = None,
        knowledge: Optional[Any] = None,
        # ── CrewAI-parity ──
        function_calling_llm: Optional[Any] = None,
        output_log_file: Optional[Any] = None,
        stream: bool = False,
        tracing: Optional[bool] = None,
        cache: bool = True,
        embedder: Optional[Any] = None,
        share_crew: bool = False,
        before_kickoff_callbacks: Optional[List[Any]] = None,
        after_kickoff_callbacks: Optional[List[Any]] = None,
        memory_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not agents:
            raise CrewError("Crew must have at least one agent")
        if not tasks:
            raise CrewError("Crew must have at least one task")

        self.crew_id = crew_id or f"crew_{uuid.uuid4().hex[:8]}"
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.verbose = verbose
        self.max_rpm = max_rpm
        self.memory = memory
        self.knowledge = knowledge
        self.function_calling_llm = function_calling_llm
        self.output_log_file = output_log_file
        self.stream = stream
        self.tracing = tracing
        self.cache = cache
        self.embedder = embedder
        self.share_crew = share_crew
        self.before_kickoff_callbacks: List[Any] = list(before_kickoff_callbacks or [])
        self.after_kickoff_callbacks: List[Any] = list(after_kickoff_callbacks or [])
        self.memory_config: Dict[str, Any] = dict(memory_config or {})
        self.usage_metrics: Dict[str, Any] = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._last_inputs: Optional[Dict[str, Any]] = None
        # memory=True builds a default Memory (CrewAI-parity); a dict
        # memory_config with provider/user_id selects the backend.
        if self.memory is True:
            self.memory = self._build_default_memory()

        # Pre-execution planning
        self.planning = planning
        self.planning_llm = planning_llm
        self.plans: Dict[str, Any] = {}

        # Hierarchical management
        self.manager_agent = manager_agent
        self.manager_llm = manager_llm

        self._validate_setup()
        self._connect_agents()
        self._share_knowledge()

        if self.verbose:
            log.info("Crew %s: %d agents, %d tasks, process=%s", self.crew_id, len(agents), len(tasks), process.value)

    # ── public API ─────────────────────────────────────────────────────

    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> CrewOutput:
        """Start the crew execution.

        Everything emitted during the run shares one trace id, so a tracing
        callback can stitch the whole crew together even when tasks run on
        separate threads.
        """
        inputs = self._run_callbacks(self.before_kickoff_callbacks, inputs or {})
        with start_trace() as trace_id:
            self.trace_id = trace_id
            result = self._execute_run(inputs)
        result = self._run_output_callbacks(self.after_kickoff_callbacks, result)
        self._write_output_log(result)
        return result

    async def akickoff(self, inputs: Optional[Dict[str, Any]] = None) -> CrewOutput:
        """Native async kickoff (CrewAI-parity). Runs blocking work in threads."""
        inputs = self._run_callbacks(self.before_kickoff_callbacks, inputs or {})
        with start_trace() as trace_id:
            self.trace_id = trace_id
            result = await asyncio.to_thread(self._execute_run, inputs)
        result = self._run_output_callbacks(self.after_kickoff_callbacks, result)
        self._write_output_log(result)
        return result

    def kickoff_for_each(self, inputs_list: List[Dict[str, Any]]) -> List[CrewOutput]:
        """Run the crew once per input dict (CrewAI-parity)."""
        return [self.kickoff(inputs) for inputs in inputs_list or [{}]]

    async def akickoff_for_each(self, inputs_list: List[Dict[str, Any]]) -> List[CrewOutput]:
        """Async version of :meth:`kickoff_for_each`."""
        results: List[CrewOutput] = []
        for inputs in inputs_list or [{}]:
            results.append(await self.akickoff(inputs))
        return results

    def replay(self, task_id: str, inputs: Optional[Dict[str, Any]] = None) -> CrewOutput:
        """Re-run from a specific task, reusing outputs of completed tasks."""
        start_idx = 0
        for i, task in enumerate(self.tasks):
            if task.task_id == task_id:
                start_idx = i
                break
        else:
            raise CrewError(f"No task with id '{task_id}' in this crew")
        # Replay = sequential run starting at start_idx
        saved = [t.output for t in self.tasks[:start_idx]]
        outputs: List[TaskOutput] = [o for o in saved if o is not None]
        for task in self.tasks[start_idx:]:
            outputs.append(task.execute(inputs or self._last_inputs or {}))
        import time as _time
        result = CrewOutput(tasks_outputs=outputs, process=self.process, duration=0.0, crew_id=self.crew_id)
        self._write_output_log(result)
        return result

    def train(self, iterations: int = 3, inputs: Optional[Dict[str, Any]] = None, filename: Optional[str] = None) -> Any:
        """Train the crew with human feedback (CrewAI-parity wrapper)."""
        from mangaba.training import CrewTrainer
        trainer = CrewTrainer(self, iterations=iterations, inputs=inputs or {}, filename=filename or "trained_agents_data.pkl", verbose=self.verbose)
        return trainer.train()

    def test(self, iterations: int = 2, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """Evaluate the crew with the LLM-as-judge evaluator."""
        from mangaba.training import CrewEvaluator
        evaluator = CrewEvaluator(self, iterations=iterations, inputs=inputs or {}, verbose=self.verbose)
        return evaluator.evaluate()

    def _execute_run(self, inputs: Optional[Dict[str, Any]] = None) -> CrewOutput:
        start = time.monotonic()
        self._last_inputs = dict(inputs or {})

        EventBus.emit(Event(
            event_type=EventType.CREW_START,
            source_id=self.crew_id,
            data={"process": self.process.value, "agents": len(self.agents), "tasks": len(self.tasks)},
        ))

        original_descriptions: Dict[str, str] = {}

        try:
            if self.planning:
                original_descriptions = self._apply_planning()

            if self.process == Process.SEQUENTIAL:
                outputs = self._run_sequential(inputs or {})
            elif self.process == Process.HIERARCHICAL:
                outputs = self._run_hierarchical(inputs or {})
            elif self.process == Process.PARALLEL:
                outputs = self._run_parallel(inputs or {})
            elif self.process == Process.CONSENSUAL:
                outputs = self._run_consensual(inputs or {})
            else:
                raise CrewError(f"Unknown process: {self.process}")

            duration = time.monotonic() - start
            token_usage = self._aggregate_token_usage()
            result = CrewOutput(tasks_outputs=outputs, process=self.process, duration=duration, crew_id=self.crew_id, token_usage=token_usage, usage_metrics=dict(token_usage))
            self.usage_metrics = dict(token_usage)

            EventBus.emit(Event(
                event_type=EventType.CREW_END,
                source_id=self.crew_id,
                data={"duration": duration, "tasks_completed": len(outputs)},
            ))
            return result

        except Exception as exc:
            EventBus.emit(Event(event_type=EventType.CREW_ERROR, source_id=self.crew_id, data={"error": str(exc)}))
            raise

        finally:
            # Leave the crew reusable — planning mutates task descriptions
            for task_id, description in original_descriptions.items():
                for task in self.tasks:
                    if task.task_id == task_id:
                        task.description = description

    # ── process implementations ────────────────────────────────────────

    def _run_sequential(self, inputs: Dict[str, Any]) -> List[TaskOutput]:
        outputs: List[TaskOutput] = []
        for i, task in enumerate(self.tasks, 1):
            if self.verbose:
                log.info("[%d/%d] %s → %s", i, len(self.tasks), task.agent.role, task.description[:60])
            output = task.execute(inputs)
            outputs.append(output)
        return outputs

    def _run_hierarchical(self, inputs: Dict[str, Any]) -> List[TaskOutput]:
        """Manager plans, picks a worker per task, then reviews the result.

        The manager is a dedicated agent when ``manager_agent`` or
        ``manager_llm`` is given; otherwise the first crew member takes the
        role, which is why that fallback still needs two agents.
        """
        manager = self._resolve_manager()
        workers = [a for a in self.agents if a is not manager]
        if not workers:
            raise CrewError(
                "Hierarchical process needs at least one worker agent besides the manager"
            )

        outputs: List[TaskOutput] = []

        for i, task in enumerate(self.tasks, 1):
            worker = self._select_worker(manager, task, workers)

            if self.verbose:
                log.info("[%d/%d] Manager %s delegates to %s", i, len(self.tasks), manager.role, worker.role)

            # Manager turns the task into instructions for this specific worker
            refined = manager.execute_task(
                f"You are the manager of this crew. Refine the task below into clear, "
                f"actionable instructions for one worker.\n"
                f"Worker role: {worker.role}\n"
                f"Worker background: {worker.backstory}\n"
                f"Task: {task.description}\n"
                f"Expected output: {task.expected_output}\n"
                f"Write only the instructions."
            )

            original_desc = task.description
            original_agent = task.agent
            task.description = f"{refined}\n\nOriginal task: {original_desc}"
            task.agent = worker

            try:
                output = task.execute(inputs)

                # Manager reviews and can send the work back once
                verdict = manager.execute_task(
                    f"Review this worker output.\n"
                    f"Task: {original_desc}\n"
                    f"Expected output: {task.expected_output}\n"
                    f"Output: {output.result[:2000]}\n\n"
                    f"If it satisfies the task, reply exactly APPROVED. "
                    f"Otherwise reply REVISE followed by what must change."
                )

                if verdict.strip().upper().startswith("REVISE"):
                    notes = verdict.strip()[len("REVISE"):].strip(" :\n")
                    log.info("Manager requested a revision on task %d", i)
                    task.description = (
                        f"{original_desc}\n\n"
                        f"Your manager rejected the previous attempt: {notes}\n"
                        f"Produce a corrected version."
                    )
                    output = task.execute(inputs)

            finally:
                task.description = original_desc
                task.agent = original_agent

            outputs.append(output)
        return outputs

    # ── management & planning ──────────────────────────────────────────

    def _resolve_manager(self) -> Agent:
        """Return the agent that runs the hierarchical process."""
        if self.manager_agent is not None:
            return self.manager_agent

        if self.manager_llm is not None:
            return Agent(
                role="Crew Manager",
                goal="Plan the work, delegate each task to the right specialist, and verify the results",
                backstory=(
                    "An experienced delivery lead who breaks work down, matches it to the "
                    "person best equipped to do it, and holds the bar on quality."
                ),
                llm=self.manager_llm,
                allow_delegation=True,
                verbose=self.verbose,
            )

        if len(self.agents) < 2:
            raise CrewError(
                "Hierarchical process needs >= 2 agents, or an explicit "
                "manager_agent / manager_llm"
            )
        return self.agents[0]

    def _select_worker(self, manager: Agent, task: Task, workers: List[Agent]) -> Agent:
        """Let the manager choose who should handle *task*.

        Falls back to the task's pre-assigned agent when the manager's answer
        doesn't name a real worker.
        """
        if len(workers) == 1:
            return workers[0]

        if task.agent in workers and self.manager_agent is None and self.manager_llm is None:
            # Historical behaviour: honour the explicit assignment
            return task.agent

        roster = "\n".join(f"- {w.role}: {w.goal}" for w in workers)
        try:
            answer = manager.execute_task(
                f"Choose which team member should handle this task.\n\n"
                f"Team:\n{roster}\n\n"
                f"Task: {task.description}\n"
                f"Expected output: {task.expected_output}\n\n"
                f"Reply with the role name only."
            )
        except Exception as exc:
            log.warning("Manager could not pick a worker (%s) — using the assigned agent", exc)
            return task.agent if task.agent in workers else workers[0]

        chosen = answer.strip().lower()
        for worker in workers:
            if worker.role.lower() in chosen or chosen in worker.role.lower():
                return worker

        log.debug("Manager's pick %r matched no worker — using the assigned agent", answer[:80])
        return task.agent if task.agent in workers else workers[0]

    def _apply_planning(self) -> Dict[str, str]:
        """Draft a plan per task and fold it into the task description.

        Returns the original descriptions so ``kickoff`` can restore them.
        """
        from mangaba.core.planner import TaskPlanner

        llm = self.planning_llm or self.agents[0].llm
        originals: Dict[str, str] = {}

        for task in self.tasks:
            tools = task.tools or (task.agent.tools if task.agent else [])
            planner = TaskPlanner(llm=llm, tools=tools)

            try:
                plan = planner.plan(f"{task.description}\n\nExpected output: {task.expected_output}")
            except Exception as exc:
                log.warning("Planning failed for task %s: %s", task.task_id, exc)
                continue

            if not plan.steps:
                continue

            self.plans[task.task_id] = plan
            originals[task.task_id] = task.description
            rendered = "\n".join(f"{s.step_number}. {s.description}" for s in plan.steps)
            task.description = f"{task.description}\n\nSuggested plan:\n{rendered}"

            EventBus.emit(Event(
                event_type=EventType.PLAN_CREATED,
                source_id=self.crew_id,
                data={"task_id": task.task_id, "steps": len(plan.steps)},
            ))

        if self.verbose and self.plans:
            log.info("Planned %d/%d tasks", len(self.plans), len(self.tasks))

        return originals

    def _share_knowledge(self) -> None:
        """Give every agent the crew's knowledge base unless it brought its own."""
        if self.knowledge is None:
            return
        for agent in self.agents:
            if getattr(agent, "knowledge", None) is None:
                agent.knowledge = self.knowledge

    def _run_parallel(self, inputs: Dict[str, Any]) -> List[TaskOutput]:
        """Execute independent tasks concurrently using asyncio."""

        async def _run() -> List[TaskOutput]:
            coros = [task.aexecute(inputs) for task in self.tasks]
            return list(await asyncio.gather(*coros))

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an event loop — fall back to sequential
            return self._run_sequential(inputs)

        return asyncio.run(_run())

    def _run_consensual(self, inputs: Dict[str, Any]) -> List[TaskOutput]:
        """All agents independently execute each task; results are merged."""
        outputs: List[TaskOutput] = []

        for task in self.tasks:
            agent_results: List[str] = []
            for agent in self.agents:
                original_agent = task.agent
                task.agent = agent
                try:
                    out = task.execute(inputs)
                    agent_results.append(f"[{agent.role}]: {out.result}")
                finally:
                    task.agent = original_agent

            # Use first agent to synthesise consensus
            synthesis_prompt = (
                "Multiple experts provided their analysis. Synthesise a consensus.\n\n"
                + "\n---\n".join(agent_results)
            )
            consensus = self.agents[0].execute_task(synthesis_prompt)
            outputs.append(TaskOutput(
                description=task.description,
                result=consensus,
                agent="consensus",
                success=True,
            ))
        return outputs

    # ── internal ───────────────────────────────────────────────────────

    def _validate_setup(self) -> None:
        # Under a manager, tasks may be left unassigned — the manager picks
        manager_assigns = self.process == Process.HIERARCHICAL

        for task in self.tasks:
            if not task.agent:
                if manager_assigns:
                    continue
                raise CrewError(f"Task '{task.description[:50]}...' has no agent assigned")
            if task.agent not in self.agents:
                raise CrewError(f"Task agent '{task.agent.role}' not in crew's agent list")
            for dep in task.context:
                if dep not in self.tasks:
                    raise CrewError("Task has a dependency on a task that is not in this crew")

    def _connect_agents(self) -> None:
        roster = list(self.agents)
        if self.manager_agent is not None and self.manager_agent not in roster:
            roster.append(self.manager_agent)

        for i, a1 in enumerate(roster):
            for a2 in roster[i + 1 :]:
                a1.connect_to(a2)

    # ── CrewAI-parity helpers ────────────────────────────────────────

    def _run_callbacks(self, callbacks: List[Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(inputs or {})
        for cb in callbacks:
            try:
                out = cb(data)
                if isinstance(out, dict):
                    data = out
            except Exception as exc:
                log.warning("before_kickoff callback failed: %s", exc)
        return data

    def _run_output_callbacks(self, callbacks: List[Any], result: CrewOutput) -> CrewOutput:
        for cb in callbacks:
            try:
                out = cb(result)
                if isinstance(out, CrewOutput):
                    result = out
            except Exception as exc:
                log.warning("after_kickoff callback failed: %s", exc)
        return result

    def _write_output_log(self, result: CrewOutput) -> None:
        if not self.output_log_file:
            return
        try:
            path = self.output_log_file if isinstance(self.output_log_file, str) else "crew_output.txt"
            if path is True:
                path = "crew_output.txt"
            import os
            directory = os.path.dirname(str(path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(str(path), "w", encoding="utf-8") as f:
                f.write(result.final_output)
        except Exception as exc:
            log.warning("Could not write output_log_file: %s", exc)

    def _aggregate_token_usage(self) -> Dict[str, Any]:
        total = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        for agent in self.agents:
            llm = getattr(agent, "llm", None)
            usage = getattr(llm, "total_usage", None) if llm is not None else None
            if usage is not None:
                try:
                    d = usage if isinstance(usage, dict) else usage.model_dump() if hasattr(usage, "model_dump") else {}
                    for k in total:
                        total[k] += int(d.get(k, 0) or 0)
                except Exception:
                    continue
        return total

    def _build_default_memory(self) -> Any:
        """Create a Memory from ``memory_config`` (provider/user_id/embedder)."""
        provider = self.memory_config.get("provider")
        # An already-built memory instance (incl. external) passes through.
        if provider is not None and not isinstance(provider, (str, dict)):
            return provider
        try:
            from mangaba.memory import Memory
        except Exception:
            return True
        kwargs: Dict[str, Any] = {}
        for key in ("user_id", "namespace", "db_path", "embedder", "embedding", "weights", "half_life_hours", "recency_half_life_days"):
            if key in self.memory_config:
                kwargs[key] = self.memory_config[key]
        if self.embedder is not None and "embedder" not in kwargs and "embedding" not in kwargs:
            kwargs["embedder"] = self.embedder
        # External provider by name: {"provider": "mem0", "config": {...}}
        if isinstance(provider, str) and provider.lower() in ("mem0", "external"):
            try:
                from mangaba.memory.external import Mem0Memory
                cfg = self.memory_config.get("config", {})
                return Mem0Memory(config=cfg)
            except Exception as exc:
                log.warning("Could not build Mem0 memory (%s) — using local Memory", exc)
        return Memory(**kwargs)

    def __repr__(self) -> str:
        return f"Crew(agents={len(self.agents)}, tasks={len(self.tasks)}, process={self.process.value})"
