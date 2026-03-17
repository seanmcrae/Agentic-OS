"""
Dzogchen CLI - Command-line interface for the Dzogchen framework
================================================================

Usage:
    dzogchen think "What is the meaning of life?"
    dzogchen act "Write a Python function to sort a list"
    dzogchen swarm "Analyze the pros and cons of quantum computing"
    dzogchen serve --port 8080
    dzogchen status
    dzogchen chat  # Interactive chat mode
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Optional
import logging


def create_cli():
    """Create and configure the CLI application."""
    try:
        import click
    except ImportError:
        raise ImportError("click not found. Install with: pip install click")

    @click.group()
    @click.option("--debug", is_flag=True, help="Enable debug logging")
    @click.option("--model", default=None, help="Override primary model")
    @click.pass_context
    def app(ctx, debug: bool, model: Optional[str]):
        """
        Dzogchen AI - Super-Intelligent AI Framework

        Named after the Tibetan Buddhist concept of 'Great Perfection'.
        A unified intelligence that integrates reasoning, memory, and agency.
        """
        ctx.ensure_object(dict)
        ctx.obj["debug"] = debug
        ctx.obj["model"] = model

        if debug:
            logging.basicConfig(level=logging.DEBUG)

    @app.command()
    @click.argument("query")
    @click.option("--strategy", "-s", default="cot",
                  type=click.Choice(["cot", "tot", "self_consistency", "deductive", "inductive"]),
                  help="Reasoning strategy")
    @click.option("--no-reflect", is_flag=True, help="Disable self-reflection")
    @click.option("--stream", is_flag=True, help="Stream reasoning steps")
    @click.option("--json", "output_json", is_flag=True, help="Output as JSON")
    @click.pass_context
    def think(ctx, query: str, strategy: str, no_reflect: bool, stream: bool, output_json: bool):
        """Apply multi-step reasoning to a query."""
        async def _run():
            from dzogchen import Dzogchen
            async with Dzogchen() as dz:
                if stream:
                    click.echo(click.style("Reasoning...", fg="cyan", bold=True))
                    async for step in dz.stream_think(query):
                        click.echo(click.style(step, fg="yellow"), nl=False)
                    click.echo()
                else:
                    with click.progressbar(length=1, label="Thinking") as bar:
                        response = await dz.think(query, strategy=strategy, reflect=not no_reflect)
                        bar.update(1)

                    if output_json:
                        click.echo(json.dumps({"query": query, "response": response}, indent=2))
                    else:
                        click.echo()
                        click.echo(click.style("Response:", fg="green", bold=True))
                        click.echo(response)

        asyncio.run(_run())

    @app.command()
    @click.argument("task")
    @click.option("--tools", "-t", multiple=True, help="Tool names to use")
    @click.option("--json", "output_json", is_flag=True, help="Output as JSON")
    @click.pass_context
    def act(ctx, task: str, tools: tuple, output_json: bool):
        """Execute a task autonomously using agents and tools."""
        async def _run():
            from dzogchen import Dzogchen
            async with Dzogchen() as dz:
                click.echo(click.style("Executing task...", fg="cyan"))
                result = await dz.act(task, tools=list(tools) if tools else None)

                if output_json:
                    click.echo(json.dumps(result, indent=2, default=str))
                else:
                    status = click.style("SUCCESS", fg="green") if result["success"] else click.style("FAILED", fg="red")
                    click.echo(f"\nStatus: {status}")
                    click.echo(f"Iterations: {result['iterations']}")
                    click.echo(f"Tools used: {', '.join(result['tools_used']) or 'none'}")
                    click.echo(f"Time: {result['execution_time']:.2f}s")
                    click.echo()
                    click.echo(click.style("Result:", fg="green", bold=True))
                    click.echo(str(result["result"]))

        asyncio.run(_run())

    @app.command()
    @click.argument("problem")
    @click.option("--size", "-n", default=3, type=int, help="Swarm size (1-10)")
    @click.option("--iterations", "-i", default=2, type=int, help="Collective iterations")
    @click.option("--json", "output_json", is_flag=True)
    @click.pass_context
    def swarm(ctx, problem: str, size: int, iterations: int, output_json: bool):
        """Apply swarm intelligence to a complex problem."""
        async def _run():
            from dzogchen import Dzogchen
            async with Dzogchen() as dz:
                click.echo(click.style(
                    f"Deploying swarm of {size} agents...",
                    fg="cyan"
                ))
                result = await dz.swarm(problem, swarm_size=size, iterations=iterations)

                if output_json:
                    click.echo(json.dumps(result, indent=2, default=str))
                else:
                    click.echo(click.style(
                        f"\nSwarm consensus (confidence={result['confidence']:.2f}):",
                        fg="green", bold=True
                    ))
                    click.echo(str(result["solution"]))
                    click.echo(click.style(
                        f"\n[{result['agents']} agents, {result['solutions_explored']} solutions explored]",
                        fg="blue"
                    ))

        asyncio.run(_run())

    @app.command()
    @click.argument("goal")
    @click.option("--execute", "-e", is_flag=True, help="Execute the plan")
    @click.option("--json", "output_json", is_flag=True)
    @click.pass_context
    def plan(ctx, goal: str, execute: bool, output_json: bool):
        """Create a multi-step execution plan for a goal."""
        async def _run():
            from dzogchen import Dzogchen
            async with Dzogchen() as dz:
                result = await dz.plan(goal, execute=execute)

                if output_json:
                    click.echo(json.dumps(result, indent=2))
                else:
                    click.echo(click.style(f"\nPlan for: {goal}", fg="blue", bold=True))
                    click.echo(click.style(f"Steps: {len(result['steps'])}", fg="cyan"))
                    click.echo()
                    for step in result["steps"]:
                        click.echo(f"  [{step['id']}] {step['goal']}")
                        if step["dependencies"]:
                            click.echo(f"       Depends on: {', '.join(step['dependencies'])}")
                    if execute:
                        progress = result.get("progress", 0)
                        click.echo()
                        click.echo(click.style(
                            f"Execution: {result.get('execution_status')} ({progress:.0%} complete)",
                            fg="green"
                        ))

        asyncio.run(_run())

    @app.command()
    @click.option("--port", "-p", default=8080, type=int, help="Server port")
    @click.option("--host", default="0.0.0.0", help="Server host")
    @click.option("--workers", "-w", default=1, type=int, help="Number of workers")
    @click.option("--reload", is_flag=True, help="Auto-reload on code changes")
    def serve(port: int, host: str, workers: int, reload: bool):
        """Start the Dzogchen API server."""
        try:
            import uvicorn
        except ImportError:
            click.echo(click.style("Error: uvicorn not found. Install with: pip install uvicorn", fg="red"))
            sys.exit(1)

        from dzogchen.api.server import create_app

        click.echo(click.style(
            f"\n  Dzogchen API Server\n  http://{host}:{port}\n",
            fg="cyan", bold=True
        ))
        click.echo(f"  Docs: http://{host}:{port}/docs")
        click.echo(f"  Status: http://{host}:{port}/status\n")

        app = create_app()
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level="info",
        )

    @app.command()
    @click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
    def status(format: str):
        """Show current system status."""
        async def _run():
            from dzogchen import Dzogchen
            async with Dzogchen() as dz:
                s = dz.get_status()
                if format == "json":
                    click.echo(json.dumps(s, indent=2, default=str))
                else:
                    click.echo(click.style("\nDzogchen Status", fg="cyan", bold=True))
                    click.echo(f"  Version:    {s['version']}")
                    click.echo(f"  Session:    {s['session_id']}")
                    click.echo(f"  Model:      {s['model']}")
                    click.echo(f"  Uptime:     {s['uptime_seconds']:.1f}s")
                    click.echo()
                    mem = s.get("memory", {})
                    click.echo(click.style("  Memory:", fg="blue"))
                    working = mem.get("working", {})
                    click.echo(f"    Working:  {working.get('size', 0)}/{working.get('capacity', 7)}")
                    episodic = mem.get("episodic", {})
                    click.echo(f"    Episodic: {episodic.get('total', 0)} episodes")
                    semantic = mem.get("semantic", {})
                    click.echo(f"    Semantic: {semantic.get('total_entries', 0)} entries")

        asyncio.run(_run())

    @app.command()
    @click.option("--session", "-s", default=None, help="Session ID to resume")
    def chat(session: Optional[str]):
        """Start an interactive chat session."""
        async def _run():
            from dzogchen import Dzogchen
            async with Dzogchen() as dz:
                if session:
                    dz.memory.set_session(session)

                click.echo(click.style("\nDzogchen Interactive Chat", fg="cyan", bold=True))
                click.echo(click.style("Type 'exit' to quit, '/status' for status, '/help' for commands\n", fg="blue"))

                while True:
                    try:
                        user_input = click.prompt(click.style("You", fg="green", bold=True))
                    except (click.exceptions.Abort, EOFError):
                        break

                    if user_input.lower() in ("exit", "quit", "/exit"):
                        break

                    if user_input.startswith("/"):
                        await _handle_command(dz, user_input)
                        continue

                    click.echo(click.style("Dzogchen: ", fg="cyan", bold=True), nl=False)
                    response = await dz.think(user_input)
                    click.echo(response)
                    click.echo()

                click.echo(click.style("\nGoodbye!", fg="cyan"))

        async def _handle_command(dz, command: str):
            """Handle special chat commands."""
            cmd = command.lower().split()[0]

            if cmd == "/status":
                s = dz.get_status()
                click.echo(json.dumps(s, indent=2, default=str))
            elif cmd == "/clear":
                dz.memory.working.clear()
                click.echo(click.style("Working memory cleared.", fg="yellow"))
            elif cmd == "/help":
                click.echo(click.style("""
Commands:
  /status    - Show system status
  /clear     - Clear working memory
  /act <task> - Execute a task with agents
  /swarm <problem> - Use swarm intelligence
  /learn <concept>: <definition> - Teach a fact
  exit       - Quit
""", fg="blue"))
            elif cmd == "/act":
                task = command[4:].strip()
                if task:
                    result = await dz.act(task)
                    click.echo(f"Result: {result['result']}")
            elif cmd == "/swarm":
                problem = command[6:].strip()
                if problem:
                    result = await dz.swarm(problem)
                    click.echo(f"Swarm solution: {result['solution']}")
            elif cmd.startswith("/learn"):
                parts = command[6:].strip().split(":", 1)
                if len(parts) == 2:
                    await dz.learn(parts[0].strip(), parts[1].strip())
                    click.echo(click.style(f"Learned: {parts[0].strip()}", fg="green"))
            else:
                click.echo(click.style(f"Unknown command: {cmd}. Type /help for help.", fg="red"))

        asyncio.run(_run())

    @app.command()
    @click.argument("concept")
    @click.argument("definition")
    @click.option("--domain", "-d", default="general")
    def learn(concept: str, definition: str, domain: str):
        """Teach Dzogchen a new fact."""
        async def _run():
            from dzogchen import Dzogchen
            async with Dzogchen() as dz:
                await dz.learn(concept, definition, domain=domain)
                click.echo(click.style(f"Learned: '{concept}'", fg="green"))

        asyncio.run(_run())

    @app.command()
    @click.argument("query")
    def remember(query: str):
        """Query Dzogchen's memory."""
        async def _run():
            from dzogchen import Dzogchen
            async with Dzogchen() as dz:
                results = await dz.remember(query)
                click.echo(json.dumps(results, indent=2, default=str))

        asyncio.run(_run())

    return app


app = create_cli()


def main():
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
