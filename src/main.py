"""
ConvFinQA Financial Agent - CLI

Command-line interface for running and evaluating the financial QA agent.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.logger import get_logger
from src.utils.data_loader import load_record

console = Console()
logger = get_logger(__name__)

app = typer.Typer(
    name="convfinqa",
    help="ConvFinQA Financial Agent",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _configure_logging(verbose: bool = True) -> None:
    """Configure logging level based on verbosity setting."""
    import logging
    
    log_level = "INFO" if verbose else "WARNING"
    os.environ["LOG_LEVEL"] = log_level
    logging.getLogger().setLevel(getattr(logging, log_level))


def _display_turn_results(
    turn_idx: int,
    total_turns: int,
    question: str,
    turn_result: dict,
    expected: str,
    expected_exec: str,
    verbose: bool,
) -> None:
    """Display results for a single conversation turn."""
    console.print(f"\n[bold yellow]{'─' * 80}[/bold yellow]")
    console.print(f"[bold yellow]Turn {turn_idx + 1}/{total_turns}[/bold yellow]")
    console.print(f"[bold yellow]{'─' * 80}[/bold yellow]")
    console.print(f"\n[cyan]Question:[/cyan] {question}\n")
    
    # Display plan if verbose
    if verbose and turn_result.get("plan"):
        import json
        plan_data = turn_result["plan"]
        if isinstance(plan_data, dict):
            plan_text = json.dumps(plan_data, indent=2)
        else:
            plan_text = str(plan_data)
        
        console.print(Panel(
            plan_text,
            title="[bold cyan]Generated Plan[/bold cyan]",
            border_style="cyan",
        ))
        console.print()
    
    # Display execution details
    if verbose:
        status = "✅ Success" if turn_result.get("success") else "❌ Failed"
        table = Table(title="Execution Details", show_header=False, box=None)
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold")
        
        table.add_row("Status", status)
        table.add_row("Time", f"{turn_result.get('execution_time_ms', 0):.1f}ms")
        
        # Extract tokens from nested dict or top-level
        tokens_info = turn_result.get("tokens", {})
        total_tokens = tokens_info.get("total_tokens", turn_result.get("total_tokens", 0))
        table.add_row("Tokens", str(total_tokens))
        
        if turn_result.get("error"):
            table.add_row("Error", f"[red]{turn_result['error']}[/red]")
        
        console.print(table)
        console.print()
    
    # Display answer and accuracy
    answer_text = str(turn_result.get("answer", "N/A"))
    console.print(f"[bold green]Answer:[/bold green] {answer_text}")
    console.print(f"[dim]Expected: {expected} (executed: {expected_exec})[/dim]\n")
    
    # Display accuracy indicators
    if turn_result.get("numerical_match"):
        console.print("[green]✓ Correct (Numerical Match)[/green]")
    elif turn_result.get("financial_match"):
        console.print("[green]✓ Correct (Financial - 1% tolerance)[/green]")
    elif turn_result.get("soft_match"):
        console.print("[yellow]≈ Soft Match[/yellow]")
    else:
        console.print("[red]✗ Incorrect[/red]")


def _display_conversation_summary(result: dict) -> None:
    """Display overall conversation metrics."""
    console.print(f"\n[bold cyan]{'═' * 80}[/bold cyan]")
    console.print("[bold cyan]Conversation Summary[/bold cyan]")
    console.print(f"[bold cyan]{'═' * 80}[/bold cyan]\n")
    
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="bold green", justify="right")
    
    table.add_row("Financial Accuracy", f"{result.get('financial_accuracy', 0):.2%}")
    table.add_row("All Correct", str(result.get("all_correct", False)))
    table.add_row("Correct Turns", result.get('correct_turns', '0/0'))  # Already formatted string
    table.add_row("Total Tokens", f"{result.get('total_tokens', 0):,}")
    table.add_row("Total Time", f"{result.get('total_response_time_ms', 0):.1f}ms")
    
    console.print(table)
    console.print()


@app.command()
def chat(
    example_id: str = typer.Argument(..., help="Example ID to test"),
    model: str = typer.Option("gpt-5-mini", "--model", help="Model name"),
    validator: bool = typer.Option(False, "--validator/--no-validator", help="Enable plan validation"),
    verifier: bool = typer.Option(True, "--verifier/--no-verifier", help="Enable result verification"),
) -> None:
    """
    Test a specific record.
    
    Example:
        uv run python -m src.main chat Single_JKHY/2009/page_28.pdf-3 --model gpt-5-mini
    """
    from src.agent.agent_v2 import FinancialAgentV2
    
    _configure_logging(True)
    
    os.environ["MODEL_NAME"] = model
    
    # Display configuration
    console.print(Panel(
        f"[cyan]Example ID:[/cyan] {example_id}\n"
        f"[cyan]Model:[/cyan] {model}\n"
        f"[cyan]Validator:[/cyan] {'Enabled' if validator else 'Disabled'}\n"
        f"[cyan]Verifier:[/cyan] {'Enabled' if verifier else 'Disabled'}",
        title="[bold]Configuration[/bold]",
        border_style="cyan",
    ))
    
    try:
        # Load record
        console.print(f"\n[cyan]Loading record...[/cyan]")
        record = load_record(example_id)
        console.print(f"[green]✓ Loaded {record.features.num_dialogue_turns} conversation turns[/green]")
        
        # Initialize agent
        console.print(f"[cyan]Initializing agent...[/cyan]")
        agent = FinancialAgentV2(
            model_name=model,
            enable_validation=validator,
            enable_judge=verifier,
        )
        console.print("[green]✓ Agent ready[/green]")
        
        # Run conversation
        result = asyncio.run(agent.run_conversation(record))
        
        # Display turn results
        for turn_idx, question in enumerate(record.dialogue.conv_questions):
            if turn_idx >= len(result.turn_results):
                console.print(f"[red]Error: No result for turn {turn_idx + 1}[/red]")
                break
            
            turn_result = result.turn_results[turn_idx]
            expected = record.dialogue.conv_answers[turn_idx]
            expected_exec = record.dialogue.executed_answers[turn_idx]
            
            _display_turn_results(
                turn_idx,
                len(record.dialogue.conv_questions),
                question,
                turn_result.to_dict(),
                expected,
                expected_exec,
                True,
            )
        
        # Display summary
        _display_conversation_summary(result.to_dict())
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Error: {str(e)}[/red]")
        logger.exception("Chat command failed")
        raise typer.Exit(1)


@app.command()
def evaluate(
    records: int = typer.Option(20, "--records", help="Number of records to test"),
    model: str = typer.Option("gpt-5-mini", "--model", help="Model name"),
    validator: bool = typer.Option(False, "--validator/--no-validator", help="Enable plan validation"),
    verifier: bool = typer.Option(True, "--verifier/--no-verifier", help="Enable result verification"),
) -> None:
    """
    Test a random N batch and save results.
    
    Example:
        uv run python -m src.main evaluate --records 20 --model gpt-5-mini
    """
    from datetime import datetime
    from src.evaluation.batch_test import BatchTestRunner
    
    os.environ["MODEL_NAME"] = model
    
    # Generate output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model.replace(":", "_").replace("/", "_").replace("-", "_")
    output_dir = f"batch_test_results_{model_safe}_{timestamp}"
    
    # Display configuration
    console.print(Panel(
        f"[cyan]Records:[/cyan] {records}\n"
        f"[cyan]Model:[/cyan] {model}\n"
        f"[cyan]Validator:[/cyan] {'Enabled' if validator else 'Disabled'}\n"
        f"[cyan]Verifier:[/cyan] {'Enabled' if verifier else 'Disabled'}\n"
        f"[cyan]Output:[/cyan] {output_dir}",
        title="[bold]Batch Evaluation[/bold]",
        border_style="cyan",
    ))
    
    try:
        # Create and run batch test
        console.print(f"\n[cyan]Initializing batch test runner...[/cyan]")
        runner = BatchTestRunner(
            dataset_path="data/convfinqa_dataset.json",
            sample_size=records,
            seed=42,
            model_name=model,
            output_dir=output_dir,
            enable_validation=validator,
            enable_judge=verifier,
        )
        
        result = runner.run_batch()
        
        # Display summary
        runner.print_summary(result["metrics"])
        
        console.print(f"\n[bold green]{'═' * 80}[/bold green]")
        console.print("[bold green]Evaluation Complete[/bold green]")
        console.print(f"[bold green]{'═' * 80}[/bold green]\n")
        console.print(f"[green]✓ Results saved to: {output_dir}[/green]\n")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Evaluation interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Error: {str(e)}[/red]")
        logger.exception("Evaluation command failed")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()