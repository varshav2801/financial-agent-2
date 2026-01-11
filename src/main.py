"""
Main typer app for ConvFinQA
"""

import asyncio
import json
import typer
from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
from rich.table import Table
from src.utils.data_loader import load_record
from src.logger import get_logger

console = Console()
logger = get_logger(__name__)

app = typer.Typer(
    name="main",
    help="ConvFinQA Agent CLI",
    add_completion=True,
    no_args_is_help=True,
)


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
    verbose: bool = typer.Option(True, "--verbose/--no-verbose", help="Show detailed execution logs"),
    enable_validation: bool = typer.Option(False, "--validation/--no-validation", help="Enable/disable validation"),
) -> None:
    """Ask questions about a specific record with detailed logging"""
    
    # Set log level to INFO for detailed output
    import logging
    import os
    os.environ["LOG_LEVEL"] = "INFO"
    logging.getLogger().setLevel(logging.INFO)
    
    console.print(f"[cyan]Loading record: {record_id}[/cyan]")
    record = load_record(record_id)
    
    console.print(f"[green]Loaded record with {record.features.num_dialogue_turns} conversation turns[/green]")
    console.print(f"[dim]Validation: {'ENABLED' if enable_validation else 'DISABLED'}[/dim]\n")
    
    # Initialize agent
    from src.agent.agent import FinancialAgent
    import os
    model_name = os.getenv('MODEL_NAME', 'gpt-4o')
    agent = FinancialAgent(model_name=model_name, enable_validation=enable_validation)
    
    # Run conversation through agent
    result = asyncio.run(agent.run_conversation(record))
    
    # Display results for each turn
    for turn_idx, question in enumerate(record.dialogue.conv_questions):
        console.print(f"\n[bold yellow]━━━ Turn {turn_idx + 1}/{len(record.dialogue.conv_questions)} ━━━[/bold yellow]")
        console.print(f"[yellow]Question:[/yellow] {question}\n")
        
        # Get turn result from agent's conversation result
        if turn_idx >= len(result.turn_results):
            console.print(f"[red]Error: No result for turn {turn_idx + 1}[/red]")
            break
        
        turn_result = result.turn_results[turn_idx]
        
        # Display plan if available
        if turn_result.plan and verbose:
            console.print("[bold cyan]📋 Generated Plan[/bold cyan]")
            import json
            console.print(json.dumps(turn_result.plan, indent=2))
            console.print()
        
        # Display execution details
        if verbose:
            console.print("[bold cyan]⚙️  Execution[/bold cyan]")
            console.print(f"Status: {'✅' if turn_result.success else '❌'}")
            console.print(f"Time: {turn_result.response_time_ms:.1f}ms")
            console.print(f"Tokens: {turn_result.total_tokens}")
            if turn_result.error:
                console.print(f"[red]Error: {turn_result.error}[/red]")
            console.print()
        
        # Display answer
        expected = record.dialogue.conv_answers[turn_idx]
        expected_exec = record.dialogue.executed_answers[turn_idx]
        
        answer_text = str(turn_result.answer)
        console.print(f"[blue]Assistant:[/blue] {answer_text}")
        console.print(f"[dim]Expected: {expected} (executed: {expected_exec})[/dim]")
        
        # Display accuracy
        if turn_result.numerical_accuracy == 1.0:
            console.print("[green]✓ Correct! (Numerical)[/green]")
        elif turn_result.financial_accuracy == 1.0:
            console.print("[green]✓ Correct! (Financial - 1% tolerance)[/green]")
        elif turn_result.soft_match_accuracy == 1.0:
            console.print("[yellow]≈ Soft Match[/yellow]")
        else:
            console.print(f"[red]✗ Incorrect[/red]")
        
        console.print()
    
    # Display overall conversation metrics
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print("[bold cyan]Conversation Summary[/bold cyan]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")
    console.print(f"Financial Accuracy: {result.financial_accuracy:.2%}")
    console.print(f"All Correct: {result.all_correct}")
    console.print(f"Correct Turns: {result.correct_turns}/{result.total_turns}")
    console.print(f"Total Tokens: {result.total_tokens}")
    console.print(f"Total Time: {result.total_response_time_ms:.1f}ms")
    console.print()


@app.command()
def evaluate(
    sample_size: int = typer.Option(100, "--sample-size", "-n", help="Total number of conversations to evaluate"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory (default: eval_runs/TIMESTAMP)"),
    enable_validation: bool = typer.Option(False, "--validation/--no-validation", help="Enable/disable validation"),
) -> None:
    """
    Run batch evaluation with comprehensive metrics tracking
    
    Outputs:
      - summary.csv: High-level metrics per conversation
      - turns.csv: Turn-by-turn detailed metrics
      - validation.csv: All validator responses
      - traces/{trace_id}.json: Detailed logs per conversation
      - statistics.json: Aggregate statistics
    """
    from pathlib import Path
    from datetime import datetime
    from src.evaluation.runner import EvaluationRunner
    
    # Create timestamped output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"eval_runs/run_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rich_print(f"[cyan]Starting evaluation run[/cyan]")
    rich_print(f"[dim]Sample size: {sample_size}[/dim]")
    rich_print(f"[dim]Random seed: {seed}[/dim]")
    rich_print(f"[dim]Validation: {'ENABLED' if enable_validation else 'DISABLED'}[/dim]")
    rich_print(f"[dim]Output directory: {output_path}[/dim]\n")
    
    # Run evaluation
    runner = EvaluationRunner(output_path, enable_validation=enable_validation)
    results = asyncio.run(runner.run_evaluation(
        sample_size=sample_size,
        random_seed=seed,
    ))
    
    # Print summary
    rich_print("\n[green]✓ Evaluation complete![/green]")
    rich_print(f"[cyan]Results saved to:[/cyan] {output_path}\n")
    
    rich_print("[yellow]Summary Statistics:[/yellow]")
    rich_print(f"  Total conversations: {results.total_conversations}")
    rich_print(f"  Total turns: {results.total_turns}")
    rich_print(f"  Numerical accuracy: [bold]{results.numerical_accuracy:.1%}[/bold]")
    rich_print(f"  General accuracy: [bold]{results.general_accuracy:.1%}[/bold]")
    rich_print(f"  Conversation-level accuracy: {results.conversation_level_accuracy:.1%}")
    rich_print(f"  Error rate: {results.error_rate:.1%}")
    rich_print(f"  Avg tokens per turn: {results.avg_tokens_per_turn:.1f}")
    rich_print(f"  Avg latency per turn: {results.avg_latency_per_turn:.0f}ms")
    
    if results.error_types:
        rich_print("\n[yellow]Error Distribution:[/yellow]")
        for error_type, count in sorted(results.error_types.items(), key=lambda x: x[1], reverse=True):
            rich_print(f"  {error_type}: {count}")
    
    rich_print(f"\n[dim]Run 'python -m src.main visualize {output_path}' to generate visualizations[/dim]")


@app.command()
def list_records(
    split: str = typer.Option("train", "--split", "-s", help="Dataset split: train, dev, or test"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of record IDs to show"),
) -> None:
    """List available record IDs in the dataset"""
    import json
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent / "data" / "convfinqa_dataset.json"
    if not data_path.exists():
        console.print(f"[red]Dataset not found at {data_path}[/red]")
        return
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if split not in data:
        console.print(f"[red]Split '{split}' not found. Available: {list(data.keys())}[/red]")
        return
    
    records = data[split]
    console.print(f"[cyan]Found {len(records)} records in '{split}' split[/cyan]")
    console.print(f"[dim]Showing first {min(limit, len(records))} record IDs:[/dim]\n")
    
    for i, record in enumerate(records[:limit]):
        num_turns = record.get('features', {}).get('num_dialogue_turns', '?')
        console.print(f"  {i+1}. {record['id']} ({num_turns} turns)")
    
    if len(records) > limit:
        console.print(f"\n[dim]... and {len(records) - limit} more records[/dim]")


if __name__ == "__main__":
    app()