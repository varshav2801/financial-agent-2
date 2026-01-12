#!/usr/bin/env python3
"""
Batch test script that mimics the chat function behavior exactly.
Randomly selects N examples and runs them sequentially, saving detailed results.
"""

import asyncio
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.agent.agent import FinancialAgent
from src.models.dataset import Document, ConvFinQARecord, Dialogue, Features

console = Console()


def select_random_examples(dataset_path: Path, n: int, seed: int) -> List[ConvFinQARecord]:
    """Randomly select N examples from the dataset."""
    console.print(f"[cyan]Loading dataset from {dataset_path}...[/cyan]")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Get examples from train split
    if 'train' in data:
        all_examples = data['train']
    elif 'examples' in data:
        all_examples = data['examples']
    else:
        raise ValueError("Dataset must contain 'train' or 'examples' key")
    
    console.print(f"[green]Found {len(all_examples)} total examples[/green]")
    
    # Set seed and sample
    random.seed(seed)
    
    if n > len(all_examples):
        console.print(f"[yellow]Warning: Requested {n} examples but only {len(all_examples)} available. Using all.[/yellow]")
        selected = all_examples
    else:
        selected = random.sample(all_examples, n)
    
    console.print(f"[green]Selected {len(selected)} examples (seed={seed})[/green]")
    
    # Convert to ConvFinQARecord objects
    records = []
    for ex in selected:
        doc = Document(
            pre_text=ex['doc'].get('pre_text', ''),
            post_text=ex['doc'].get('post_text', ''),
            table=ex['doc'].get('table', {})
        )
        
        dialogue = Dialogue(
            conv_questions=ex['dialogue']['conv_questions'],
            conv_answers=ex['dialogue']['conv_answers'],
            turn_program=ex['dialogue'].get('turn_program', []),
            executed_answers=ex['dialogue'].get('executed_answers', []),
            qa_split=ex['dialogue'].get('qa_split', [])
        )
        
        features = Features(
            num_dialogue_turns=ex['features']['num_dialogue_turns'],
            has_type2_question=ex['features']['has_type2_question'],
            has_duplicate_columns=ex['features']['has_duplicate_columns'],
            has_non_numeric_values=ex['features']['has_non_numeric_values']
        )
        
        record = ConvFinQARecord(
            id=ex['id'],
            doc=doc,
            dialogue=dialogue,
            features=features
        )
        records.append(record)
    
    return records


async def run_single_example(
    record: ConvFinQARecord,
    agent: FinancialAgent,
    example_idx: int,
    total: int
) -> Dict[str, Any]:
    """Run a single example exactly like the chat command does."""
    
    example_id = record.id
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold cyan]Example {example_idx + 1}/{total}: {example_id}[/bold cyan]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")
    
    console.print(f"[green]Loaded record with {record.features.num_dialogue_turns} conversation turns[/green]\n")
    
    # Run conversation through agent (exactly like chat command)
    result = await agent.run_conversation(record)
    
    # Build turn-by-turn results matching o3_eval format
    turns = []
    for turn_idx, question in enumerate(record.dialogue.conv_questions):
        console.print(f"[yellow]Turn {turn_idx + 1}/{len(record.dialogue.conv_questions)}:[/yellow] {question}")
        
        if turn_idx >= len(result.turn_results):
            console.print(f"[red]Error: No result for turn {turn_idx + 1}[/red]")
            break
        
        turn_result = result.turn_results[turn_idx]
        expected = record.dialogue.conv_answers[turn_idx]
        
        # Build turn data matching o3_eval format
        turn_data = {
            "success": turn_result.success,
            "answer": turn_result.answer,
            "expected": expected,
            "question": question,
            "turn": turn_idx + 1,
            "execution_time_ms": turn_result.response_time_ms,
            "plan": turn_result.plan,  # Already in dict format from agent
            "step_results": turn_result.step_results if hasattr(turn_result, 'step_results') else {},
            "error": turn_result.error,
            "numerical_match": turn_result.numerical_accuracy == 1.0,
            "financial_match": turn_result.financial_accuracy == 1.0,
            "soft_match": turn_result.soft_match_accuracy == 1.0,
            "tokens": {
                "prompt_tokens": turn_result.prompt_tokens,
                "completion_tokens": turn_result.completion_tokens,
                "total_tokens": turn_result.total_tokens
            }
        }
        
        turns.append(turn_data)
        
        # Display result
        answer_text = str(turn_result.answer)
        console.print(f"  [blue]Answer:[/blue] {answer_text}")
        console.print(f"  [dim]Expected: {expected}[/dim]")
        
        if turn_result.numerical_accuracy == 1.0:
            console.print("  [green]✓ Correct! (Numerical)[/green]")
        elif turn_result.financial_accuracy == 1.0:
            console.print("  [green]✓ Correct! (Financial - 1% tolerance)[/green]")
        elif turn_result.soft_match_accuracy == 1.0:
            console.print("  [yellow]≈ Soft Match[/yellow]")
        else:
            console.print("  [red]✗ Incorrect[/red]")
    
    # Build example result matching o3_eval format
    example_result = {
        "example_id": example_id,
        "model": agent.model_name,
        "features": {
            "num_dialogue_turns": record.features.num_dialogue_turns,
            "has_type2_question": record.features.has_type2_question,
            "has_duplicate_columns": record.features.has_duplicate_columns,
            "has_non_numeric_values": record.features.has_non_numeric_values
        },
        "num_turns": len(turns),
        "turns": turns,
        "numerical_accuracy": result.numerical_accuracy,
        "financial_accuracy": result.financial_accuracy,
        "soft_match_accuracy": result.soft_match_accuracy,
        "all_correct": result.all_correct,
        "correct_turns": result.correct_turns,
        "total_turns": result.total_turns,
        "total_tokens": result.total_tokens,
        "total_response_time_ms": result.total_response_time_ms,
        "avg_tokens_per_turn": result.total_tokens / result.total_turns if result.total_turns > 0 else 0,
        "avg_response_time_ms": result.total_response_time_ms / result.total_turns if result.total_turns > 0 else 0
    }
    
    # Display summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Financial Accuracy: {result.financial_accuracy:.2%}")
    console.print(f"  Correct Turns: {result.correct_turns}/{result.total_turns}")
    console.print(f"  Total Tokens: {result.total_tokens}")
    console.print(f"  Total Time: {result.total_response_time_ms:.1f}ms")
    
    return example_result


async def run_batch_test(
    n: int,
    seed: int,
    model: str,
    output_dir: Path,
    dataset_path: Path
) -> None:
    """Run batch test on N randomly selected examples."""
    
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]                         BATCH TEST RUNNER                           [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]\n")
    
    console.print(f"[yellow]Dataset:[/yellow] {dataset_path}")
    console.print(f"[yellow]Sample Size:[/yellow] {n}")
    console.print(f"[yellow]Seed:[/yellow] {seed}")
    console.print(f"[yellow]Model:[/yellow] {model}")
    console.print(f"[yellow]Output Dir:[/yellow] {output_dir}\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select random examples
    examples = select_random_examples(dataset_path, n, seed)
    
    # Save sample metadata
    sample_ids = [record.id for record in examples]
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'dataset_path': str(dataset_path),
        'sample_size': len(examples),
        'seed': seed,
        'model': model,
        'sample_ids': sample_ids
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    console.print(f"[green]✓ Saved metadata to {output_dir / 'metadata.json'}[/green]\n")
    
    # Initialize agent (exactly like chat command)
    import os
    os.environ['MODEL_NAME'] = model
    
    console.print(f"[cyan]Initializing agent with model: {model}...[/cyan]")
    agent = FinancialAgent(model_name=model, enable_validation=False)
    console.print("[green]✓ Agent initialized[/green]\n")
    
    # Run examples sequentially
    results = []
    
    for idx, record in enumerate(examples):
        try:
            result = await run_single_example(record, agent, idx, len(examples))
            results.append(result)
            
            # Rate limiting between examples
            if idx < len(examples) - 1:
                console.print("\n[dim]Waiting 1 second before next example...[/dim]")
                await asyncio.sleep(1.0)
            
        except Exception as e:
            console.print(f"\n[red]✗ Error processing {record.id}: {str(e)}[/red]")
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]")
            
            # Save error result
            results.append({
                "example_id": record.id,
                "model": model,
                "features": {
                    "num_dialogue_turns": record.features.num_dialogue_turns,
                    "has_type2_question": record.features.has_type2_question,
                    "has_duplicate_columns": record.features.has_duplicate_columns,
                    "has_non_numeric_values": record.features.has_non_numeric_values
                },
                "error": str(e),
                "error_trace": traceback.format_exc()
            })
    
    # Save results matching o3_eval format
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n[bold green]{'='*80}[/bold green]")
    console.print("[bold green]✓ BATCH TEST COMPLETE[/bold green]")
    console.print(f"[bold green]{'='*80}[/bold green]\n")
    console.print(f"[green]Results saved to: {results_file}[/green]\n")
    
    # Calculate and display aggregate metrics
    successful = [r for r in results if 'error' not in r]
    
    if successful:
        avg_financial_acc = sum(r['financial_accuracy'] for r in successful) / len(successful)
        avg_numerical_acc = sum(r['numerical_accuracy'] for r in successful) / len(successful)
        perfect_count = sum(1 for r in successful if r['all_correct'])
        avg_tokens = sum(r['total_tokens'] for r in successful) / len(successful)
        avg_time = sum(r['total_response_time_ms'] for r in successful) / len(successful)
        
        console.print("[bold]Aggregate Metrics:[/bold]")
        console.print(f"  Successful: {len(successful)}/{len(results)}")
        console.print(f"  Financial Accuracy: {avg_financial_acc:.2%}")
        console.print(f"  Numerical Accuracy: {avg_numerical_acc:.2%}")
        console.print(f"  Perfect Conversations: {perfect_count} ({perfect_count/len(successful):.1%})")
        console.print(f"  Avg Tokens/Example: {avg_tokens:.1f}")
        console.print(f"  Avg Time/Example: {avg_time:.1f}ms")
        
        # Save aggregate metrics
        aggregate = {
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'total_examples': len(results),
            'successful_examples': len(successful),
            'failed_examples': len(results) - len(successful),
            'avg_financial_accuracy': avg_financial_acc,
            'avg_numerical_accuracy': avg_numerical_acc,
            'perfect_conversations': perfect_count,
            'perfect_rate': perfect_count / len(successful) if successful else 0,
            'avg_tokens_per_example': avg_tokens,
            'avg_time_per_example_ms': avg_time
        }
        
        with open(output_dir / 'aggregate_metrics.json', 'w') as f:
            json.dump(aggregate, f, indent=2)
        
        console.print(f"\n[green]✓ Saved aggregate metrics to {output_dir / 'aggregate_metrics.json'}[/green]\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run batch test on N randomly selected examples'
    )
    parser.add_argument(
        '-n', '--num-examples',
        type=int,
        required=True,
        help='Number of examples to randomly select and test'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o',
        help='Model to use (default: gpt-4o)'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory (default: batch_test_results_<model>_<timestamp>)'
    )
    parser.add_argument(
        '--dataset',
        default='data/convfinqa_dataset.json',
        help='Path to dataset (default: data/convfinqa_dataset.json)'
    )
    
    args = parser.parse_args()
    
    # Generate output directory name if not provided
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_safe = args.model.replace(':', '_').replace('/', '_').replace('-', '_')
        args.output_dir = f'batch_test_results_{model_safe}_{timestamp}'
    
    output_dir = Path(args.output_dir)
    dataset_path = Path(__file__).parent / args.dataset
    
    # Run batch test
    try:
        asyncio.run(run_batch_test(
            n=args.num_examples,
            seed=args.seed,
            model=args.model,
            output_dir=output_dir,
            dataset_path=dataset_path
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Test interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]✗ Fatal error: {str(e)}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
