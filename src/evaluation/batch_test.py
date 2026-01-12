#!/usr/bin/env python3
"""
Batch test script for evaluating financial agent on random samples.
Allows configurable sample size, random seed, and model selection.
"""

import json
import os
import sys
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import asyncio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent.agent_v2 import FinancialAgentV2
from src.models.dataset import Document, ConvFinQARecord, Dialogue, Features

console = Console()


class BatchTestRunner:
    """Runs batch tests on random samples from the dataset."""
    
    def __init__(
        self,
        dataset_path: str,
        sample_size: int,
        seed: int,
        model_name: str,
        output_dir: str,
        enable_validation: bool = False,
        enable_judge: bool = True,
    ):
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.seed = seed
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.enable_validation = enable_validation
        self.enable_judge = enable_judge
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Load full dataset
        console.print(f"[cyan]Loading dataset from {dataset_path}...[/cyan]")
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Get all examples (from 'train' or 'examples' key)
        if 'examples' in dataset:
            all_examples = dataset['examples']
        elif 'train' in dataset:
            all_examples = dataset['train']
        else:
            raise ValueError("Dataset must contain 'examples' or 'train' key")
        
        console.print(f"[green]Loaded {len(all_examples)} total examples[/green]")
        
        # Sample examples
        if sample_size > len(all_examples):
            console.print(f"[yellow]Warning: Sample size ({sample_size}) exceeds dataset size ({len(all_examples)}). Using all examples.[/yellow]")
            self.examples = all_examples
        else:
            self.examples = random.sample(all_examples, sample_size)
        
        console.print(f"[green]Selected {len(self.examples)} examples (seed={seed})[/green]")
        
        # Save sample metadata
        sample_ids = [ex['id'] for ex in self.examples]
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': dataset_path,
            'total_examples': len(all_examples),
            'sample_size': len(self.examples),
            'seed': seed,
            'model_name': model_name,
            'enable_validation': enable_validation,
            'enable_judge': enable_judge,
            'sample_ids': sample_ids
        }
        
        metadata_file = self.output_dir / 'sample_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        console.print(f"[green]Saved sample metadata to {metadata_file}[/green]")
    
    def _convert_to_record(self, example: Dict) -> ConvFinQARecord:
        """Convert example dict to ConvFinQARecord."""
        doc_dict = example['doc']
        document = Document(
            pre_text=doc_dict.get('pre_text', ''),
            post_text=doc_dict.get('post_text', ''),
            table=doc_dict.get('table', {})
        )
        
        dialogue = Dialogue(
            conv_questions=example['dialogue']['conv_questions'],
            conv_answers=example['dialogue']['conv_answers'],
            turn_program=example['dialogue'].get('turn_program', []),
            executed_answers=example['dialogue'].get('executed_answers', []),
            qa_split=example['dialogue'].get('qa_split', [])
        )
        
        features = Features(
            num_dialogue_turns=example['features']['num_dialogue_turns'],
            has_type2_question=example['features']['has_type2_question'],
            has_duplicate_columns=example['features']['has_duplicate_columns'],
            has_non_numeric_values=example['features']['has_non_numeric_values']
        )
        
        return ConvFinQARecord(
            id=example['id'],
            doc=document,
            dialogue=dialogue,
            features=features
        )
    
    async def evaluate_example(
        self,
        example: Dict,
        agent: FinancialAgentV2
    ) -> Dict[str, Any]:
        """Evaluate a single example."""
        record = self._convert_to_record(example)
        result = await agent.run_conversation(record)
        return result.to_dict()
    
    async def run_batch_async(self) -> Dict[str, Any]:
        """Run batch evaluation asynchronously."""
        console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
        console.print(f"[bold cyan]BATCH TEST - {self.model_name}[/bold cyan]")
        console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")
        
        # Set model in environment
        os.environ['MODEL_NAME'] = self.model_name
        
        # Initialize agent
        console.print(f"[cyan]Initializing agent with model: {self.model_name}...[/cyan]")
        agent = FinancialAgentV2(
            model_name=self.model_name,
            enable_validation=self.enable_validation,
            enable_judge=self.enable_judge
        )
        console.print("[green]Agent initialized successfully[/green]\n")
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[cyan]Evaluating {len(self.examples)} examples...",
                total=len(self.examples)
            )
            
            for i, example in enumerate(self.examples, 1):
                example_id = example['id']
                features = example['features']
                
                progress.update(
                    task,
                    description=f"[cyan]Evaluating {example_id} ({i}/{len(self.examples)})"
                )
                
                try:
                    # Timeout: 5 minutes per example
                    result = await asyncio.wait_for(
                        self.evaluate_example(example, agent),
                        timeout=300.0
                    )
                    results.append(result)
                    
                    # Print summary
                    console.print(
                        f"  [{i}/{len(self.examples)}] {example_id}: "
                        f"Acc={result['financial_accuracy']:.2%}, "
                        f"Turns={features['num_dialogue_turns']}, "
                        f"Tokens={result.get('total_tokens', 0)}"
                    )
                    
                except asyncio.TimeoutError:
                    console.print(f"  [red]TIMEOUT on {example_id} after 5 minutes[/red]")
                    results.append({
                        'example_id': example_id,
                        'model': self.model_name,
                        'features': example['features'],
                        'error': 'Evaluation timeout (300s)'
                    })
                    
                except Exception as e:
                    console.print(f"  [red]ERROR on {example_id}: {str(e)}[/red]")
                    import traceback
                    error_trace = traceback.format_exc()
                    results.append({
                        'example_id': example_id,
                        'model': self.model_name,
                        'features': example['features'],
                        'error': str(e),
                        'error_trace': error_trace
                    })
                
                # Rate limiting
                if i < len(self.examples):
                    await asyncio.sleep(1.0)
                
                progress.advance(task)
        
        # Save detailed results
        results_file = self.output_dir / 'detailed_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"\n[green]✓ Saved detailed results to {results_file}[/green]")
        
        # Calculate and save metrics
        metrics = self._calculate_metrics(results)
        
        metrics_file = self.output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        console.print(f"[green]✓ Saved metrics to {metrics_file}[/green]")
        
        return {
            'results': results,
            'metrics': metrics
        }
    
    def run_batch(self) -> Dict[str, Any]:
        """Synchronous wrapper for run_batch_async."""
        return asyncio.run(self.run_batch_async())
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive metrics from results."""
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {
                'error': 'No successful evaluations',
                'total_examples': 0,
                'errors': len(results)
            }
        
        total = len(successful_results)
        
        # Accuracy metrics
        numerical_accuracies = [r.get('numerical_accuracy', 0) for r in successful_results]
        financial_accuracies = [r.get('financial_accuracy', 0) for r in successful_results]
        soft_match_accuracies = [r.get('soft_match_accuracy', 0) for r in successful_results]
        all_correct_count = sum(1 for r in successful_results if r.get('all_correct', False))
        
        # Token and timing metrics
        total_tokens = sum(r.get('total_tokens', 0) for r in successful_results)
        avg_tokens_per_example = total_tokens / total if total > 0 else 0
        avg_tokens_per_turn = sum(r.get('avg_tokens_per_turn', 0) for r in successful_results) / total if total > 0 else 0
        
        total_response_time = sum(r.get('total_response_time_ms', 0) for r in successful_results)
        avg_response_time_per_example = total_response_time / total if total > 0 else 0
        avg_response_time_per_turn = sum(r.get('avg_response_time_ms', 0) for r in successful_results) / total if total > 0 else 0
        
        # By feature analysis
        by_turns_metrics = {}
        by_type2_metrics = {'true': [], 'false': []}
        by_non_numeric_metrics = {'true': [], 'false': []}
        by_duplicate_cols_metrics = {'true': [], 'false': []}
        
        for r in successful_results:
            features = r.get('features', {})
            turns = features.get('num_dialogue_turns', 0)
            type2 = str(features.get('has_type2_question', False)).lower()
            non_numeric = str(features.get('has_non_numeric_values', False)).lower()
            dup_cols = str(features.get('has_duplicate_columns', False)).lower()
            
            financial_acc = r.get('financial_accuracy', 0)
            
            # By turns
            if turns <= 2:
                turn_key = '1-2'
            elif turns <= 4:
                turn_key = '3-4'
            else:
                turn_key = '5+'
            
            if turn_key not in by_turns_metrics:
                by_turns_metrics[turn_key] = []
            by_turns_metrics[turn_key].append(financial_acc)
            
            # By features
            by_type2_metrics[type2].append(financial_acc)
            by_non_numeric_metrics[non_numeric].append(financial_acc)
            by_duplicate_cols_metrics[dup_cols].append(financial_acc)
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'sample_size': len(self.examples),
            'seed': self.seed,
            'total_examples': total,
            'errors': len(results) - total,
            'overall': {
                'numerical_accuracy': sum(numerical_accuracies) / total if total > 0 else 0,
                'financial_accuracy': sum(financial_accuracies) / total if total > 0 else 0,
                'soft_match_accuracy': sum(soft_match_accuracies) / total if total > 0 else 0,
                'perfect_conversations': all_correct_count,
                'perfect_rate': all_correct_count / total if total > 0 else 0,
                'avg_tokens_per_example': avg_tokens_per_example,
                'avg_tokens_per_turn': avg_tokens_per_turn,
                'avg_response_time_ms_per_example': avg_response_time_per_example,
                'avg_response_time_ms_per_turn': avg_response_time_per_turn
            },
            'by_dialogue_turns': {
                k: sum(v) / len(v) if v else 0 
                for k, v in by_turns_metrics.items()
            },
            'by_type2_question': {
                k: sum(v) / len(v) if v else 0 
                for k, v in by_type2_metrics.items()
            },
            'by_non_numeric_values': {
                k: sum(v) / len(v) if v else 0 
                for k, v in by_non_numeric_metrics.items()
            },
            'by_duplicate_columns': {
                k: sum(v) / len(v) if v else 0 
                for k, v in by_duplicate_cols_metrics.items()
            }
        }
        
        return metrics
    
    def print_summary(self, metrics: Dict[str, Any]) -> None:
        """Print summary table of metrics."""
        console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
        console.print("[bold cyan]BATCH TEST SUMMARY[/bold cyan]")
        console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")
        
        # Overall metrics
        table = Table(title="Overall Performance", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", justify="right", style="green")
        
        overall = metrics['overall']
        table.add_row("Model", metrics['model_name'])
        table.add_row("Sample Size", str(metrics['sample_size']))
        table.add_row("Seed", str(metrics['seed']))
        table.add_row("Successful Examples", str(metrics['total_examples']))
        table.add_row("Errors", str(metrics['errors']))
        table.add_row("", "")
        table.add_row("Numerical Accuracy", f"{overall['numerical_accuracy']:.2%}")
        table.add_row("Financial Accuracy", f"{overall['financial_accuracy']:.2%}")
        table.add_row("Soft Match Accuracy", f"{overall['soft_match_accuracy']:.2%}")
        table.add_row("Perfect Rate", f"{overall['perfect_rate']:.2%}")
        table.add_row("", "")
        table.add_row("Avg Tokens/Example", f"{overall['avg_tokens_per_example']:.1f}")
        table.add_row("Avg Tokens/Turn", f"{overall['avg_tokens_per_turn']:.1f}")
        table.add_row("Avg Response Time/Example (ms)", f"{overall['avg_response_time_ms_per_example']:.1f}")
        table.add_row("Avg Response Time/Turn (ms)", f"{overall['avg_response_time_ms_per_turn']:.1f}")
        
        console.print(table)
        
        # By dialogue turns
        if metrics.get('by_dialogue_turns'):
            console.print("\n[bold]Performance by Dialogue Turns:[/bold]")
            for turn_range, acc in sorted(metrics['by_dialogue_turns'].items()):
                console.print(f"  {turn_range} turns: {acc:.2%}")
        
        # By features
        if metrics.get('by_type2_question'):
            console.print("\n[bold]Performance by Type2 Question:[/bold]")
            for has_type2, acc in metrics['by_type2_question'].items():
                label = "Has Type2" if has_type2 == 'true' else "No Type2"
                console.print(f"  {label}: {acc:.2%}")
        
        if metrics.get('by_non_numeric_values'):
            console.print("\n[bold]Performance by Non-Numeric Values:[/bold]")
            for has_non_numeric, acc in metrics['by_non_numeric_values'].items():
                label = "Has Non-Numeric" if has_non_numeric == 'true' else "All Numeric"
                console.print(f"  {label}: {acc:.2%}")


def main():
    parser = argparse.ArgumentParser(
        description='Run batch test on random sample from dataset'
    )
    parser.add_argument(
        '--dataset',
        default='data/convfinqa_dataset.json',
        help='Path to dataset JSON file'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=20,
        help='Number of examples to sample'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o',
        help='Model to evaluate (e.g., gpt-4o, claude-3-5-sonnet-20241022)'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory to save results (default: batch_test_results_<model>_<timestamp>)'
    )
    parser.add_argument(
        '--validation',
        action='store_true',
        help='Enable plan validation'
    )
    
    args = parser.parse_args()
    
    # Generate output directory name if not provided
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_safe = args.model.replace(':', '_').replace('/', '_').replace('-', '_')
        args.output_dir = f'batch_test_results_{model_safe}_{timestamp}'
    
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]                         BATCH TEST RUNNER                           [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]\n")
    
    console.print(f"[yellow]Dataset:[/yellow] {args.dataset}")
    console.print(f"[yellow]Sample Size:[/yellow] {args.sample_size}")
    console.print(f"[yellow]Seed:[/yellow] {args.seed}")
    console.print(f"[yellow]Model:[/yellow] {args.model}")
    console.print(f"[yellow]Validation:[/yellow] {'Enabled' if args.validation else 'Disabled'}")
    console.print(f"[yellow]Output Dir:[/yellow] {args.output_dir}\n")
    
    # Create runner and execute
    try:
        runner = BatchTestRunner(
            dataset_path=args.dataset,
            sample_size=args.sample_size,
            seed=args.seed,
            model_name=args.model,
            output_dir=args.output_dir,
            enable_validation=args.validation,
        )
        
        result = runner.run_batch()
        
        # Print summary
        runner.print_summary(result['metrics'])
        
        console.print(f"\n[bold green]{'='*80}[/bold green]")
        console.print("[bold green]                       BATCH TEST COMPLETE                              [/bold green]")
        console.print(f"[bold green]{'='*80}[/bold green]\n")
        console.print(f"[green]Results saved to: {args.output_dir}[/green]\n")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Test interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Error: {str(e)}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    main()
