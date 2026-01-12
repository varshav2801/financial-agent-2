#!/usr/bin/env python3
"""
Quick test script for evaluating specific examples from test_examples_20_good.txt
Useful for rapid testing during development.
"""

import json
import os
import sys
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

from src.agent.agent import FinancialAgent
from src.models.dataset import Document, ConvFinQARecord, Dialogue, Features

console = Console()


class QuickTestRunner:
    """Runs quick tests on specific examples."""
    
    def __init__(
        self,
        dataset_path: str,
        example_ids: List[str],
        model_name: str,
        output_dir: str
    ):
        self.dataset_path = dataset_path
        self.example_ids = example_ids
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load full dataset
        console.print(f"[cyan]Loading dataset from {dataset_path}...[/cyan]")
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Get all examples
        if 'examples' in dataset:
            all_examples = dataset['examples']
        elif 'train' in dataset:
            all_examples = dataset['train']
        else:
            raise ValueError("Dataset must contain 'examples' or 'train' key")
        
        # Create lookup dict
        example_dict = {ex['id']: ex for ex in all_examples}
        
        # Find specified examples
        self.examples = []
        missing_ids = []
        
        for example_id in example_ids:
            if example_id in example_dict:
                self.examples.append(example_dict[example_id])
            else:
                missing_ids.append(example_id)
        
        if missing_ids:
            console.print(f"[yellow]Warning: Could not find {len(missing_ids)} example(s):[/yellow]")
            for mid in missing_ids:
                console.print(f"  [yellow]- {mid}[/yellow]")
        
        if not self.examples:
            raise ValueError("No valid examples found")
        
        console.print(f"[green]Found {len(self.examples)} examples to test[/green]")
        
        # Save test metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': dataset_path,
            'model_name': model_name,
            'requested_ids': example_ids,
            'found_ids': [ex['id'] for ex in self.examples],
            'missing_ids': missing_ids
        }
        
        metadata_file = self.output_dir / 'test_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        console.print(f"[green]Saved test metadata to {metadata_file}[/green]")
    
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
        agent: FinancialAgent
    ) -> Dict[str, Any]:
        """Evaluate a single example."""
        record = self._convert_to_record(example)
        result = await agent.run_conversation(record)
        return result.to_dict()
    
    async def run_tests_async(self) -> Dict[str, Any]:
        """Run quick tests asynchronously."""
        console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
        console.print(f"[bold cyan]QUICK TEST - {self.model_name}[/bold cyan]")
        console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")
        
        # Set model in environment
        os.environ['MODEL_NAME'] = self.model_name
        
        # Initialize agent
        console.print(f"[cyan]Initializing agent with model: {self.model_name}...[/cyan]")
        agent = FinancialAgent(model_name=self.model_name, enable_validation=False)
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
                f"[cyan]Testing {len(self.examples)} examples...",
                total=len(self.examples)
            )
            
            for i, example in enumerate(self.examples, 1):
                example_id = example['id']
                features = example['features']
                
                progress.update(
                    task,
                    description=f"[cyan]Testing {example_id} ({i}/{len(self.examples)})"
                )
                
                try:
                    # Timeout: 5 minutes per example
                    result = await asyncio.wait_for(
                        self.evaluate_example(example, agent),
                        timeout=300.0
                    )
                    results.append(result)
                    
                    # Print detailed summary
                    console.print(f"\n[bold cyan]Example {i}/{len(self.examples)}: {example_id}[/bold cyan]")
                    console.print(f"  Turns: {features['num_dialogue_turns']}")
                    console.print(f"  Type2: {features['has_type2_question']}")
                    console.print(f"  Numerical Accuracy: {result['numerical_accuracy']:.2%}")
                    console.print(f"  Financial Accuracy: {result['financial_accuracy']:.2%}")
                    console.print(f"  Soft Match Accuracy: {result['soft_match_accuracy']:.2%}")
                    console.print(f"  All Correct: {result.get('all_correct', False)}")
                    console.print(f"  Total Tokens: {result.get('total_tokens', 0)}")
                    console.print(f"  Avg Tokens/Turn: {result.get('avg_tokens_per_turn', 0):.1f}")
                    console.print(f"  Response Time: {result.get('total_response_time_ms', 0):.1f}ms")
                    
                    # Show per-turn results
                    if 'turn_results' in result:
                        console.print(f"  Turn Results:")
                        for turn_idx, turn in enumerate(result['turn_results'], 1):
                            status = "✓" if turn.get('correct', False) else "✗"
                            console.print(
                                f"    Turn {turn_idx}: {status} "
                                f"(predicted: {turn.get('predicted_answer', 'N/A')}, "
                                f"actual: {turn.get('ground_truth', 'N/A')})"
                            )
                    
                except asyncio.TimeoutError:
                    console.print(f"\n[red]TIMEOUT on {example_id} after 5 minutes[/red]")
                    results.append({
                        'example_id': example_id,
                        'model': self.model_name,
                        'features': example['features'],
                        'error': 'Evaluation timeout (300s)'
                    })
                    
                except Exception as e:
                    console.print(f"\n[red]ERROR on {example_id}: {str(e)}[/red]")
                    import traceback
                    error_trace = traceback.format_exc()
                    console.print(f"[red]{error_trace}[/red]")
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
    
    def run_tests(self) -> Dict[str, Any]:
        """Synchronous wrapper for run_tests_async."""
        return asyncio.run(self.run_tests_async())
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics from results."""
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
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
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
            }
        }
        
        return metrics
    
    def print_summary(self, metrics: Dict[str, Any]) -> None:
        """Print summary table of metrics."""
        console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
        console.print("[bold cyan]QUICK TEST SUMMARY[/bold cyan]")
        console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")
        
        # Overall metrics
        table = Table(title="Overall Performance", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", justify="right", style="green")
        
        overall = metrics['overall']
        table.add_row("Model", metrics['model_name'])
        table.add_row("Examples Tested", str(metrics['total_examples']))
        table.add_row("Errors", str(metrics['errors']))
        table.add_row("", "")
        table.add_row("Numerical Accuracy", f"{overall['numerical_accuracy']:.2%}")
        table.add_row("Financial Accuracy", f"{overall['financial_accuracy']:.2%}")
        table.add_row("Soft Match Accuracy", f"{overall['soft_match_accuracy']:.2%}")
        table.add_row("Perfect Conversations", f"{overall['perfect_conversations']}/{metrics['total_examples']}")
        table.add_row("Perfect Rate", f"{overall['perfect_rate']:.2%}")
        table.add_row("", "")
        table.add_row("Avg Tokens/Example", f"{overall['avg_tokens_per_example']:.1f}")
        table.add_row("Avg Tokens/Turn", f"{overall['avg_tokens_per_turn']:.1f}")
        table.add_row("Avg Response Time/Example (ms)", f"{overall['avg_response_time_ms_per_example']:.1f}")
        table.add_row("Avg Response Time/Turn (ms)", f"{overall['avg_response_time_ms_per_turn']:.1f}")
        
        console.print(table)


def load_test_examples_file(filepath: str) -> List[str]:
    """Load example IDs from test_examples_20_good.txt format."""
    example_ids = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            example_ids.append(line)
    
    return example_ids


def main():
    parser = argparse.ArgumentParser(
        description='Quick test on specific examples'
    )
    parser.add_argument(
        '--dataset',
        default='data/convfinqa_dataset.json',
        help='Path to dataset JSON file'
    )
    parser.add_argument(
        '--examples-file',
        default='../test_examples_20_good.txt',
        help='Path to file containing example IDs (one per line)'
    )
    parser.add_argument(
        '--examples',
        nargs='+',
        help='Specific example IDs to test (overrides --examples-file)'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o',
        help='Model to evaluate (e.g., gpt-4o, claude-3-5-sonnet-20241022)'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory to save results (default: quick_test_results_<timestamp>)'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Only test simple examples (1-2 turns)'
    )
    parser.add_argument(
        '--medium',
        action='store_true',
        help='Only test medium examples (3-4 turns)'
    )
    parser.add_argument(
        '--complex',
        action='store_true',
        help='Only test complex examples (4-5 turns or type2)'
    )
    
    args = parser.parse_args()
    
    # Determine which examples to test
    if args.examples:
        example_ids = args.examples
    else:
        # Load from file
        if not Path(args.examples_file).exists():
            console.print(f"[red]Error: Examples file not found: {args.examples_file}[/red]")
            sys.exit(1)
        
        all_example_ids = load_test_examples_file(args.examples_file)
        
        # Filter by complexity if requested
        if args.simple or args.medium or args.complex:
            # Parse the file to get categories
            example_ids = []
            current_category = None
            
            with open(args.examples_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        # Check for category markers
                        if 'SIMPLE' in line.upper():
                            current_category = 'simple'
                        elif 'MEDIUM' in line.upper():
                            current_category = 'medium'
                        elif 'COMPLEX' in line.upper():
                            current_category = 'complex'
                        continue
                    
                    # Add example if it matches selected categories
                    if (args.simple and current_category == 'simple') or \
                       (args.medium and current_category == 'medium') or \
                       (args.complex and current_category == 'complex'):
                        example_ids.append(line)
            
            if not example_ids:
                console.print("[yellow]Warning: No examples match the selected complexity filters[/yellow]")
                example_ids = all_example_ids
        else:
            example_ids = all_example_ids
    
    # Generate output directory name if not provided
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'quick_test_results_{timestamp}'
    
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]                         QUICK TEST RUNNER                           [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]\n")
    
    console.print(f"[yellow]Dataset:[/yellow] {args.dataset}")
    console.print(f"[yellow]Model:[/yellow] {args.model}")
    console.print(f"[yellow]Examples:[/yellow] {len(example_ids)}")
    console.print(f"[yellow]Output Dir:[/yellow] {args.output_dir}\n")
    
    # Create runner and execute
    try:
        runner = QuickTestRunner(
            dataset_path=args.dataset,
            example_ids=example_ids,
            model_name=args.model,
            output_dir=args.output_dir
        )
        
        result = runner.run_tests()
        
        # Print summary
        runner.print_summary(result['metrics'])
        
        console.print(f"\n[bold green]{'='*80}[/bold green]")
        console.print("[bold green]                       QUICK TEST COMPLETE                              [/bold green]")
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
