#!/usr/bin/env python3
"""
Evaluation script for model eval dataset across multiple models.
Tests multiple models on balanced test cases.

This script uses the FinancialAgent interface to run evaluations.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Any
import asyncio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent.agent import FinancialAgent
from src.models.dataset import Document, ConvFinQARecord, Dialogue, Features

console = Console()


class ModelEvalDatasetEvaluator:
    """Evaluator for model eval dataset across multiple models."""
    
    def __init__(self, model_eval_dataset_path: str, output_dir: str):
        self.model_eval_dataset_path = model_eval_dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model eval dataset
        try:
            with open(model_eval_dataset_path, 'r') as f:
                self.model_eval_data = json.load(f)
        except FileNotFoundError:
            console.print(f"[red]Error: Dataset file not found: {model_eval_dataset_path}[/red]")
            raise
        except json.JSONDecodeError as e:
            console.print(f"[red]Error: Invalid JSON in dataset file: {e}[/red]")
            raise
        
        self.examples = self.model_eval_data.get('examples', [])
        if not self.examples:
            console.print(f"[red]Error: No examples found in dataset[/red]")
            raise ValueError("Dataset contains no examples")
        
        console.print(f"[green]Loaded {len(self.examples)} examples from model eval dataset[/green]")
    
    async def evaluate_example(
        self,
        example: Dict,
        agent: FinancialAgent
    ) -> Dict[str, Any]:
        """Evaluate a single example using the agent."""
        
        # Convert example dict to ConvFinQARecord
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
        
        record = ConvFinQARecord(
            id=example['id'],
            doc=document,
            dialogue=dialogue,
            features=features
        )
        
        # Run conversation through agent
        result = await agent.run_conversation(record)
        
        return result.to_dict()
    
    async def evaluate_model_async(self, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model on the model eval dataset."""
        console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
        console.print(f"[bold cyan]Evaluating model: {model_name}[/bold cyan]")
        console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")
        
        # Set model in environment
        os.environ['MODEL_NAME'] = model_name
        
        # Create model-specific output directory
        model_output_dir = self.output_dir / model_name.replace(":", "_").replace("/", "_").replace("-", "_")
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize agent for this model
        try:
            agent = FinancialAgent(model_name=model_name, enable_validation=False)
        except Exception as e:
            console.print(f"[red]Failed to initialize agent for {model_name}: {e}[/red]")
            raise
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[cyan]Evaluating {model_name}...",
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
                    # Add timeout for long-running evaluations (5 minutes per example)
                    result = await asyncio.wait_for(
                        self.evaluate_example(example, agent),
                        timeout=300.0
                    )
                    results.append(result)
                    
                    # Print summary
                    console.print(
                        f"  [{i}/{len(self.examples)}] {example_id}: "
                        f"Accuracy={result['financial_accuracy']:.2%}, "
                        f"Turns={features['num_dialogue_turns']}, "
                        f"Tokens={result.get('total_tokens', 0)}"
                    )
                    
                except asyncio.TimeoutError:
                    console.print(f"  [red]TIMEOUT on {example_id} after 5 minutes[/red]")
                    results.append({
                        'example_id': example_id,
                        'model': model_name,
                        'features': example['features'],
                        'error': 'Evaluation timeout (300s)'
                    })
                    
                except Exception as e:
                    console.print(f"  [red]ERROR on {example_id}: {str(e)}[/red]")
                    import traceback
                    error_trace = traceback.format_exc()
                    results.append({
                        'example_id': example_id,
                        'model': model_name,
                        'features': example['features'],
                        'error': str(e),
                        'error_trace': error_trace
                    })
                
                # Rate limiting: 1 second between examples to avoid overwhelming API
                if i < len(self.examples):
                    await asyncio.sleep(1.0)
                
                progress.advance(task)
        
        # Save results
        results_file = model_output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"\n[green]✓ Saved results to {results_file}[/green]")
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        
        return {
            'model': model_name,
            'results': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """Synchronous wrapper for evaluate_model_async."""
        return asyncio.run(self.evaluate_model_async(model_name))
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive aggregate metrics across all examples."""
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            console.print("[yellow]Warning: No successful evaluations to calculate metrics[/yellow]")
            return {
                'error': 'No successful evaluations',
                'total_examples': 0,
                'errors': len(results)
            }
        
        total = len(successful_results)
        console.print(f"[green]Calculating metrics from {total} successful evaluations[/green]")
        
        # Calculate accuracy metrics (all three types) - use .get() for safety
        numerical_accuracies = [r.get('numerical_accuracy', 0) for r in successful_results]
        financial_accuracies = [r.get('financial_accuracy', 0) for r in successful_results]
        soft_match_accuracies = [r.get('soft_match_accuracy', 0) for r in successful_results]
        all_correct_count = sum(1 for r in successful_results if r.get('all_correct', False))
        
        # Diagnostic: Check if all three metrics are identical
        avg_numerical = sum(numerical_accuracies) / total if total > 0 else 0
        avg_financial = sum(financial_accuracies) / total if total > 0 else 0
        avg_soft = sum(soft_match_accuracies) / total if total > 0 else 0
        
        if avg_numerical == avg_financial == avg_soft:
            console.print(f"[yellow]Note: All three accuracy metrics are identical ({avg_numerical:.2%})[/yellow]")
            console.print("[yellow]This typically means answers are either completely correct or completely wrong.[/yellow]")
            console.print("[yellow]Check if your dataset has borderline cases (e.g., off by 1%, unit mismatches)[/yellow]")
        
        # Calculate token and timing metrics
        total_tokens = sum(r.get('total_tokens', 0) for r in successful_results)
        avg_tokens_per_example = total_tokens / total if total > 0 else 0
        avg_tokens_per_turn = sum(r.get('avg_tokens_per_turn', 0) for r in successful_results) / total if total > 0 else 0
        
        total_response_time = sum(r.get('total_response_time_ms', 0) for r in successful_results)
        avg_response_time_per_example = total_response_time / total if total > 0 else 0
        avg_response_time_per_turn = sum(r.get('avg_response_time_ms', 0) for r in successful_results) / total if total > 0 else 0
        
        # Calculate by feature category
        by_turns_numerical = {}
        by_turns_financial = {}
        by_turns_soft = {}
        by_turns_tokens = {}
        by_turns_time = {}
        
        by_type2_numerical = {'true': [], 'false': []}
        by_type2_financial = {'true': [], 'false': []}
        by_type2_soft = {'true': [], 'false': []}
        
        by_non_numeric_numerical = {'true': [], 'false': []}
        by_non_numeric_financial = {'true': [], 'false': []}
        by_non_numeric_soft = {'true': [], 'false': []}
        
        for r in successful_results:
            features = r.get('features', {})
            turns = features.get('num_dialogue_turns', 0)
            type2 = str(features.get('has_type2_question', False)).lower()
            non_numeric = str(features.get('has_non_numeric_values', False)).lower()
            
            numerical_acc = r.get('numerical_accuracy', 0)
            financial_acc = r.get('financial_accuracy', 0)
            soft_acc = r.get('soft_match_accuracy', 0)
            tokens = r.get('avg_tokens_per_turn', 0)
            response_time = r.get('avg_response_time_ms', 0)
            
            # By turns
            if turns <= 2:
                turn_key = '1-2'
            elif turns <= 4:
                turn_key = '3-4'
            else:
                turn_key = '5+'
            
            if turn_key not in by_turns_numerical:
                by_turns_numerical[turn_key] = []
                by_turns_financial[turn_key] = []
                by_turns_soft[turn_key] = []
                by_turns_tokens[turn_key] = []
                by_turns_time[turn_key] = []
            
            by_turns_numerical[turn_key].append(numerical_acc)
            by_turns_financial[turn_key].append(financial_acc)
            by_turns_soft[turn_key].append(soft_acc)
            by_turns_tokens[turn_key].append(tokens)
            by_turns_time[turn_key].append(response_time)
            
            # By type2
            by_type2_numerical[type2].append(numerical_acc)
            by_type2_financial[type2].append(financial_acc)
            by_type2_soft[type2].append(soft_acc)
            
            # By non_numeric
            by_non_numeric_numerical[non_numeric].append(numerical_acc)
            by_non_numeric_financial[non_numeric].append(financial_acc)
            by_non_numeric_soft[non_numeric].append(soft_acc)
        
        return {
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
            'by_turns': {
                'numerical_accuracy': {k: sum(v) / len(v) if v else 0 for k, v in by_turns_numerical.items()},
                'financial_accuracy': {k: sum(v) / len(v) if v else 0 for k, v in by_turns_financial.items()},
                'soft_match_accuracy': {k: sum(v) / len(v) if v else 0 for k, v in by_turns_soft.items()},
                'avg_tokens_per_turn': {k: sum(v) / len(v) if v else 0 for k, v in by_turns_tokens.items()},
                'avg_response_time_ms': {k: sum(v) / len(v) if v else 0 for k, v in by_turns_time.items()}
            },
            'by_type2_question': {
                'numerical_accuracy': {k: sum(v) / len(v) if v else 0 for k, v in by_type2_numerical.items()},
                'financial_accuracy': {k: sum(v) / len(v) if v else 0 for k, v in by_type2_financial.items()},
                'soft_match_accuracy': {k: sum(v) / len(v) if v else 0 for k, v in by_type2_soft.items()}
            },
            'by_non_numeric_values': {
                'numerical_accuracy': {k: sum(v) / len(v) if v else 0 for k, v in by_non_numeric_numerical.items()},
                'financial_accuracy': {k: sum(v) / len(v) if v else 0 for k, v in by_non_numeric_financial.items()},
                'soft_match_accuracy': {k: sum(v) / len(v) if v else 0 for k, v in by_non_numeric_soft.items()}
            }
        }
    
    def compare_models(self, model_results: List[Dict]) -> None:
        """Generate comprehensive comparison report across all models."""
        console.print(f"\n[bold cyan]{'='*100}[/bold cyan]")
        console.print("[bold cyan]MODEL COMPARISON REPORT - COMPREHENSIVE METRICS[/bold cyan]")
        console.print(f"[bold cyan]{'='*100}[/bold cyan]\n")
        
        # Overall comparison with all accuracy types
        console.print("[bold]Overall Performance - Accuracy Metrics:[/bold]")
        console.print(f"{'Model':<30} {'Numerical':>12} {'Financial':>12} {'Soft Match':>12} {'Perfect Rate':>14}")
        console.print("-" * 100)
        
        for result in model_results:
            model = result['model']
            metrics = result['aggregate_metrics']
            overall = metrics.get('overall', {})
            
            numerical = overall.get('numerical_accuracy', 0)
            financial = overall.get('financial_accuracy', 0)
            soft = overall.get('soft_match_accuracy', 0)
            perfect_rate = overall.get('perfect_rate', 0)
            
            console.print(f"{model:<30} {numerical:>11.2%} {financial:>11.2%} {soft:>11.2%} {perfect_rate:>13.2%}")
        
        # Token and timing metrics
        console.print("\n\n[bold]Overall Performance - Efficiency Metrics:[/bold]")
        console.print(f"{'Model':<30} {'Avg Tokens/Turn':>18} {'Avg Response (ms)':>18}")
        console.print("-" * 100)
        
        for result in model_results:
            model = result['model']
            metrics = result['aggregate_metrics']
            overall = metrics.get('overall', {})
            
            tokens = overall.get('avg_tokens_per_turn', 0)
            response_time = overall.get('avg_response_time_ms_per_turn', 0)
            
            console.print(f"{model:<30} {tokens:>17.1f} {response_time:>17.1f}")
        
        # By dialogue turns (Financial Accuracy)
        console.print("\n\n[bold]Performance by Dialogue Turns (Financial Accuracy):[/bold]")
        console.print(f"{'Model':<30} {'1-2 turns':>12} {'3-4 turns':>12} {'5+ turns':>12}")
        console.print("-" * 100)
        
        for result in model_results:
            model = result['model']
            by_turns = result['aggregate_metrics'].get('by_turns', {})
            financial_by_turns = by_turns.get('financial_accuracy', {})
            
            turns_1_2 = financial_by_turns.get('1-2', 0)
            turns_3_4 = financial_by_turns.get('3-4', 0)
            turns_5_plus = financial_by_turns.get('5+', 0)
            
            console.print(f"{model:<30} {turns_1_2:>11.2%} {turns_3_4:>11.2%} {turns_5_plus:>11.2%}")
        
        # By type2 question (Financial Accuracy)
        console.print("\n\n[bold]Performance by Type2 Question (Financial Accuracy):[/bold]")
        console.print(f"{'Model':<30} {'No Type2':>12} {'Has Type2':>12}")
        console.print("-" * 100)
        
        for result in model_results:
            model = result['model']
            by_type2 = result['aggregate_metrics'].get('by_type2_question', {})
            financial_by_type2 = by_type2.get('financial_accuracy', {})
            
            no_type2 = financial_by_type2.get('false', 0)
            has_type2 = financial_by_type2.get('true', 0)
            
            console.print(f"{model:<30} {no_type2:>11.2%} {has_type2:>11.2%}")
        
        # By non-numeric values (Financial Accuracy)
        console.print("\n\n[bold]Performance by Non-Numeric Values (Financial Accuracy):[/bold]")
        console.print(f"{'Model':<30} {'No Non-Numeric':>16} {'Has Non-Numeric':>18}")
        console.print("-" * 100)
        
        for result in model_results:
            model = result['model']
            by_non_numeric = result['aggregate_metrics'].get('by_non_numeric_values', {})
            financial_by_non_numeric = by_non_numeric.get('financial_accuracy', {})
            
            no_non_numeric = financial_by_non_numeric.get('false', 0)
            has_non_numeric = financial_by_non_numeric.get('true', 0)
            
            console.print(f"{model:<30} {no_non_numeric:>15.2%} {has_non_numeric:>17.2%}")
        
        # Save comparison report
        comparison_file = self.output_dir / 'comparison_report.json'
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'models': [r['model'] for r in model_results],
            'model_results': [
                {
                    'model': r['model'],
                    'aggregate_metrics': r['aggregate_metrics']
                }
                for r in model_results
            ]
        }
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        console.print(f"\n[green]✓ Saved comparison report to {comparison_file}[/green]")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate financial agent on model eval dataset across multiple models'
    )
    parser.add_argument(
        '--model-eval-dataset',
        default='data/model_eval_dataset.json',
        help='Path to model eval dataset JSON file'
    )
    parser.add_argument(
        '--output-dir',
        default='model_eval_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['gpt-4o', 'gpt-4o-mini', 'o3-mini', 'claude-3-5-sonnet-20241022'],
        help='Models to evaluate (space-separated)'
    )
    
    args = parser.parse_args()
    
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]                         MODEL EVALUATION                            [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════════════════════[/bold cyan]\n")
    
    console.print(f"[yellow]Dataset:[/yellow] {args.model_eval_dataset}")
    console.print(f"[yellow]Output:[/yellow] {args.output_dir}")
    console.print(f"[yellow]Models:[/yellow] {', '.join(args.models)}\n")
    
    # Create evaluator
    try:
        evaluator = ModelEvalDatasetEvaluator(
            model_eval_dataset_path=args.model_eval_dataset,
            output_dir=args.output_dir
        )
    except Exception as e:
        console.print(f"\n[red]✗ Failed to initialize evaluator: {str(e)}[/red]")
        sys.exit(1)
    
    # Evaluate each model
    model_results = []
    failed_models = []
    
    for model_name in args.models:
        try:
            console.print(f"\n[bold]Starting evaluation for: {model_name}[/bold]")
            result = evaluator.evaluate_model(model_name)
            
            # Check if evaluation was successful
            if result.get('aggregate_metrics', {}).get('total_examples', 0) > 0:
                model_results.append(result)
                console.print(f"[green]✓ Successfully evaluated {model_name}[/green]")
            else:
                console.print(f"[yellow]⚠ {model_name} evaluation completed but produced no valid results[/yellow]")
                failed_models.append((model_name, "No valid results"))
                
        except KeyboardInterrupt:
            console.print(f"\n[yellow]⚠ Evaluation interrupted by user[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]✗ Error evaluating {model_name}: {str(e)}[/red]")
            import traceback
            error_trace = traceback.format_exc()
            console.print(f"[red]{error_trace}[/red]")
            failed_models.append((model_name, str(e)))
            continue
    
    # Generate comparison report
    if model_results:
        evaluator.compare_models(model_results)
    else:
        console.print("\n[red]✗ No models were successfully evaluated[/red]")
        if failed_models:
            console.print("\n[red]Failed models:[/red]")
            for model_name, error in failed_models:
                console.print(f"  - {model_name}: {error}")
        sys.exit(1)
    
    # Print summary of failed models if any
    if failed_models:
        console.print(f"\n[yellow]{'='*80}[/yellow]")
        console.print("[yellow]Warning: Some models failed to evaluate:[/yellow]")
        for model_name, error in failed_models:
            console.print(f"  [yellow]- {model_name}: {error}[/yellow]")
        console.print(f"[yellow]{'='*80}[/yellow]\n")
    
    console.print(f"\n[bold green]{'='*80}[/bold green]")
    console.print("[bold green]                       EVALUATION COMPLETE                              [/bold green]")
    console.print(f"[bold green]{'='*80}[/bold green]\n")
    console.print(f"[green]Successfully evaluated {len(model_results)}/{len(args.models)} models[/green]")
    console.print(f"[green]Results saved to: {args.output_dir}[/green]\n")


if __name__ == '__main__':
    main()
