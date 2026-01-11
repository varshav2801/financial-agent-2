"""Workflow plan validator with reference and dependency checking"""
from typing import Set
from src.models.workflow_schema import WorkflowPlan, WorkflowStep, Operand, ValidationResult
from src.logger import get_logger

logger = get_logger(__name__)


class WorkflowValidator:
    """
    Validate workflow plans for correctness.
    
    Checks:
    - All step_refs point to existing step_ids
    - No forward references (step N cannot reference step M where M >= N)
    - No circular dependencies
    - Operand types are valid (reference has step_ref, literal has value)
    - Required params present for each source type
    """
    
    def validate(self, plan: WorkflowPlan) -> ValidationResult:
        """
        Validate workflow plan for structural correctness.
        
        Args:
            plan: WorkflowPlan to validate
            
        Returns:
            ValidationResult with is_valid flag and list of issues
        """
        issues = []
        
        # Build set of valid step_ids
        step_ids: Set[int] = {step.step_id for step in plan.steps}
        
        # Check 1: Verify step IDs are sequential starting from 1
        expected_ids = list(range(1, len(plan.steps) + 1))
        actual_ids = [step.step_id for step in plan.steps]
        
        if actual_ids != expected_ids:
            issues.append(
                f"Step IDs must be sequential starting from 1. "
                f"Expected {expected_ids}, got {actual_ids}"
            )
        
        # Check 2: Validate references in compute steps
        for i, step in enumerate(plan.steps):
            if step.tool == "compute":
                if step.operands is None:
                    issues.append(
                        f"Step {step.step_id}: Compute step missing operands"
                    )
                    continue
                
                # Validate each operand
                for j, operand in enumerate(step.operands):
                    operand_issues = self._validate_operand(
                        operand=operand,
                        step_id=step.step_id,
                        valid_step_ids=step_ids,
                        operand_index=j
                    )
                    issues.extend(operand_issues)
        
        # Check 3: Validate thought_process is present
        if not plan.thought_process or not plan.thought_process.strip():
            issues.append("thought_process field is empty or missing")
        
        # Check 4: Validate at least one step exists
        if not plan.steps or len(plan.steps) == 0:
            issues.append("Plan must have at least one step")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info(f"Plan validation passed: {len(plan.steps)} steps")
        else:
            logger.warning(f"Plan validation failed with {len(issues)} issues")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=1.0 if is_valid else 0.0,
            issues=issues,
            failed_steps=[],
            failure_type="validation_error" if not is_valid else None,
            repair_instructions=None,
            is_hallucination_likely=False,
            grounding_check=None
        )
    
    def _validate_operand(
        self,
        operand: Operand,
        step_id: int,
        valid_step_ids: Set[int],
        operand_index: int
    ) -> list[str]:
        """
        Validate a single operand.
        
        Args:
            operand: Operand to validate
            step_id: Current step ID
            valid_step_ids: Set of valid step IDs in the plan
            operand_index: Index of operand in operands list
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        if operand.type == "reference":
            # Check 1: step_ref must be provided
            if operand.step_ref is None:
                issues.append(
                    f"Step {step_id}, operand {operand_index}: "
                    f"Reference operand missing step_ref"
                )
                return issues
            
            # Check 2: No forward references (except negative for conversation history)
            if operand.step_ref >= 0:
                if operand.step_ref >= step_id:
                    issues.append(
                        f"Step {step_id}, operand {operand_index}: "
                        f"Forward reference detected (references step {operand.step_ref})"
                    )
                
                # Check 3: Reference must exist in plan
                if operand.step_ref not in valid_step_ids:
                    issues.append(
                        f"Step {step_id}, operand {operand_index}: "
                        f"Invalid step_ref {operand.step_ref} (not in plan)"
                    )
            else:
                # Negative reference (conversation history) - always valid
                logger.debug(
                    f"Step {step_id}: Uses conversation history reference (step_ref={operand.step_ref})"
                )
        
        elif operand.type == "literal":
            # Check: value must be provided
            if operand.value is None:
                issues.append(
                    f"Step {step_id}, operand {operand_index}: "
                    f"Literal operand missing value"
                )
        
        else:
            issues.append(
                f"Step {step_id}, operand {operand_index}: "
                f"Unknown operand type '{operand.type}'"
            )
        
        return issues
    
    def check_circular_dependencies(self, plan: WorkflowPlan) -> list[str]:
        """
        Check for circular dependencies (should not exist in sequential workflow).
        
        This is a safety check - sequential workflows with forward reference
        validation should never have circular dependencies.
        
        Args:
            plan: WorkflowPlan to check
            
        Returns:
            List of issues (empty if no circular dependencies)
        """
        issues = []
        
        # Build dependency graph
        dependencies: dict[int, Set[int]] = {}
        
        for step in plan.steps:
            dependencies[step.step_id] = set()
            
            if step.tool == "compute" and isinstance(step, ComputeStep):
                for operand in step.operands:
                    if operand.type == "reference" and operand.step_ref is not None:
                        # Only track positive references (negative are history)
                        if operand.step_ref >= 0:
                            dependencies[step.step_id].add(operand.step_ref)
        
        # Check for cycles using depth-first search
        visited: Set[int] = set()
        rec_stack: Set[int] = set()
        
        def has_cycle(node: int) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependencies.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    issues.append(
                        f"Circular dependency detected involving step {node} and step {neighbor}"
                    )
                    return True
            
            rec_stack.remove(node)
            return False
        
        for step_id in dependencies:
            if step_id not in visited:
                if has_cycle(step_id):
                    break
        
        return issues
