"""Workflow plan validator with reference and dependency checking"""
from typing import Set, List, Dict
from src.models.workflow_schema import WorkflowPlan, WorkflowStep, Operand, ValidationResult, StepCritique
from src.logger import get_logger

logger = get_logger(__name__)


class WorkflowValidator:
    """
    Validate workflow plans for correctness with structured critiques.
    
    Returns StepCritique objects with:
    - step_id: Where the issue occurred
    - issue_type: Category of the problem
    - reason: Why it's an issue
    - fix_suggestion: How to resolve it
    """
    
    # Operation requirements
    UNARY_OPERATIONS = {"percentage"}
    BINARY_OPERATIONS = {"add", "subtract", "multiply", "divide", "percentage_change"}
    
    def validate(self, plan: WorkflowPlan) -> ValidationResult:
        """
        Validate workflow plan for structural and logical correctness.
        
        Returns:
            ValidationResult with structured critiques
        """
        critiques: List[StepCritique] = []
        failed_steps: List[int] = []
        failure_types: List[str] = []
        
        # Build set of valid step_ids
        step_ids: Set[int] = {step.step_id for step in plan.steps}
        
        # Check 0: Validate at least one step exists
        if not plan.steps or len(plan.steps) == 0:
            critiques.append(StepCritique(
                step_id=None,
                issue_type="EmptyPlan",
                reason="The workflow plan contains no steps. A valid plan must have at least one extraction or computation step.",
                fix_suggestion="Add at least one step: use 'extract_value' to get data from tables/text, or 'compute' to perform calculations."
            ))
            failure_types.append("EMPTY_PLAN")
        
        # Check 1: Validate thought_process
        if not plan.thought_process or not plan.thought_process.strip():
            critiques.append(StepCritique(
                step_id=None,
                issue_type="MissingThoughtProcess",
                reason="The 'thought_process' field is empty or missing. This field should explain the strategy behind the workflow.",
                fix_suggestion="Add a clear explanation of the plan strategy, including what data to extract and what computations to perform (e.g., 'Extract 2023 and 2024 revenue, then calculate year-over-year growth')."
            ))
            failure_types.append("MISSING_THOUGHT_PROCESS")
        
        # Check 2: Verify step IDs are sequential
        expected_ids = list(range(1, len(plan.steps) + 1))
        actual_ids = [step.step_id for step in plan.steps]
        
        if actual_ids != expected_ids:
            critiques.append(StepCritique(
                step_id=None,
                issue_type="InvalidStepSequence",
                reason=f"Step IDs must be sequential starting from 1. Expected {expected_ids}, but got {actual_ids}. Sequential IDs ensure proper execution order.",
                fix_suggestion="Renumber all steps sequentially: first step should have step_id=1, second step step_id=2, third step step_id=3, and so on."
            ))
            failure_types.append("INVALID_STEP_SEQUENCE")
        
        # Check 3: Validate each step
        for step in plan.steps:
            step_critiques = self._validate_step(step, step_ids)
            if step_critiques:
                failed_steps.append(step.step_id)
                critiques.extend(step_critiques)
        
        # Check 4: Validate circular dependencies
        circular_critiques = self._check_circular_dependencies(plan)
        if circular_critiques:
            critiques.extend(circular_critiques)
            failure_types.append("CIRCULAR_DEPENDENCY")
            for c in circular_critiques:
                if c.step_id:
                    failed_steps.append(c.step_id)
        
        # Check 5: Validate unreachable steps
        unreachable_critiques = self._check_unreachable_steps(plan)
        if unreachable_critiques:
            critiques.extend(unreachable_critiques)
            failure_types.append("UNREACHABLE_STEP")
        
        # Deduplicate failed_steps
        failed_steps = sorted(list(set(failed_steps)))
        
        # Generate legacy issues format
        issues = [f"[{c.issue_type}] Step {c.step_id}: {c.reason} → FIX: {c.fix_suggestion}" for c in critiques]
        
        # Generate repair instructions
        repair_instructions = self._generate_repair_instructions(critiques, failed_steps, failure_types)
        
        is_valid = len(critiques) == 0
        
        if is_valid:
            logger.info(f"✓ Plan validation passed: {len(plan.steps)} steps")
        else:
            logger.warning(f"✗ Plan validation failed with {len(critiques)} issues")
            for c in critiques:
                logger.warning(f"  [{c.issue_type}] Step {c.step_id}: {c.reason}")
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=1.0 if is_valid else 0.0,
            issues=issues,
            critiques=critiques,
            failed_steps=failed_steps,
            failure_type=failure_types[0] if failure_types else None,
            repair_instructions=repair_instructions,
            is_hallucination_likely=False
        )
    
    def _validate_step(self, step: WorkflowStep, valid_step_ids: Set[int]) -> List[StepCritique]:
        """Validate a single workflow step"""
        critiques = []
        
        if step.tool == "extract_value":
            critiques.extend(self._validate_extract_step(step))
        elif step.tool == "compute":
            critiques.extend(self._validate_compute_step(step, valid_step_ids))
        else:
            critiques.append(StepCritique(
                step_id=step.step_id,
                issue_type="InvalidTool",
                reason=f"Unknown tool '{step.tool}'. Only 'extract_value' and 'compute' are supported.",
                fix_suggestion="Use 'extract_value' for data extraction or 'compute' for calculations."
            ))
        
        return critiques
    
    def _validate_extract_step(self, step: WorkflowStep) -> List[StepCritique]:
        """Validate an extraction step"""
        critiques = []
        
        if step.source is None:
            critiques.append(StepCritique(
                step_id=step.step_id,
                issue_type="MissingSource",
                reason="Extract step missing 'source' field. Cannot determine whether to extract from table or text.",
                fix_suggestion="Set source='table' or source='text' depending on where the data is located."
            ))
            return critiques
        
        if step.source == "table":
            if step.table_params is None:
                critiques.append(StepCritique(
                    step_id=step.step_id,
                    issue_type="MissingTableParams",
                    reason="Table extraction requires table_params to specify which row and column to extract.",
                    fix_suggestion="Add table_params with 'row_query' (e.g., 'Total Revenue'), 'col_query' (e.g., '2023'), and optional 'table_id' and 'unit_normalization'."
                ))
            else:
                if not step.table_params.row_query or not step.table_params.row_query.strip():
                    critiques.append(StepCritique(
                        step_id=step.step_id,
                        issue_type="EmptyRowQuery",
                        reason="table_params.row_query is empty. Cannot locate the correct row without a query.",
                        fix_suggestion="Provide a descriptive row name to search for (e.g., 'Total Revenue', 'Net Income', 'Operating Expenses')."
                    ))
                if not step.table_params.col_query or not step.table_params.col_query.strip():
                    critiques.append(StepCritique(
                        step_id=step.step_id,
                        issue_type="EmptyColQuery",
                        reason="table_params.col_query is empty. Cannot locate the correct column without a query.",
                        fix_suggestion="Provide a column name or year to search for (e.g., '2023', '2024', 'Q1 2024')."
                    ))
            
            if step.text_params is not None:
                critiques.append(StepCritique(
                    step_id=step.step_id,
                    issue_type="ConflictingParams",
                    reason="text_params should not be set when source='table'. This creates ambiguity about the data source.",
                    fix_suggestion="Remove text_params or change source to 'text' if you want to extract from prose."
                ))
        
        elif step.source == "text":
            if step.text_params is None:
                critiques.append(StepCritique(
                    step_id=step.step_id,
                    issue_type="MissingTextParams",
                    reason="Text extraction requires text_params to specify search keywords and context window.",
                    fix_suggestion="Add text_params with 'context_window' ('pre_text' or 'post_text'), 'search_keywords' (2-4 keywords), and 'unit' (e.g., 'million')."
                ))
            else:
                if not step.text_params.search_keywords or len(step.text_params.search_keywords) == 0:
                    critiques.append(StepCritique(
                        step_id=step.step_id,
                        issue_type="EmptyKeywords",
                        reason="text_params.search_keywords is empty. Cannot locate values in text without search hints.",
                        fix_suggestion="Provide 2-4 semantic keywords to help locate the value in text (e.g., ['revenue', 'increase', '2023'], ['costs', 'reduction'])."
                    ))
                elif len(step.text_params.search_keywords) < 2:
                    critiques.append(StepCritique(
                        step_id=step.step_id,
                        issue_type="FewKeywords",
                        reason=f"text_params.search_keywords has only {len(step.text_params.search_keywords)} keyword(s). More keywords improve extraction accuracy.",
                        fix_suggestion="Add 1-3 more relevant keywords. Aim for 2-4 total keywords that describe the value you're looking for."
                    ))
            
            if step.table_params is not None:
                critiques.append(StepCritique(
                    step_id=step.step_id,
                    issue_type="ConflictingParams",
                    reason="table_params should not be set when source='text'. This creates ambiguity about the data source.",
                    fix_suggestion="Remove table_params or change source to 'table' if you want to extract from a table."
                ))
        
        return critiques
    
    def _validate_compute_step(self, step: WorkflowStep, valid_step_ids: Set[int]) -> List[StepCritique]:
        """Validate a computation step"""
        critiques = []
        
        if step.operation is None:
            critiques.append(StepCritique(
                step_id=step.step_id,
                issue_type="MissingOperation",
                reason="Compute step missing 'operation' field. Cannot perform calculation without knowing the operation.",
                fix_suggestion="Specify operation: 'add', 'subtract', 'multiply', 'divide', 'percentage', or 'percentage_change'."
            ))
            return critiques
        
        if step.operands is None or len(step.operands) == 0:
            critiques.append(StepCritique(
                step_id=step.step_id,
                issue_type="MissingOperands",
                reason="Compute step missing 'operands'. Operations need input values to work with.",
                fix_suggestion="Provide 1-2 operands depending on the operation (1 for 'percentage', 2 for 'add'/'subtract'/'multiply'/'divide'/'percentage_change')."
            ))
            return critiques
        
        # Validate operand count
        operand_count = len(step.operands)
        
        if step.operation in self.UNARY_OPERATIONS:
            if operand_count != 1:
                critiques.append(StepCritique(
                    step_id=step.step_id,
                    issue_type="InvalidOperandCount",
                    reason=f"Operation '{step.operation}' requires exactly 1 operand, but {operand_count} provided. Unary operations work on a single value.",
                    fix_suggestion=f"For '{step.operation}', provide a single operand (the value to convert to percentage)."
                ))
        elif step.operation in self.BINARY_OPERATIONS:
            if operand_count != 2:
                critiques.append(StepCritique(
                    step_id=step.step_id,
                    issue_type="InvalidOperandCount",
                    reason=f"Operation '{step.operation}' requires exactly 2 operands, but {operand_count} provided. Binary operations need two values.",
                    fix_suggestion=f"For '{step.operation}', provide two operands (e.g., [{{type: 'reference', step_ref: 1}}, {{type: 'reference', step_ref: 2}}])."
                ))
        
        # Validate each operand
        for j, operand in enumerate(step.operands):
            critiques.extend(self._validate_operand(operand, step.step_id, valid_step_ids, j))
        
        # Warn about conflicting params
        if step.source is not None or step.table_params is not None or step.text_params is not None:
            critiques.append(StepCritique(
                step_id=step.step_id,
                issue_type="ConflictingParams",
                reason="Compute steps should not have 'source', 'table_params', or 'text_params'. These are only for extraction steps.",
                fix_suggestion="Remove extraction parameters (source, table_params, text_params) from this compute step."
            ))
        
        return critiques
    
    def _validate_operand(self, operand: Operand, step_id: int, valid_step_ids: Set[int], operand_index: int) -> List[StepCritique]:
        """Validate a single operand"""
        critiques = []
        
        if operand.type == "reference":
            if operand.step_ref is None:
                critiques.append(StepCritique(
                    step_id=step_id,
                    issue_type="MissingStepRef",
                    reason=f"Operand[{operand_index}] has type='reference' but missing 'step_ref'. Cannot resolve which step's result to use.",
                    fix_suggestion=f"Set step_ref to the ID of the step whose result you want to use (e.g., step_ref=1 to reference step 1's result)."
                ))
                return critiques
            
            if operand.step_ref >= 0:
                if operand.step_ref >= step_id:
                    critiques.append(StepCritique(
                        step_id=step_id,
                        issue_type="ForwardReference",
                        reason=f"Operand[{operand_index}] references step {operand.step_ref}, but we're in step {step_id}. Steps can only reference previous steps, not future ones.",
                        fix_suggestion=f"Reorder steps so step {operand.step_ref} comes before step {step_id}, or reference an earlier step."
                    ))
                
                if operand.step_ref not in valid_step_ids:
                    available_steps = sorted(list(valid_step_ids))
                    critiques.append(StepCritique(
                        step_id=step_id,
                        issue_type="InvalidReference",
                        reason=f"Operand[{operand_index}] references step {operand.step_ref}, which does not exist in the plan. Available steps: {available_steps}.",
                        fix_suggestion=f"Change step_ref to one of the existing step IDs: {available_steps}."
                    ))
            else:
                logger.debug(f"Step {step_id}: Uses conversation history reference (step_ref={operand.step_ref})")
        
        elif operand.type == "literal":
            if operand.value is None:
                critiques.append(StepCritique(
                    step_id=step_id,
                    issue_type="MissingLiteralValue",
                    reason=f"Operand[{operand_index}] has type='literal' but missing 'value'. Cannot use a literal operand without a value.",
                    fix_suggestion="Set value to a numeric constant (e.g., value=100.0, value=0.5)."
                ))
        
        else:
            critiques.append(StepCritique(
                step_id=step_id,
                issue_type="InvalidOperandType",
                reason=f"Operand[{operand_index}] has unknown type '{operand.type}'. Only 'reference' and 'literal' are supported.",
                fix_suggestion="Use type='reference' with step_ref to reference another step, or type='literal' with value for a constant."
            ))
        
        return critiques
    
    def _check_circular_dependencies(self, plan: WorkflowPlan) -> List[StepCritique]:
        """Check for circular dependencies"""
        critiques = []
        dependencies: Dict[int, Set[int]] = {}
        
        for step in plan.steps:
            dependencies[step.step_id] = set()
            if step.tool == "compute" and step.operands:
                for operand in step.operands:
                    if operand.type == "reference" and operand.step_ref is not None and operand.step_ref >= 0:
                        dependencies[step.step_id].add(operand.step_ref)
        
        visited: Set[int] = set()
        rec_stack: Set[int] = set()
        
        def has_cycle(node: int, path: List[int]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in dependencies.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    cycle_start_idx = path.index(neighbor)
                    cycle = path[cycle_start_idx:] + [neighbor]
                    cycle_str = " → ".join([f"step {s}" for s in cycle])
                    critiques.append(StepCritique(
                        step_id=node,
                        issue_type="CircularDependency",
                        reason=f"Circular dependency detected: {cycle_str}. Steps cannot reference each other in a cycle.",
                        fix_suggestion="Restructure the workflow to remove the circular reference. Each step should only reference previous steps."
                    ))
                    return True
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        for step_id in sorted(dependencies.keys()):
            if step_id not in visited:
                if has_cycle(step_id, []):
                    break
        
        return critiques
    
    def _check_unreachable_steps(self, plan: WorkflowPlan) -> List[StepCritique]:
        """Check for unreachable/orphaned steps"""
        critiques = []
        
        if len(plan.steps) <= 1:
            return critiques
        
        referenced_steps: Set[int] = set()
        for step in plan.steps:
            if step.tool == "compute" and step.operands:
                for operand in step.operands:
                    if operand.type == "reference" and operand.step_ref is not None and operand.step_ref >= 0:
                        referenced_steps.add(operand.step_ref)
        
        unreferenced_steps = []
        for step in plan.steps[:-1]:  # Exclude final step
            if step.step_id not in referenced_steps:
                unreferenced_steps.append(step.step_id)
        
        if unreferenced_steps:
            for step_id in unreferenced_steps:
                critiques.append(StepCritique(
                    step_id=step_id,
                    issue_type="UnreachableStep",
                    reason=f"Step {step_id} is never referenced by subsequent steps. Its computed value is not used in the final result.",
                    fix_suggestion=f"Either remove step {step_id}, or add a compute step that uses its result. If you need this intermediate value, make sure it feeds into the final computation."
                ))
        
        return critiques
    
    def _generate_repair_instructions(self, critiques: List[StepCritique], failed_steps: List[int], failure_types: List[str]) -> str:
        """Generate repair instructions from critiques"""
        if not critiques:
            return ""
        
        instructions = []
        issue_types = {c.issue_type for c in critiques}
        
        if any(t in issue_types for t in ["EmptyPlan", "InvalidStepSequence", "MissingThoughtProcess"]):
            instructions.append("1. FIX STRUCTURE: Ensure sequential step IDs (1, 2, 3...) and a non-empty thought_process field.")
        
        if any("Missing" in t or "Empty" in t for t in issue_types):
            instructions.append("2. FIX MISSING PARAMETERS: Add required params (table_params for table extraction, text_params for text extraction, operation and operands for compute steps).")
        
        if "InvalidOperandCount" in issue_types:
            instructions.append("3. FIX OPERAND COUNTS: 1 operand for 'percentage', 2 for 'add'/'subtract'/'multiply'/'divide'/'percentage_change'.")
        
        if any(t in issue_types for t in ["ForwardReference", "InvalidReference", "CircularDependency"]):
            instructions.append("4. FIX REFERENCES: Steps can only reference previous steps (lower IDs). Remove circular references.")
        
        if failed_steps:
            instructions.append(f"5. FOCUS ON: Steps {failed_steps} need attention.")
        
        return " ".join(instructions) if instructions else "Review critiques and apply suggested fixes."
    
    def check_circular_dependencies(self, plan: WorkflowPlan) -> List[StepCritique]:
        """Public method for backwards compatibility"""
        return self._check_circular_dependencies(plan)
