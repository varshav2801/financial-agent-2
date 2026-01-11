"""Custom exceptions for structured error handling"""


class ToolExecutionError(Exception):
    """Base exception for all tool execution errors"""
    def __init__(self, message: str, tool_name: str | None = None, details: dict | None = None):
        self.message = message
        self.tool_name = tool_name
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Convert error to structured dict for logging/display"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "tool": self.tool_name,
            "details": self.details
        }


# Table Extraction Errors
class TableExtractionError(ToolExecutionError):
    """Base exception for table extraction failures"""
    pass


# Text Extraction Errors
class TextExtractionError(ToolExecutionError):
    """Base exception for text extraction failures"""
    pass


class TextParsingError(TextExtractionError):
    """Failed to parse numeric value from text"""
    pass


# Math Operation Errors
class MathOperationError(ToolExecutionError):
    """Base exception for math operation failures"""
    pass


class DivisionByZeroError(MathOperationError):
    """Division by zero attempted"""
    pass


class InvalidOperationError(MathOperationError):
    """Invalid operation name or parameters"""
    pass


class OperandTypeError(MathOperationError):
    """Operand has wrong type or cannot be converted"""
    pass


# Plan Execution Errors
class PlanExecutionError(ToolExecutionError):
    """Base exception for plan execution failures"""
    pass


class StepExecutionError(PlanExecutionError):
    """Specific step failed to execute"""
    def __init__(self, message: str, step_id: str | int, original_error: Exception | None = None):
        self.step_id = step_id
        self.original_error = original_error
        details = {"step_id": step_id}
        if original_error:
            details["original_error"] = str(original_error)
            details["original_error_type"] = type(original_error).__name__
        super().__init__(message, tool_name="executor", details=details)


def format_error_message(error: Exception, include_traceback: bool = False) -> str:
    """
    Format error into user-friendly message.
    
    Args:
        error: Exception to format
        include_traceback: Whether to include technical details
        
    Returns:
        Formatted error message string
    """
    if isinstance(error, ToolExecutionError):
        msg = f"[{error.__class__.__name__}] {error.message}"
        if error.tool_name:
            msg = f"{error.tool_name}: {msg}"
        if include_traceback and error.details:
            details_str = ", ".join(f"{k}={v}" for k, v in error.details.items())
            msg = f"{msg} ({details_str})"
        return msg
    else:
        # Generic exception
        error_type = type(error).__name__
        return f"[{error_type}] {str(error)}"
