"""ConvFinQA dataset models"""
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Financial document"""
    pre_text: str
    post_text: str
    table: dict[str, dict[str, float | str | int]]


class Dialogue(BaseModel):
    """Conversational dialogue"""
    conv_questions: list[str]
    conv_answers: list[str]
    turn_program: list[str]
    executed_answers: list[float | str]
    qa_split: list[bool]


class Features(BaseModel):
    """Dataset features"""
    num_dialogue_turns: int
    has_type2_question: bool
    has_duplicate_columns: bool
    has_non_numeric_values: bool


class ConvFinQARecord(BaseModel):
    """Single record from ConvFinQA dataset"""
    id: str
    doc: Document
    dialogue: Dialogue
    features: Features