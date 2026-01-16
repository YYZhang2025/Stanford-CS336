import logging

from .adapters import run_parse_gsm8k_response, run_parse_mmlu_response

logger = logging.getLogger(__name__)


def test_parse_mmlu_response():
    mmlu_example = {
        "subject": "virology",
        "question": "How many human polyomaviruses are known at present?",
        "options": ["100", "1", "10", "unknown"],
        "answer": "A",
    }
    model_output = (
        "The correct answer is B. "
        "There is only one human polyomavirus known at present, which is the BK virus."
    )
    parsed_response = run_parse_mmlu_response(
        mmlu_example=mmlu_example, model_output=model_output
    )
    assert parsed_response == "B"


def test_parse_mmlu_response_unknown():
    mmlu_example = {
        "subject": "virology",
        "question": "How many human polyomaviruses are known at present?",
        "options": ["100", "1", "10", "unknown"],
        "answer": "A",
    }
    model_output = "The correct answer is 10000 polyomaviruses."
    parsed_response = run_parse_mmlu_response(
        mmlu_example=mmlu_example, model_output=model_output
    )
    assert not parsed_response


def test_parse_gsm8k_response():
    model_output = (
        "Natalia sold 48/2 = 24 clips in May. "
        "Natalia sold 48+24 = 72 clips altogether in April and May."
    )
    parsed_response = run_parse_gsm8k_response(model_output=model_output)
    assert parsed_response == "72"


def test_parse_gsm8k_response_unknown():
    model_output = (
        "Natalia sold twenty-four clips in May. "
        "Thus, Natalia sold seventy-two clips altogether in April and May."
    )
    parsed_response = run_parse_gsm8k_response(model_output=model_output)
    assert not parsed_response
