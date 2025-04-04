from unittest.mock import Mock, patch

import pytest

from app import MCQGradioApp
from src.model.qwen_handler import QwenModelHandler
from src.prompt_processors.prompt_creator import PromptCreator
from src.prompt_processors.response_parser import ResponseParser


@pytest.fixture
def mock_model_handler():
    """Create a mock model handler"""
    with patch("app.QwenModelHandler") as mock:
        handler = Mock()
        mock.return_value = handler
        yield handler


@pytest.fixture
def app(mock_model_handler):
    """Create an app instance with mocked model handler"""
    return MCQGradioApp()


def test_inference_success(app, mock_model_handler):
    """Test successful inference"""
    # Setup mock response
    mock_model_handler.generate_with_streaming.return_value = """
    understanding: |
        The question asks about Python list iteration.
    analysis: |
        A. for i in range(len(lst)): Valid
        B. for i in lst: Valid
        C. while i < len(lst): Valid
        D. lst.iterate(): Invalid
    reasoning: |
        Python lists can be iterated using for loops with range(),
        direct iteration, or while loops. There is no iterate() method.
    conclusion: |
        Option D is invalid as lists don't have an iterate() method.
    answer: D
    """

    # Test inference
    prompt, response = app.inference(
        "Which of the following is NOT a valid way to iterate through a list in Python?",
        "A. for i in range(len(lst))\nB. for i in lst\nC. while i < len(lst)\nD. lst.iterate()",
        temperature=0.1,
    )

    # Verify response format
    assert "Predicted Answer: D" in response
    assert "Reasoning:" in response
    assert "Raw Model Output:" in response
    assert "understanding:" in prompt
    assert "analysis:" in prompt
    assert "reasoning:" in prompt
    assert "conclusion:" in prompt
    assert "answer:" in prompt


def test_inference_empty_choices(app, mock_model_handler):
    """Test inference with empty choices"""
    with pytest.raises(ValueError, match="Choices cannot be empty"):
        app.inference("Test question", "")


def test_inference_model_error(app, mock_model_handler):
    """Test inference when model raises an error"""
    # Setup mock to raise an exception
    mock_model_handler.generate_with_streaming.side_effect = Exception("Model error")

    # Test inference
    with pytest.raises(Exception, match="Model error"):
        app.inference("Test question", "A. Choice 1\nB. Choice 2", temperature=0.1)


def test_inference_invalid_response(app, mock_model_handler):
    """Test inference with invalid model response format"""
    # Setup mock to return invalid response format
    mock_model_handler.generate_with_streaming.return_value = "Invalid response format"

    # Test inference
    prompt, response = app.inference("Test question", "A. Choice 1\nB. Choice 2", temperature=0.1)

    # Verify response still includes raw output even if parsing fails
    assert "Raw Model Output:" in response
    assert "Invalid response format" in response


def test_inference_long_response(app, mock_model_handler):
    """Test inference with very long response"""
    # Setup mock to return a very long response
    long_response = "understanding: |\n" + "Test " * 1000
    mock_model_handler.generate_with_streaming.return_value = long_response

    # Test inference
    prompt, response = app.inference("Test question", "A. Choice 1\nB. Choice 2", temperature=0.1)

    # Verify response is truncated if needed
    assert len(response) <= 4096  # Assuming max length of 4096 tokens


def test_inference_special_characters(app, mock_model_handler):
    """Test inference with special characters in input"""
    # Setup mock response
    mock_model_handler.generate_with_streaming.return_value = """
    understanding: |
        Testing special chars: !@#$%^&*()
    analysis: |
        A. Test1
        B. Test2
    reasoning: |
        Testing reasoning
    conclusion: |
        Testing conclusion
    answer: A
    """

    # Test inference with special characters
    prompt, response = app.inference(
        "Test question with special chars: !@#$%^&*()",
        "A. Choice with !@#$%^&*()\nB. Normal choice",
        temperature=0.1,
    )

    # Verify response handles special characters
    assert "Predicted Answer: A" in response
    assert "!@#$%^&*()" in prompt


def test_inference_unicode_characters(app, mock_model_handler):
    """Test inference with unicode characters"""
    # Setup mock response
    mock_model_handler.generate_with_streaming.return_value = """
    understanding: |
        Testing unicode: 你好世界
    analysis: |
        A. 选项1
        B. 选项2
    reasoning: |
        测试推理
    conclusion: |
        测试结论
    answer: A
    """

    # Test inference with unicode characters
    prompt, response = app.inference("测试问题", "A. 选项1\nB. 选项2", temperature=0.1)

    # Verify response handles unicode characters
    assert "Predicted Answer: A" in response
    assert "你好世界" in prompt
    assert "选项1" in prompt
    assert "选项2" in prompt
