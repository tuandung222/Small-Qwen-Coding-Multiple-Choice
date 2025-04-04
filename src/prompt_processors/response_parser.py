import re
from typing import Any, Dict, Optional, Tuple

import yaml

try:
    from .prompt_creator import PromptCreator
except ImportError:
    pass


class ResponseParser:
    """
    Parser for model responses with support for different formats
    Extracts answers and reasoning from model outputs
    """

    # Parser modes
    BASIC = "basic"  # Extract single letter answer
    YAML = "yaml"  # Parse YAML formatted response with reasoning

    def __init__(self, parser_mode: str = BASIC):
        """
        Initialize with specified parser mode

        Args:
            parser_mode: Mode of parsing to use (BASIC or YAML)
        """
        if parser_mode not in [self.BASIC, self.YAML]:
            raise ValueError(f"Unknown parser mode: {parser_mode}")
        self.parser_mode = parser_mode

    def parse(self, response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the response text to extract answer and reasoning

        Args:
            response_text: Raw response text from the model

        Returns:
            Tuple of (answer, reasoning)
        """
        if not response_text:
            return None, None

        if self.parser_mode == self.BASIC:
            return self._parse_basic_response(response_text)
        elif self.parser_mode == self.YAML:
            return self._parse_yaml_response(response_text)
        else:
            raise ValueError(f"Unknown parser mode: {self.parser_mode}")

    def _parse_basic_response(self, response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse basic response format (just the answer)"""
        # Clean up the response
        response_text = response_text.strip()

        # Try to find a single letter answer
        answer_match = re.search(r"^[A-Za-z]$", response_text)
        if answer_match:
            return answer_match.group(0).upper(), None

        # Try to find answer after "Answer:" or similar
        answer_match = re.search(r"(?:answer|Answer):\s*([A-Za-z])", response_text)
        if answer_match:
            return answer_match.group(1).upper(), None

        # Try to find any single letter in the response
        answer_match = re.search(r"[A-Za-z]", response_text)
        if answer_match:
            return answer_match.group(0).upper(), None

        return None, None

    def _parse_yaml_response(self, response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse YAML-formatted response with reasoning"""
        # Clean up the response
        response_text = response_text.strip()

        # Remove any markdown code block markers
        response_text = re.sub(r"```yaml\s*", "", response_text)
        response_text = re.sub(r"```\s*", "", response_text)

        try:
            # Try to parse as YAML
            yaml_content = yaml.safe_load("---\n" + response_text)
            if isinstance(yaml_content, dict):
                answer = yaml_content.get("answer")
                reasoning = self._extract_reasoning_from_yaml(yaml_content)

                # Clean up answer if needed
                if answer:
                    answer = answer.strip().upper()
                    if len(answer) > 1:
                        # Extract first letter if multiple characters
                        answer = answer[0]

                return answer, reasoning
        except yaml.YAMLError:
            # If YAML parsing fails, try to extract using regex
            answer_match = re.search(r"answer:\s*([A-Za-z])", response_text)
            reasoning_match = re.search(r"reasoning:\s*\|\s*([\s\S]+?)(?:\n\w+:|$)", response_text)

            answer = answer_match.group(1).upper() if answer_match else None
            reasoning = reasoning_match.group(1).strip() if reasoning_match else None

            return answer, reasoning

        return None, None

    def _extract_reasoning_from_yaml(self, yaml_content: Dict[str, Any]) -> Optional[str]:
        """Extract and format reasoning from YAML content"""
        reasoning_parts = []

        # Add understanding if present
        if "understanding" in yaml_content:
            reasoning_parts.append(f"Understanding:\n{yaml_content['understanding']}")

        # Add analysis if present
        if "analysis" in yaml_content:
            reasoning_parts.append(f"Analysis:\n{yaml_content['analysis']}")

        # Add reasoning if present
        if "reasoning" in yaml_content:
            reasoning_parts.append(f"Reasoning:\n{yaml_content['reasoning']}")

        # Add conclusion if present
        if "conclusion" in yaml_content:
            reasoning_parts.append(f"Conclusion:\n{yaml_content['conclusion']}")

        return "\n\n".join(reasoning_parts) if reasoning_parts else None

    def set_parser_mode(self, parser_mode: str) -> "ResponseParser":
        """Set the parser mode"""
        if parser_mode not in [self.BASIC, self.YAML]:
            raise ValueError(f"Unknown parser mode: {parser_mode}")
        self.parser_mode = parser_mode
        return self

    @classmethod
    def from_prompt_type(cls, prompt_type: str) -> "ResponseParser":
        """
        Create a ResponseParser instance from a prompt type

        Args:
            prompt_type: Type of prompt (from PromptCreator)

        Returns:
            ResponseParser instance with appropriate mode
        """
        if prompt_type == PromptCreator.BASIC:
            return cls(cls.BASIC)
        elif prompt_type in [PromptCreator.YAML_REASONING, PromptCreator.TEACHER_REASONED]:
            return cls(cls.YAML)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
