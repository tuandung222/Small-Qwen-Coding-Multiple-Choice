from typing import List, Optional, Union


class PromptCreator:
    """
    Creates and formats prompts for multiple choice questions
    Supports different prompt styles for training and inference
    """

    # Prompt types
    BASIC = "basic"  # Simple answer-only format
    YAML_REASONING = "yaml"  # YAML formatted reasoning
    TEACHER_REASONED = (
        "teacher"  # Same YAML format as YAML_REASONING but using teacher completions for training
    )
    OPTIONS = "options"  # Includes only lettered options in prompt

    # Additional reasoning paradigms
    SOCRATIC = "socratic"  # Uses Socratic questioning to explore options
    SCIENTIST = "scientist"  # Uses scientific method and hypothesis testing
    LAWYER = "lawyer"  # Uses legal arguments and evidence evaluation
    DEBUGGER = "debugger"  # Uses programmer debugging methodology
    PHILOSOPHER = "philosopher"  # Uses philosophical analysis frameworks
    EXPERT_NOVICE = "expert_novice"  # Dialogues between expert and novice
    PROS_CONS = "pros_cons"  # Evaluates pros and cons for each option
    CODE_REVIEW = "code_review"  # Uses code review paradigm for code questions
    MATH_PROOF = "math_proof"  # Uses mathematical proof structure

    VALID_PROMPT_TYPES = [
        BASIC,
        YAML_REASONING,
        TEACHER_REASONED,
        OPTIONS,
        SOCRATIC,
        SCIENTIST,
        LAWYER,
        DEBUGGER,
        PHILOSOPHER,
        EXPERT_NOVICE,
        PROS_CONS,
        CODE_REVIEW,
        MATH_PROOF,
    ]

    def __init__(self, prompt_type: str = BASIC):
        """
        Initialize with specified prompt type

        Args:
            prompt_type: Type of prompt to use

        Raises:
            ValueError: If prompt_type is not one of the valid types
        """
        if prompt_type not in self.VALID_PROMPT_TYPES:
            raise ValueError(
                f"Invalid prompt type: {prompt_type}. Must be one of {self.VALID_PROMPT_TYPES}"
            )

        # For prompt formatting, teacher_reasoned is equivalent to yaml_reasoning
        # The difference only matters during training when using teacher completions
        if prompt_type == self.TEACHER_REASONED:
            prompt_type = self.YAML_REASONING

        self.prompt_type = prompt_type
        # Store the original prompt type to track if we're using teacher mode
        self.original_type = prompt_type

    def format_choices(self, choices: Union[List[str], str]) -> str:
        """
        Format choices into a string

        Args:
            choices: List of choices or pre-formatted string

        Returns:
            Formatted string of choices

        Raises:
            ValueError: If choices is empty or invalid
        """
        if not choices:
            raise ValueError("Choices cannot be empty")

        if isinstance(choices, str):
            return choices

        if not isinstance(choices, list):
            raise ValueError(f"Choices must be a list or string, got {type(choices)}")

        if not all(isinstance(choice, str) for choice in choices):
            raise ValueError("All choices must be strings")

        return "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))

    def get_max_letter(self, choices: Union[List[str], str]) -> str:
        """
        Get the maximum letter for the given number of choices

        Args:
            choices: List of choices or pre-formatted string

        Returns:
            Maximum letter (A, B, C, etc.)

        Raises:
            ValueError: If choices is empty or invalid
        """
        if not choices:
            raise ValueError("Choices cannot be empty")

        if isinstance(choices, str):
            # Try to count the number of lines in the formatted string
            num_choices = len([line for line in choices.split("\n") if line.strip()])
            if num_choices == 0:
                raise ValueError("No valid choices found in string")
            return chr(64 + num_choices)

        if not isinstance(choices, list):
            raise ValueError(f"Choices must be a list or string, got {type(choices)}")

        if not all(isinstance(choice, str) for choice in choices):
            raise ValueError("All choices must be strings")

        return chr(64 + len(choices))

    def create_inference_prompt(self, question: str, choices: Union[List[str], str]) -> str:
        """
        Create a prompt for inference

        Args:
            question: The question text
            choices: List of choices or pre-formatted string

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If question or choices are empty or invalid
        """
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")

        formatted_choices = self.format_choices(choices)
        max_letter = self.get_max_letter(choices)

        # Basic prompt types
        if self.prompt_type == self.BASIC:
            return self._create_basic_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type in [self.YAML_REASONING, self.TEACHER_REASONED]:
            return self._create_yaml_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.OPTIONS:
            return self._create_options_prompt(question, formatted_choices, max_letter)

        # Advanced reasoning paradigms
        elif self.prompt_type == self.SOCRATIC:
            return self._create_socratic_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.SCIENTIST:
            return self._create_scientist_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.LAWYER:
            return self._create_lawyer_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.DEBUGGER:
            return self._create_debugger_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.PHILOSOPHER:
            return self._create_philosopher_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.EXPERT_NOVICE:
            return self._create_expert_novice_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.PROS_CONS:
            return self._create_pros_cons_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.CODE_REVIEW:
            return self._create_code_review_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.MATH_PROOF:
            return self._create_math_proof_prompt(question, formatted_choices, max_letter)
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_type}")

    def _create_basic_prompt(self, question: str, formatted_choices: str, max_letter: str) -> str:
        """Create a basic prompt that only asks for the answer"""
        return f"""Question: {question}

Choices:
{formatted_choices}

Answer with a single letter from A through {max_letter} without any additional explanation or commentary."""

    def _create_yaml_prompt(self, question: str, formatted_choices: str, max_letter: str) -> str:
        """Create a YAML-formatted prompt that asks for reasoning"""
        return f"""Question: {question}

Choices:
{formatted_choices}

Analyze this question step-by-step and provide a detailed explanation.
Your response MUST be in YAML format as follows:

understanding: |
  <your understanding of what the question is asking>
analysis: |
  <your analysis of each option>
reasoning: |
  <your step-by-step reasoning process>
conclusion: |
  <your final conclusion>
answer: <single letter A through {max_letter}>

The answer field MUST contain ONLY a single character letter."""

    def _create_options_prompt(self, question: str, formatted_choices: str, max_letter: str) -> str:
        """Create a prompt that focuses on lettered options"""
        return f"""Question: {question}

Choices:
{formatted_choices}

Please select the best answer from the options above. Provide a brief explanation for your choice and clearly state the letter of your answer (A through {max_letter})."""

    def create_training_prompt(self, question: str, choices: Union[List[str], str]) -> str:
        """
        Create a prompt for training

        Args:
            question: The question text
            choices: List of choices or pre-formatted string

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If question or choices are empty or invalid
        """
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")

        formatted_choices = self.format_choices(choices)
        max_letter = self.get_max_letter(choices)

        # Basic prompt types
        if self.prompt_type == self.BASIC:
            return self._create_basic_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type in [self.YAML_REASONING, self.TEACHER_REASONED]:
            return self._create_yaml_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.OPTIONS:
            return self._create_options_training_prompt(question, formatted_choices, max_letter)

        # Advanced reasoning paradigms
        elif self.prompt_type == self.SOCRATIC:
            return self._create_socratic_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.SCIENTIST:
            return self._create_scientist_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.LAWYER:
            return self._create_lawyer_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.DEBUGGER:
            return self._create_debugger_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.PHILOSOPHER:
            return self._create_philosopher_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.EXPERT_NOVICE:
            return self._create_expert_novice_training_prompt(
                question, formatted_choices, max_letter
            )
        elif self.prompt_type == self.PROS_CONS:
            return self._create_pros_cons_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.CODE_REVIEW:
            return self._create_code_review_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.MATH_PROOF:
            return self._create_math_proof_training_prompt(question, formatted_choices, max_letter)
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_type}")

    def _create_basic_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a basic training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

The answer is a single letter (A, B, C, etc.). Only provide ONE character as your answer:"""

    def _create_yaml_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a YAML-formatted training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

Analyze this question step-by-step and provide a detailed explanation.
Follow the YAML format in your response:

understanding: |
  <your understanding of the question>
analysis: |
  <your analysis of each option>
reasoning: |
  <your reasoning about the correct answer>
conclusion: |
  <your final conclusion>
answer: <single letter A through {max_letter}>"""

    def _create_options_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a training prompt for options format"""
        return f"""Question: {question}

Choices:
{formatted_choices}

Please select the best answer from the options above. Provide a brief explanation for your choice and clearly state the letter of your answer (A through {max_letter})."""

    def set_prompt_type(self, prompt_type: str) -> "PromptCreator":
        """
        Set the prompt type

        Args:
            prompt_type: Type of prompt to use (BASIC, YAML_REASONING, or TEACHER_REASONED)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If prompt_type is not one of the valid types
        """
        if prompt_type not in self.VALID_PROMPT_TYPES:
            raise ValueError(
                f"Invalid prompt type: {prompt_type}. Must be one of {self.VALID_PROMPT_TYPES}"
            )

        # Store the original type
        self.original_type = prompt_type

        # For prompt formatting, teacher_reasoned is equivalent to yaml_reasoning
        if prompt_type == self.TEACHER_REASONED:
            prompt_type = self.YAML_REASONING

        self.prompt_type = prompt_type
        return self

    def is_teacher_mode(self) -> bool:
        """Check if using teacher-reasoned mode"""
        return self.original_type == self.TEACHER_REASONED

    # Advanced reasoning paradigm prompt methods

    def _create_socratic_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a Socratic dialogue prompt that leads through questioning"""
        return f"""Question: {question}

Choices:
{formatted_choices}

To solve this problem, I'll use Socratic questioning to examine each option:
1. What do I already know about this topic?
2. What assumptions am I making?
3. What evidence would prove or disprove each option?
4. What are the implications of each option?
5. Are there alternative perspectives I should consider?

After answering these questions for each option, I will conclude with my answer letter (A through {max_letter})."""

    def _create_socratic_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a Socratic dialogue training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

To solve this problem, I'll use Socratic questioning to examine each option:
1. What do I already know about this topic?
2. What assumptions am I making?
3. What evidence would prove or disprove each option?
4. What are the implications of each option?
5. Are there alternative perspectives I should consider?

After answering these questions for each option, I will conclude with my answer letter (A through {max_letter})."""

    def _create_scientist_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a scientific method prompt that tests each option as a hypothesis"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll approach this using the scientific method:

1. Observation: Understanding what the question is asking
2. Hypothesis: Treating each option (A through {max_letter}) as a potential hypothesis
3. Testing: Analyzing the validity of each option
4. Analysis: Evaluating the evidence for each option
5. Conclusion: Determining which option is supported by the evidence

For my final answer, I'll clearly state which letter (A through {max_letter}) corresponds to the correct option."""

    def _create_scientist_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a scientific method training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll approach this using the scientific method:

1. Observation: Understanding what the question is asking
2. Hypothesis: Treating each option (A through {max_letter}) as a potential hypothesis
3. Testing: Analyzing the validity of each option
4. Analysis: Evaluating the evidence for each option
5. Conclusion: Determining which option is supported by the evidence

For my final answer, I'll clearly state which letter (A through {max_letter}) corresponds to the correct option."""

    def _create_lawyer_prompt(self, question: str, formatted_choices: str, max_letter: str) -> str:
        """Create a legal argument prompt that evaluates evidence"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll approach this question as a lawyer evaluating evidence:

FACTS:
- What are the key facts presented in the question?
- What established principles or knowledge apply to this situation?

ARGUMENTS:
- For each option (A through {max_letter}):
  - What arguments support this option?
  - What arguments oppose this option?
  - What is the strength of evidence for each?

RULING:
- Based on the weight of evidence, which option has the strongest case?
- Are there any reasonable doubts about my conclusion?

VERDICT:
My answer is option [letter A through {max_letter}]."""

    def _create_lawyer_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a legal argument training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll approach this question as a lawyer evaluating evidence:

FACTS:
- What are the key facts presented in the question?
- What established principles or knowledge apply to this situation?

ARGUMENTS:
- For each option (A through {max_letter}):
  - What arguments support this option?
  - What arguments oppose this option?
  - What is the strength of evidence for each?

RULING:
- Based on the weight of evidence, which option has the strongest case?
- Are there any reasonable doubts about my conclusion?

VERDICT:
My answer is option [letter A through {max_letter}]."""

    def _create_debugger_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a debugging prompt that treats options as code paths"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll debug this problem systematically:

1. PROBLEM STATEMENT:
   - What is the core issue or question?
   - What is the expected correct behavior/answer?

2. TRACE EXECUTION:
   - For each option (A through {max_letter}):
     - If this option were correct, what logical steps would follow?
     - Are there any logical errors or edge cases in this option?

3. TEST CASES:
   - What examples can I think of to test each option?
   - Do any options fail under certain conditions?

4. ROOT CAUSE:
   - Which option correctly addresses the core problem?
   - Why do the other options fail?

5. FIX:
   - My answer is option [letter A through {max_letter}]
   - Explanation: [brief justification]"""

    def _create_debugger_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a debugging training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll debug this problem systematically:

1. PROBLEM STATEMENT:
   - What is the core issue or question?
   - What is the expected correct behavior/answer?

2. TRACE EXECUTION:
   - For each option (A through {max_letter}):
     - If this option were correct, what logical steps would follow?
     - Are there any logical errors or edge cases in this option?

3. TEST CASES:
   - What examples can I think of to test each option?
   - Do any options fail under certain conditions?

4. ROOT CAUSE:
   - Which option correctly addresses the core problem?
   - Why do the other options fail?

5. FIX:
   - My answer is option [letter A through {max_letter}]
   - Explanation: [brief justification]"""

    def _create_philosopher_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a philosophical analysis prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll analyze this philosophically:

CONCEPTUAL ANALYSIS:
- What are the key concepts in this question?
- How might different philosophical frameworks interpret these concepts?

LOGICAL STRUCTURE:
- What is the logical form of each option?
- Are there any logical fallacies or contradictions in the options?

THOUGHT EXPERIMENT:
- What hypothetical scenarios could test the validity of each option?
- What would be the implications if each option were true?

SYNTHESIS:
- Which option best aligns with sound reasoning?
- What might opponents of this view argue?

CONCLUSION:
Therefore, the correct answer is option [letter A through {max_letter}]."""

    def _create_philosopher_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a philosophical analysis training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll analyze this philosophically:

CONCEPTUAL ANALYSIS:
- What are the key concepts in this question?
- How might different philosophical frameworks interpret these concepts?

LOGICAL STRUCTURE:
- What is the logical form of each option?
- Are there any logical fallacies or contradictions in the options?

THOUGHT EXPERIMENT:
- What hypothetical scenarios could test the validity of each option?
- What would be the implications if each option were true?

SYNTHESIS:
- Which option best aligns with sound reasoning?
- What might opponents of this view argue?

CONCLUSION:
Therefore, the correct answer is option [letter A through {max_letter}]."""

    def _create_expert_novice_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a dialogue between expert and novice"""
        return f"""Question: {question}

Choices:
{formatted_choices}

NOVICE: I'm trying to solve this multiple-choice question but I'm not sure how to approach it. Can you help me?

EXPERT: Of course! Let's break it down step by step. First, let's understand what the question is asking.

NOVICE: Okay, so the question is asking about [{question}]. And there are {max_letter - 64} possible answers.

EXPERT: That's right. Let's analyze each option one by one:

[Analysis of each option]

NOVICE: That makes sense. So which option do you think is correct?

EXPERT: Based on our analysis, I believe the correct answer is option [letter A through {max_letter}] because [explanation].

NOVICE: Got it! So the answer is [letter A through {max_letter}]."""

    def _create_expert_novice_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create an expert-novice dialogue training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

NOVICE: I'm trying to solve this multiple-choice question but I'm not sure how to approach it. Can you help me?

EXPERT: Of course! Let's break it down step by step. First, let's understand what the question is asking.

NOVICE: Okay, so the question is asking about [{question}]. And there are {max_letter - 64} possible answers.

EXPERT: That's right. Let's analyze each option one by one:

[Analysis of each option]

NOVICE: That makes sense. So which option do you think is correct?

EXPERT: Based on our analysis, I believe the correct answer is option [letter A through {max_letter}] because [explanation].

NOVICE: Got it! So the answer is [letter A through {max_letter}]."""

    def _create_pros_cons_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a pros and cons analysis prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll analyze each option by listing its pros and cons:

OPTION A:
- Pros: [list of advantages]
- Cons: [list of disadvantages]

[Continue for all options through {max_letter}]

DECISION MATRIX:
- Option with most pros: ?
- Option with fewest cons: ?
- Option with best overall balance: ?

CONCLUSION:
After weighing the pros and cons of each option, the answer is [letter A through {max_letter}]."""

    def _create_pros_cons_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a pros and cons analysis training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll analyze each option by listing its pros and cons:

OPTION A:
- Pros: [list of advantages]
- Cons: [list of disadvantages]

[Continue for all options through {max_letter}]

DECISION MATRIX:
- Option with most pros: ?
- Option with fewest cons: ?
- Option with best overall balance: ?

CONCLUSION:
After weighing the pros and cons of each option, the answer is [letter A through {max_letter}]."""

    def _create_code_review_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a code review prompt for programming questions"""
        return f"""Question: {question}

Choices:
{formatted_choices}

CODE REVIEW PROCESS:

REQUIREMENTS ANALYSIS:
- What is the expected behavior/output?
- What are the constraints or edge cases to consider?

CODE INSPECTION:
- For each option (A through {max_letter}):
  - Is the syntax correct?
  - Are there any potential bugs or edge cases?
  - Does it follow best practices?
  - Is it efficient and maintainable?

TESTING PERSPECTIVE:
- What test cases would validate or invalidate each option?
- How would each option handle those tests?

REVIEWER FEEDBACK:
Based on this review, option [letter A through {max_letter}] is the most correct because [explanation]."""

    def _create_code_review_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a code review training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

CODE REVIEW PROCESS:

REQUIREMENTS ANALYSIS:
- What is the expected behavior/output?
- What are the constraints or edge cases to consider?

CODE INSPECTION:
- For each option (A through {max_letter}):
  - Is the syntax correct?
  - Are there any potential bugs or edge cases?
  - Does it follow best practices?
  - Is it efficient and maintainable?

TESTING PERSPECTIVE:
- What test cases would validate or invalidate each option?
- How would each option handle those tests?

REVIEWER FEEDBACK:
Based on this review, option [letter A through {max_letter}] is the most correct because [explanation]."""

    def _create_math_proof_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a mathematical proof structure prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll approach this as a mathematical proof:

GIVEN:
- Information provided in the question
- Mathematical principles relevant to this problem

PROVE:
- I need to determine which option (A through {max_letter}) is correct

PROOF:
1. First, I'll establish the key principles needed
2. For each option:
   - Assume the option is true
   - Determine if this leads to a valid result or a contradiction
   - Note any special cases or conditions

CONCLUSION:
Therefore, option [letter A through {max_letter}] is proven to be correct.
"""

    def _create_math_proof_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a mathematical proof training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll approach this as a mathematical proof:

GIVEN:
- Information provided in the question
- Mathematical principles relevant to this problem

PROVE:
- I need to determine which option (A through {max_letter}) is correct

PROOF:
1. First, I'll establish the key principles needed
2. For each option:
   - Assume the option is true
   - Determine if this leads to a valid result or a contradiction
   - Note any special cases or conditions

CONCLUSION:
Therefore, option [letter A through {max_letter}] is proven to be correct.
"""
