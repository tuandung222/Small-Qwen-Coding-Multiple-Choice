import json
import time
from typing import Any, Dict, Optional

import requests


class GradioAppTester:
    """Test class for interacting with the Gradio app server"""

    def __init__(self, base_url: str = "http://127.0.0.1:7860"):
        """Initialize the tester with the app's base URL"""
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make a POST request to the Gradio API endpoint"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {endpoint}: {e}")
            return None

    def test_model_response(
        self, question: str, choices: list, temperature: float = 0.1
    ) -> Optional[Dict[str, Any]]:
        """Test the model's response to a question"""
        # Format choices as newline-separated string
        choices_text = "\n".join(choices)

        # Prepare the request data
        data = {
            "data": [question, choices_text, temperature],
            "fn_index": 2,  # Index of the inference function in the Gradio interface
            "session_hash": "test_session",
        }

        # Make the request
        response = self._make_request("/api/predict", data)
        if not response:
            return None

        # Wait for the response to be ready
        while True:
            if response.get("status") == "COMPLETE":
                return response
            elif response.get("status") == "FAILED":
                print(f"Request failed: {response.get('error')}")
                return None

            time.sleep(1)  # Wait before polling again
            response = self._make_request("/api/predict", data)

    def test_example_selection(self, example_idx: int) -> Optional[Dict[str, Any]]:
        """Test selecting an example from the dropdown"""
        data = {
            "data": [
                f"Example {example_idx}: "
            ],  # The actual example text will be filled by the server
            "fn_index": 1,  # Index of the process_example function
            "session_hash": "test_session",
        }

        return self._make_request("/api/predict", data)

    def test_category_selection(self, category: str) -> Optional[Dict[str, Any]]:
        """Test selecting a category from the dropdown"""
        data = {
            "data": [category],
            "fn_index": 0,  # Index of the get_category_examples function
            "session_hash": "test_session",
        }

        return self._make_request("/api/predict", data)


def main():
    """Main function to run the tests"""
    tester = GradioAppTester()

    # Test 1: Basic model response
    print("\nTesting basic model response...")
    question = "What is the difference between '==' and '===' in JavaScript?"
    choices = [
        "'==' performs type coercion while '===' does not",
        "'===' performs type coercion while '==' does not",
        "There is no difference between them",
        "Both perform type coercion but in different ways",
    ]

    response = tester.test_model_response(question, choices)
    if response:
        print("Model response test successful!")
        print(f"Response: {json.dumps(response, indent=2)}")
    else:
        print("Model response test failed!")

    # Test 2: Example selection
    print("\nTesting example selection...")
    example_response = tester.test_example_selection(1)
    if example_response:
        print("Example selection test successful!")
        print(f"Response: {json.dumps(example_response, indent=2)}")
    else:
        print("Example selection test failed!")

    # Test 3: Category selection
    print("\nTesting category selection...")
    category_response = tester.test_category_selection("JavaScript")
    if category_response:
        print("Category selection test successful!")
        print(f"Response: {json.dumps(category_response, indent=2)}")
    else:
        print("Category selection test failed!")


if __name__ == "__main__":
    main()
