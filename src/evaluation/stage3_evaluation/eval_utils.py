import json
import re
from typing import Any

from pydantic import BaseModel


def parse_language_block(output: str, language: str | list[str]) -> str:
    """
    Parse content from a language code block with the specified language.

    Args:
        output: The string output containing language blocks
        language: The language identifier to look for (e.g., 'python', 'json')
                 or a list of language identifiers to try in order

    Returns:
        The content inside the language block, or the original text if not found
    """
    # Convert single language to list for uniform processing
    languages = [language] if isinstance(language, str) else language

    # Try each language in the list
    for lang in languages:
        # Look for content between triple backticks with the specified language
        pattern = r'```' + lang + r'\s*(.*?)\s*```'
        lang_match = re.search(pattern, output, re.DOTALL)

        if lang_match:
            return lang_match.group(1)

    # If no match found for any language, return the original output
    return output


def parse_model_output_json(output: str, parse_into: type[BaseModel]) -> Any:
    """
    Parse the evaluation output in JSON format and return a Pydantic instance.

    Args:
        output: The JSON string output from the evaluation
        parse_into: The Pydantic model class to validate and parse the data into

    Returns:
        A Pydantic model instance of the specified type containing the parsed data
    """
    # Look for JSON content between triple backticks
    json_str = parse_language_block(output, "json")

    # If the returned string is the same as the original (no language block found),
    # try to find a JSON object directly
    if json_str == output:
        json_match = re.search(r'(\{.*\})', output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            raise ValueError("Could not find valid JSON in the output")

    try:
        json_data = json.loads(json_str)
        return parse_into.model_validate(json_data)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing evaluation output: {e}")