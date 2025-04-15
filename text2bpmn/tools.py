
from langchain_core.tools import tool

FILE_NAME = "output.txt"

@tool
def save_to_file(content: str) -> str:
    """Save the content to a file."""
    with open(FILE_NAME, "w") as f:
        f.write(content)
    return f"Content saved to {FILE_NAME}"