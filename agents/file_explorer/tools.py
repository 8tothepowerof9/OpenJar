from pathlib import Path

from langchain.tools import tool


@tool
async def list_directory(path: str) -> str:
    """List the contents of a directory, showing files and subdirectories.

    Args:
        path (str): The absolute path to the directory to list.

    Returns:
        str: A formatted listing of directory contents with type indicators.
    """
    target = Path(path)
    if not target.exists():
        return f"Error: path does not exist: {path}"
    if not target.is_dir():
        return f"Error: path is not a directory: {path}"

    try:
        entries = sorted(
            target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())
        )
    except PermissionError:
        return f"Error: permission denied: {path}"

    if not entries:
        return f"{path} is empty."

    lines = []
    for entry in entries:
        prefix = "[DIR]  " if entry.is_dir() else "[FILE] "
        lines.append(f"{prefix}{entry.name}")
    return "\n".join(lines)


@tool
async def current_path() -> str:
    """Get the current working directory.

    Returns:
        str: The absolute path of the current working directory.
    """
    return str(Path.cwd().resolve())
