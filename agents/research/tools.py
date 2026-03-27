from langchain.tools import tool


@tool
async def search_web(query: str) -> str:
    """Search the web for information on a given query.

    Args:
        query (str): The search query string.

    Returns:
        str: Search results as a string.
    """

    return f"Placeholder search results for: {query}"
