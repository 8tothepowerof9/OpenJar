from langchain.tools import tool


@tool
def book_meeting(query: str):
    """
    Book a meeting based on the query

    Args:
        query (str): The details of the meeting to be booked
    """
    return "Meeting booked successfully"
