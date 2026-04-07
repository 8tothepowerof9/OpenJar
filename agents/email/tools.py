from langchain.tools import tool


@tool
def get_email(query: str):
    """
    Search for relevent mail from query

    Args:
        query (str): The query to search for in the email
    """
    return "Subject: NVIDIA Internship. Body: We are pleased to inform you that you have been selected for an interview for the NVIDIA internship position. We have scheduled a meeting in 10/8/2026"
