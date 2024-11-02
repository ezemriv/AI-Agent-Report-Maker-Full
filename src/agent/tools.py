# tools.py
import re
from fpdf import FPDF

def preprocess_input(input_text):
    """
    Removes unwanted phrases from the input text to help the AI model generate dates.

    Parameters
    ----------
    input_text : str
        The input text to preprocess.

    Returns
    -------
    str
        The preprocessed input text with unwanted phrases removed.
    """
    
    unwanted_phrases = ["create a pdf report", "generate a PDF", "make a PDF", "pdf"]

    for phrase in unwanted_phrases:
        # Create a regex pattern that ignores case
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        # Replace the phrase with an empty string
        input_text = pattern.sub("", input_text)
    # Remove extra whitespace
    return input_text.strip()


def extract_dates(chain, input):
    """
    Extracts the start and end dates from the given input text.

    Parameters
    ----------
    chain : langchain_ollama.ChatOllama
        The Langchain AI model to use for text processing.
    input : str
        The input text to extract the dates from.

    Returns
    -------
    tuple
        A tuple of (start_date, end_date) as strings in the format "YYYY-MM-DD".
        If the input is invalid or the dates are not found, returns (None, None).
    """

    cleaned_input = preprocess_input(input)
    
    if not cleaned_input:
        return None, None  # Return None for both dates if input is invalid

    response = chain.invoke({"input_prompt": cleaned_input})
    
    # Use regex to find dates after START_DATE and END_DATE labels
    start_date_match = re.search(r"START_DATE:\s*(\d{4}-\d{2}-\d{2})", response.content)
    end_date_match = re.search(r"END_DATE:\s*(\d{4}-\d{2}-\d{2})", response.content)
    
    if start_date_match and end_date_match:
        start_date = start_date_match.group(1)
        end_date = end_date_match.group(1)
        return start_date, end_date  # Return as separate values
    
    return None, None  # Return None if dates are not found

class CustomPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Client Report", align="C", ln=True)
        self.ln(10)

    def add_table(self, data_frame, title):
        # Table title
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)
        
        # Column headers
        self.set_font("Arial", "B", 10)
        col_width = self.epw / len(data_frame.columns)  # Column width is evenly distributed
        for col_name in data_frame.columns:
            self.cell(col_width, 10, col_name, border=1, align="C")
        self.ln()
        
        # Table rows
        self.set_font("Arial", "", 8)
        for _, row in data_frame.iterrows():
            for item in row:
                self.cell(col_width, 10, str(item), border=1, align="C")
            self.ln()
        self.ln(10)  # Space after table

if __name__ == "__main__":
    ...
