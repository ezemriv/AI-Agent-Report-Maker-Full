import pandas as pd
import re
from datetime import datetime
from fpdf import FPDF

# Import the functions from Task 2
from src.data.data_functions import (
    earnings_and_expenses,
    expenses_summary,
    cash_flow_summary,
)

import polars as pl
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from .tools import extract_dates, CustomPDF

from fpdf import FPDF, XPos, YPos
import os

def run_agent(df: pd.DataFrame, client_id: int, input: str) -> dict:

    """
    Returns
    -------
    variables_dict : dict
        Dictionary containing the date range, client ID, report flag, dataframes, and PDF path.
    """
    # Set up the model and prompt for date extraction
    model = ChatOllama(model="llama3.2:1b", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful assistant that extracts start and end dates from the given text."
                    "For each date found, output it in the format START_DATE and END_DATE."
                    "Dates should be formatted as 'YYYY-MM-DD'."
                    "If no day is provided in each date, assume the day is 01."
                    "If there is reference to only one month, assume the start day is 01 and the end day is the last day of the month."
                    "Return only 2 dates. Do not include any additional text."
                ),
            ),
            ("human", "{input_prompt}"),
        ]
    )

    # Create the chain and extract dates
    chain = prompt | model
    START, END = extract_dates(chain, input)
    print("Extracted Dates:", START, END)

    # Process data for reporting
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    earnings_and_expenses_df = earnings_and_expenses(df, client_id, START, END)
    expenses_df = expenses_summary(df, client_id, START, END)
    cash_flow_df = cash_flow_summary(df, client_id, START, END)
    
    # Store in variables_dict
    variables_dict = {
        "start_date": START,
        "end_date": END,
        "client_id": client_id,
        "create_report": True, # Default to True
    }

    # Check if all DataFrames are empty and set create_report to False if so
    if earnings_and_expenses_df.empty and expenses_df.empty and cash_flow_df.empty:
        variables_dict["create_report"] = False
        print("No data available for the specified client or date range. Skipping report generation.")
        print(variables_dict)
        return variables_dict  # Return early without generating PDF

    # Generate PDF
    pdf = CustomPDF()
    pdf.add_page()

    # Paths to the plot images saved by your functions
    earnings_expenses_plot_path = "reports/figures/earnings_and_expenses.png"
    expenses_summary_plot_path = "reports/figures/expenses_summary.png"

    # Assuming `earnings_and_expenses_df`, `expenses_df`, and `cash_flow_df` are your DataFrames
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Report for Client {client_id} | Date Range: {START} to {END}", ln=True, align="C")
    pdf.ln(10)  # Adds some space after the title

    pdf.add_table(earnings_and_expenses_df, "Earnings and Expenses Summary")
    # Add Earnings and Expenses Plot
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Earnings and Expenses Plot", ln=True)
    pdf.image(earnings_expenses_plot_path, x=10, w=180)  # Adjust 'w' as needed to fit page width
    pdf.ln(10)  # Space after the image

    pdf.add_table(expenses_df, "Expenses Summary by Merchant Category")
    # Add Expenses Summary Plot
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Expenses Summary Plot", ln=True)
    pdf.image(expenses_summary_plot_path, x=10, w=180)
    pdf.ln(10)

    pdf.add_table(cash_flow_df, "Cash Flow Summary")

    # Save PDF
    pdf_output_folder = "reports/"
    os.makedirs(pdf_output_folder, exist_ok=True)
    pdf_filename = os.path.join(pdf_output_folder, f"client_{client_id}_report.pdf")
    pdf.output(pdf_filename)
    print(f"PDF report saved at {pdf_filename}")

    return variables_dict


if __name__ == "__main__":
    ...
