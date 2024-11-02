import os
import pandas as pd
import shutil
from src.agent.agent import run_agent

# Define folder paths
parent_folder = os.path.dirname(os.path.abspath(__file__))
report_folder = os.path.join(parent_folder, "reports")
data_folder = os.path.join(parent_folder, "data")

# Load data
sample_data = pd.read_csv(
    f"{data_folder}/raw/transactions_data.csv", parse_dates=["date"]
)

# Function to delete and recreate report folders
def delete_reports():
    if os.path.exists(report_folder):
        shutil.rmtree(report_folder)
    os.makedirs(os.path.join(report_folder, "figures"), exist_ok=True)

# Main function to run agent
def main():
    delete_reports()
    input_text = "Create a pdf report for the year 2016"
    client_id = 1556
    try:
        submitted_output = run_agent(
            input=input_text,
            client_id=client_id,
            df=sample_data.copy(deep=True),
        )
        print("Agent output:", submitted_output)
    except Exception as e:
        print("Error running agent:", e)

if __name__ == "__main__":
    main()
