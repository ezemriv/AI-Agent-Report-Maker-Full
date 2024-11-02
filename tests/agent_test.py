from agent.agent import run_agent
import os
import pandas as pd
import shutil

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
report_folder = os.path.join(parent_folder, "reports")
data_folder = os.path.join(parent_folder, "data")
sample_data = pd.read_csv(
    f"{data_folder}/raw/transactions_data.csv", parse_dates=["date"]
)


def delete_reports():
    if os.path.exists(report_folder):
        shutil.rmtree(report_folder)
    os.mkdir("reports")
    os.mkdir("reports/figures")


def check_all_values(expected_dict, submitted_dict):
    for k in expected_dict:
        if k not in submitted_dict or expected_dict[k] != submitted_dict[k]:
            return False
    return True


def test_agent_1():
    delete_reports()
    input = "Create a pdf report for the fourth month of 2017"
    client_id = 122
    output = {
        "start_date": "2017-04-01",
        "end_date": "2017-04-30",
        "client_id": client_id,
        "create_report": True,
    }
    try:
        submitted_output = run_agent(
            input=input,
            client_id=client_id,
            df=sample_data.copy(deep=True),
        )
    except Exception as e:
        print(e)
        submitted_output = {}

    test_passed = check_all_values(output, submitted_output)
    expected_length = 1 if output["create_report"] else 0
    try:
        pdf_files = [
            file for file in os.listdir(report_folder) if file.endswith(".pdf")
        ]
        condition_pdf = len(pdf_files) == expected_length
    except Exception as e:
        print(e)
        condition_pdf = False
    assert condition_pdf and test_passed


def test_agent_2():
    delete_reports()
    input = "Create a pdf report from 2018-01-01 to 2018-05-31"
    client_id = 1352
    output = {
        "start_date": "2018-01-01",
        "end_date": "2018-05-31",
        "client_id": client_id,
        "create_report": True,
    }
    try:
        submitted_output = run_agent(
            input=input,
            client_id=client_id,
            df=sample_data.copy(deep=True),
        )
    except Exception as e:
        print(e)
        submitted_output = {}

    test_passed = check_all_values(output, submitted_output)
    expected_length = 1 if output["create_report"] else 0
    try:
        pdf_files = [
            file for file in os.listdir(report_folder) if file.endswith(".pdf")
        ]
        condition_pdf = len(pdf_files) == expected_length
    except Exception as e:
        print(e)
        condition_pdf = False
    assert condition_pdf and test_passed


def test_agent_3():
    delete_reports()
    input = "Create a pdf report from 2018-01-01 to 2018-05-31"
    client_id = 7000
    output = {
        "start_date": "2018-01-01",
        "end_date": "2018-05-31",
        "client_id": client_id,
        "create_report": False,
    }
    try:
        submitted_output = run_agent(
            input=input,
            client_id=client_id,
            df=sample_data.copy(deep=True),
        )
    except Exception as e:
        print(e)
        submitted_output = {}

    test_passed = check_all_values(output, submitted_output)
    expected_length = 1 if output["create_report"] else 0
    try:
        pdf_files = [
            file for file in os.listdir(report_folder) if file.endswith(".pdf")
        ]
        condition_pdf = len(pdf_files) == expected_length
    except Exception as e:
        print(e)
        condition_pdf = False
    assert condition_pdf and test_passed
