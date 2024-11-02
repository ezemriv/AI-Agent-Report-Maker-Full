from data.data_functions import (
    earnings_and_expenses,
    expenses_summary,
    cash_flow_summary,
)

import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path

plt.switch_backend("Agg")
sample_data = pd.read_csv("data/raw/transactions_data.csv", parse_dates=["date"])
image_folder = "reports/figures"


def test_earnings_and_expenses_1():
    df, user_id, start_date, end_date = (
        sample_data.copy(deep=True),
        126,
        "2013-01-01",
        "2020-01-31",
    )
    try:
        answer = earnings_and_expenses(df, user_id, start_date, end_date)
        if not isinstance(answer, pd.DataFrame):
            print("Returned object is not a dataframe.")
            answer = pd.DataFrame()
        expected_answer = pd.DataFrame(
            {"Earnings": [176279.33], "Expenses": [-18246.0]}
        )

        pd.testing.assert_frame_equal(answer, expected_answer)
        test_passed = True
    except Exception as e:
        print("Not the expected output.")
        test_passed = False
    assert test_passed


def test_earnings_and_expenses_2():
    # test png image is generated
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)
    os.mkdir(image_folder)
    df, user_id, start_date, end_date = (
        sample_data.copy(deep=True),
        126,
        "2013-01-01",
        "2020-01-31",
    )
    try:
        answer = earnings_and_expenses(df, user_id, start_date, end_date)
    except:
        answer = pd.DataFrame()
    condition = Path(f"{image_folder}/earnings_and_expenses.png").is_file()

    assert condition


def test_expenses_summary_1():
    df, user_id, start_date, end_date = (
        sample_data.copy(deep=True),
        126,
        "2013-01-01",
        "2020-01-31",
    )
    try:
        answer = expenses_summary(df, user_id, start_date, end_date)
        if not isinstance(answer, pd.DataFrame):
            print("Returned_object if not a DataFrame.")
            answer = pd.DataFrame()
        expected_answer = pd.DataFrame(
            {
                "Expenses Type": [
                    "Electroplating, Plating, Polishing Services",
                    "Gardening Supplies",
                    "Heat Treating Metal Services",
                    "Lighting, Fixtures, Electrical Supplies",
                    "Miscellaneous Food Stores",
                    "Miscellaneous Machinery and Parts Manufacturing",
                    "Non-Ferrous Metal Foundries",
                    "Railroad Passenger Transport",
                    "Service Stations",
                    "Ship Chandlers",
                ],
                "Total Amount": [
                    426.0,
                    446.0,
                    457.0,
                    477.0,
                    6871.0,
                    156.0,
                    179.0,
                    437.0,
                    8241.0,
                    556.0,
                ],
                "Average": [
                    426.0,
                    446.0,
                    457.0,
                    477.0,
                    75.51,
                    156.0,
                    179.0,
                    437.0,
                    73.58,
                    278.0,
                ],
                "Max": [
                    426.0,
                    446.0,
                    457.0,
                    477.0,
                    50.0,
                    156.0,
                    179.0,
                    437.0,
                    51.0,
                    225.0,
                ],
                "Min": [
                    426.0,
                    446.0,
                    457.0,
                    477.0,
                    99.0,
                    156.0,
                    179.0,
                    437.0,
                    100.0,
                    331.0,
                ],
                "Num. Transactions": [1, 1, 1, 1, 91, 1, 1, 1, 112, 2],
            }
        )

        pd.testing.assert_frame_equal(answer, expected_answer)

        test_passed = True
    except Exception as e:
        test_passed = False

    assert test_passed


def test_expenses_summary_2():
    # test png image is generated
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)
    os.mkdir(image_folder)
    df, user_id, start_date, end_date = (
        sample_data.copy(deep=True),
        32,
        "2011-05-01",
        "2016-11-30",
    )
    try:
        answer = expenses_summary(df, user_id, start_date, end_date)
    except:
        answer = pd.DataFrame()
    condition = Path(f"{image_folder}/expenses_summary.png").is_file()

    assert condition


def test_cash_flow_summary_1():
    df, user_id, start_date, end_date = (
        sample_data.copy(deep=True),
        126,
        "2013-01-01",
        "2013-02-28",
    )
    try:
        answer = cash_flow_summary(df, user_id, start_date, end_date)
        if not isinstance(answer, pd.DataFrame):
            print("Returned object is not a DataFrame")
            answer = pd.DataFrame()

        expected_answer = pd.DataFrame(
            {
                "Date": [
                    "2013-01-06",
                    "2013-01-13",
                    "2013-01-20",
                    "2013-01-27",
                    "2013-02-03",
                    "2013-02-10",
                    "2013-02-17",
                    "2013-02-24",
                    "2013-03-03",
                ],
                "Inflows": [
                    432.74,
                    518.67,
                    297.63,
                    450.91,
                    863.16,
                    288.75,
                    466.82,
                    1019.02,
                    138.35,
                ],
                "Outflows": [129.0, 0.0, 66.0, 0.0, 156.0, 0.0, 0.0, 71.0, 0.0],
                "Net Cash Flow": [
                    303.74,
                    518.67,
                    231.63,
                    450.91,
                    707.16,
                    288.75,
                    466.82,
                    948.02,
                    138.35,
                ],
                "% Savings": [
                    70.19,
                    100.0,
                    77.82,
                    100.0,
                    81.93,
                    100.0,
                    100.0,
                    93.03,
                    100.0,
                ],
            }
        )

        pd.testing.assert_frame_equal(answer, expected_answer)

        test_passed = True
    except Exception as e:
        test_passed = False

    assert test_passed


def test_cash_flow_summary_2():
    df, user_id, start_date, end_date = (
        sample_data.copy(deep=True),
        50,
        "2011-01-01",
        "2011-04-30",
    )
    try:
        answer = cash_flow_summary(df, user_id, start_date, end_date)
        if not isinstance(answer, pd.DataFrame):
            print("Returned object is not a DataFrame.")
            answer = pd.DataFrame()
        expected_answer = pd.DataFrame(
            {
                "Date": ["2011-01", "2011-02", "2011-03", "2011-04"],
                "Inflows": [2488.64, 2154.61, 2721.85, 2725.5],
                "Outflows": [183.0, 578.0, 628.0, 483.0],
                "Net Cash Flow": [2305.64, 1576.61, 2093.85, 2242.5],
                "% Savings": [92.65, 73.17, 76.93, 82.28],
            }
        )
        pd.testing.assert_frame_equal(answer, expected_answer)

        test_passed = True
    except Exception as e:
        test_passed = False

    assert test_passed
