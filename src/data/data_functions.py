# src/data/data_functions.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Get the directory of the current script
script_dir = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(script_dir, '..', '..'))

def clean_transactions_df(df):
    # Create an explicit copy to avoid chained indexing
    relevant_columns = ['client_id', 'date', 'amount', 'mcc']
    df_clean = df[relevant_columns].copy()
    df_clean['date'] = pd.to_datetime(df['date'])
    # Now modify the copy
    df_clean['amount_clean'] = pd.to_numeric(df['amount'].str.replace(r'[\$,]', '', regex=True))
    
    return df_clean

def save_earnings_expenses_plot(df: pd.DataFrame):
    """
    Generate a bar plot for earnings and expenses using pastel blue shades.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the summary data with columns for earnings and expenses.
    """
    # Define two shades of pastel blue
    pastel_blues = ['#ACC6DE', '#4A6A94']  # Light pastel blue and slightly darker pastel blue
    
    # Create the Bar Plot with improved styling
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = df.plot(kind='bar', ax=ax, color=pastel_blues)
    
    # Enhance the plot styling
    plt.title('Earnings and Expenses', fontsize=16, pad=20, fontweight='bold')
    plt.ylabel('Amount', fontsize=12)
    plt.xticks(ticks=[0], labels=["Total"], rotation=0)
    
    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add value labels on top of each individual bar
    for i, column in enumerate(df.columns):
        value = df[column].values[0]
        ax.text(0 + (i-0.5)*0.25, value, f'${value:,.0f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Customize the frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Define the path and ensure the directory exists
    figures_dir = os.path.join(ROOT, 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, 'earnings_and_expenses.png')
    
    # Save the plot with higher DPI for better quality
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Earnings and expenses plot saved at {plot_path}")

def save_expenses_summary_plot(df: pd.DataFrame):
    """
    Generate a bar plot for expenses by merchant category using a single pastel color.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the summary data with columns 'Expenses Type' and 'Total Amount'.
    """
    # Define single pastel color for all bars
    pastel_color = '#4A6A94'  # Light pastel blue
    
    # Create the Bar Plot with improved styling
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = df.plot.bar(x='Expenses Type', y='Total Amount', ax=ax, 
                      color=pastel_color, 
                      width=0.7)
    
    # Enhance the plot styling
    plt.title('Expenses by Merchant Category', fontsize=16, pad=20, fontweight='bold')
    plt.ylabel('Total Amount', fontsize=12)
    plt.xlabel('', fontsize=12)  # Remove x-label as it's self-explanatory
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add value labels on top of bars
    for i, v in enumerate(df['Total Amount']):
        ax.text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Customize the frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Define the path and ensure the directory exists
    figures_dir = os.path.join(ROOT, 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, 'expenses_summary.png')
    
    # Save the plot with higher DPI for better quality
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Expenses summary plot saved at {plot_path}")

def earnings_and_expenses(
    df: pd.DataFrame, client_id: int, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    For the period defined in between start_date and end_date (both included), get the client data available and return
    a pandas DataFrame with the Earnings and Expenses total amount for the period range and user given.The expected columns are:
        - Earnings
        - Expenses
    The DataFrame should have the columns in this order ['Earnings','Expenses']. Round the amounts to 2 decimals.

    Create a Bar Plot with the Earnings and Expenses absolute values and save it as "reports/figures/earnings_and_expenses.png" .

    Parameters
    ----------
    df : pandas DataFrame
       DataFrame of the data to be used for the agent.
    client_id : int
        Id of the client.
    start_date : str
        Start date for the date period. In the format "YYYY-MM-DD".
    end_date : str
        End date for the date period. In the format "YYYY-MM-DD".


    Returns
    -------
    Pandas Dataframe with the earnings and expenses rounded to 2 decimals.

    """
    
    df = clean_transactions_df(df)

    # Filter data for the client and date range
    mask = (
        (df['client_id'] == client_id) &
        (df['date'] >= start_date) &
        (df['date'] <= end_date)
    )
    client_data = df.loc[mask]

    # Return an empty DataFrame if no data is available for the client in the date range
    if client_data.empty:
        print("No data available for the specified client ID and date range.")
        # Prepare the DataFrame
        zero_result_df = pd.DataFrame(columns=['Earnings', 'Expenses'])
        # save_earnings_expenses_plot(zero_result_df)
        return zero_result_df

    # Calculate Earnings and Expenses
    earnings = client_data[client_data['amount_clean'] > 0]['amount_clean'].sum()
    expenses = client_data[client_data['amount_clean'] < 0]['amount_clean'].sum()

    # Prepare the DataFrame
    data = {
        'Earnings': [round(earnings, 2)],
        'Expenses': [round(expenses, 2)]
    }
    result_df = pd.DataFrame(data)
    save_earnings_expenses_plot(result_df)

    return result_df

def expenses_summary(
    df: pd.DataFrame, client_id: int, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    For the period defined in between start_date and end_date (both included), get the client data available and return
    a Pandas Data Frame with the Expenses by merchant category. The expected columns are:
        - Expenses Type --> (merchant category names)
        - Total Amount
        - Average
        - Max
        - Min
        - Num. Transactions
    The DataFrame should be sorted alphabeticaly by Expenses Type and values have to be rounded to 2 decimals. Return the dataframe with the columns in the given order.
    The merchant category names can be found in data/raw/mcc_codes.json .

    Create a Bar Plot with the data in absolute values and save it as "reports/figures/expenses_summary.png" .

    Parameters
    ----------
    df : pandas DataFrame
       DataFrame  of the data to be used for the agent.
    client_id : int
        Id of the client.
    start_date : str
        Start date for the date period. In the format "YYYY-MM-DD".
    end_date : str
        End date for the date period. In the format "YYYY-MM-DD".

    Returns
    -------
    Pandas Dataframe with the Expenses by merchant category.

    """

    # Load merchant category codes
    mcc_codes_path = os.path.join(ROOT, 'data', 'raw', 'mcc_codes.json')
    with open(mcc_codes_path, 'r') as f:
        mcc_codes = json.load(f)

    df = clean_transactions_df(df)

    # Filter data for the client and date range
    mask = (
        (df['client_id'] == client_id) &
        (df['date'] >= start_date) &
        (df['date'] <= end_date) &
        (df['amount_clean'] < 0)  # Only expenses
    )
    client_data = df.loc[mask]

    # Return an empty DataFrame if no data is available for the client in the date range
    if client_data.empty:
        print("No data available for the specified client ID and date range.")
        zero_result_df = pd.DataFrame(columns=['Expenses Type', 'Total Amount', 'Average', 'Max', 'Min', 'Num. Transactions'])
        
        # save_expenses_summary_plot(zero_result_df)
        return zero_result_df

    client_data = client_data.copy()
    client_data.loc[:, 'Expenses Type'] = client_data['mcc'].astype(str).map(mcc_codes)
    client_data.loc[:, 'Expenses Type'] = client_data['Expenses Type'].fillna('Unknown')
    client_data.loc[:, 'amount_clean'] = client_data['amount_clean'].abs()

    summary = client_data.groupby('Expenses Type')['amount_clean'].agg([
        ('Total Amount', lambda x: round(x.sum(), 2)),
        ('Average', lambda x: round(x.mean(), 2)),
        ('Max', lambda x: round(x.min(), 2)),  # min because amounts are negative
        ('Min', lambda x: round(x.max(), 2)),  # max because amounts are negative
        ('Num. Transactions', 'count')
    ]).reset_index()

    # Sort alphabetically by 'Expenses Type'
    summary.sort_values('Expenses Type', inplace=True)
    save_expenses_summary_plot(summary)

    return summary

def cash_flow_summary(
    df: pd.DataFrame, client_id: int, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    For the period defined by start_date and end_date (both inclusive), retrieve the available client data and return a Pandas DataFrame containing cash flow information.

    If the period exceeds 60 days, group the data by month, using the end of each month for the date. If the period is 60 days or shorter, group the data by week.

        The expected columns are:
            - Date --> the date for the period. YYYY-MM if period larger than 60 days, YYYY-MM-DD otherwise.
            - Inflows --> the sum of the earnings (positive amounts)
            - Outflows --> the sum of the expenses (absolute values of the negative amounts)
            - Net Cash Flow --> Inflows - Outflows
            - % Savings --> Percentage of Net Cash Flow / Inflows

        The DataFrame should be sorted by ascending date and values rounded to 2 decimals. The columns should be in the given order.

        Parameters
        ----------
        df : pandas DataFrame
           DataFrame  of the data to be used for the agent.
        client_id : int
            Id of the client.
        start_date : str
            Start date for the date period. In the format "YYYY-MM-DD".
        end_date : str
            End date for the date period. In the format "YYYY-MM-DD".


        Returns
        -------
        Pandas Dataframe with the cash flow summary.

    """

    # Calculate the period length
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    period_length = (end - start).days

    df = clean_transactions_df(df)

    # Filter data for the client and date range
    mask = (
        (df['client_id'] == client_id) &
        (df['date'] >= start) &
        (df['date'] <= end)
    )
    client_data = df.loc[mask]

    # Return an empty DataFrame if no data is available for the client in the date range
    if client_data.empty:
        print("No data available for the specified client ID and date range.")
        # Prepare a simple DataFrame with zero values
        zero_result_df = pd.DataFrame(columns=['Date', 'Inflows', 'Outflows', 'Net Cash Flow', '% Savings'])
        return zero_result_df

    # Determine grouping frequency
    if period_length > 60:
        freq = 'ME'
        date_format = '%Y-%m'
    else:
        freq = 'W'
        date_format = '%Y-%m-%d'

    # Group and aggregate data
    client_data.set_index('date', inplace=True)
    grouped = client_data.groupby(pd.Grouper(freq=freq))

    summary = grouped['amount_clean'].agg([
        ('Inflows', lambda x: x[x > 0].sum()),
        ('Outflows', lambda x: abs(x[x < 0].sum()))
    ])

    summary['Net Cash Flow'] = summary['Inflows'] - summary['Outflows']
    summary['% Savings'] = (summary['Net Cash Flow'] / summary['Inflows']) * 100
    summary.reset_index(inplace=True)

    # Format the 'Date' column
    summary['Date'] = summary['date'].dt.strftime(date_format)
    summary.drop('date', axis=1, inplace=True)

    # Round values to 2 decimals
    summary = summary.round(2)

    # Sort by Date
    summary.sort_values('Date', inplace=True)

    return summary[['Date', 'Inflows', 'Outflows', 'Net Cash Flow', '% Savings']]


if __name__ == "__main__":
    ...
