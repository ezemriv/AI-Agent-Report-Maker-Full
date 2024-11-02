# 📈 AI Agent Report Maker - Hackathon Project (2nd Place 🏆)

## 📜 Project Overview

This project focuses on processing and analyzing financial transaction data for insights generation. The primary goals were to develop an intelligent AI agent capable of integrating multiple data sources, cleaning and transforming the data, and applying machine learning models for fraud detection and expense forecasting. The final solution combines these models into a report-generating AI agent to aid financial institutions in decision-making and customer insights.

### 🔑 Key Objectives

1. **Data Collection and API Integration**: Retrieve client and card data via API, and transaction data from a CSV file.
2. **Data Preprocessing and Transformation**: Clean and merge datasets, engineer features, and handle categorical encoding.
3. **Machine Learning Models**:
   - **Fraud Detection**: A model to classify transactions as fraudulent or non-fraudulent.
   - **Expense Forecasting**: A time-series model to forecast future monthly expenses for clients.
4. **AI Agent for Report Generation**: An agent built using LangChain and a small Llama-based language model to summarize and present insights based on user prompts.

---

## 📂 Project Structure

```text
├── data/                      
│   ├── raw/                   # Raw data files
│   └── processed/             # Processed datasets              
│
├── models/                    # Saved models for reuse and testing
│
├── notebooks/                 # Jupyter notebooks with experiments and tests
│   ├── 01_task1_presetup.ipynb
│   ├── 02_task1.ipynb
│   ├── 03_task2.ipynb
│   ├── 04_fraud_prepro.ipynb
│   ├── 05_fraud_model.ipynb
│   ├── 06_forecast_prepro.ipynb
│   ├── 07_Forecast_LGBM.ipynb
│   ├── 08_Forecast_AutoGluon.ipynb
│   └── 09_agent_experiments.ipynb
│
├── predictions/               # Predictions output files for evaluation
│   ├── predictions_1.json     # Task 1 predictions
│   ├── predictions_3.json     # Task 3 fraud detection predictions
│   └── predictions_4.json     # Task 4 expense forecast predictions
│
├── reports/                   # Generated reports and figures
│   ├── figures/               # Plots and figures
│   └── {summary_report}.pdf   # Final PDF summary report      
│
├── src/                       # Source code
│   ├── agent/                 # AI agent scripts
│   │   ├── __init__.py
│   │   ├── agent.py           # AI agent implementation
│   │   └── tools.py           # Supporting tools for agent functions
│   ├── data/                  # Data processing scripts
│   │   ├── __init__.py
│   │   ├── api_calls.py       # API interaction scripts (not implemented due to server limitations)
│   │   ├── data_functions.py  # Data processing functions
│   │   └── data_processing.py # Custom module for fraud detection preprocessing and train-test split
│   └── models/                # Model scripts and configurations
│
├── tests/                     # Unit tests for verification
│   ├── agent_test.py          # Unit tests for the AI agent
│   ├── statistics_test.py     # Unit tests for data functions
│   └── conftest.py            # Pytest configuration                    
│
├── Challenge.md               # Challenge description
├── README.md                  # Project documentation
├── LICENSE                    # License information
└── requirements.txt           # Project dependencies
```

---

## 📝 Data and API Usage

The following APIs and datasets are used in this project:

- **Client Data API**: Retrieves demographic and account-related client information. *(Note: API calls are not fully implemented due to server limitations.)*
- **Card Data API**: Fetches card-related details such as card type, limit, and activation date for each client.
- **Transaction Data**: Large CSV file containing all transaction records for analysis, efficiently handled using **Polars** library due to its size and the need for optimized processing.

---

## 🧑‍💻 Modeling and AI Agent Creation

This project involved extensive data processing, feature engineering, and modeling to build a solution for fraud detection, expense forecasting, and AI-based report generation. Below are the key components of the modeling and agent creation process.

### Fraud Detection Model

- **Data Processing**: The `data_processing.py` module was created to handle data cleaning, feature engineering, and train-test splitting. Downsampling was applied to balance the fraud detection classes.
- **Modeling**: LightGBM (LGBM) was chosen for its efficiency and performance. The model was trained using cross-validation, achieving:
  - **Training Balanced Accuracy**: `[0.999, 0.999, 0.999, 0.999, 1.000]`
  - **Validation Balanced Accuracy**: `[0.947, 0.950, 0.952, 0.943, 0.944]`
  - **Mean Validation Score**: `0.947`

### Expense Forecasting Model

For expense forecasting, transaction data was combined with client demographics to enhance feature richness. Key modeling steps included:

- **Feature Engineering**: Rolling features and lagged values were created using **MLForecast** to capture temporal spending patterns. Static client features like `current_age` and `monthly_income` were also used. More advanced features were explored but not implemented due to time constraints.
- **Hyperparameter Tuning**: Conducted through `LGBM-Forecast_Hyperparameter_tunning.py`, optimizing LightGBM model parameters for improved performance.
- **Alternative Model**: **AutoGluon’s TimeSeriesPredictor** was tested in a separate Conda environment and yielded similar results.
- **Model Results**: Single-fold validation was employed due to inconsistent date availability across clients, with the following metrics:
  - **Validation R2**: `0.772`
  - **Validation RMSE**: `314.361`

### AI Agent for Report Generation

The AI agent, implemented in `agent.py`, leverages LangChain and a locally hosted Llama 3.2:1b model. The agent processes natural language prompts to generate client-specific financial reports, using extracted date ranges to focus the analysis.

**Key Features**:
- **Date Extraction**: Using prompt engineering, the agent extracts start and end dates from text prompts, defaulting to the first and last days of a month if dates are partially specified.
- **Data Processing**: The agent calls functions from `data_functions.py` to generate summaries on earnings, expenses, and cash flow over the specified date range.
- **Report Generation**: The processed data is compiled into a PDF report that includes insights such as financial summaries and graphics, providing actionable information for financial decision-making.

This combination of models and an AI-driven reporting agent allows for dynamic, data-driven insights generation, enhancing the utility of transaction and client data for financial analysis.

---

## 🧪 Testing and Evaluation

- **Unit Tests**: Tests are available in the `tests/` folder. Key tests include:
  - `agent_test.py`: Tests for the AI agent's functionality.
  - `statistics_test.py`: Tests for data processing and query functions.

- **Evaluation Metrics**:
  - **Fraud Detection**: Balanced Accuracy Score.
  - **Expense Forecasting**: R2 Score for prediction accuracy.

Run the tests using the following command:
```bash
python -m pytest tests/test_module.py
```

---

