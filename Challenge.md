#  📈 The Game is HackathON - AI Agent Report Maker 📊

Category   ➡️   Data Science

Subcategory   ➡️   Data Scientist

Difficulty   ➡️   Hard

---

## 🌐 Background

In today's digital economy, banks and financial institutions process massive amounts of transactional data daily, which includes purchases, payments, transfers, and other activities. This data is not only pivotal for monitoring individual financial behavior but also critical for detecting fraudulent transactions, understanding customer spending patterns, and optimizing personalized financial services. The goal of this challenge is to leverage transaction data from various sources, such as APIs and pre-existing datasets—merge and process it for meaningful analysis, and train machine learning models that can predict outcomes like fraud detection, credit scoring, or customer segmentation. In the final step, an AI agent will be developed to combine these analytical and predictive processes, offering an automated and intelligent solution for decision-making and insights generation.


### 🗂️ Dataset

You will have to work with diverse datasets containing customer and transaction information, originating from various sources. The primary data include:

- `client data`: This contains demographic and account-related information about clients, accessible through an API.
  -  API Endpoint: `https://faas-lon1-917a94a7.doserverless.co/api/v1/web/fn-a1f52b59-3551-477f-b8f3-de612fbf2769/default/clients-data`
  -  Parameters required: `client_id`
  -  Result: A dictionary with client details as values.

``` 
{
  "client_id":  "1556",
  "values":
    {
      "address":" 594 Mountain View Street",
      "birth_month": "7",
      "birth_year": "1989", 
      ...
    }
}
```

- `card data`: This dataset includes information about a customer's credit and debit cards, such as card limits, types, and activation dates, and is also accessible via an API.
  -  API Endpoint: `https://faas-lon1-917a94a7.doserverless.co/api/v1/web/fn-a1f52b59-3551-477f-b8f3-de612fbf2769/default/cards-data`
  -  Parameters required: `client_id`
  -  Result:  A dictionary with all the cards of the user, where the keys are the `card_id` and the values contain card details.

```
{
  "client_id": "1556",
  "values": {
    "412": 
      {
        "acct_open_date": "01/2020",
        "card_brand" : "Amex",
        "card_number": "384780809062464",
        ...
      },
    "3623": 
      {
        "acct_open_date": "03/2014",
        "card_brand" : "Visa",
        "card_number": "793288564792063",
        ...
      }
    }
}
```
- `transactions dataset`: A comprehensive dataset that tracks credit card transactions for the 2010's decade. This dataset includes transaction amounts, merchant details, timestamps, and more, providing a foundation for analyzing spending behaviors and detecting fraudulent activities.
  - [Download transaction data](https://cdn.nuwe.io/challenges-ds-datasets/hackathon-caixabank-data-24/transactions_data.csv)
  
Additionally, the merchant category codes are provided in the `mcc_codes.json` file.


### 📊 Data Processing

To effectively analyze and model the data, you will need to implement the following steps:

- **API Integration:** First, you'll need to implement API calls to fetch all necessary data from the given endpoints, including customer information and card data.

- **Data Merging and Cleaning:** After retrieving the data, merge the customer, card, and transaction datasets using relevant keys (such as `client_id` or `card_id`). Ensure that the data is cleaned and preprocessed to handle any missing or inconsistent values.

- **Data Transformation:** Apply further transformations, such as creating new features, aggregating data (e.g., monthly expenses), and encoding categorical variables. These preprocessing steps  are crucial to preparing the data for machine learning models.

### 🤖 Model

For the modeling phase, you are tasked with developing two different machine learning models, each serving a distinct purpose:

- **Fraud Detection Model:** This model will classify transactions as either fraudulent or non-fraudulent. The goal is to identify suspicious transactions in real-time by analyzing patterns such as transaction frequency, amounts, and locations. Supervised learning techniques, such as logistic regression, decision trees, or ensemble methods, can be employed. Since fraud is often rare, handling class imbalance effectively (e.g., through resampling or custom loss functions) will be key to building a robust model.

- **Expenses Forecast Model:** This model will forecast future customer expenses based on their historical spending patterns. Time series forecasting models can be used to predict future expenditures, helping banks provide personalized budget management tools. The model will learn from past transaction data and predict the monthly expenses over the next three months from each client's last transaction.


## 📂 Repository Structure

The repository structure is provided and must be adhered to strictly:

```

├── data/                      
│   ├── raw/
│   │   └── mcc_codes.json    
│   └── processed/              
│
|
│── predictions/   
│   ├── predictions_1.json 
│   ├── predictions_3.json 
│   └── predictions_4.json
|
├── reports/   
│   ├── figures/ 
│   └── {summary_report}.pdf      
│
├── src/                       
│   ├── data/                   
│   │   ├── api_calls.py       
│   │   ├── data_functions.py       
│   │   └── data_questions.py     
│   │
│   ├── models/                 
│   │   ├── train_model.py      
│   │   └── predict_model.py   
│   │
│   └── agent/                  
│       ├── agent.py            
│       └── tools.py            
│
├── tests/                      
│   ├── agent_test.py       
│   ├── statistics_test.py 
│   └── conftest.py                    
│
├── README.md  
└── requirements.txt     

```

The `/predictions` folder should contain the tasks outputs for Task 1 , 3 and 4.


## 🎯 Tasks:

The challenge will be divided in three subparts: Data Statistics, Data Prediction and IA Agent creation. For each part you have a set of tasks. You have examples of the expected submission for tasks 1, 3 and 4 in the `/predictions` folder.

- **Part 1:**
  - **Task 1:** Submit the answers to the following queries: 
    - **query_1:** The `card_id` with the latest expiry date and the lowest credit limit amount. 
    - **query_2:** The `client_id` that will retire within a year that has the lowest credit score and highest debt. 
    - **query_3:** The `transaction_id` of an Online purchase on a 31st of December with the highest absolute amount (either earnings or expenses). 
    - **query_4:** Which client over the age of 40 made the most transactions with a Visa card in February 2016? Please return the `client_id`, the `card_id` involved, and the total number of transactions. 

  - **Task 2:** Implement the functions stated on the file `src/data/data_functions.py`. The input dataset will be the `transactions.csv` file in its given raw format with the dates parsed into datetime format. The expected functionality of each function is described within the `src/data/data_functions.py` file:
    - **Function:** earnings_and_expenses(data,client_id,start_date,end_date)
    - **Function:** expenses_summary(data,client_id,start_date,end_date)
    - **Function:** cash_flow_summary(data,client_id,start_date,end_date)

- **Part 2:**
  - **Task 3:** Create a fraud detection model capable of detecting fraudulent transactions. The dataset `transactions_data.csv` will be split for training and prediction. Predictions should only be made for the transaction IDs listed in `predictions/predictions_3.json`. The fraud labels for training can be found at:
    - [Download transaction fraud labels](https://cdn.nuwe.io/challenges-ds-datasets/hackathon-caixabank-data-24/train_fraud_labels.json)
  - **Task 4:** Develop a forecast model to predict monthly **expenses** (negative amounts) for each client in `predictions/predictions_4.json`. Forecast the monthly expenses for the next three months based on each client's most recent transaction date. The output format can be found in the `predictions/predictions_4.json` file.

- **Part 3:**
  - **Task 5:** In this task, your objective is to implement a simple AI Agent with `Langchain` and the Language Model [llama3.2:1b](https://ollama.com/library/llama3.2:1b), hosted locally via Ollama. The Agent should be able to process a natural language input and generate a summary report in PDF format by using the functions developed in Task 2. The input of the functions `run_agent()` will include the `client_id` , `df:pd.DataFrame` and the text prompt. The Agent must extract the relevant start and end dates from the prompt and be able to invoke the three functions to generate the report. You can find this function and its definition in `src/agent/agent.py`.
    - Take into account the non-deterministic nature of language models, as this model is a Small Language Model, use the adequate prompt engineering techniques to ensure the optimal results.

### 💫 Guides
Ensure that the project environment is configured to use Python 3.10.12. The libraries listed in `requirements.txt` must not be altered, although you are permitted to add additional libraries as needed.

## 📤 Submission

For this challenge, submissions will be made in different formats depending on the task. Below are the details for each task’s submission process:

- **Task 1, Task 3, and Task 4:**
You are required to submit your results in JSON format. The expected format for each task’s output can be found in the `/predictions` folder.

- **Task 2 and Task 5:**
Submissions will be tested via **unit** tests. You need to ensure that your functions produce the correct outputs for a range of input cases. Sample test cases are provided in the `/tests` folder to help you validate your implementation. Note that the final evaluation will be done using a different set of tests beyond the provided examples. Please refer to the Python files for the function descriptions and parameter details.


## 📊 Evaluation

The evaluation of Task 1 , Task 3 and Task 4 will be done via comparison of the results in the json file. You can find the expected format in the `/predictions` folder.
- The predictions for Task 3 will be evaluated using the [Balanced Accuracy score](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.balanced_accuracy_score.html) .
- The predictions for Task 4 will be evaluated using the [R2 Score](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.r2_score.html).

The evaluation for Task 2 and Task 5 will be done via Pytest. Your functions should have the expected input and outputs. You have available some sample tests in the `/test` folder. Take into account that these are just example tests, the evaluation will be done with a larger sample of tests.
The grading system will be the following:
- Task 1: 200 / 1000 points
- Task 2: 200 / 1000 points
- Task 3: 200 / 1000 points
- Task 4: 200 / 1000 points
- Task 5: 200 / 1000 points


**⚠️ Please note:**  
All submissions might undergo a manual code review process to ensure that the work has been conducted honestly and adheres to the highest standards of academic integrity. Any form of dishonesty or misconduct will be addressed seriously, and may lead to disqualification from the challenge.

Ensure that all data manipulation and model training strictly use the libraries mentioned in `requirements.txt`.


## ❓ FAQs

### Q1: Where do I find the client ids? 
A1: The clients you need information from are the ones found in the transactions dataset.

### Q2: What is considered a monthly expense? 
A2: The sum of the negative transaction's amounts for each client and month across all their credit cards.

### Q3: How do I run the sample tests?
A3: In the command line, write the command `python -m pytest tests/statistics_test.py` for Task 2 tests and `python -m pytest tests/agent_test.py` for Task 5. Ensure the `transactions_data.csv` file is located in the `/data/raw/` directory.


### Q4: How do I submit my solution? 
A4: Submit your solution via Git. Once your code and predictions are ready, push your repository. Your submission will be graded automatically within a few minutes. Make sure to write meaningful commit messages.