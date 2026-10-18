# Module4_ITAI2372
# Financial News Sentiment & Insight Analyzer

## Project Overview

The Financial News Sentiment & Insight Analyzer is a Python-based application that analyzes finance-related news articles using sentiment analysis, sector classification, and AI-generated text summarization. It provides an interactive interface that allows users to select a news article and view its key information and analysis.

The project uses the HuffPost News Category Dataset, VADER sentiment analysis, a locally running DistilBART summarization model, and Gradio to create an interactive application without requiring a paid AI API.

## Features

- **Financial News Filtering:** Filters finance-related articles from the HuffPost News Category Dataset using keyword-based rules.
- **Sector Classification:** Categorizes articles into sectors such as Stocks & Investments, Banking & Credit, Business & Workplace, and Economy & Employment.
- **Sentiment Analysis:** Uses VADER to classify article sentiment as positive, neutral, or negative and calculate a compound sentiment score.
- **AI-Generated Summarization:** Uses the pretrained DistilBART model to generate concise summaries of selected articles.
- **Interactive User Interface:** Uses Gradio to let users select articles and view their headline, sector, sentiment, score, and AI-generated summary.

## Technologies Used

- Python
- Pandas
- VADER Sentiment Analysis
- Hugging Face Transformers
- PyTorch
- DistilBART (`sshleifer/distilbart-cnn-12-6`)
- Gradio
- Google Colab

## Dataset

This project uses the HuffPost News Category Dataset, specifically `News_Category_Dataset_v3.json`.

The dataset contains news headlines, short descriptions, publication information, and news categories. The project combines each headline with its short description and uses keyword-based filtering to identify finance-related articles.

The current dataset-processing workflow produced approximately 8,361 finance-related articles.

Because the original dataset covers a broad range of news topics, the filtered subset may contain articles that mention financial subjects without being exclusively about financial markets.

## Installation and Setup

### Option 1: Run in Google Colab

Google Colab is the recommended environment for this project.

1. Open the project notebook in Google Colab.
2. Upload `News_Category_Dataset_v3.json` or place it in the location expected by the notebook.
3. Install the required dependencies by running the following command in a notebook cell:

   ```python
   %pip install -q pandas vaderSentiment transformers torch gradio
   ```

4. Run the notebook cells in order to load the dataset, preprocess the text, classify sectors, perform sentiment analysis, and load the summarization model.
5. Run the cell containing the Gradio interface.
6. Open the URL displayed by Gradio to access the application.

The first model download may take several minutes. The summarization model runs locally and does not require a paid API key.

### Option 2: Run Locally

1. Download or clone this repository.
2. Install Python and create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment.

   **Windows PowerShell:**

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   **macOS/Linux:**

   ```bash
   source .venv/bin/activate
   ```

4. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install pandas vaderSentiment transformers torch gradio jupyter
   ```

5. Place `News_Category_Dataset_v3.json` in the location expected by the notebook.
6. Open the notebook using Jupyter or upload it to Google Colab.
7. Run the notebook cells in order.

**Note:** The project was developed and tested in Google Colab. Package compatibility may vary across Python versions. The commands above install the main dependencies, but the exact versions have not been pinned.

## How to Run the Application

The current application is implemented in a Jupyter/Google Colab notebook.

1. Load the dataset and create the finance-related subset.
2. Run the text preprocessing and sector classification steps.
3. Run VADER sentiment analysis.
4. Load the pretrained DistilBART tokenizer and model using Hugging Face Transformers.
5. Define the article analysis function.
6. Run the Gradio interface cell.
7. Open the Gradio URL displayed in the notebook.
8. Select a news article from the dropdown menu.
9. Click **Analyze Article**.

The application displays:

- Article headline
- Classified financial sector
- Sentiment label
- VADER compound sentiment score
- AI-generated summary

The Gradio sharing URL is temporary and is intended for testing and demonstration rather than permanent deployment.

## Testing the Application

### 1. Test Dataset Processing

Run the following code to verify that the finance-related dataset has been created:

```python
print("Number of finance-related articles:", len(finance_df))

print(
    finance_df[
        ["headline", "primary_sector", "sentiment", "compound"]
    ].head()
)
```

Expected result: The dataframe contains article headlines, assigned sectors, sentiment labels, and compound sentiment scores.

### 2. Test Sentiment Analysis

Run:

```python
print(finance_df["sentiment"].value_counts())

print(
    finance_df.groupby("primary_sector")["compound"]
    .mean()
    .sort_values()
)
```

Expected result: The output shows sentiment counts and the average compound score for each sector.

### 3. Test AI Summarization

Run:

```python
analyze_news(finance_df.index[10])
```

Expected result: The function displays the selected article's headline, sector, sentiment, sentiment score, and AI-generated summary.

### 4. Test the Interactive Interface

Verify the following:

- The Gradio interface opens successfully.
- The article dropdown displays available headlines.
- Clicking **Analyze Article** generates the analysis.
- The headline, sector, sentiment, sentiment score, and summary are displayed.
- Selecting a different article updates the results.

The first summarization request may take several seconds, particularly when running on a CPU.

## Results and Interpretation

The application combines rule-based sector classification, VADER sentiment analysis, and pretrained language-model summarization to provide a concise overview of selected news articles.

Sentiment labels describe the linguistic tone of the article rather than predicting market performance. For example, a negative sentiment score does not necessarily mean that a stock or financial market will decline.

Sector classification is based on keyword rules and may not always identify the correct sector. Generated summaries may also omit important context or contain inaccuracies, so users should consult the original article when necessary.

## Limitations

- The finance-related dataset is created through keyword filtering and may contain irrelevant articles or miss relevant ones.
- Sector classification relies on predefined keyword rules.
- VADER is a general-purpose sentiment analysis tool rather than a finance-specific sentiment model.
- DistilBART summarization may be slow on CPU.
- The application analyzes article text and does not retrieve live stock prices or predict financial-market movements.
- The current implementation is notebook-based and has not been packaged as a standalone production application.

## Future Improvements

- Add interactive charts showing sentiment distribution and sentiment by sector.
- Improve sector classification and evaluate its accuracy.
- Compare VADER with a finance-specific sentiment model.
- Add automated tests for data preprocessing, sector classification, sentiment analysis, and summarization.
- Record tested dependency versions in a `requirements.txt` file.
- Develop a standalone web application for easier deployment.

## Disclaimer

This project is intended for educational and research purposes only. Sentiment analysis and AI-generated summaries are not financial advice and should not be used as the sole basis for investment decisions.
