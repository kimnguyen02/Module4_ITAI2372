# Module4_ITAI2372
Financial News Sentiment & Insight Analyzer
An interactive Python application that analyzes finance-related news articles by classifying their topic sector, estimating the sentiment of their wording, and generating an AI summary. It uses a Gradio interface and a locally loaded DistilBART summarization model, so it does not require a paid LLM API or API key.
Features
- News filtering: Selects finance-related articles from the HuffPost News Category Dataset using keyword-based filtering.
- Sector classification: Assigns each selected article to a rule-based sector, such as Stocks & Investments, Banking & Credit, or Economy & Employment.
- Sentiment analysis: Uses VADER to label article sentiment as positive, neutral, or negative and provide a compound score.
- AI-generated summaries: Uses sshleifer/distilbart-cnn-12-6 through Hugging Face Transformers.
- Interactive UI: Uses Gradio to let users select an article and view its headline, sector, sentiment, score, and generated summary.
Technology stack
- Python
- pandas
- VADER Sentiment (vaderSentiment)
- Hugging Face Transformers
- PyTorch
- Gradio
- Google Colab (recommended environment)
Dataset
This project uses the HuffPost News Category Dataset, specifically News_Category_Dataset_v3.json. The notebook combines each article's headline and short_description into text and applies keyword-based filtering to create a finance-related subset.
The current notebook run produced approximately 8,361 articles. Because the original dataset covers many news categories and the finance subset is keyword-filtered, it may include articles that mention financial topics without being strictly financial-market news.
Place News_Category_Dataset_v3.json in the location expected by the notebook, or update the notebook's dataset path. Make sure you have the right to access and use the dataset under its applicable terms.
Requirements
- A Python environment compatible with the installed versions of PyTorch and Transformers.
- Internet access on the first run to download model files (and the dataset, if needed).
- Enough RAM and disk space to load the dataset and model. Summarization can run on CPU if no supported GPU is available, but inference may take several seconds per article.
The working notebook uses AutoTokenizer and AutoModelForSeq2SeqLM directly rather than the Transformers summarization pipeline, because that pipeline task was unavailable in the environment used during development.
Installation
Option 1: Google Colab (recommended)
1. Open the project notebook in Google Colab.
2. Upload or make News_Category_Dataset_v3.json available at the path used by the notebook.
3. Install the packages in a notebook cell:
   %pip install -q pandas vaderSentiment transformers torch gradio
4. Run the notebook cells in order to load and filter the dataset, create sector and sentiment columns, load the tokenizer and model, define the analyzer, and launch the Gradio interface.
5. The first model load downloads the tokenizer and model weights, so it can take a few minutes.
If your notebook already has a working environment, avoid reinstalling or downgrading Transformers unnecessarily.
Option 2: Run locally
1. Clone or download this repository.
2. Create and activate a virtual environment:
   python -m venv .venv
   Windows PowerShell:
   .venv\Scripts\Activate.ps1
   macOS/Linux:
   source .venv/bin/activate
3. Install dependencies:
   python -m pip install --upgrade pip
   python -m pip install pandas vaderSentiment transformers torch gradio jupyter
4. Place News_Category_Dataset_v3.json where the notebook expects it.
5. Open the notebook with Jupyter or upload it to Google Colab, then run the cells in order.
Reproducibility note: Exact dependency versions have not been pinned in this README. For a reproducible setup, record the versions from the environment where the notebook successfully runs and save them in requirements.txt.

How to run the application
The current implementation is notebook-based and expects finance_df, tokenizer, and model to have been created.
1. Run the data-loading and preprocessing cells.
2. Run the sector classification and VADER sentiment-analysis cells.
3. Load the summarization tokenizer and model.
4. Run the cell that defines the Gradio interface.
5. When demo.launch(share=True) runs, open the Gradio URL printed in the output.
6. Choose a news headline from the dropdown and click Analyze Article.
7. Review the headline, sector, sentiment label, VADER compound score, and AI-generated summary.
The Gradio share link is temporary; it is not a permanent deployment.
Testing the application
1. Check the filtered dataset
print("Number of finance-related articles:", len(finance_df))
print(finance_df[["headline", "primary_sector", "sentiment", "compound"]].head())
Confirm that the dataframe is populated and the expected columns exist.
2. Check sentiment results
print(finance_df["sentiment"].value_counts())
print(finance_df.groupby("primary_sector")["compound"].mean().sort_values())
Confirm that sentiment labels and compound scores are present. Sector counts may be uneven because the classifier uses keyword rules.
3. Check summarization
analyze_news(finance_df.index[10])
Confirm that the function displays a headline, sector, sentiment label and score, and generated summary. Inference may be slower on CPU.
4. Check the Gradio interface
- Confirm the interface opens from the URL printed by Gradio.
- Select a headline and click Analyze Article.
- Confirm all five outputs appear: headline, sector, sentiment, VADER score, and AI-generated summary.
- Select a second article and confirm the displayed results change.
Interpreting results
- VADER sentiment measures the linguistic tone of article text. It does not determine whether an asset price will rise or fall, and it is not investment advice.
- Sector labels are rule-based classifications and may be imperfect.
- Generated summaries may omit context or contain errors. Check the original article before relying on a summary.
- The dataset is general news filtered by keywords, so not every selected article is necessarily a market report.
Limitations
- Keyword filtering may include irrelevant articles and miss relevant ones that use different wording.
- Sector assignment depends on keyword rules.
- VADER is a general-purpose sentiment tool, not a finance-specific sentiment model.
- DistilBART can be slow on CPU.
- The project analyzes article text; it does not use live stock prices or predict market movements.
Suggested repository structure
financial-news-sentiment-analyzer/
├── README.md
├── News_Category_Dataset_v3.json   # keep local; do not commit unless licensing permits
└── financial_news_analyzer.ipynb   # replace with your actual notebook filename
Future improvements
- Add charts for sentiment distribution and sentiment by sector.
- Validate and improve sector classification.
- Compare VADER with a finance-specific sentiment model.
- Add automated tests for preprocessing, sector classification, and sentiment output.
- Record tested package versions in requirements.txt.
Disclaimer
This project is for educational and research purposes only. Sentiment labels and generated summaries are not financial advice and should not be used alone to make investment decisions.
