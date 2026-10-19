import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Reuse your existing processed dataset
app_df = finance_df.copy()

# Preserve the existing dataframe indices used in your notebook
app_df["app_id"] = app_df.index

sectors = ["All Sectors"] + sorted(
    app_df["primary_sector"].dropna().unique().tolist()
)

sentiments = [
    "All Sentiments",
    "Positive",
    "Neutral",
    "Negative"
]


def filter_articles(sector, sentiment, keyword):
    data = app_df.copy()

    if sector != "All Sectors":
        data = data[data["primary_sector"] == sector]

    if sentiment != "All Sentiments":
        data = data[data["sentiment"] == sentiment]

    keyword = (keyword or "").strip()

    if keyword:
        mask = (
            data["headline"].fillna("").str.contains(
                keyword, case=False, regex=False
            )
            | data["short_description"].fillna("").str.contains(
                keyword, case=False, regex=False
            )
        )
        data = data[mask]

    return data


def sentiment_chart(data):
    counts = data["sentiment"].value_counts().reindex(
        ["Positive", "Neutral", "Negative"],
        fill_value=0
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("News Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Articles")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def sector_chart(data):
    counts = data["primary_sector"].value_counts().head(12)

    fig, ax = plt.subplots(figsize=(8, 5))
    counts.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Financial News by Sector")
    ax.set_xlabel("Number of Articles")
    ax.set_ylabel("Primary Sector")
    fig.tight_layout()
    return fig


def sector_statistics(data):
    if data.empty:
        return pd.DataFrame()

    result = data.groupby("primary_sector").agg(
        Articles=("headline", "size"),
        Average_Sentiment=("compound", "mean"),
        Positive=("sentiment", lambda x: (x == "Positive").sum()),
        Neutral=("sentiment", lambda x: (x == "Neutral").sum()),
        Negative=("sentiment", lambda x: (x == "Negative").sum())
    ).reset_index()

    result["Average_Sentiment"] = (
        result["Average_Sentiment"].round(3)
    )

    return result.sort_values("Articles", ascending=False)


def update_dashboard(sector, sentiment, keyword):
    data = filter_articles(sector, sentiment, keyword)

    if data.empty:
        return (
            "## No matching articles. Try different filters.",
            None,
            None,
            pd.DataFrame(),
            pd.DataFrame(),
            gr.update(choices=[], value=None)
        )

    counts = data["sentiment"].value_counts()

    overview = (
        "## Financial News Dashboard\n\n"
        f"**Articles:** {len(data):,}  \n"
        f"**Positive:** {counts.get('Positive', 0):,}  \n"
        f"**Neutral:** {counts.get('Neutral', 0):,}  \n"
        f"**Negative:** {counts.get('Negative', 0):,}  \n"
        f"**Average VADER score:** {data['compound'].mean():.3f}"
    )

    table = data[
        ["app_id", "headline", "primary_sector", "sentiment", "compound"]
    ].copy()

    table["compound"] = table["compound"].round(3)
    table = table.rename(columns={
        "app_id": "Article ID",
        "headline": "Headline",
        "primary_sector": "Sector",
        "sentiment": "Sentiment",
        "compound": "VADER Score"
    }).head(500)

    choices = [
        (f"{row['headline'][:100]} | {row['sentiment']}", int(idx))
        for idx, row in data.iterrows()
    ]

    return (
        overview,
        sentiment_chart(data),
        sector_chart(data),
        sector_statistics(data),
        table,
        gr.update(
            choices=choices,
            value=choices[0][1] if choices else None
        )
    )


def analyze_article(article_id):
    if article_id is None:
        return "", "", "", "", ""

    article = app_df.loc[int(article_id)]

    return (
        str(article["headline"]),
        str(article["short_description"]),
        str(article["primary_sector"]),
        str(article["sentiment"]),
        f"{float(article['compound']):.3f}"
    )


def generate_summary(article_id):
    if article_id is None:
        return "Select an article first."

    article = app_df.loc[int(article_id)]

    text = (
        str(article["headline"]) + ". " +
        str(article["short_description"])
    ).strip()

    try:
        model.eval()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        # Support either CPU or GPU model placement
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=2,
                do_sample=False
            )

        summary = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        return "### AI-Generated Summary\n\n" + summary

    except Exception as e:
        return f"Summary error: {type(e).__name__}: {e}"


with gr.Blocks(
    title="Financial News Insight Analyzer",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # Financial News Insight Analyzer

        Explore financial news through sentiment analysis, sector
        classification, and AI-powered summarization.

        *Sentiment scores describe article language, not future
        market performance or investment recommendations.*
        """
    )

    with gr.Row():
        sector_filter = gr.Dropdown(
            choices=sectors,
            value="All Sectors",
            label="News Sector"
        )

        sentiment_filter = gr.Dropdown(
            choices=sentiments,
            value="All Sentiments",
            label="Sentiment"
        )

    search_box = gr.Textbox(
        label="Search Articles",
        placeholder="Search headlines or descriptions..."
    )

    refresh_button = gr.Button(
        "Update Dashboard",
        variant="primary"
    )

    overview = gr.Markdown()

    with gr.Tab("Dashboard"):
        with gr.Row():
            sentiment_plot = gr.Plot()
            sector_plot = gr.Plot()

        statistics = gr.Dataframe(
            label="Sentiment Statistics by Sector",
            interactive=False
        )

    with gr.Tab("Article Explorer"):
        article_table = gr.Dataframe(
            label="Matching Financial News",
            interactive=False
        )

        article_picker = gr.Dropdown(
            label="Select an Article",
            choices=[]
        )

        analyze_button = gr.Button("View Article Details")

        headline_out = gr.Textbox(label="Headline")
        description_out = gr.Textbox(
            label="Description",
            lines=4
        )

        with gr.Row():
            sector_out = gr.Textbox(label="Primary Sector")
            sentiment_out = gr.Textbox(label="Sentiment")
            score_out = gr.Textbox(label="VADER Compound Score")

        summary_button = gr.Button("Generate AI Summary")
        summary_out = gr.Markdown()

    dashboard_outputs = [
        overview,
        sentiment_plot,
        sector_plot,
        statistics,
        article_table,
        article_picker
    ]

    refresh_button.click(
        update_dashboard,
        inputs=[sector_filter, sentiment_filter, search_box],
        outputs=dashboard_outputs
    )

    sector_filter.change(
        update_dashboard,
        inputs=[sector_filter, sentiment_filter, search_box],
        outputs=dashboard_outputs
    )

    sentiment_filter.change(
        update_dashboard,
        inputs=[sector_filter, sentiment_filter, search_box],
        outputs=dashboard_outputs
    )

    search_box.submit(
        update_dashboard,
        inputs=[sector_filter, sentiment_filter, search_box],
        outputs=dashboard_outputs
    )

    analyze_button.click(
        analyze_article,
        inputs=article_picker,
        outputs=[
            headline_out,
            description_out,
            sector_out,
            sentiment_out,
            score_out
        ]
    )

    summary_button.click(
        generate_summary,
        inputs=article_picker,
        outputs=summary_out
    )

    demo.load(
        update_dashboard,
        inputs=[sector_filter, sentiment_filter, search_box],
        outputs=dashboard_outputs
    )

demo.launch(share=True)
