
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# -------------------------------
# 1. Page settings
# -------------------------------
st.set_page_config(
    page_title="Social Media Brand Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Social Media Brand Analyzer")
st.write(
    "Explore customer feedback, sentiment, platforms, "
    "brands, and feedback groups."
)

# -------------------------------
# 2. Load the CSV file
# -------------------------------
CSV_FILE = Path(__file__).parent / "clustered_comments.csv"

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

if not CSV_FILE.exists():
    st.error(
        "CSV file not found. Place clustered_comments.csv "
        "in the same folder as app.py."
    )
    st.stop()

df = load_data(str(CSV_FILE))

# -------------------------------
# 3. Check required columns
# -------------------------------
required_columns = [
    "platform", "brand", "comment", "sentiment", "cluster"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing CSV columns: {missing_columns}")
    st.stop()

# Clean text columns for display and analysis.
for col in required_columns:
    df[col] = df[col].astype("string").str.strip()

# Treat blank text values as missing.
df = df.replace(r"^\s*$", pd.NA, regex=True)

# Normalize sentiment to lowercase.
df["sentiment"] = df["sentiment"].str.lower()

# Keep original cluster labels and show missing labels separately.
df["cluster_display"] = df["cluster"].fillna(
    "Unclustered feedback"
)

# -------------------------------
# 4. Sidebar filters
# -------------------------------
st.sidebar.header("Filter feedback")

platform_options = sorted(df["platform"].dropna().unique())
selected_platforms = st.sidebar.multiselect(
    "Select platform",
    platform_options,
    default=platform_options
)

sentiment_options = sorted(df["sentiment"].dropna().unique())
selected_sentiments = st.sidebar.multiselect(
    "Select sentiment",
    sentiment_options,
    default=sentiment_options
)

brand_options = sorted(df["brand"].dropna().unique())
selected_brands = st.sidebar.multiselect(
    "Select brand",
    brand_options,
    default=brand_options
)

search_text = st.sidebar.text_input(
    "Search comments",
    placeholder="Enter a word or phrase"
)

# Apply filters
filtered = df[
    df["platform"].isin(selected_platforms)
    & df["sentiment"].isin(selected_sentiments)
    & df["brand"].isin(selected_brands)
].copy()

if search_text:
    filtered = filtered[
        filtered["comment"]
        .fillna("")
        .str.contains(search_text, case=False, regex=False)
    ]

# -------------------------------
# 5. Summary metrics
# -------------------------------
st.subheader("Customer Feedback Overview")

total_comments = len(filtered)
negative_count = (filtered["sentiment"] == "negative").sum()
positive_count = (filtered["sentiment"] == "positive").sum()
neutral_count = (filtered["sentiment"] == "neutral").sum()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Comments", f"{total_comments:,}")
c2.metric("Positive Comments", f"{positive_count:,}")
c3.metric("Negative Comments", f"{negative_count:,}")
c4.metric("Neutral Comments", f"{neutral_count:,}")

if total_comments == 0:
    st.warning("No comments match your selected filters.")
    st.stop()

# -------------------------------
# 6. Sentiment analysis
# -------------------------------
st.subheader("Customer Sentiment Overview")

left, right = st.columns(2)

sentiment_counts = (
    filtered["sentiment"]
    .value_counts()
    .rename_axis("Sentiment")
    .reset_index(name="Comments")
)

with left:
    fig = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Comments",
        title="Sentiment Distribution",
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Comments",
        color="Sentiment",
        title="Comments by Sentiment",
        text="Comments"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 7. Platform analysis
# -------------------------------
st.subheader("Feedback Across Platforms")

platform_counts = (
    filtered["platform"]
    .value_counts()
    .rename_axis("Platform")
    .reset_index(name="Comments")
)

fig = px.bar(
    platform_counts,
    x="Platform",
    y="Comments",
    color="Platform",
    title="Number of Comments by Platform",
    text="Comments"
)
st.plotly_chart(fig, use_container_width=True)

# Sentiment comparison by platform
platform_sentiment = (
    filtered.groupby(["platform", "sentiment"], dropna=False)
    .size()
    .reset_index(name="Comments")
)

fig = px.bar(
    platform_sentiment,
    x="platform",
    y="Comments",
    color="sentiment",
    barmode="group",
    title="Sentiment Comparison by Platform"
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 8. Brand analysis
# -------------------------------
st.subheader("Brand Feedback")

brand_counts = (
    filtered["brand"]
    .dropna()
    .value_counts()
    .head(15)
    .rename_axis("Brand")
    .reset_index(name="Comments")
)

if not brand_counts.empty:
    fig = px.bar(
        brand_counts,
        x="Comments",
        y="Brand",
        orientation="h",
        title="Top 15 Brands by Number of Comments"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No brand data is available for these filters.")

# -------------------------------
# 9. Customer feedback groups
# -------------------------------
st.subheader("Customer Feedback Groups")

cluster_counts = (
    filtered["cluster_display"]
    .value_counts()
    .rename_axis("Feedback Group")
    .reset_index(name="Comments")
)

fig = px.bar(
    cluster_counts,
    x="Feedback Group",
    y="Comments",
    title="Comments in Each Feedback Group",
    text="Comments"
)
fig.update_layout(xaxis_tickangle=-35)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 10. Negative feedback
# -------------------------------
st.subheader("Negative Customer Feedback")

negative_comments = filtered[
    filtered["sentiment"] == "negative"
]

st.write(f"Negative comments matching filters: {len(negative_comments):,}")

st.dataframe(
    negative_comments[
        ["platform", "brand", "comment", "cluster"]
    ].head(100),
    use_container_width=True,
    hide_index=True
)

# -------------------------------
# 11. Searchable comments table
# -------------------------------
st.subheader("Explore Customer Comments")

st.caption("Showing up to 500 matching comments.")

st.dataframe(
    filtered[
        ["platform", "brand", "comment", "sentiment", "cluster"]
    ].head(500),
    use_container_width=True,
    hide_index=True
)

# -------------------------------
# 12. Download filtered data
# -------------------------------
csv_data = filtered[
    ["platform", "brand", "comment", "sentiment", "cluster"]
].to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Comments",
    data=csv_data,
    file_name="filtered_comments.csv",
    mime="text/csv"
)

st.caption(
    "Analysis is based on the supplied CSV. "
    "Missing cluster labels are shown as Unclustered feedback."
)


# ----------------------------------
# Topic-wise Recommendations
# ----------------------------------

st.divider()
st.header("💡 Topic-wise Recommendations")
st.caption(
    "Recommended actions based on customer feedback topics."
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
recommendation_file = BASE_DIR / "ai_recommendations.csv"

try:
    ai_df = pd.read_csv(recommendation_file)
    ai_df.columns = ai_df.columns.str.strip()

    # Prefer the real topic or cluster name
    topic_col = next(
        (
            col for col in ["cluster", "topic", "complaint_category"]
            if col in ai_df.columns
        ),
        None
    )

    if topic_col is None:
        st.error(
            "No topic name found. The CSV must contain "
            "a cluster, topic, or complaint_category column."
        )

    elif "ai_recommendation" not in ai_df.columns:
        st.error(
            "The ai_recommendation column is missing "
            "from ai_recommendations.csv."
        )

    else:
        # Remove empty recommendations
        ai_df = ai_df.dropna(
            subset=[topic_col, "ai_recommendation"]
        )

        ai_df["ai_recommendation"] = (
            ai_df["ai_recommendation"].astype(str).str.strip()
        )

        ai_df = ai_df[
            ai_df["ai_recommendation"] != ""
        ]

        # Exclude generic or missing topic names
        invalid_names = [
            "category needs review",
            "nan",
            "none",
            "unassigned",
            "unknown",
            ""
        ]

        ai_df = ai_df[
            ~ai_df[topic_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(invalid_names)
        ]

        if ai_df.empty:
            st.info(
                "No valid topic-wise recommendations found. "
                "Check the topic names in ai_recommendations.csv."
            )

        else:
            # Optional topic filter
            topic_names = sorted(
                ai_df[topic_col].astype(str).unique().tolist()
            )

            selected_topic = st.selectbox(
                "Filter by topic",
                ["All Topics"] + topic_names,
                key="recommendation_topic_filter"
            )

            if selected_topic != "All Topics":
                display_df = ai_df[
                    ai_df[topic_col].astype(str) == selected_topic
                ]
            else:
                display_df = ai_df

            # Show one recommendation card per topic
            for topic, group in display_df.groupby(
                topic_col, sort=True
            ):
                with st.container(border=True):
                    # Heading is only the real topic name
                    st.markdown(f"### {topic}")

                    # Show number of comments, if available
                    if "number_of_comments" in group.columns:
                        counts = pd.to_numeric(
                            group["number_of_comments"],
                            errors="coerce"
                        ).dropna()

                        if not counts.empty:
                            st.caption(
                                f"Comments analyzed: "
                                f"{int(counts.sum()):,}"
                            )

                    st.markdown("**Recommended actions**")

                    # Display each saved recommendation
                    for recommendation in group[
                        "ai_recommendation"
                    ].drop_duplicates():
                        st.write(recommendation)

except FileNotFoundError:
    st.warning(
        "ai_recommendations.csv was not found. "
        "Generate the recommendation file first."
    )

except Exception as e:
    st.error(f"Could not load recommendations: {e}")
