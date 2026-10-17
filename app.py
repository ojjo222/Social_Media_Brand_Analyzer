
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ----------------------------------
# Page Settings
# ----------------------------------

st.set_page_config(
    page_title="Brand Complaint Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Social Media Brand Sentiment & Complaint Clusterer")

st.write(
    "Analyze social media feedback, explore comment clusters, "
    "identify complaint categories, and view recommendations."
)

st.divider()

# ----------------------------------
# File Loading
# ----------------------------------

DATA_FILE = "clustered_comments.csv"
RESULT_FILE = "final_results.csv"
AI_FILE = "ai_recommendations.csv"


@st.cache_data
def load_csv(filename):
    return pd.read_csv(filename)


if not Path(DATA_FILE).exists():
    st.error(
        "clustered_comments.csv not found. "
        "Place Member 3's new CSV in the project folder."
    )
    st.stop()

cluster_df = load_csv(DATA_FILE)

required_columns = [
    "platform", "brand", "comment", "sentiment", "cluster"
]

missing = [c for c in required_columns if c not in cluster_df.columns]

if missing:
    st.error(f"Missing columns in clustered_comments.csv: {missing}")
    st.stop()

# Clean text fields without changing the original file.
for col in ["platform", "brand", "comment", "sentiment", "cluster"]:
    cluster_df[col] = cluster_df[col].astype("string").str.strip()

cluster_df["sentiment_clean"] = (
    cluster_df["sentiment"].str.lower()
)

# ----------------------------------
# Sidebar Filters
# ----------------------------------

st.sidebar.header("Filters")

platforms = sorted(cluster_df["platform"].dropna().unique().tolist())

selected_platforms = st.sidebar.multiselect(
    "Platforms",
    platforms,
    default=platforms
)

sentiments = ["Positive", "Negative", "Neutral"]

selected_sentiments = st.sidebar.multiselect(
    "Sentiment",
    sentiments,
    default=sentiments
)

filtered_all = cluster_df[
    cluster_df["platform"].isin(selected_platforms)
    & cluster_df["sentiment_clean"].isin(
        [s.lower() for s in selected_sentiments]
    )
].copy()

search = st.text_input("🔎 Search comments")

if search:
    filtered_all = filtered_all[
        filtered_all["comment"].str.contains(
            search, case=False, na=False
        )
    ]

# ----------------------------------
# Overall Sentiment
# ----------------------------------

st.header("😊 Overall Sentiment")

sentiment_counts = (
    filtered_all["sentiment_clean"]
    .value_counts()
    .reindex(["positive", "negative", "neutral"], fill_value=0)
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Comments", f"{len(filtered_all):,}")
c2.metric("Positive", f"{int(sentiment_counts['positive']):,}")
c3.metric("Negative", f"{int(sentiment_counts['negative']):,}")
c4.metric("Neutral", f"{int(sentiment_counts['neutral']):,}")

sentiment_chart = pd.DataFrame({
    "Sentiment": ["Positive", "Negative", "Neutral"],
    "Comments": [
        int(sentiment_counts["positive"]),
        int(sentiment_counts["negative"]),
        int(sentiment_counts["neutral"])
    ]
})

fig_sentiment = px.pie(
    sentiment_chart,
    names="Sentiment",
    values="Comments",
    hole=0.4,
    title="Sentiment Distribution"
)

st.plotly_chart(fig_sentiment, use_container_width=True)

st.download_button(
    "Download Filtered Comments",
    data=filtered_all.drop(
        columns=["sentiment_clean"], errors="ignore"
    ).to_csv(index=False).encode("utf-8"),
    file_name="filtered_comments.csv",
    mime="text/csv"
)

st.divider()

# ----------------------------------
# Member 3: Clustering Analysis
# ----------------------------------

st.header("🧠 Member 3's Clustering Analysis")

st.caption(
    "These are the cluster labels provided in Member 3's CSV. "
    "Missing cluster labels are excluded from the cluster chart."
)

clustered = filtered_all[
    filtered_all["cluster"].notna()
    & (filtered_all["cluster"] != "")
].copy()

if clustered.empty:
    st.warning("No clustered comments match the current filters.")
else:
    cluster_counts = (
        clustered.groupby("cluster")
        .size()
        .reset_index(name="Comment Count")
        .sort_values("Comment Count", ascending=False)
    )

    st.dataframe(
        cluster_counts,
        use_container_width=True,
        hide_index=True
    )

    fig_clusters = px.bar(
        cluster_counts,
        x="cluster",
        y="Comment Count",
        text="Comment Count",
        title="Comments per Cluster",
        labels={
            "cluster": "Cluster Label",
            "Comment Count": "Number of Comments"
        }
    )

    st.plotly_chart(fig_clusters, use_container_width=True)

    cluster_options = cluster_counts["cluster"].tolist()

    chosen_cluster = st.selectbox(
        "Inspect a cluster",
        cluster_options
    )

    cluster_examples = clustered[
        clustered["cluster"] == chosen_cluster
    ]

    st.subheader(f"Comments: {chosen_cluster}")

    st.dataframe(
        cluster_examples[
            ["platform", "brand", "comment", "sentiment", "cluster"]
        ].head(500),
        use_container_width=True,
        hide_index=True
    )

    st.caption("Showing at most 500 comments for readability.")

    st.download_button(
        "Download Cluster Comments",
        data=cluster_examples.to_csv(index=False).encode("utf-8"),
        file_name="selected_cluster_comments.csv",
        mime="text/csv"
    )

st.divider()

# ----------------------------------
# Complaint Categories
# ----------------------------------

st.header("📋 Complaint Categories")

# Negative comments with a cluster label are the initial
# complaint-analysis subset. Their labels should be validated.
negative = filtered_all[
    (filtered_all["sentiment_clean"] == "negative")
    & filtered_all["cluster"].notna()
    & (filtered_all["cluster"] != "")
].copy()

if negative.empty:
    st.info("No negative comments with cluster labels match the filters.")
else:
    complaint_counts = (
        negative.groupby("cluster")
        .size()
        .reset_index(name="Comments")
        .sort_values("Comments", ascending=False)
        .rename(columns={"cluster": "Complaint Category"})
    )

    total = int(complaint_counts["Comments"].sum())

    complaint_counts["Percentage"] = (
        complaint_counts["Comments"] / total * 100
    ).round(1)

    a, b, c = st.columns(3)

    a.metric("Negative Clustered Comments", f"{total:,}")
    b.metric("Categories / Cluster Labels", complaint_counts.shape[0])
    c.metric("Platforms", negative["platform"].nunique())

    fig_complaints = px.bar(
        complaint_counts,
        x="Complaint Category",
        y="Comments",
        text="Comments",
        title="Negative Comments by Cluster Label"
    )

    st.plotly_chart(fig_complaints, use_container_width=True)

    st.dataframe(
        complaint_counts,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Negative Comment Details")

    st.dataframe(
        negative[
            ["platform", "brand", "comment", "sentiment", "cluster"]
        ].head(1000),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Negative Clustered Comments",
        data=negative.to_csv(index=False).encode("utf-8"),
        file_name="negative_clustered_comments.csv",
        mime="text/csv"
    )

st.divider()

# ----------------------------------
# Processed Topic Results
# ----------------------------------

st.header("🔎 Processed Topic Results")

if Path(RESULT_FILE).exists():
    results = load_csv(RESULT_FILE)

    if "complaint_category" in results.columns:
        st.caption(
            "This section reads final_results.csv. "
            "Regenerate that file after updating the processing scripts."
        )

        st.dataframe(
            results.head(1000),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Processed Results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="final_results.csv",
            mime="text/csv"
        )
    else:
        st.warning(
            "final_results.csv does not contain complaint_category. "
            "Run the updated topic processing script."
        )
else:
    st.info("final_results.csv not found. Run topic_model.py first.")

st.divider()

# ----------------------------------
# AI Recommendations
# ----------------------------------

st.header("🤖 Recommendations by Cluster")

if not Path(AI_FILE).exists():
    st.info(
        "ai_recommendations.csv not found. "
        "Run recommendations.py after processing the new data."
    )
else:
    ai_df = load_csv(AI_FILE)

    required_ai = [
        "cluster", "complaint_category", "ai_recommendation"
    ]

    missing_ai = [c for c in required_ai if c not in ai_df.columns]

    if missing_ai:
        st.error(f"Missing columns in ai_recommendations.csv: {missing_ai}")
    else:
        if "status" in ai_df.columns:
            ai_df = ai_df[
                ai_df["status"].astype(str).str.lower().eq("success")
            ]

        ai_df = ai_df.drop_duplicates(
            subset=["cluster"], keep="last"
        )

        if ai_df.empty:
            st.warning(
                "No successful recommendations found. "
                "Check recommendations.py and its terminal output."
            )
        else:
            st.metric(
                "Clusters with Recommendations",
                ai_df["cluster"].nunique()
            )

            for _, row in ai_df.iterrows():
                with st.container(border=True):
                    st.subheader(
                        f"{row['cluster']} — {row['complaint_category']}"
                    )

                    if "number_of_comments" in ai_df.columns:
                        st.caption(
                            f"Comments analyzed: "
                            f"{row['number_of_comments']}"
                        )

                    st.markdown(
                        str(row["ai_recommendation"]).replace(
                            "\\n", "\n"
                        )
                    )

            st.download_button(
                "Download Recommendations",
                data=ai_df.to_csv(index=False).encode("utf-8"),
                file_name="ai_recommendations.csv",
                mime="text/csv"
            )

st.caption(
    "Note: Cluster labels are supplied by Member 3. "
    "They should be reviewed before being treated as confirmed complaint categories."
)
