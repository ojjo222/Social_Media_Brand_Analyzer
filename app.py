
import streamlit as st
import pandas as pd
import plotly.express as px

# Page settings
st.set_page_config(
    page_title="Brand Complaint Analyzer",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("final_results.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("final_results.csv not found. Run topic_model.py first.")
    st.stop()

# Check required columns
required = [
    "platform", "brand", "comment",
    "sentiment", "complaint_category"
]

missing = [col for col in required if col not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# Title
st.title("📊 Social Media Brand Sentiment & Complaint Clusterer")
st.write(
    "Analyze negative feedback and discover "
    "the most common customer complaints."
)

st.divider()

# Sidebar filters
st.sidebar.header("Filters")

platforms = sorted(df["platform"].dropna().unique())

selected_platforms = st.sidebar.multiselect(
    "Select platforms",
    platforms,
    default=platforms
)

categories = sorted(
    df["complaint_category"].dropna().unique()
)

selected_categories = st.sidebar.multiselect(
    "Select complaint categories",
    categories,
    default=categories
)

# Apply filters
filtered_df = df[
    df["platform"].isin(selected_platforms)
    & df["complaint_category"].isin(selected_categories)
].copy()

# Search comments
search = st.text_input("🔎 Search comments")

if search:
    filtered_df = filtered_df[
        filtered_df["comment"].str.contains(
            search, case=False, na=False
        )
    ]

# Metrics
st.subheader("Complaint Overview")

total = len(filtered_df)
category_count = filtered_df["complaint_category"].nunique()
platform_count = filtered_df["platform"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("Negative Comments", total)
col2.metric("Complaint Categories", category_count)
col3.metric("Platforms", platform_count)

st.caption(
    "These metrics reflect the currently selected filters."
)

st.divider()

# Complaint category chart
st.subheader("Complaints by Category")

if not filtered_df.empty:
    counts = filtered_df["complaint_category"].value_counts()

    fig = px.bar(
        x=counts.index,
        y=counts.values,
        text=counts.values,
        labels={
            "x": "Complaint Category",
            "y": "Number of Comments"
        },
        title="Number of Complaints in Each Category"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Percentage report
    report = counts.reset_index()
    report.columns = ["Complaint Category", "Comments"]

    report["Percentage"] = (
        report["Comments"] / report["Comments"].sum() * 100
    ).round(1)

    st.subheader("Complaint Summary")
    st.dataframe(report, use_container_width=True)

    # Platform chart
    st.subheader("Complaints by Platform")

    platform_counts = filtered_df["platform"].value_counts()

    fig2 = px.pie(
        names=platform_counts.index,
        values=platform_counts.values,
        title="Share of Filtered Complaints by Platform"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Complaint table
    st.subheader("Complaint Details")

    st.dataframe(
        filtered_df[
            [
                "platform",
                "brand",
                "comment",
                "sentiment",
                "complaint_category"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # Download filtered data
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered Complaints CSV",
        data=csv_data,
        file_name="filtered_complaints.csv",
        mime="text/csv"
    )

else:
    st.warning("No comments match your selected filters.")


# ----------------------------------
# Improved BERTopic Analysis
# ----------------------------------

st.divider()
st.header("🧠 Complaint Topic Analysis")

try:
    topic_df = pd.read_csv("integrated_results.csv")
    summary_df = pd.read_csv("topic_summary.csv")

    # Remove rows with missing topic IDs
    topic_df = topic_df.dropna(
        subset=["discovered_topic"]
    ).copy()

    topic_df["discovered_topic"] = (
        topic_df["discovered_topic"].astype(int)
    )

    # Keep only real discovered topics, not outliers
    valid_topics = summary_df[
        summary_df["Topic"] != -1
    ].copy()

    if valid_topics.empty:
        st.warning(
            "No regular topics were discovered. "
            "Try collecting more varied comments."
        )

    else:
        # Count comments in each topic
        counts = (
            topic_df[
                topic_df["discovered_topic"] != -1
            ]
            .groupby("discovered_topic")
            .size()
            .to_dict()
        )

        # Add comment counts to the topic summary
        valid_topics["Comment Count"] = (
            valid_topics["Topic"]
            .map(counts)
            .fillna(0)
            .astype(int)
        )

        # Display topic names and descriptions
        st.subheader("Discovered Topics")

        display_columns = [
            col for col in [
                "Topic",
                "Name",
                "Representation",
                "Comment Count"
            ]
            if col in valid_topics.columns
        ]

        st.dataframe(
            valid_topics[display_columns],
            use_container_width=True,
            hide_index=True
        )

        # Interactive chart
        st.subheader("Topic Comparison")

        chart_df = valid_topics[
            ["Topic", "Comment Count"]
        ].copy()

        chart_df["Topic"] = (
            "Topic " + chart_df["Topic"].astype(str)
        )

        fig = px.bar(
            chart_df,
            x="Topic",
            y="Comment Count",
            text="Comment Count",
            title="Number of Comments per Topic"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Inspect comments in a selected topic
        topic_ids = sorted(
            valid_topics["Topic"].astype(int).tolist()
        )

        selected_topic = st.selectbox(
            "Choose a topic to inspect",
            topic_ids
        )

        selected_rows = topic_df[
            topic_df["discovered_topic"] == selected_topic
        ]

        st.subheader(f"Comments in Topic {selected_topic}")

        st.dataframe(
            selected_rows[
                [
                    col for col in [
                        "platform",
                        "brand",
                        "comment",
                        "discovered_topic"
                    ]
                    if col in selected_rows.columns
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        # Download topic report
        report_csv = valid_topics.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Topic Summary",
            data=report_csv,
            file_name="topic_report.csv",
            mime="text/csv"
        )

    # Show outliers separately
    outliers = topic_df[
        topic_df["discovered_topic"] == -1
    ]

    with st.expander("View unassigned comments"):
        st.write(
            f"Unassigned comments: {len(outliers)}"
        )

        if not outliers.empty:
            st.dataframe(
                outliers[
                    [
                        col for col in [
                            "platform",
                            "brand",
                            "comment"
                        ]
                        if col in outliers.columns
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

except FileNotFoundError:
    st.warning(
        "Topic files not found. Run bertopic_model.py "
        "and integrate_topics.py first."
    )

# ----------------------------------
# Overall Sentiment Analysis
# ----------------------------------

st.divider()
st.header("😊 Overall Sentiment Analysis")

try:
    # Load the complete sample dataset
    sentiment_df = pd.read_csv(
        "sample_clustering_results.csv"
    )

    # Count positive, negative, and neutral comments
    sentiment_counts = (
        sentiment_df["sentiment"]
        .str.strip()
        .str.title()
        .value_counts()
        .reindex(
            ["Positive", "Negative", "Neutral"],
            fill_value=0
        )
    )

    # Display sentiment chart
    fig_sentiment = px.pie(
        names=sentiment_counts.index,
        values=sentiment_counts.values,
        title="Distribution of All Sentiments",
        hole=0.35
    )

    st.plotly_chart(
        fig_sentiment,
        use_container_width=True
    )

    # Display sentiment counts
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Positive",
        int(sentiment_counts["Positive"])
    )

    col2.metric(
        "Negative",
        int(sentiment_counts["Negative"])
    )

    col3.metric(
        "Neutral",
        int(sentiment_counts["Neutral"])
    )

except FileNotFoundError:
    st.warning("Sample dataset was not found.")

# ----------------------------------
# Complaint Insights
# ----------------------------------

st.divider()
st.header("💡 Complaint Insights")

if not df.empty:
    # Find the most common complaint category
    complaint_counts = (
        df["complaint_category"].value_counts()
    )

    top_category = complaint_counts.index[0]
    top_count = int(complaint_counts.iloc[0])

    percentage = (
        top_count / len(df) * 100
    )

    st.subheader("Most Common Complaint")

    st.metric(
        "Top Complaint Category",
        top_category,
        f"{percentage:.1f}% of negative comments"
    )

    # Display example comments
    st.subheader("Example Comments")

    examples = df[
        df["complaint_category"] == top_category
    ]

    st.dataframe(
        examples[
            [
                "platform",
                "comment",
                "complaint_category"
            ]
        ].head(5),
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("No complaint data is available.")

# Complaint Recommendation System

st.header("Recommended Actions for the Brand")

if "complaint_category" in df.columns and not df.empty:

    counts = df["complaint_category"].value_counts()

    recommendations = {
        "Battery Issues": "Investigate battery usage and improve battery life.",
        "Customer Service": "Improve customer support response times.",
        "Product Quality": "Strengthen product testing and quality checks.",
        "Delivery Issues": "Improve shipping and delivery tracking.",
        "Pricing": "Review pricing and improve value for money."
    }

    for category, count in counts.items():
        st.subheader(category)
        st.write("Total complaints:", count)
        st.write(
            "Suggested action:",
            recommendations.get(
                category,
                "Review comments to identify suitable improvements."
            )
        )
        st.divider()

else:
    st.info("No complaint categories available yet.")
