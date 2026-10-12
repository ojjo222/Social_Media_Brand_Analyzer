
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
