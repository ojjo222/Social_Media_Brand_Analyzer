
import streamlit as st
import pandas as pd
import plotly.express as px

# Page settings
st.set_page_config(
    page_title="Brand Complaint Analyzer",
    layout="wide"
)

st.title("Social Media Brand Sentiment & Complaint Clusterer")
st.write(
    "Analyze social media comments and identify "
    "the most common complaint categories."
)

# Load processed results
df = pd.read_csv("final_results.csv")

# Summary numbers
total_comments = len(df)
total_categories = df["complaint_category"].nunique()
total_platforms = df["platform"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("Negative Comments", total_comments)
col2.metric("Complaint Categories", total_categories)
col3.metric("Platforms", total_platforms)

# Complaint category chart
st.subheader("Most Common Complaint Categories")

counts = df["complaint_category"].value_counts()

fig = px.bar(
    x=counts.index,
    y=counts.values,
    labels={
        "x": "Complaint Category",
        "y": "Number of Comments"
    },
    title="Complaints by Category"
)

st.plotly_chart(fig, use_container_width=True)

# Platform chart
st.subheader("Complaints by Platform")

platform_counts = df["platform"].value_counts()

fig2 = px.bar(
    x=platform_counts.index,
    y=platform_counts.values,
    labels={
        "x": "Platform",
        "y": "Number of Comments"
    },
    title="Negative Comments by Platform"
)

st.plotly_chart(fig2, use_container_width=True)

# Category filter
st.subheader("Explore Complaint Details")

categories = sorted(
    df["complaint_category"].dropna().unique()
)

selected_category = st.selectbox(
    "Choose a complaint category",
    ["All"] + categories
)

filtered_df = df.copy()

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["complaint_category"]
        == selected_category
    ]

# Comment search
search = st.text_input("Search for a word in comments")

if search:
    filtered_df = filtered_df[
        filtered_df["comment"].str.contains(
            search,
            case=False,
            na=False
        )
    ]

# Display table
st.dataframe(
    filtered_df[
        [
            "platform",
            "brand",
            "comment",
            "complaint_category"
        ]
    ],
    use_container_width=True
)
