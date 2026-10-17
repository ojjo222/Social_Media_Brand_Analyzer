
import pandas as pd
from pathlib import Path
import re

# Input and output files
input_file = Path("clustered_comments.csv")
output_file = Path("final_results.csv")

# Check whether the input file exists
if not input_file.exists():
    print("Error: clustered_comments.csv was not found.")
    raise SystemExit(1)

# Read Member 3's clustering results
df = pd.read_csv(input_file)

# Check required columns
required = [
    "platform",
    "brand",
    "comment",
    "sentiment",
    "cluster"
]

missing = [col for col in required if col not in df.columns]

if missing:
    print("Missing required columns:", missing)
    raise SystemExit(1)

# Normalize sentiment
df["sentiment"] = (
    df["sentiment"]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Keep only negative comments with valid text
negative_df = df[
    df["sentiment"].eq("negative")
    & df["comment"].notna()
].copy()

negative_df["comment"] = (
    negative_df["comment"].astype(str).str.strip()
)

negative_df = negative_df[
    negative_df["comment"] != ""
].copy()


# Assign a provisional category using keywords
def get_category(comment):
    text = str(comment).lower()
    text = re.sub(r"\s+", " ", text).strip()

    # Refund and payment issues
    if any(word in text for word in [
        "refund", "money back", "payment failed",
        "money was deducted", "charged", "billing",
        "double payment", "wrong amount"
    ]):
        return "Refund & Payment Issues"

    # Delivery issues
    elif any(word in text for word in [
        "delivery", "delivered", "shipping",
        "shipment", "driver", "tracking",
        "late order", "not arrived", "never arrived"
    ]):
        return "Delivery Issues"

    # Customer service issues
    elif any(word in text for word in [
        "customer support", "customer service",
        "representative", "unresponsive",
        "useless bot", "hold for",
        "rude and unhelpful", "support team",
        "no response"
    ]):
        return "Customer Service"

    # App and technical issues
    elif any(word in text for word in [
        "app", "website", "login", "log in",
        "crash", "bug", "glitch", "checkout",
        "error 500", "buttons don't respond",
        "not loading", "keeps freezing"
    ]):
        return "App & Technical Issues"

    # Product quality issues
    elif any(word in text for word in [
        "broken", "defective", "spoiled", "damaged",
        "quality", "counterfeit", "crushed",
        "stale", "cold food", "leaking",
        "stopped working", "poorly made"
    ]):
        return "Product Quality Issues"

    # Pricing and subscription issues
    elif any(word in text for word in [
        "expensive", "overpriced", "price",
        "cost", "subscription", "auto-renew",
        "hidden fee", "price increase"
    ]):
        return "Pricing & Subscription Issues"

    else:
        return "Other Complaints"


# Apply categorization
negative_df["complaint_category"] = (
    negative_df["comment"].apply(get_category)
)

# Display category counts
print("\nComplaint Counts:")
print(
    negative_df["complaint_category"]
    .value_counts()
    .to_string()
)

# Show sample results
print("\nSample Results:")
print(
    negative_df[
        ["platform", "brand", "comment",
         "cluster", "complaint_category"]
    ].head(10).to_string(index=False)
)

# Save results for the dashboard
negative_df.to_csv(output_file, index=False)

# Display summary
print("\nSaved negative complaints to final_results.csv")
print("Total input comments:", len(df))
print("Total negative complaints categorized:", len(negative_df))

print("\nCategory distribution (%):")
print(
    (negative_df["complaint_category"]
     .value_counts(normalize=True) * 100)
    .round(2)
    .to_string()
)
