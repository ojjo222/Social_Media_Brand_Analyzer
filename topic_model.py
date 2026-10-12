
import pandas as pd

# Read the sample data
df = pd.read_csv("sample_clustering_results.csv")

# Keep only negative comments
negative_df = df[
    df["sentiment"].str.lower() == "negative"
].copy()

# Give each cluster a readable complaint category
category_names = {
    0: "Battery Issues",
    1: "Customer Service",
    2: "Product Quality",
    3: "Delivery Issues",
    4: "Pricing"
}

# Add the complaint category column
negative_df["complaint_category"] = (
    negative_df["cluster"].map(category_names)
)

# Display the results
print("\nComplaint Details:")
print(
    negative_df[
        ["comment", "cluster", "complaint_category"]
    ].to_string(index=False)
)

# Count complaints in each category
print("\nComplaint Counts:")
print(negative_df["complaint_category"].value_counts())

# Save the processed results
negative_df.to_csv("final_results.csv", index=False)

print("\nResults saved to final_results.csv")
