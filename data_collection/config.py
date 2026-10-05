BRANDS = [
    "Samsung", "Apple", "OnePlus", "Xiaomi", "Realme", "iQOO", "Poco",
    "Flipkart", "Meesho", "Myntra", "Amazon India",
    "Swiggy", "Zomato", "Blinkit",
    "Netflix", "Disney+ Hotstar", "Prime Video",
]

SENTIMENT_WEIGHTS = {
    "positive": 0.40,
    "negative": 0.35,
    "neutral": 0.25,
}

PLATFORM_TARGETS = {
    "youtube":   10_000,
    "twitter":   15_000,
    "instagram": 50_000,    
    
}

GEMINI_SHARE = 0.5
GROQ_SHARE   = 0.5

COMMENTS_PER_REQUEST = 30

GEMINI_OUTPUT = "synthetic_gemini.csv"
GROQ_OUTPUT   = "synthetic_groq.csv"
PROGRESS_FILE = "synthetic_progress.json"  