import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment.distilbert_analyzer import clean_text_for_transformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("train_distilbert")

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False
        )
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def clean_and_deduplicate(df: pd.DataFrame, text_col: str = "text", label_col: str = "sentiment") -> pd.DataFrame:
    initial_len = len(df)
    df = df.copy()

    # 1. Clean text
    df["clean_text"] = df[text_col].fillna("").astype(str).apply(clean_text_for_transformer)
    df = df[df["clean_text"].str.strip().str.len() > 0]

    # 2. Map sentiment to int IDs
    if label_col in df.columns:
        if df[label_col].dtype == object or isinstance(df[label_col].iloc[0], str):
            df["label"] = df[label_col].str.lower().map(LABEL2ID)
        else:
            df["label"] = df[label_col].astype(int)
        df = df[df["label"].isin([0, 1, 2])]

    # 3. Deduplication & Conflict removal
    # Group by clean_text; if multiple different labels exist for same text, drop them
    label_counts = df.groupby("clean_text")["label"].nunique()
    conflicting_texts = set(label_counts[label_counts > 1].index)
    if conflicting_texts:
        logger.info("Dropping %d texts with conflicting sentiment labels", len(conflicting_texts))
        df = df[~df["clean_text"].isin(conflicting_texts)]

    # Deduplicate identical text with same label
    df = df.drop_duplicates(subset=["clean_text", "label"])
    logger.info("Cleaned dataset: from %d rows to %d unique valid rows", initial_len, len(df))
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_w)
    }


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    output_dir: str = "models/distilbert/gfg_distilbert_sentiment_v2",
    model_name: str = "distilbert-base-uncased",
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    fp16: bool = False
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    train_dataset = SentimentDataset(
        train_df["clean_text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        max_length=max_length
    )
    val_dataset = SentimentDataset(
        val_df["clean_text"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        max_length=max_length
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(out_path / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        fp16=fp16 and torch.cuda.is_available(),
        logging_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    logger.info("Starting fine-tuning for %d epochs...", epochs)
    trainer.train()

    logger.info("Saving final model and tokenizer to %s", out_path)
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    # Evaluate on untouched test set if provided
    final_metrics = {}
    if test_df is not None:
        logger.info("Evaluating on untouched test set (%d samples)...", len(test_df))
        test_dataset = SentimentDataset(
            test_df["clean_text"].tolist(),
            test_df["label"].tolist(),
            tokenizer,
            max_length=max_length
        )
        test_preds_raw = trainer.predict(test_dataset)
        preds = np.argmax(test_preds_raw.predictions, axis=-1)
        labels = test_df["label"].values

        acc = accuracy_score(labels, preds)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        report = classification_report(labels, preds, target_names=["negative", "neutral", "positive"], output_dict=True)
        conf_mat = confusion_matrix(labels, preds).tolist()

        final_metrics = {
            "test_accuracy": float(acc),
            "test_macro_precision": float(p_m),
            "test_macro_recall": float(r_m),
            "test_macro_f1": float(f1_m),
            "classification_report": report,
            "confusion_matrix": conf_mat
        }

        with open(out_path / "metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)

        print("\n" + "=" * 50)
        print("TEST SET EVALUATION RESULTS")
        print("=" * 50)
        print(f"Accuracy:        {acc * 100:.2f}%")
        print(f"Macro Precision: {p_m * 100:.2f}%")
        print(f"Macro Recall:    {r_m * 100:.2f}%")
        print(f"Macro F1:        {f1_m * 100:.2f}%\n")
        print("Per-class Metrics:")
        for cls_name in ["negative", "neutral", "positive"]:
            c_metrics = report[cls_name]
            print(f"  {cls_name:8s} -> P: {c_metrics['precision']:.4f}, R: {c_metrics['recall']:.4f}, F1: {c_metrics['f1-score']:.4f}")
        print("=" * 50)

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on Sentiment Classification dataset")
    parser.add_argument("--train-data", default="data/benchmark/train.csv", help="Path to training CSV")
    parser.add_argument("--val-data", default="data/benchmark/validation.csv", help="Path to validation CSV")
    parser.add_argument("--test-data", default="data/benchmark/test.csv", help="Path to untouched test CSV")
    parser.add_argument("--output-dir", default="models/distilbert/gfg_distilbert_sentiment_v2", help="Model output directory")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 on GPU")

    args = parser.parse_args()

    train_df = clean_and_deduplicate(pd.read_csv(args.train_data))
    val_df = clean_and_deduplicate(pd.read_csv(args.val_data))
    test_df = clean_and_deduplicate(pd.read_csv(args.test_data)) if os.path.exists(args.test_data) else None

    train_model(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=args.fp16
    )


if __name__ == "__main__":
    main()
