import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def load_jsonl(file_path):
    """Loads a JSONL file into a list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_labels(golden_path, predicted_path):
    """Extracts gold and predicted labels from JSONL files."""
    golden_data = {entry["problem"]: entry["label"] for entry in load_jsonl(golden_path)}
    predicted_data = {entry["problem"]: entry["classification"] for entry in load_jsonl(predicted_path)}

    problems = set(golden_data.keys()) & set(predicted_data.keys())
    references = [golden_data[p] for p in problems]
    predictions = [predicted_data[p] for p in problems]
    return references, predictions


def plot_confusion_matrix(references, predictions, output_path="confusion_matrix.png"):
    """Generates and saves a confusion matrix."""
    cm = confusion_matrix(references, predictions, normalize="true").T
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=["No Hate Speech", "Hate Speech"],
                yticklabels=["No Hate Speech", "Hate Speech"], fmt=".2f")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_misclassifications(references, predictions, output_path="misclassifications.txt"):
    """Saves misclassified cases to a file."""
    with open(output_path, "w") as f:
        for idx, (ref, pred) in enumerate(zip(references, predictions)):
            if ref != pred:
                f.write(f"Index: {idx}, Gold: {ref}, Predicted: {pred}\n")


def main():
    references, predictions = extract_labels("data/hate-speech/test.jsonl", "output_copy.jsonl")
    plot_confusion_matrix(references, predictions)
    print("\nClassification Report:")
    print(classification_report(np.array(references), np.array(predictions)))
    save_misclassifications(references, predictions)


if __name__ == "__main__":
    main()
