import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from ast import literal_eval

VALID_TAGS = [
    "NO_SYMPTOMS"
    "DEPRESSED_MOOD",
    "ANHEDONIA",
    "APPETITE_CHANGE",
    "SLEEP_ISSUES",
    "PSYCHOMOTOR",
    "FATIGUE",
    "WORTHLESSNESS",
    "COGNITIVE_ISSUES",
    "SUICIDAL_THOUGHTS",
]


def load_golden_truth(file_path):
    golden = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            golden[parts[0]] = {"depressed_golden": bool(int(parts[1])), "criteria": []}
    return golden


def load_predictions(file_path, golden):
    with open(file_path, "r", encoding="utf-8") as file:
        user_posts_seen = 1
        processed_users = []

        for line in file:
            record = json.loads(line.strip())

            if (len(record.get("text")) > 300):
                user_id = record.get("subject")

                if user_id not in processed_users:
                    user_posts_seen = 1
                    processed_users.append(user_id)

                user_posts_seen += 1
                criteria = record.get("classification")

                criteria = [item for item in criteria if item != "NO_SYMPTOMS"]

                golden[user_id]["criteria"] = list(
                    set(golden[user_id]["criteria"] + criteria)
                )
                golden[user_id]["judged"] = True

                criteria = golden[user_id]["criteria"]

                decision = len(criteria) >= 5 and (
                    "DEPRESSED_MOOD" in criteria or "ANHEDONIA" in criteria
                )

                golden[user_id]["decision"] = decision

                if not golden[user_id]["decision"] or not "delay" in golden[user_id]:
                    golden[user_id]["delay"] = user_posts_seen

    return golden


def calculate_predictions(golden):
    references, predictions, delays = [], [], []
    for subject in golden.values():
        references.append(subject["depressed_golden"])
        predictions.append(subject["decision"])
        delays.append(subject["delay"])
    return references, predictions, delays


def erisk_eval(references, predictions, delays, o):
    """
    Evaluate eRisk metrics using boolean np.arrays of references and predictions.

    Args:
            references (np.array): True risk labels (ground truth), a np.array of booleans.
            predictions (np.array): Predicted risk decisions, a np.array of booleans.
            delays (np.array): A np.array of delay values corresponding to predictions.
            o (float): Parameter for the ERDE formula.

    Returns:
            None
    """
    # Ensure input lists are of the same length
    assert len(references) == len(predictions)
    assert len(predictions) == len(delays)

    # Total number of users
    total_users = len(references)

    # Count true positives, predicted positives, and correctly predicted positives
    true_pos = np.sum(references)
    pos_decisions = np.sum(predictions)
    pos_hits = np.sum(references & predictions)

    # Initialize ERDE array
    erde = np.ones(total_users, dtype=float)

    # Compute individual ERDE values
    for i in range(total_users):
        if predictions[i] and not references[i]:
            erde[i] = float(true_pos) / total_users
        elif not predictions[i] and references[i]:
            erde[i] = 1.0
        elif predictions[i] and references[i]:
            erde[i] = 1.0 - (1.0 / (1.0 + np.exp(delays[i] - o)))
        elif not predictions[i] and not references[i]:
            erde[i] = 0.0

    # Calculate Precision, Recall, F1 Score, and global ERDE
    precision = float(pos_hits) / pos_decisions if pos_decisions > 0 else 0.0
    recall = float(pos_hits) / true_pos if true_pos > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    erde_global = erde.mean() * 100

    # Print results
    # print("Individual ERDE values:")
    # for i, e in enumerate(erde):
    #     print(f"User {i}: ERDE = {e:.4f}")
    print(f"\nGlobal ERDE (with o = {o}): {erde_global:.2f}")
    print(f"F1: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


def generate_confusion_matrix(
    references, predictions, output_path="confusion_matrix.png"
):
    cm = confusion_matrix(references, predictions, normalize="true")
    cm = cm.T
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        # fmt="d",
        cmap="Blues",
        xticklabels=["Control", "Depressed"],
        yticklabels=["Control", "Depressed"],
    )
    # plt.title("Confusion Matrix")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_misclassifications(
    golden, references, predictions, output_path="misclassifications.txt"
):
    with open(output_path, "w") as f:
        for idx, (ref, pred) in enumerate(zip(references, predictions)):
            if ref != pred:
                case = golden[list(golden.keys())[idx]]
                f.write(
                    f"User ID: {list(golden.keys())[idx]}, Golden: {ref}, Predicted: {pred}\n"
                )
                f.write(f"Criteria: {case['criteria']}\n")
                f.write("\n")


def main():

    golden = load_golden_truth("data/raw/erisk_t2_2022/test/risk_golden_truth.txt")

    golden = load_predictions("output_copy.jsonl", golden)

    # TODO: remove
    golden = {k: v for k, v in golden.items() if "judged" in v}
    print(golden)

    references, predictions, delays = calculate_predictions(golden)

    generate_confusion_matrix(references, predictions)

    print("\nClassification Report:")
    print(classification_report(np.array(references), np.array(predictions)))

    print("\neRisk Report:")
    for o in [5, 50]:
        erisk_eval(np.array(references), np.array(predictions), np.array(delays), o)

    analyze_misclassifications(golden, references, predictions)


if __name__ == "__main__":
    main()
