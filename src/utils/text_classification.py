#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections import defaultdict
from typing import Any, Dict, List


def subsample_completions(x: Dict[str, List[Any]], n: int) -> Dict[str, List[Any]]:
    completions = x["completions"]
    agg_scores = x["agg_scores"]
    if len(completions) != len(agg_scores):
        raise ValueError(
            f"The number of completions and agg_scores should be the same. Got {len(completions)} completions and {len(agg_scores)} agg_scores."
        )

    # Take the first n samples, as the completions are ordered in groups of size m e.g [0,0,0,0, 1,1,1,1, 2,2,2,2, ...]
    # We need to ensure these groups are not broken up in order to have a valid comparison at smaller n
    return {
        f"completions@{n}": completions[:n],
        f"agg_scores@{n}": agg_scores[:n],
    }


def extract_completion_answers(
    x: Dict[str, List[Any]], n: int | None = None
) -> Dict[str, List[str]]:

    def __extract_answer(pred_str):

        VALID_TAGS = {
            "DEPRESSED_MOOD",
            "ANHEDONIA",
            "APPETITE_CHANGE",
            "SLEEP_ISSUES",
            "PSYCHOMOTOR",
            "FATIGUE",
            "WORTHLESSNESS",
            "COGNITIVE_ISSUES",
            "SUICIDAL_THOUGHTS",
        }

        match = re.search(
            r"final answer is:\s*\[([A-Z_,\s]*)\]", pred_str, re.IGNORECASE
        )

        if match:
            list_str = match.group(1)
            extracted_list = [tag.strip() for tag in list_str.split(",") if tag.strip()]
            return [tag for tag in extracted_list if tag in VALID_TAGS]

        return []

    if n is None:
        return {"preds": [__extract_answer(p) for p in x["completions"]]}
    else:
        return {f"preds@{n}": [__extract_answer(p) for p in x[f"completions@{n}"]]}


def compute_naive_pred(x: Dict[str, List[Any]], n: int) -> Dict[str, List[str]]:
    preds = x[f"preds@{n}"]
    scores = x[f"agg_scores@{n}"]
    preds = [
        (p, s) for p, s in sorted(zip(preds, scores), key=lambda x: x[1], reverse=True)
    ]
    return {f"pred_naive@{n}": preds[0][0]}


def compute_weighted_pred(x: Dict[str, List[Any]], n: int) -> Dict[str, List[str]]:
    preds = x[f"preds@{n}"]
    scores = x[f"agg_scores@{n}"]
    return {f"pred_weighted@{n}": find_answer_with_largest_sum(preds, scores)}


def compute_maj_pred(x: Dict[str, List[Any]], n: int) -> Dict[str, List[str]]:
    preds = x[f"preds@{n}"]
    return {f"pred_maj@{n}": find_majority_answer(preds)}


def find_answer_with_largest_sum(
    answers: List[List[str]], scores: List[float]
) -> List[str]:
    """
    Groups answers (lists of words) based on their canonical forms and finds the group with the largest sum of scores.

    Args:
        answers (list of list of str): A list of answers, where each answer is a list of words.
        scores (list of float): A list of scores corresponding to each answer (list of words).

    Returns:
        list of str: The list of words representing the group with the largest sum of scores.
    """
    if len(answers) == 0 or len(scores) == 0:
        raise ValueError("answers and scores cannot be empty")

    # Grouping using canonical forms
    canonical_groups = defaultdict(
        float
    )  # Stores cumulative scores for each canonical group
    canonical_to_original = {}  # Maps canonical form back to an original answer

    for answer, score in zip(answers, scores):
        # Create a canonical form by sorting and removing duplicates
        canonical_form = tuple(sorted(set(answer)))

        # Aggregate scores and track the original answer
        canonical_groups[canonical_form] += score
        if canonical_form not in canonical_to_original:
            canonical_to_original[canonical_form] = answer

    # Find the canonical form with the largest cumulative score
    max_canonical = max(canonical_groups, key=canonical_groups.get)
    return canonical_to_original[max_canonical]


def find_majority_answer(answers: List[List[str]]) -> List[str]:
    """
    Groups answers (lists of words) based on their canonical forms and finds the group with the largest number of elements.
    In case of a tie, returns the first occurring group with the largest size.

    Args:
        answers (list of list of str): A list of answers, where each answer is a list of words.

    Returns:
        list of str: The list of words representing the group with the largest number of elements.

    Example:
        answers = [["a", "b"], ["a", "b"], ["c"], ["a", "b", "c"]]
        result = find_majority_answer(answers)
        # result would be ["a", "b"] since it occurs most frequently.
    """
    if len(answers) == 0:
        raise ValueError("answers cannot be empty")

    # Group answers using canonical forms
    canonical_groups = defaultdict(int)  # Count occurrences for each canonical form
    canonical_to_original = {}  # Map canonical form back to an original answer

    for answer in answers:
        # Create a canonical form by sorting and removing duplicates
        canonical_form = tuple(sorted(set(answer)))

        # Increment count for the canonical form
        canonical_groups[canonical_form] += 1

        # Track the original answer for this canonical form
        if canonical_form not in canonical_to_original:
            canonical_to_original[canonical_form] = answer

    # Find the canonical form with the largest count
    max_count = max(canonical_groups.values())
    for canonical_form, count in canonical_groups.items():
        if count == max_count:
            # Return the first occurring group in case of a tie
            return canonical_to_original[canonical_form]
