#!/bin/python
import argparse
import sys
from typing import Tuple

import pandas as pd


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "type",
        choices=["a", "p"],
        help="a for answers correction and p for paragraphs correction",
    )
    parser.add_argument("solution_file")
    parser.add_argument("submission_file")
    return parser.parse_args()


def compute_f1(correct: str, answer: str) -> float:
    if correct == "<No Answer>":
        return answer == "<No Answer>"
    else:
        correct_ = correct.split()
        answer_ = answer.split()
        tp = sum(word in correct_ for word in answer_)
        fp = sum(word not in correct_ for word in answer_)
        fn = sum(word not in answer_ for word in correct_)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)


def correct_answers(
    solution: pd.DataFrame, submission: pd.DataFrame
) -> Tuple[float, float]:
    if any(col not in solution.columns for col in ("id", "answer")) or any(
        col not in submission.columns for col in ("id", "answer")
    ):
        print(
            "[!] Both the solution and submission files need to have the "
            "`id` and `answer` columns."
        )
        sys.exit(1)

    data = pd.merge(
        solution.rename(columns={"answer": "correct"}), submission, on="id"
    )

    exact_match = (data["correct"] == data["answer"]).mean()
    f1 = data.apply(
        lambda x: compute_f1(x["correct"], x["answer"]), axis=1
    ).mean()
    print(f"Exact match: {exact_match:.3f}")
    print(f"F1:          {f1:.3f}")
    return exact_match, f1


def correct_paragraphs(
    solution: pd.DataFrame, submission: pd.DataFrame
) -> Tuple[float, float]:
    if any(
        col not in solution.columns for col in ("id", "paragraph_id")
    ) or any(
        col not in submission.columns for col in ("id", "top_10", "top_n")
    ):
        print(
            "[!] The solution file needs the `id` and `paragraph_id` columns "
            "and the submission file the `id`, `top_10` and `top_n` columns."
        )
        sys.exit(1)

    if submission["top_10"].apply(lambda x: len(x.split(";")) != 10).sum() > 1:
        print(
            "[!] The top_10 paragraphs must be in the format "
            "`0;1;2;3;4;5;6;7;8;9`"
        )
        sys.exit(1)
    if submission["top_n"].apply(lambda x: len(x.split(";"))).nunique() != 1:
        print(
            "[!] The top_n paragraphs must be in the format `0;1;...;n` "
            "and must all have the same number of paragraphs (n)."
        )

    data = pd.merge(solution, submission, on="id")

    precision_top_10 = data.apply(
        lambda x: (str(x["paragraph_id"]) in x["top_10"].split(";")),
        axis=1,
    ).mean()
    precision_top_n = data.apply(
        lambda x: (str(x["paragraph_id"]) in x["top_n"].split(";")), axis=1
    ).mean()

    print(f"Precision top 10: {precision_top_10:.3f}")
    print(f"Precision top n:  {precision_top_n:.3f}")
    return precision_top_10, precision_top_n


if __name__ == "__main__":
    args = read_args()

    solution = pd.read_csv(args.solution_file)
    submission = pd.read_csv(args.submission_file)

    if args.type == "a":
        correct_answers(solution, submission)
    else:
        correct_paragraphs(solution, submission)
