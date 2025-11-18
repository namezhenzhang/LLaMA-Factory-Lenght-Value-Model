import argparse
import json
import os
import random
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample a block-structured dataset of size (num_groups * group_size) to (m * n). "
                    "Assumes entries are ordered in contiguous blocks of length group_size."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file (list of records).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "-m",
        "--num-questions",
        type=int,
        required=True,
        help="Number of questions to keep. Must be <= total questions.",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        required=True,
        help="Number of samples per question to keep. Must be <= group_size.",
    )
    parser.add_argument(
        "-g",
        "--group-size",
        type=int,
        default=64,
        help="Number of records per group (default: 64).",
    )
    parser.add_argument(
        "--group-mode",
        choices=["head", "random"],
        default="head",
        help="How to choose which groups to keep: 'head' (first m) or 'random' (sample m groups).",
    )
    parser.add_argument(
        "--within-mode",
        choices=["head", "random"],
        default="head",
        help="How to choose which samples within each group to keep: 'head' (first n) or 'random' (sample n).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when mode is 'random'.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent for output JSON (default: 2).",
    )
    return parser.parse_args()


def choose_indices_head(count: int, k: int) -> List[int]:
    return list(range(min(k, count)))


def choose_indices_random(count: int, k: int, rng: random.Random) -> List[int]:
    k = min(k, count)
    if k <= 0:
        return []
    return rng.sample(range(count), k)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")

    total_records = len(data)
    if total_records == 0:
        raise ValueError("Input dataset is empty.")

    if args.group_size <= 0:
        raise ValueError("group_size must be positive.")

    overall_num_questions = total_records // args.group_size
    remainder = total_records % args.group_size
    if remainder != 0:
        raise ValueError(
            f"Dataset length ({total_records}) is not divisible by group_size ({args.group_size}). "
            f"Remainder: {remainder}. Adjust --group-size or fix the input."
        )

    if args.num_questions <= 0 or args.num_questions > overall_num_questions:
        raise ValueError(f"num_questions must be in [1, {overall_num_questions}], got {args.num_questions}.")
    if args.num_samples <= 0 or args.num_samples > args.group_size:
        raise ValueError(f"num_samples must be in [1, {args.group_size}], got {args.num_samples}.")

    if args.group_mode == "head":
        selected_groups = choose_indices_head(overall_num_questions, args.num_questions)
    else:
        selected_groups = choose_indices_random(overall_num_questions, args.num_questions, rng)

    output_records: List[dict] = []
    for question_index in selected_groups:
        start = question_index * args.group_size
        end = start + args.group_size
        question_records = data[start:end]

        if args.within_mode == "head":
            within_indices = choose_indices_head(args.group_size, args.num_samples)
        else:
            within_indices = choose_indices_random(args.group_size, args.num_samples, rng)

        for idx in within_indices:
            output_records.append(question_records[idx])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=args.indent)

    print(
        f"Downsampled dataset written to: {args.output}\n"
        f"- Input records: {total_records} (groups: {overall_num_questions}, group_size: {args.group_size})\n"
        f"- Output records: {len(output_records)} (num_questions={args.num_questions}, num_samples={args.num_samples})\n"
        f"- group_mode: {args.group_mode}, within_mode: {args.within_mode}, seed: {args.seed}"
    )


if __name__ == "__main__":
    main()


