"""Prepare DAPO MATH-17k dataset samples with configurable generation options."""

import argparse
import asyncio
import json
import os
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import datasets
import httpx
import matplotlib.pyplot as plt
import openai
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

MATH_QUERY_TEMPLATE = """
{Question}
""".strip()


logger = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = {408, 409, 423, 425, 429, 499}


def calculate_backoff_delay(attempt: int, initial_delay: float, max_delay: float) -> float:
    if attempt <= 0:
        return 0.0
    initial_delay = max(0.0, initial_delay)
    max_delay = max(0.0, max_delay)
    delay = initial_delay * (2 ** (attempt - 1))
    if max_delay:
        delay = min(delay, max_delay)
    if delay <= 0.0:
        return 0.0
    jitter_multiplier = random.uniform(0.8, 1.2)
    return delay * jitter_multiplier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="open-r1/DAPO-Math-17k-Processed", help="Hugging Face dataset identifier")
    parser.add_argument("--dataset-split", default="train", help="Dataset split to sample")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of questions to sample")
    parser.add_argument("--samples-per-question", type=int, default=3, help="Number of generations per question")
    parser.add_argument("--max-concurrency", type=int, default=1000, help="Maximum concurrent generation requests")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct", help="Model name for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Maximum tokens per completion")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling value")
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:10001/v1", help="OpenAI compatible endpoint base URL")
    parser.add_argument("--openai-api-key", default=None, help="API key for the OpenAI compatible endpoint")
    parser.add_argument("--save-path", default=None, help="Output file path (default: data/dapo_math_17k_<num_samples>.jsonl)")
    parser.add_argument("--random-seed", type=int, default=42, help="Shuffle seed when sampling questions")
    parser.add_argument("--plot-difficulty", action="store_true", help="Plot histogram of question difficulties")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum number of retries for recoverable request errors")
    parser.add_argument("--retry-initial-delay", type=float, default=1.0, help="Initial delay in seconds before the first retry")
    parser.add_argument("--retry-max-delay", type=float, default=30.0, help="Maximum backoff delay in seconds between retries")
    parser.add_argument("--request-timeout", type=float, default=60.0, help="Per-request timeout in seconds for the OpenAI client")
    parser.add_argument("--save-batch-size", type=int, default=0, help="Save intermediate results every N successful completions (0 = only save at the end)")
    return parser.parse_args()


def build_prompt(question: str) -> str:
    return MATH_QUERY_TEMPLATE.format(Question=question)


def maybe_get(obj: Any, key: str, default: Optional[Any] = None) -> Optional[Any]:
    if obj is None:
        return default
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def load_existing_data(path: Path) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    if not path.exists():
        return [], {}
    
    data = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_idx, line in enumerate(file):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON at line %d in %s; skipping", line_idx + 1, path)
    except Exception as exc:
        logger.warning("Failed to read existing data from %s: %s", path, exc, exc_info=True)
        # We return what we have loaded so far instead of empty list
        # This prevents re-generating already saved valid data
    
    # Key is a normalized sample identifier (typically meta_info.extra_info["index"])
    counts: Dict[str, int] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        meta = maybe_get(entry, "meta_info")
        extra = maybe_get(meta, "extra_info")
        index = None
        if isinstance(extra, dict):
            index = extra.get("index")
        if index is None:
            raise ValueError(f"Found entry in existing data without an index: {entry}")

        # Normalize to string so that we can handle both int and UUID-like ids
        key = str(index)
        counts[key] = counts.get(key, 0) + 1
    return data, counts


async def generate_single_completion(
    client: openai.AsyncClient,
    sample: Dict[str, Any],
    prompt: str,
    *,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> Dict[str, Any]:
    message = [{"role": "user", "content": prompt}]
    response = await client.chat.completions.create(
        model=model_name,
        messages=message,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    answer = response.choices[0].message.content
    usage = getattr(response, "usage", None)
    completion_tokens = maybe_get(usage, "completion_tokens")
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
        "meta_info": {
            "answer_token_length": completion_tokens,
            "extra_info": sample.get("extra_info"),
            "reward_model": sample.get("reward_model")
        },
    }


async def collect_dataset(
    samples: Iterable[Tuple[int, Dict[str, Any]]],
    *,
    samples_per_question: int,
    max_concurrency: int,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    openai_base_url: str,
    openai_api_key: str,
    max_retries: int,
    retry_initial_delay: float,
    retry_max_delay: float,
    request_timeout: float,
    existing_counts: Dict[int, int],
    save_batch_size: int = 0,
    batch_callback: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    timeout = None
    if request_timeout and request_timeout > 0:
        timeout = httpx.Timeout(request_timeout)
    client = openai.AsyncClient(base_url=openai_base_url, api_key=openai_api_key, timeout=timeout)
    retryable_exceptions = tuple(
        exc_type
        for exc_type in (
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "APITimeoutError", None),
            getattr(openai, "RateLimitError", None),
            httpx.ReadTimeout,
        )
        if exc_type is not None
    )

    def should_retry_exception(exc: BaseException) -> bool:
        if isinstance(exc, retryable_exceptions):
            return True
        status_code = getattr(exc, "status_code", None)
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
        if status_code is None:
            return False
        if status_code in RETRYABLE_STATUS_CODES:
            return True
        return 500 <= status_code < 600

    max_retries_clamped = max(0, max_retries)

    async def generate_with_limit(sample_idx: str, sample: Dict[str, Any], prompt: str) -> Optional[Dict[str, Any]]:
        sample_extra = sample.get("extra_info") or {}
        sample_hint: Any = None
        if isinstance(sample_extra, dict):
            sample_hint = sample_extra.get("index")
        if sample_hint is None:
            fallback = sample.get("question")
            if isinstance(fallback, str):
                sample_hint = fallback[:80].strip()
            else:
                sample_hint = repr(fallback)
        async with semaphore:
            attempt = 0
            while True:
                try:
                    return await generate_single_completion(
                        client,
                        sample,
                        prompt,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    is_retryable = should_retry_exception(exc)
                    if attempt >= max_retries_clamped or not is_retryable:
                        logger.error(
                            "Giving up on sample %s after %d attempt(s): %s",
                            sample_hint,
                            attempt + 1,
                            exc,
                            exc_info=True,
                        )
                        return None
                    attempt += 1
                    delay = calculate_backoff_delay(attempt, retry_initial_delay, retry_max_delay)
                    status_code = getattr(exc, "status_code", None)
                    if isinstance(exc, httpx.HTTPStatusError):
                        status_code = exc.response.status_code
                    extra = f" status={status_code}" if status_code is not None else ""
                    logger.warning(
                        "Retrying completion after %s%s (attempt %d/%d) in %.2fs",
                        exc.__class__.__name__,
                        extra,
                        attempt,
                        max_retries_clamped,
                        delay,
                    )
                    if delay > 0.0:
                        await asyncio.sleep(delay)

    tasks: List[asyncio.Task] = []
    total_existing = 0
    total_needed = 0

    for sample in samples:
        sample_extra = sample.get("extra_info") or {}
        raw_idx = None
        if isinstance(sample_extra, dict):
            raw_idx = sample_extra.get("index")
        # If index is missing, fall back to the added "idx" column (if present)
        
        if raw_idx is None:
            raise ValueError(f"Sample missing both 'extra_info.index' and 'idx'. Sample keys: {list(sample.keys())}")

        # Ensure index is preserved in extra_info so it gets saved
        if sample.get("extra_info") is None:
            sample["extra_info"] = {}
        if isinstance(sample["extra_info"], dict) and sample["extra_info"].get("index") is None:
            sample["extra_info"]["index"] = raw_idx

        # Normalize to string so it can match what load_existing_data stored
        sample_idx = str(raw_idx)

        assigned = existing_counts.get(sample_idx, 0)
        missing_count = max(0, samples_per_question - assigned)

        total_existing += min(assigned, samples_per_question)
        total_needed += samples_per_question

        if missing_count <= 0:
            continue
        prompt = build_prompt(sample.get("source_prompt")[0].get("content"))
        for _ in range(missing_count):
            tasks.append(asyncio.create_task(generate_with_limit(sample_idx, sample, prompt)))
        
        # Update existing_counts to prevent over-generation if the dataset has duplicate indices
        existing_counts[sample_idx] = assigned + missing_count

    print(f"Dataset coverage: {total_existing} / {total_needed} ({total_existing/total_needed*100:.1f}%) samples found.")
    print(f"Queueing {len(tasks)} new generation tasks.")

    results: List[Optional[Dict[str, Any]]] = []
    batch: List[Dict[str, Any]] = []

    try:
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Generating completions",
        ):
            res = await task
            results.append(res)

            if res is not None and save_batch_size > 0 and batch_callback is not None:
                batch.append(res)
                if len(batch) >= save_batch_size:
                    try:
                        await batch_callback(batch)
                    except Exception:
                        logger.exception("Error while executing batch_callback; continuing.")
                    batch = []
    finally:
        # Cancel all running tasks immediately to allow quick exit on Ctrl+C or error
        for t in tasks:
            if not t.done():
                t.cancel()
        
        # Flush remaining partial batch, if any
        if batch and save_batch_size > 0 and batch_callback is not None:
            try:
                await batch_callback(batch)
            except Exception:
                logger.exception("Error while executing final batch_callback; continuing.")
        await client.close()

    successful_results = [item for item in results if item is not None]
    failed_count = len(results) - len(successful_results)
    if failed_count:
        logger.warning("Skipped %d sample(s) due to repeated request failures", failed_count)
    return successful_results


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def prepare_dataset_with_idx(
    dataset_split: datasets.Dataset, *, num_samples: int, random_seed: int
) -> datasets.Dataset:
    if "idx" not in dataset_split.column_names:
        dataset_split = dataset_split.add_column("idx", list(range(len(dataset_split))))
    dataset_split = dataset_split.shuffle(seed=random_seed)
    if 0 < num_samples < len(dataset_split):
        dataset_split = dataset_split.select(range(num_samples))
    return dataset_split


# def iter_samples_with_idx(dataset_split: datasets.Dataset) -> Iterable[Tuple[int, Dict[str, Any]]]:
#     for sample in dataset_split:
#         raw_idx = maybe_get(sample, "idx")
#         if raw_idx is None:
#             raise ValueError("Sample is missing required 'idx' column")
#         try:
#             idx = int(raw_idx)
#         except (TypeError, ValueError) as exc:
#             raise ValueError(f"Sample 'idx' value {raw_idx!r} is not convertible to int") from exc
#         normalized_sample = dict(sample)
#         normalized_sample["idx"] = idx
#         yield idx, normalized_sample


# def plot_difficulty_distribution(difficulties: List[float]) -> None:
#     if not difficulties:
#         return
#     bins = [i * 0.5 for i in range(int(min(difficulties) * 2), int(max(difficulties) * 2) + 2)]
#     plt.hist(difficulties, bins=bins, edgecolor="black")
#     plt.title("Distribution of Difficulty")
#     plt.xlabel("Difficulty")
#     plt.ylabel("Frequency")
#     plt.show()


def resolve_save_path(save_path: Optional[str], num_samples: int) -> Path:
    if save_path:
        return Path(save_path).expanduser().resolve()
    default_dir = Path(__file__).resolve().parent.parent / "data"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir / f"dapo_math_17k_{num_samples}.jsonl"


def main() -> None:
    args = parse_args()

    dataset = datasets.load_dataset(args.dataset_name)

    dataset_split = dataset[args.dataset_split]
    print(f"Length of dataset_split: {len(dataset_split)}")
    if args.num_samples > len(dataset_split):
        print(f"Warning: num_samples is greater than the length of the dataset, setting num_samples to {len(dataset_split)}")
        args.num_samples = len(dataset_split)

    prepared_dataset = prepare_dataset_with_idx(
        dataset_split,
        num_samples=args.num_samples,
        random_seed=args.random_seed,
    )

    # if args.plot_difficulty and "difficulty" in prepared_dataset.column_names:
    #     plot_difficulty_distribution([
    #         difficulty
    #         for difficulty in prepared_dataset["difficulty"]
    #         if isinstance(difficulty, (int, float))
    #     ])

    output_path = resolve_save_path(args.save_path, args.num_samples)

    # Save sampled indices
    indices_path = output_path.with_name(output_path.stem + "_indices.json")
    try:
        # Extract 'idx' column which is guaranteed to exist by prepare_dataset_with_idx
        sampled_indices = list(prepared_dataset["idx"])
        ensure_parent_dir(indices_path)
        with indices_path.open("w", encoding="utf-8") as f:
            json.dump(sampled_indices, f)
        print(f"Saved sampled indices to {indices_path}")
    except Exception as exc:
        logger.warning("Failed to save sampled indices to %s: %s", indices_path, exc)

    existing_data, existing_counts = load_existing_data(output_path)

    # For incremental saving, we accumulate new data and re-write the merged file
    # new_data: List[Dict[str, Any]] = []

    async def batch_callback(batch: List[Dict[str, Any]]) -> None:
        """
        Synchronously persist intermediate results using append-only JSONL:
        - Write batch to output_path immediately
        """
        if not batch:
            return
        ensure_parent_dir(output_path)
        
        # Append to the file (thread-safe for single writer, atomic enough for lines)
        try:
            with output_path.open("a", encoding="utf-8") as file:
                for item in batch:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Error appending batch to %s", output_path)

    data = asyncio.run(
        collect_dataset(
            prepared_dataset,
            samples_per_question=args.samples_per_question,
            max_concurrency=args.max_concurrency,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            openai_base_url=args.openai_base_url,
            openai_api_key=args.openai_api_key,
            max_retries=args.max_retries,
            retry_initial_delay=args.retry_initial_delay,
            retry_max_delay=args.retry_max_delay,
            request_timeout=args.request_timeout,
            existing_counts=existing_counts,
            save_batch_size=max(0, args.save_batch_size),
            batch_callback=batch_callback if args.save_batch_size > 0 else None,
        )
    )

    # Final save for any remaining data that wasn't covered by batch callback
    # (collect_dataset calls batch_callback internally for the final batch, 
    # so we usually don't need to do anything else if save_batch_size > 0)
    
    # If save_batch_size == 0, we haven't saved anything yet.
    if args.save_batch_size <= 0 and data:
        asyncio.run(batch_callback(data))
    
    # ensure_parent_dir(output_path)
    # merged_data = existing_data + data
    # with output_path.open("w", encoding="utf-8") as file:
    #    json.dump(merged_data, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()

