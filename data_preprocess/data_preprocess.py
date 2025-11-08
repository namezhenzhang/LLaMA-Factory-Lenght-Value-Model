"""Prepare DeepMath samples with configurable generation options."""

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import datasets
import httpx
import matplotlib.pyplot as plt
import openai
from tqdm.asyncio import tqdm_asyncio

MATH_QUERY_TEMPLATE = """
Please reason step by step, and put your final answer within \\boxed{{}}.

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
    parser.add_argument("--dataset-name", default="zwhe99/DeepMath-103K", help="Hugging Face dataset identifier")
    parser.add_argument("--dataset-split", default="train", help="Dataset split to sample")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of questions to sample")
    parser.add_argument("--samples-per-question", type=int, default=3, help="Number of generations per question")
    parser.add_argument("--max-concurrency", type=int, default=1000, help="Maximum concurrent generation requests")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507", help="Model name for generation")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Maximum tokens per completion")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling value")
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:10001/v1", help="OpenAI compatible endpoint base URL")
    parser.add_argument("--openai-api-key", default=None, help="API key for the OpenAI compatible endpoint")
    parser.add_argument("--save-path", default=None, help="Output file path (default: data/deepmath_<num_samples>.json)")
    parser.add_argument("--random-seed", type=int, default=42, help="Shuffle seed when sampling questions")
    parser.add_argument("--plot-difficulty", action="store_true", help="Plot histogram of question difficulties")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum number of retries for recoverable request errors")
    parser.add_argument("--retry-initial-delay", type=float, default=1.0, help="Initial delay in seconds before the first retry")
    parser.add_argument("--retry-max-delay", type=float, default=30.0, help="Maximum backoff delay in seconds between retries")
    parser.add_argument("--request-timeout", type=float, default=60.0, help="Per-request timeout in seconds for the OpenAI client")
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
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as exc:
        logger.warning("Failed to load existing data from %s: %s", path, exc, exc_info=True)
        return [], {}

    if not isinstance(data, list):
        logger.warning("Existing data file %s does not contain a list; ignoring its contents", path)
        return [], {}

    counts: Dict[int, int] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        meta = maybe_get(entry, "meta_info")
        idx = maybe_get(meta, "idx")
        if isinstance(idx, int):
            counts[idx] = counts.get(idx, 0) + 1
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
            "difficulty": sample.get("difficulty"),
            "answer_token_length": completion_tokens,
            "idx": sample.get("idx"),
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

    async def generate_with_limit(sample_idx: int, sample: Dict[str, Any], prompt: str) -> Optional[Dict[str, Any]]:
        sample_hint: Any = sample.get("idx")
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
    for sample_idx, sample in samples:
        assigned = existing_counts.get(sample_idx, 0)
        missing_count = max(0, samples_per_question - assigned)
        if missing_count <= 0:
            continue
        prompt = build_prompt(sample.get("question", ""))
        for _ in range(missing_count):
            tasks.append(asyncio.create_task(generate_with_limit(sample_idx, sample, prompt)))

    try:
        results: List[Optional[Dict[str, Any]]] = await tqdm_asyncio.gather(
            *tasks, total=len(tasks), desc="Generating completions"
        )
    finally:
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
    if 0 < num_samples < len(dataset_split):
        dataset_split = dataset_split.shuffle(seed=random_seed).select(range(num_samples))
    return dataset_split


def iter_samples_with_idx(dataset_split: datasets.Dataset) -> Iterable[Tuple[int, Dict[str, Any]]]:
    for sample in dataset_split:
        raw_idx = maybe_get(sample, "idx")
        if raw_idx is None:
            raise ValueError("Sample is missing required 'idx' column")
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Sample 'idx' value {raw_idx!r} is not convertible to int") from exc
        normalized_sample = dict(sample)
        normalized_sample["idx"] = idx
        yield idx, normalized_sample


def plot_difficulty_distribution(difficulties: List[float]) -> None:
    if not difficulties:
        return
    bins = [i * 0.5 for i in range(int(min(difficulties) * 2), int(max(difficulties) * 2) + 2)]
    plt.hist(difficulties, bins=bins, edgecolor="black")
    plt.title("Distribution of Difficulty")
    plt.xlabel("Difficulty")
    plt.ylabel("Frequency")
    plt.show()


def resolve_save_path(save_path: Optional[str], num_samples: int) -> Path:
    if save_path:
        return Path(save_path).expanduser().resolve()
    default_dir = Path(__file__).resolve().parent.parent / "data"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir / f"deepmath_{num_samples}.json"


def main() -> None:
    args = parse_args()

    dataset = datasets.load_dataset(args.dataset_name)
    dataset_split = dataset[args.dataset_split]
    prepared_dataset = prepare_dataset_with_idx(
        dataset_split,
        num_samples=args.num_samples,
        random_seed=args.random_seed,
    )

    if args.plot_difficulty and "difficulty" in prepared_dataset.column_names:
        plot_difficulty_distribution([
            difficulty
            for difficulty in prepared_dataset["difficulty"]
            if isinstance(difficulty, (int, float))
        ])

    output_path = resolve_save_path(args.save_path, args.num_samples)
    existing_data, existing_counts = load_existing_data(output_path)

    data = asyncio.run(
        collect_dataset(
            iter_samples_with_idx(prepared_dataset),
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
        )
    )

    ensure_parent_dir(output_path)
    merged_data = existing_data + data
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(merged_data, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
