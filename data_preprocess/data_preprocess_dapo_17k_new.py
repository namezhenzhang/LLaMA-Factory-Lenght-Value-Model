"""Prepare DAPO MATH-17k dataset samples with configurable generation options."""

import argparse
import asyncio
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import datasets
import httpx
import openai
from tqdm.asyncio import tqdm_asyncio

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.WARNING,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants
MATH_QUERY_TEMPLATE = "{Question}"
RETRYABLE_STATUS_CODES = {408, 409, 423, 425, 429, 499}


@dataclass
class GenerationConfig:
    """Configuration for text generation and API requests."""
    model_name: str
    temperature: float
    max_tokens: int
    top_p: float
    openai_base_url: str
    openai_api_key: Optional[str]
    max_retries: int
    retry_initial_delay: float
    retry_max_delay: float
    request_timeout: float


def calculate_backoff_delay(attempt: int, initial_delay: float, max_delay: float) -> float:
    """Calculates exponential backoff delay with jitter."""
    if attempt <= 0:
        return 0.0
    delay = initial_delay * (2 ** (attempt - 1))
    if max_delay:
        delay = min(delay, max_delay)
    return delay * random.uniform(0.8, 1.2)


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get value from object or dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class DataGenerator:
    """Handles async generation of completions using OpenAI-compatible API."""
    
    def __init__(self, config: GenerationConfig, max_concurrency: int):
        self.config = config
        self.semaphore = asyncio.Semaphore(max(1, max_concurrency))
        
        timeout = httpx.Timeout(config.request_timeout) if config.request_timeout > 0 else None
        self.client = openai.AsyncClient(
            base_url=config.openai_base_url,
            api_key=config.openai_api_key,
            timeout=timeout,
        )

    async def close(self):
        """Closes the underlying HTTP client."""
        await self.client.close()

    def _should_retry(self, exc: Exception) -> bool:
        """Determines if the exception is retryable."""
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, httpx.ReadTimeout)):
            return True
        
        status_code = getattr(exc, "status_code", None)
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
        
        if status_code in RETRYABLE_STATUS_CODES:
            return True
        return 500 <= (status_code or 0) < 600

    async def process_sample(self, sample: Dict[str, Any], prompt: str) -> Optional[Dict[str, Any]]:
        """Generates a completion for a single sample with retry logic."""
        async with self.semaphore:
            for attempt in range(self.config.max_retries + 1):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    response = await self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        top_p=self.config.top_p,
                    )
                    
                    answer = response.choices[0].message.content
                    completion_tokens = getattr(response.usage, "completion_tokens", None) if response.usage else None
                    
                    return {
                        "conversations": [
                            {"from": "human", "value": prompt},
                            {"from": "gpt", "value": answer},
                        ],
                        "meta_info": {
                            "answer_token_length": completion_tokens,
                            "extra_info": sample.get("extra_info"),
                            "reward_model": sample.get("reward_model"),
                        },
                    }
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if attempt >= self.config.max_retries or not self._should_retry(exc):
                        logger.error(f"Failed to process sample after {attempt+1} attempts: {exc}")
                        return None
                    
                    delay = calculate_backoff_delay(attempt + 1, self.config.retry_initial_delay, self.config.retry_max_delay)
                    logger.warning(f"Retry {attempt+1}/{self.config.max_retries} in {delay:.2f}s due to {type(exc).__name__}")
                    await asyncio.sleep(delay)
            return None


def load_existing_data(path: Path) -> Dict[str, int]:
    """Scans existing output file to count processed samples."""
    counts: Dict[str, int] = {}
    if not path.exists():
        return counts

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Extract index from deep structure
                    meta = safe_get(entry, "meta_info")
                    extra = safe_get(meta, "extra_info")
                    idx = safe_get(extra, "index")
                    
                    if idx is not None:
                        key = str(idx)
                        counts[key] = counts.get(key, 0) + 1
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        logger.warning(f"Error reading existing data from {path}: {exc}")
    
    return counts


async def save_batch(path: Path, batch: List[Dict[str, Any]]):
    """Appends a batch of results to the output file."""
    if not batch:
        return
    
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as f:
            for item in batch:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error(f"Failed to save batch to {path}: {exc}")


async def process_dataset(
    generator: DataGenerator,
    samples: List[Dict[str, Any]],
    samples_per_question: int,
    existing_counts: Dict[str, int],
    save_path: Path,
    save_batch_size: int = 10,
):
    """Main processing loop: schedules tasks and saves results."""
    tasks = []
    
    # Create tasks only for missing samples
    for sample in samples:
        # We ensure 'index' exists in extra_info during preprocessing
        idx = str(sample.get("extra_info", {}).get("index"))
        if not idx:
             continue
             
        current_count = existing_counts.get(idx, 0)
        needed = max(0, samples_per_question - current_count)
        
        if needed > 0:
            prompt = MATH_QUERY_TEMPLATE.format(Question=sample["source_prompt"][0]["content"])
            for _ in range(needed):
                tasks.append(generator.process_sample(sample, prompt))
                # Update local count to avoid over-scheduling if duplicates exist in source
                existing_counts[idx] = existing_counts.get(idx, 0) + 1

    total_needed = len(samples) * samples_per_question
    already_done = total_needed - len(tasks)
    print(f"Dataset coverage: {already_done} / {total_needed} ({(already_done/total_needed*100) if total_needed else 0:.1f}%) found.")
    print(f"Queueing {len(tasks)} new generation tasks.")

    if not tasks:
        return []

    results = []
    batch = []
    
    # Execute tasks
    for future in tqdm_asyncio.as_completed(tasks, desc="Generating"):
        result = await future
        if result:
            results.append(result)
            batch.append(result)
            
            if save_batch_size > 0 and len(batch) >= save_batch_size:
                await save_batch(save_path, batch)
                batch = []
    
    # Save remaining items
    if batch and save_batch_size > 0:
        await save_batch(save_path, batch)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAPO Math Data Generation")
    
    # Dataset args
    parser.add_argument("--dataset-name", default="open-r1/DAPO-Math-17k-Processed")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--samples-per-question", type=int, default=3)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--save-batch-size", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--plot-difficulty", action="store_true", help="Unused but preserved for compatibility")

    # Model/API args
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:10001/v1")
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--max-concurrency", type=int, default=1000)
    
    # Reliability args
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-initial-delay", type=float, default=1.0)
    parser.add_argument("--retry-max-delay", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Setup paths
    save_path = Path(args.save_path) if args.save_path else \
        Path(__file__).resolve().parent.parent / "data" / f"dapo_math_17k_{args.num_samples}.jsonl"
    
    # 2. Load and Preprocess Dataset
    try:
        ds = datasets.load_dataset(args.dataset_name, split=args.dataset_split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Add 'idx' if missing
    if "idx" not in ds.column_names:
        ds = ds.add_column("idx", list(range(len(ds))))
        
    # Shuffle and select samples
    ds = ds.shuffle(seed=args.random_seed)
    if 0 < args.num_samples < len(ds):
        ds = ds.select(range(args.num_samples))
    elif args.num_samples > len(ds):
        logger.warning(f"Requested {args.num_samples} samples but dataset only has {len(ds)}.")
        
    # Ensure extra_info.index exists for tracking
    def ensure_index(example):
        # Handle dict/obj differences if any (though mapped example is dict)
        extra = example.get("extra_info") or {}
        if "index" not in extra:
            extra["index"] = example["idx"]
        example["extra_info"] = extra
        return example
        
    ds = ds.map(ensure_index)
    
    # 3. Save Indices
    indices_path = save_path.with_name(f"{save_path.stem}_indices.json")
    indices_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with indices_path.open("w") as f:
            json.dump(list(ds["idx"]), f)
        logger.info(f"Saved indices to {indices_path}")
    except Exception as e:
        logger.error(f"Failed to save indices: {e}")
    
    # 4. Initialize Generation
    existing_counts = load_existing_data(save_path)
    
    config = GenerationConfig(
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
    )
    
    # 5. Run
    generator = DataGenerator(config, args.max_concurrency)
    try:
        asyncio.run(process_dataset(
            generator, 
            ds, 
            args.samples_per_question, 
            existing_counts, 
            save_path, 
            args.save_batch_size
        ))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    finally:
        asyncio.run(generator.close())


if __name__ == "__main__":
    main()
