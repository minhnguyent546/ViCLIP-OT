#!/usr/bin/env python

"""Adapted from: https://github.com/huggingface/large-scale-image-deduplication/blob/main/dedup_dataset.py"""

import argparse
import glob
import json
import os
import time

import numpy as np
from compute_embeddings import compute_embeddings
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity


class Timer:
    """Context manager for timing operations."""

    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        assert self.start_time is not None
        elapsed = time.time() - self.start_time
        self.elapsed = elapsed


def find_duplicates_against_precomputed(
    new_embeddings_file: str,
    new_image_ids_file: str,
    precomputed_dir: str,
    batch_size: int = 32,
    threshold: float = 0.9,
):
    """Find duplicates by comparing new embeddings against all precomputed embeddings."""
    # Load new dataset embeddings
    new_embeddings = np.load(new_embeddings_file)
    new_image_ids = np.load(new_image_ids_file)

    logger.info(f"Loaded {len(new_embeddings)} new embeddings")

    # Get all precomputed embedding files
    precomputed_files = glob.glob(os.path.join(precomputed_dir, "*_embeddings.npy"))
    logger.info(f"Found {len(precomputed_files)} precomputed embedding files")

    duplicate_indices = set()
    duplicate_details = []
    loading_time = 0.0
    similarity_time = 0.0

    # Compare against each precomputed file
    for i, precomputed_file in enumerate(precomputed_files):
        logger.info(
            f"Comparing against precomputed file {i + 1}/{len(precomputed_files)}: {os.path.basename(precomputed_file)}"
        )

        # Time loading of precomputed embeddings and image_ids
        with Timer("Loading precomputed embeddings") as load_timer:
            precomputed_embeddings = np.load(precomputed_file)
            # Load corresponding image_ids file
            precomputed_image_ids_file = precomputed_file.replace(
                "_embeddings.npy", "_image_ids.npy"
            )
            precomputed_image_ids = np.load(precomputed_image_ids_file)
        loading_time += load_timer.elapsed

        # Process new embeddings in batches against this precomputed file
        for batch_start in range(0, len(new_embeddings), batch_size):
            batch_end = min(batch_start + batch_size, len(new_embeddings))
            batch_embeddings = new_embeddings[batch_start:batch_end]

            # Time similarity computation
            with Timer("Computing similarities") as sim_timer:
                similarities = cosine_similarity(batch_embeddings, precomputed_embeddings)
                # Find duplicates above threshold
                batch_indices, precomputed_indices = np.where(similarities >= threshold)
            similarity_time += sim_timer.elapsed

            # Record duplicates
            for batch_idx, precomputed_idx in zip(batch_indices, precomputed_indices, strict=True):
                global_idx = batch_start + batch_idx
                duplicate_indices.add(global_idx)

                duplicate_details.append(
                    {
                        "new_idx": int(global_idx),
                        "new_image_id": int(new_image_ids[global_idx]),
                        "source_file": os.path.basename(precomputed_file),
                        "source_idx": int(precomputed_idx),
                        "source_image_id": int(precomputed_image_ids[precomputed_idx]),
                        "similarity": float(similarities[batch_idx, precomputed_idx]),
                    }
                )

    return sorted(duplicate_indices), duplicate_details, loading_time, similarity_time


def deduplicate_dataset(
    dataset_dir: str,
    precomputed_dir: str,
    threshold: float = 0.9,
    output_dir: str = "duplicates",
    device="auto",
    batch_size: int = 32,
    split_name: str = "test",
):
    """Deduplicate a dataset against precomputed embeddings with timing information."""

    with Timer("Total execution time") as total_timer:
        # Step 1: Compute embeddings for new dataset
        logger.info("Step 1: Computing embeddings for new dataset...")
        tmp_embedding_dir = "embeddings-tmp"
        with Timer("Computing embeddings") as embedding_timer:
            compute_embeddings(
                dataset_dir=dataset_dir,
                output_dir=tmp_embedding_dir,
                device=device,
                batch_size=batch_size,
                split_name=split_name,
            )

        # Step 2: Find duplicate files
        save_file_name = os.path.basename(tmp_embedding_dir.rstrip("/"))
        new_embeddings_file = os.path.join(tmp_embedding_dir, f"{save_file_name}_embeddings.npy")
        new_image_ids_file = os.path.join(tmp_embedding_dir, f"{save_file_name}_image_ids.npy")

        # Step 3: Find duplicates
        logger.info("Step 2: Finding duplicates against precomputed embeddings...")
        with Timer("Duplicate detection") as duplicate_timer:
            duplicate_indices, duplicate_details, loading_time, similarity_time = (
                find_duplicates_against_precomputed(
                    new_embeddings_file, new_image_ids_file, precomputed_dir, threshold=threshold
                )
            )

    # Step 4: Save results (outside timer context to access elapsed times)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"duplicates_{save_file_name}.json")

    new_image_ids = np.load(new_image_ids_file)
    total_images = int(len(new_image_ids))
    duplicate_image_ids = [int(new_image_ids[idx]) for idx in duplicate_indices]
    results = {
        "dataset_dir": dataset_dir,
        "threshold": float(threshold),
        "total_images": total_images,
        "timing": {
            "total_time": total_timer.elapsed,
            "embedding_computation": embedding_timer.elapsed,
            "duplicate_detection": duplicate_timer.elapsed,
            "loading_precomputed": loading_time,
            "similarity_search": similarity_time,
        },
        "duplicate_count": len(duplicate_indices),
        "duplicate_indices": [int(idx) for idx in duplicate_indices],
        "duplicate_image_ids": duplicate_image_ids,
        "duplicate_details": duplicate_details,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print timing summary
    logger.info("\nTiming Summary:")
    logger.info(f"Total execution: {total_timer.elapsed:.5f} seconds")
    logger.info(
        f"Embedding computation: {embedding_timer.elapsed:.5f} seconds ({embedding_timer.elapsed / total_timer.elapsed * 100:.1f}%)"
    )
    logger.info(
        f"Duplicate detection: {duplicate_timer.elapsed:.5f} seconds ({duplicate_timer.elapsed / total_timer.elapsed * 100:.1f}%)"
    )
    logger.info(
        f"  - Loading precomputed: {loading_time:.5f} seconds ({loading_time / total_timer.elapsed * 100:.1f}%)"
    )
    logger.info(
        f"  - Similarity search: {similarity_time:.5f} seconds ({similarity_time / total_timer.elapsed * 100:.1f}%)"
    )
    logger.info(f"\nFound {len(duplicate_indices)} duplicate images out of {total_images}")
    logger.info(f"Duplicate results saved to: {output_file}")

    return duplicate_indices, duplicate_details


def add_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--precomputed_dir",
        type=str,
        required=True,
        help="Directory containing precomputed embeddings",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for duplicate detection",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="duplicates",
        help="Output directory for duplicate results",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for computation",
        choices=["cpu", "cuda", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        help="Dataset split to process",
        default="train",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate dataset against precomputed embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_opts(parser)
    args = parser.parse_args()

    deduplicate_dataset(
        dataset_dir=args.dataset_dir,
        precomputed_dir=args.precomputed_dir,
        threshold=args.threshold,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        split_name=args.split_name,
    )


if __name__ == "__main__":
    main()
