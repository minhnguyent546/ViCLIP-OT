import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm


def _extract_valid_records_from_shard(
    shards_dir: str,
    shard_file: str,
) -> tuple[list[str], list[str], list[str], list[str], int]:
    shard_name = os.path.splitext(shard_file)[0]
    df = pd.read_parquet(os.path.join(shards_dir, shard_file))

    captions: list[str] = []
    image_paths: list[str] = []
    uids: list[str] = []
    keys: list[str] = []
    missing_images = 0

    for _, row in df.iterrows():
        if row["status"] != "success":
            continue

        caption = row["caption"]
        image_name = row["key"]
        if pd.isna(caption) or pd.isna(image_name) or pd.isna(row["uid"]):
            continue

        image_path = os.path.join(shards_dir, shard_name, f"{image_name}.jpg")
        if not os.path.isfile(image_path):
            missing_images += 1
            continue

        captions.append(caption)
        image_paths.append(image_path)
        uids.append(row["uid"])
        keys.append(row["key"])

    return captions, image_paths, uids, keys, missing_images


def compute_mclip_score(args: argparse.Namespace) -> None:
    # Initialize the model
    model_kwargs = {}
    if args.use_flash_attn:
        model_kwargs["use_flash_attention"] = True

    image_model = SentenceTransformer("clip-ViT-B-32", model_kwargs=model_kwargs)
    text_model = SentenceTransformer(
        "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        model_kwargs=model_kwargs,
    )

    shard_files = [f for f in os.listdir(args.shards_dir) if f.endswith(".parquet")]
    shard_files = sorted(shard_files)
    logger.info(f"Found {len(shard_files)} shard files in {args.shards_dir}")

    if args.score_output_dir is None:
        score_dir = os.path.join(os.path.dirname(args.shards_dir), "scores")
    else:
        score_dir = args.score_output_dir
    os.makedirs(score_dir, exist_ok=True)

    all_captions: list[str] = []
    all_image_paths: list[str] = []
    all_uids: list[str] = []
    all_keys: list[str] = []
    missing_image_rows = 0
    if args.num_workers > 1 and len(shard_files) > 1:
        logger.info(f"Using multiprocessing for shard extraction with {args.num_workers} workers")
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            extraction_results = list(
                tqdm(
                    executor.map(
                        _extract_valid_records_from_shard,
                        [args.shards_dir] * len(shard_files),
                        shard_files,
                    ),
                    total=len(shard_files),
                    desc=f"Extracting shards (num_workers={args.num_workers})",
                )
            )
    else:
        extraction_results = [
            _extract_valid_records_from_shard(args.shards_dir, shard_file)
            for shard_file in tqdm(shard_files, desc="Extracting shards")
        ]

    for shard_file, (captions, image_paths, uids, keys, shard_missing_images) in zip(
        shard_files,
        extraction_results,
        strict=True,
    ):
        assert (
            len(all_captions) == len(all_image_paths)
            and len(captions) == len(uids)
            and len(captions) == len(keys)
        )
        missing_image_rows += shard_missing_images

        if not captions:
            logger.warning(f"No valid records found in shard {shard_file}. Skipping this shard.")
            continue

        all_captions.extend(captions)
        all_image_paths.extend(image_paths)
        all_uids.extend(uids)
        all_keys.extend(keys)

    logger.info(f"Found total {len(all_captions)} valid records across all shards for processing.")

    _permutation = np.argsort([-len(caption) for caption in all_captions])
    _inverse_permutation = np.argsort(_permutation)
    all_captions = [all_captions[idx] for idx in _permutation]

    caption_embedding_batches: list[torch.Tensor] = []
    image_embedding_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in tqdm(
            range(0, len(all_captions), args.batch_size),
            desc="Computing caption embeddings",
        ):
            batch_caption = all_captions[i : i + args.batch_size]
            batch_caption_embeddings = text_model.encode(
                sentences=batch_caption,
                normalize_embeddings=True,
                convert_to_tensor=True,
            )
            caption_embedding_batches.append(batch_caption_embeddings.cpu())
            del batch_caption_embeddings

        caption_embeddings = torch.cat(caption_embedding_batches, dim=0)
        inverse_permutation_tensor = torch.from_numpy(_inverse_permutation).long()
        caption_embeddings = caption_embeddings.index_select(0, inverse_permutation_tensor)
        del caption_embedding_batches

        for i in tqdm(
            range(0, len(all_image_paths), args.batch_size),
            desc="Computing image embeddings",
        ):
            batch_image = all_image_paths[i : i + args.batch_size]

            batch_image_pils = []
            for image_path in batch_image:
                try:
                    image = Image.open(image_path)
                    # handle palette images with transparency
                    if image.mode == "P" and "transparency" in image.info:
                        image = image.convert("RGBA")

                    image = image.convert("RGB")

                    batch_image_pils.append(image)
                except Exception as e:
                    logger.error(f"Error loading image {image_path}: {e}")
                    raise e

            batch_image_embeddings = image_model.encode(
                batch_image_pils,
                normalize_embeddings=True,
                convert_to_tensor=True,
            )

            del batch_image_pils  # free up memory

            image_embedding_batches.append(batch_image_embeddings.cpu())

            del batch_image_embeddings  # free up memory

        image_embeddings = torch.cat(image_embedding_batches, dim=0)
        del image_embedding_batches

    assert caption_embeddings is not None
    assert image_embeddings is not None
    assert len(caption_embeddings) == len(image_embeddings), (
        "Caption and image embedding counts do not match"
    )

    # Compute cosine similarity for each pair in chunks to reduce peak memory
    similarity_batch_size = max(args.batch_size * 8, 1)
    similarity_chunks: list[np.ndarray] = []
    for i in tqdm(
        range(0, len(caption_embeddings), similarity_batch_size),
        desc="Computing cosine similarities",
    ):
        caption_chunk = caption_embeddings[i : i + similarity_batch_size]
        image_chunk = image_embeddings[i : i + similarity_batch_size]
        similarity_chunk = torch.einsum("bd,bd->b", caption_chunk, image_chunk)
        similarity_chunks.append(similarity_chunk.cpu().numpy())

    cosine_similarities = np.concatenate(similarity_chunks, axis=0)
    scores_df = pd.DataFrame(
        {
            "uid": all_uids,
            "key": all_keys,
            "mclip_score": cosine_similarities,
        }
    )
    score_file = os.path.join(score_dir, "scores.parquet")
    scores_df.to_parquet(score_file, index=False)
    logger.info(f"Saved scores for {len(scores_df)} records to {score_file}")
    logger.info(f"Total failed rows due to missing images: {missing_image_rows}")


def add_opts(parser: argparse.ArgumentParser) -> None:
    cpu_count = os.cpu_count()
    default_num_workers = max(cpu_count // 2, 1) if cpu_count is not None else 1

    parser.add_argument(
        "--shards_dir",
        type=str,
        help="Path to the directory containing the parquet shards.",
        default="./data/datacomp_large_vie_filtered2/shards",
    )
    parser.add_argument(
        "--score_output_dir",
        type=str,
        help="Path to the directory where the scored shards will be saved",
        default="./data/datacomp_large_vie_filtered2/scores",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for encoding.",
        default=128,
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use flash attention for encoding.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        help="Number of records per shard for saving intermediate scores.",
        default=50_000,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of worker processes for shard extraction. Use 1 to disable multiprocessing.",
        default=default_num_workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute M-CLIP score for a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_opts(parser)
    args = parser.parse_args()

    compute_mclip_score(args)


if __name__ == "__main__":
    main()
