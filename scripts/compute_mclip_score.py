import argparse
import os

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm


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

    score_df = pd.DataFrame(columns=["uid", "key", "mclip_score"])

    if args.score_output_dir is None:
        score_dir = os.path.join(os.path.dirname(args.shards_dir), "scores")
    else:
        score_dir = args.score_output_dir
    os.makedirs(score_dir, exist_ok=True)

    missing_image_row = 0
    for shard_file in shard_files:
        shard_name = os.path.splitext(shard_file)[0]
        logger.info(f"Processing shard file: {shard_file}")
        df = pd.read_parquet(os.path.join(args.shards_dir, shard_file))

        captions = []
        image_paths = []
        row_indices = []
        uids = []
        keys = []
        for idx, row in df.iterrows():
            if row["status"] != "success":
                continue

            caption = row["caption"]
            image_name = row["key"]
            if pd.isna(caption) or pd.isna(image_name) or pd.isna(row["uid"]):
                continue

            image_path = os.path.join(args.shards_dir, shard_name, f"{image_name}.jpg")
            if not os.path.isfile(image_path):
                logger.warning(f"Image file {image_path} does not exist. Skipping this record.")
                missing_image_row += 1
                continue

            captions.append(caption)
            image_paths.append(image_path)
            row_indices.append(idx)
            uids.append(row["uid"])
            keys.append(row["key"])

        assert len(captions) == len(image_paths)

        if not captions:
            logger.warning(f"No valid records found in shard {shard_file}. Skipping this shard.")
            continue

        _permutation = np.argsort([-len(caption) for caption in captions])
        _inverse_permutation = np.argsort(_permutation)
        captions = [captions[idx] for idx in _permutation]

        image_embeddings = None
        caption_embeddings = None
        with torch.inference_mode():
            for i in tqdm(
                range(0, len(captions), args.batch_size),
                desc="Computing caption embeddings",
            ):
                batch_caption = captions[i : i + args.batch_size]
                batch_caption_embeddings = text_model.encode(
                    sentences=batch_caption,
                    normalize_embeddings=True,
                    convert_to_tensor=True,
                )
                if i == 0:
                    caption_embeddings = batch_caption_embeddings
                else:
                    caption_embeddings = torch.cat(
                        (caption_embeddings, batch_caption_embeddings),  # pyright: ignore
                        dim=0,
                    )

            assert caption_embeddings is not None
            caption_embeddings = torch.stack(
                [caption_embeddings[idx] for idx in _inverse_permutation]
            )

            for i in tqdm(
                range(0, len(image_paths), args.batch_size),
                desc="Computing image embeddings",
            ):
                batch_image = image_paths[i : i + args.batch_size]

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

                # expand image embeddings according to caption counts
                if i == 0:
                    image_embeddings = batch_image_embeddings
                else:
                    image_embeddings = torch.cat(
                        (image_embeddings, batch_image_embeddings),  # pyright: ignore
                        dim=0,
                    )

                del batch_image_embeddings  # free up memory

        # Compute cosine similarity for each pair
        cosine_similarities = (caption_embeddings * image_embeddings).sum(dim=1).cpu().numpy()  # pyright: ignore[reportOperatorIssue]
        shard_score_df = pd.DataFrame(
            {
                "uid": uids,
                "key": keys,
                "mclip_score": cosine_similarities,
            }
        )
        score_df = pd.concat([score_df, shard_score_df], ignore_index=True)

        if len(score_df) >= args.shard_size:
            score_file = os.path.join(score_dir, f"{shard_name}_scores.parquet")
            score_df.to_parquet(score_file, index=False)
            logger.info(f"Saved scores for {len(score_df)} records to {score_file}")
            score_df = pd.DataFrame(columns=["uid", "key", "mclip_score"])

    logger.info(
        f"Finished processing all shards. Total failed rows due to missing images: {missing_image_row}"
    )


def add_opts(parser: argparse.ArgumentParser) -> None:
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
