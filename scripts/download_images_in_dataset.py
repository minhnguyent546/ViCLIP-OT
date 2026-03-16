import argparse
import os
import shutil

import img2dataset
from huggingface_hub import snapshot_download
from loguru import logger


def download_image(args: argparse.Namespace) -> None:
    metadata_dir = args.metadata_dir
    if metadata_dir is None:
        metadata_dir = os.path.join(args.data_dir, "metadata")

    # Download the metadata files if needed.
    if args.overwrite_metadata or not os.path.isdir(metadata_dir):
        if os.path.isdir(metadata_dir):
            logger.info(f"Cleaning up {metadata_dir}")
            shutil.rmtree(metadata_dir)
        os.makedirs(metadata_dir, exist_ok=True)

        logger.info(f"Downloading metadata to {metadata_dir}...")

        cache_dir = os.path.join(metadata_dir, "hf")

        snapshot_download(
            repo_id=args.dataset_path,
            allow_patterns="*.parquet",
            local_dir=metadata_dir,
            cache_dir=cache_dir,
            local_dir_use_symlinks=False,
            repo_type="dataset",
        )

        # Flatten directory structure in case of xlarge
        shutil.rmtree(cache_dir)

        logger.info("Done downloading metadata.")
    else:
        logger.info(
            f"Skipping download of metadata because {metadata_dir} exists. Use --overwrite_metadata to force re-downloading."
        )

    if not args.skip_shards:
        # Download images.
        shard_dir = os.path.join(args.data_dir, "shards")
        os.makedirs(shard_dir, exist_ok=True)
        logger.info(f"Downloading images to {shard_dir}")

        bbox_col = None if args.skip_bbox_blurring else "face_bboxes"

        url_list_dir = metadata_dir
        if os.path.isdir(os.path.join(metadata_dir, "data")):
            url_list_dir = os.path.join(metadata_dir, "data")

        img2dataset.download(
            url_list=url_list_dir,
            image_size=args.image_size,
            output_folder=shard_dir,
            processes_count=args.process_count,
            thread_count=args.thread_count,
            resize_mode=args.resize_mode,
            resize_only_if_bigger=not args.no_resize_only_if_bigger,
            encode_format=args.encode_format,
            output_format=args.output_format,
            input_format="parquet",
            url_col="url",
            caption_col="text",
            bbox_col=bbox_col,
            save_additional_columns=["uid"],
            number_sample_per_shard=args.num_samples_per_shard,
            oom_shard_count=8,
            retries=args.retries,
            enable_wandb=args.enable_wandb,
            wandb_project=args.wandb_project,
        )
    else:
        print("Skipping image data download.")

    print("Done!")


def add_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the Hugging Face dataset.",
        default="minhnguyent546/datacomp_large_vie_filtered2",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory where the data (webdataset shards) will be stored.",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        help="Path to directory where the metadata will be stored. If not set, infer from data_dir.",
        default=None,
    )
    parser.add_argument(
        "--skip_shards",
        action="store_true",
        help="If true, only download metadata.",
    )
    parser.add_argument(
        "--overwrite_metadata",
        action="store_true",
        help="If true, force re-download of the metadata files.",
    )
    parser.add_argument(
        "--skip_bbox_blurring",
        action="store_true",
        help="If true, skip bounding box blurring on images while downloading.",
    )
    parser.add_argument(
        "--process_count",
        type=int,
        help="Number of processes for download.",
        default=16,
    )
    parser.add_argument(
        "--thread_count",
        type=int,
        help="Number of threads for download.",
        default=128,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        help="Size images need to be downloaded to.",
        default=512,
    )
    parser.add_argument(
        "--num_samples_per_shard",
        type=int,
        help="Number of samples per shard.",
        default=10_000,
    )
    parser.add_argument(
        "--resize_mode",
        type=str,
        choices=["no", "border", "keep_ratio", "keep_ratio_largest", "center_crop"],
        help="Resizing mode used by img2dataset when downloading images.",
        default="keep_ratio_largest",
    )
    parser.add_argument(
        "--no_resize_only_if_bigger",
        action="store_true",
        help="If true, do not resize only if images are bigger than target size.",
    )
    parser.add_argument(
        "--encode_format",
        type=str,
        choices=["png", "jpg", "webp"],
        help="Images encoding format.",
        default="jpg",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["webdataset", "tfrecord", "parquet", "files"],
        help="Output format used by img2dataset when downloading images.",
        default="webdataset",
    )
    parser.add_argument(
        "--retries",
        type=int,
        help="Number of time a download should be retried (default 2)",
        default=2,
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        help="Whether to enable wandb logging (default False)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="Name of W&B project used (default datacomp_vi)",
        default="datacomp_vi",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download images using img2dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_opts(parser)
    args = parser.parse_args()

    download_image(args)


if __name__ == "__main__":
    main()
