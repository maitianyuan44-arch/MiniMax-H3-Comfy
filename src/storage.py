"""S3-compatible object storage upload."""

import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import boto3
from botocore.config import Config as BotoConfig

from src import config
from src.errors import UploadError

logger = logging.getLogger("minimax-h3-comfy")


def _get_s3_client():
    kwargs = {
        "service_name": "s3",
        "region_name": config.S3_REGION,
        "aws_access_key_id": config.S3_ACCESS_KEY_ID,
        "aws_secret_access_key": config.S3_SECRET_ACCESS_KEY,
        "config": BotoConfig(
            connect_timeout=10,
            read_timeout=config.UPLOAD_TIMEOUT,
            retries={"max_attempts": 2},
        ),
    }
    if config.S3_ENDPOINT:
        kwargs["endpoint_url"] = config.S3_ENDPOINT
    return boto3.client(**kwargs)


def upload_video(
    local_path: str,
    job_id: str,
    bucket: str,
    prefix: str,
    public: bool = True,
    ttl_seconds: int = 0,
) -> str:
    """Upload an mp4 to S3-compatible storage. Returns the public or signed URL."""
    if not bucket:
        raise UploadError(
            "No S3 bucket configured.",
            hint="Set S3_BUCKET_DEFAULT or pass output.bucket in the request.",
        )

    now = datetime.now(timezone.utc)
    date_prefix = now.strftime("%Y/%m/%d")
    key = f"{prefix}{date_prefix}/{job_id}.mp4"

    client = _get_s3_client()
    file_size = os.path.getsize(local_path)
    logger.info("Uploading %s (%.1f MB) -> s3://%s/%s", local_path, file_size / (1024 * 1024), bucket, key)

    try:
        client.upload_file(
            Filename=local_path,
            Bucket=bucket,
            Key=key,
            ExtraArgs={"ContentType": "video/mp4"},
        )
    except Exception as e:
        raise UploadError(
            f"S3 upload failed: {e}",
            hint="Check S3 credentials, bucket name, and network connectivity.",
        )

    if public and config.S3_PUBLIC_BASE_URL:
        url = f"{config.S3_PUBLIC_BASE_URL.rstrip('/')}/{key}"
    elif not public and ttl_seconds > 0:
        try:
            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=ttl_seconds,
            )
        except Exception as e:
            raise UploadError(
                f"Failed to generate presigned URL: {e}",
                hint="Check S3 credentials and permissions.",
            )
    elif config.S3_ENDPOINT:
        url = f"{config.S3_ENDPOINT.rstrip('/')}/{bucket}/{key}"
    else:
        url = f"https://{bucket}.s3.{config.S3_REGION}.amazonaws.com/{key}"

    logger.info("Upload complete: %s", url)
    return url


def _download_one_lora(key: str, dest_dir: str) -> str:
    """Fetch one LoRA object from the default bucket into dest_dir.
    Returns the local filename. Raises UploadError when unfetchable."""
    bucket = config.S3_BUCKET_DEFAULT
    if not bucket:
        raise UploadError(
            "An extra LoRA is configured but no S3 bucket is set.",
            hint="Set S3_BUCKET_DEFAULT so the worker can fetch the LoRA file.",
        )
    name = os.path.basename(key)
    dest = os.path.join(dest_dir, name)
    if os.path.exists(dest):
        logger.info("Extra LoRA already present: %s", dest)
        return name
    os.makedirs(dest_dir, exist_ok=True)
    client = _get_s3_client()
    logger.info("Downloading extra LoRA s3://%s/%s -> %s", bucket, key, dest)
    try:
        client.download_file(bucket, key, dest)
    except Exception as e:
        raise UploadError(
            f"Extra LoRA download failed: {e}",
            hint=f"Check the key {key!r} matches the object key in the bucket.",
        )
    logger.info(
        "Extra LoRA ready: %s (%.1f MB)",
        dest, os.path.getsize(dest) / (1024 * 1024),
    )
    return name


def _download_lora_from_url(url: str, dest_dir: str) -> str:
    """Fetch one LoRA over HTTP(S) into dest_dir. Returns the local
    filename (the basename of the URL path). For LoRAs too large for the
    storage web UI's upload limit, the worker pulls the file itself at
    boot. Raises UploadError when unfetchable."""
    name = os.path.basename(urllib.parse.urlparse(url).path)
    if not name:
        raise UploadError(
            f"Extra LoRA URL has no filename: {url!r}",
            hint="Use a URL that ends with the .safetensors filename.",
        )
    dest = os.path.join(dest_dir, name)
    if os.path.exists(dest):
        logger.info("Extra LoRA already present: %s", dest)
        return name
    os.makedirs(dest_dir, exist_ok=True)
    tmp = dest + ".part"
    logger.info("Downloading extra LoRA %s -> %s", url, dest)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, dest)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise UploadError(
            f"Extra LoRA URL download failed: {e}",
            hint=f"Check the URL {url!r} is reachable.",
        )
    logger.info(
        "Extra LoRA ready: %s (%.1f MB)",
        dest, os.path.getsize(dest) / (1024 * 1024),
    )
    return name


def download_extra_loras(dest_dir: str) -> list:
    """Fetch every configured extra LoRA (config.EXTRA_LORAS /
    config.EXTRA_LORA_URLS) into dest_dir (ComfyUI's models/loras).
    A slot with a URL set downloads from that URL; otherwise a set key
    downloads from the default bucket. Returns a list with one entry per
    slot: the local filename, or None when that slot is not configured.
    A configured-but-unfetchable LoRA raises UploadError: silently
    generating without a LoRA the caller explicitly set up would hand
    back the wrong video."""
    names = []
    for (key, _strength), url in zip(config.EXTRA_LORAS, config.EXTRA_LORA_URLS):
        if url:
            names.append(_download_lora_from_url(url, dest_dir))
        elif key:
            names.append(_download_one_lora(key, dest_dir))
        else:
            names.append(None)
    return names
