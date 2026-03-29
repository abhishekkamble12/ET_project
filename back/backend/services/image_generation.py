from __future__ import annotations
import json
import logging

logger = logging.getLogger(__name__)


def _get_bedrock():
    try:
        import boto3
        return boto3.client("bedrock-runtime", region_name="us-east-1")
    except Exception:
        return None


def generate_image(prompt: str) -> str:
    bedrock = _get_bedrock()
    if bedrock is None:
        raise RuntimeError("AWS boto3 not available")

    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "standard",
            "cfgScale": 8.0,
            "height": 1024,
            "width": 1024,
            "seed": 0
        }
    })

    try:
        response = bedrock.invoke_model(
            modelId="amazon.titan-image-generator-v2:0",
            body=body,
            accept="application/json",
            contentType="application/json"
        )
        result = json.loads(response["body"].read())
        return result["images"][0]
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {e}")
