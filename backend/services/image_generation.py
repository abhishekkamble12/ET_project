import boto3
import json
import base64

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")


def generate_image(prompt: str):
    body = json.dumps({
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 10,
        "steps": 50
    })

    try:
        response = bedrock.invoke_model(
            modelId="stability.stable-diffusion-xl-v1",
            body=body
        )
        result = json.loads(response["body"].read())
        image_base64 = result["artifacts"][0]["base64"]
        return image_base64
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {e}")