"""
Smoke-test InternVL3-1B on several benchmark families (CPU only).

Requires:
  pip install "transformers>=4.37.2" datasets torchvision pillow
  pip install decord  # only if you later extend to video

Usage examples:
  python internvl3_smoketests.py --task math --num-samples 2
  python internvl3_smoketests.py --task ocr --num-samples 3
  python internvl3_smoketests.py --task hallucination --num-samples 3
  python internvl3_smoketests.py --task grounding --num-samples 3
  python internvl3_smoketests.py --task gui --num-samples 3
"""

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# -------------------------------------------------------------------
# Model & image preprocessing (CPU-friendly)
# -------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff")


def build_transform(input_size: int = 448):
    """Standard ViT-style ImageNet normalization + resize."""
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # tie-breaker using area, following InternVL3 quick-start
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image,
                       min_num: int = 1,
                       max_num: int = 12,
                       image_size: int = 448,
                       use_thumbnail: bool = True) -> List[Image.Image]:
    """
    Tile the image into up to max_num 448x448 patches, as in InternVL3 quick-start.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # candidate tilings
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images: List[Image.Image] = []
    grid_w = target_width // image_size
    grid_h = target_height // image_size

    for i in range(blocks):
        x = (i % grid_w) * image_size
        y = (i // grid_w) * image_size
        box = (x, y, x + image_size, y + image_size)
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def preprocess_pil_image(
    image: Image.Image,
    input_size: int = 448,
    max_num: int = 12,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert a PIL image -> stacked tiles tensor [T, 3, H, W] on given device/dtype.
    """
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    pixel_values = [transform(tile) for tile in tiles]
    pixel_values = torch.stack(pixel_values, dim=0)
    return pixel_values.to(device=device, dtype=dtype)


def load_model_and_tokenizer(
    model_path: str = "OpenGVLab/InternVL3-1B",
    device: str = "cpu",
):
    """
    Load InternVL3-1B as InternVLChatModel via remote code, on CPU.
    """
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,          # CPU-friendly
        low_cpu_mem_usage=True,
        use_flash_attn=False,               # no FlashAttention on CPU
        trust_remote_code=True,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    return model, tokenizer


def chat_single_image(
    model,
    tokenizer,
    image: Image.Image,
    question: str,
    device: str = "cpu",
    max_new_tokens: int = 128,
    do_sample: bool = False,
) -> str:
    """
    Helper: preprocess one image, call model.chat, return the string response.
    """
    pixel_values = preprocess_pil_image(image, device=device)
    # InternVL3 expects <image> token in the prompt
    prompt = f"<image>\n{question}"
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)

    # model.chat is provided by InternVLChatModel remote code
    response = model.chat(
        tokenizer,
        pixel_values,
        prompt,
        generation_config=generation_config,
    )
    return response


# -------------------------------------------------------------------
# Dataset-specific runners (one per experiment family)
# -------------------------------------------------------------------

def _try_extract_pil_image(value) -> Optional[Image.Image]:
    """
    Recursively search for a PIL.Image.Image inside arbitrary containers (list/dict)
    or load it from a filesystem path if the string points to an existing file.
    """
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            img = _try_extract_pil_image(item)
            if img is not None:
                return img
        return None
    if isinstance(value, dict):
        for item in value.values():
            img = _try_extract_pil_image(item)
            if img is not None:
                return img
        return None
    if isinstance(value, (str, os.PathLike)):
        text_value = str(value).strip()
        if not text_value:
            return None
        looks_like_path = (
            os.path.sep in text_value
            or text_value.lower().endswith(IMAGE_EXTENSIONS)
        )
        if not looks_like_path:
            return None
        candidate = Path(text_value).expanduser()
        try:
            if candidate.exists():
                with Image.open(candidate) as img:
                    return img.convert("RGB")
        except OSError:
            return None
    return None


def get_image_from_example(example, image_keys=("image", "images", "img")) -> Image.Image:
    """
    Try to grab a PIL image from common keys.
    """
    for k in image_keys:
        if k in example and example[k] is not None:
            # HF `Image` feature already returns PIL, but if you get bytes or path,
            # you can add additional handling here.
            img = _try_extract_pil_image(example[k])
            if img is not None:
                return img
    # fallback: scan the whole example for any PIL objects or loadable paths
    img = _try_extract_pil_image(example)
    if img is not None:
        return img
    raise KeyError(
        f"Could not find an image key in example. Tried: {image_keys}. "
        f"Available keys: {tuple(example.keys())}"
    )


def get_question_from_example(example, text_keys=("question", "query", "caption", "prompt", "sent", "sentence")) -> str:
    """
    Try to grab a question/prompt-like field from the example.
    """
    for k in text_keys:
        if k in example and example[k] is not None:
            return str(example[k])
    # last resort: stringify the whole example (you can customize)
    return str(example)


def get_answer_from_example(example, answer_keys=("answer", "answers", "label", "target", "solution")) -> Optional[str]:
    """
    Best-effort extraction of the ground-truth answer for comparison.
    """
    for k in answer_keys:
        if k in example and example[k]:
            val = example[k]
            if isinstance(val, (list, tuple)):
                if len(val) == 0:
                    continue
                val = val[0]
            return str(val)
    return None


# 1) Multimodal reasoning & math → MathVista
def run_mathvista(model, tokenizer, device: str = "cpu", num_samples: int = 3):
    """
    AI4Math/MathVista: visual math reasoning benchmark.
    We’ll just sample a few questions from the main split.
    """
    # If this is too large on your machine, you can switch to a lighter subset
    # such as a custom MathVista-200 variant.
    ds = load_dataset("AI4Math/MathVista", split="testmini[:50]")  # small slice

    print("\n=== MathVista (multimodal reasoning & math) ===")
    for i, ex in enumerate(ds.select(range(num_samples))):
        img = get_image_from_example(ex)
        q = get_question_from_example(ex, text_keys=("question", "query", "instruction", "prompt"))
        print(f"\n[Sample {i}]")
        print("Q:", q)
        gt = get_answer_from_example(ex)
        if gt is not None:
            print("GT:", gt)
        resp = chat_single_image(model, tokenizer, img, q, device=device, max_new_tokens=128)
        print("A:", resp)


# 2) OCR / chart / document understanding → TextVQA
def run_textvqa(model, tokenizer, device: str = "cpu", num_samples: int = 3):
    """
    lmms-lab/textvqa: text-centric VQA with strong OCR demands.
    """
    ds = load_dataset("lmms-lab/textvqa", split="validation[:50]")  # small subset

    print("\n=== TextVQA (OCR / text-centric VQA) ===")
    for i, ex in enumerate(ds.select(range(num_samples))):
        img = get_image_from_example(ex, image_keys=("image",))
        q = ex["question"]
        print(f"\n[Sample {i}]")
        print("Q:", q)
        resp = chat_single_image(model, tokenizer, img, q, device=device, max_new_tokens=64)
        print("A:", resp)


# 3) Hallucination / robustness → HallusionBench
def run_hallusionbench(model, tokenizer, device: str = "cpu", num_samples: int = 3):
    """
    lmms-lab/HallusionBench: hallucination-focused benchmark.
    We only use samples with images.
    """
    ds = load_dataset("lmms-lab/HallusionBench", split="image")  # image split

    print("\n=== HallusionBench (hallucination / robustness) ===")
    for i, ex in enumerate(ds.select(range(num_samples))):
        img = get_image_from_example(ex, image_keys=("image", "images"))
        q = get_question_from_example(ex, text_keys=("question", "query", "prompt"))
        print(f"\n[Sample {i}]")
        print("Q:", q)
        resp = chat_single_image(
            model,
            tokenizer,
            img,
            # you can strengthen the instruction to probe hallucinations
            question=q + "\nPlease answer carefully based only on the image.",
            device=device,
            max_new_tokens=64,
        )
        print("A:", resp)


# 4) Visual grounding / referring expressions → RefCOCO
def run_refcoco(model, tokenizer, device: str = "cpu", num_samples: int = 3):
    """
    lmms-lab/RefCOCO: referring expressions / grounding.

    Instead of evaluating IoU boxes, we just ask the model to *describe*
    the referenced object and its location, as a qualitative smoke test.
    """
    ds = load_dataset("lmms-lab/RefCOCO", split="validation[:100]")

    print("\n=== RefCOCO (visual grounding / referring) ===")
    for i, ex in enumerate(ds.select(range(num_samples))):
        img = get_image_from_example(ex, image_keys=("image", "img", "images"))
        expr = get_question_from_example(ex, text_keys=("expression", "sent", "sentence", "caption", "query", "question"))
        q = (
            f"The referring expression is: \"{expr}\".\n"
            "Please identify the referred object in the image and briefly describe its "
            "appearance and where it is located (e.g., top-left, center, bottom-right)."
        )
        print(f"\n[Sample {i}]")
        print("Expression:", expr)
        resp = chat_single_image(model, tokenizer, img, q, device=device, max_new_tokens=80)
        print("A:", resp)


# 5) GUI grounding → ScreenSpot-v2
def run_screenspot(model, tokenizer, device: str = "cpu", num_samples: int = 3):
    """
    lmms-lab/ScreenSpot-v2: GUI element grounding.

    As a smoke test, we ask the model to locate or describe the GUI target
    based on the question provided.
    """
    ds = load_dataset("lmms-lab/ScreenSpot-v2", split="validation[:100]")

    print("\n=== ScreenSpot-v2 (GUI grounding) ===")
    for i, ex in enumerate(ds.select(range(num_samples))):
        img = get_image_from_example(ex, image_keys=("image", "img", "images"))
        q_raw = get_question_from_example(
            ex,
            text_keys=("question", "query", "instruction", "caption", "prompt"),
        )
        q = (
            f"{q_raw}\n\n"
            "Please answer by describing what the user should click or where it is "
            "located on the screen (e.g., 'the blue button at the bottom right')."
        )
        print(f"\n[Sample {i}]")
        print("Q:", q_raw)
        resp = chat_single_image(model, tokenizer, img, q, device=device, max_new_tokens=80)
        print("A:", resp)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

TASK_TO_FUNC = {
    "math": run_mathvista,
    "ocr": run_textvqa,
    "hallucination": run_hallusionbench,
    "grounding": run_refcoco,
    "gui": run_screenspot,
}


def main():
    parser = argparse.ArgumentParser(description="InternVL3-1B benchmark smoke tests (CPU).")
    parser.add_argument(
        "--model-path",
        type=str,
        default="OpenGVLab/InternVL3-1B",
        help="HF model id or local path for InternVL3-1B.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=sorted(TASK_TO_FUNC.keys()),
        help="Which experiment family / dataset to run.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of examples to run for the selected task.",
    )
    args = parser.parse_args()

    device = "cpu"  # keep everything on CPU for your machine
    print(f"Loading model {args.model_path} on {device} ...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=device)

    runner = TASK_TO_FUNC[args.task]
    runner(model, tokenizer, device=device, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
