# File: services/llm_prompting.py

import requests
import re
import time

LMSTUDIO_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
USE_LLM = True  # Toggle False to run without LLM
session = requests.Session()  # Reuse session for efficiency

# Global throttle (seconds) to avoid hammering LM Studio
REQUEST_DELAY = 0.5  

def _call_llm_chat(messages: list[dict], max_tokens: int = 300, temperature: float = 0.7) -> str:
    """
    Calls LM Studio's local API with retries, timeout, and connection reuse.
    """
    payload = {
        "model": "mistral",  # Must match the model name shown in LM Studio UI
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    retries = 3
    for attempt in range(1, retries + 1):
        try:
            response = session.post(
                LMSTUDIO_API_URL,
                json=payload,
                timeout=60  # Increase timeout to handle long generations
            )
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            time.sleep(REQUEST_DELAY)  # Throttle to reduce load
            return result
        except requests.exceptions.Timeout:
            print(f"[LLM ERROR] Timeout on attempt {attempt}/{retries}, retrying...")
        except Exception as e:
            print(f"[LLM ERROR] {e}, retrying ({attempt}/{retries})...")

        time.sleep(2 * attempt)  # exponential backoff before retry

    return ""  # Fallback if all retries fail


def explain_sentence_meaning(input_text: str) -> str:
    if not USE_LLM:
        return "Stub explanation: gender-based harm."
    messages = [{
        "role": "user",
        "content": f"Explain the meaning of this sentence in plain language:\n\"{input_text}\""
    }]
    return _call_llm_chat(messages)


def classify_label(input_text: str) -> tuple[int, str]:
    if not USE_LLM:
        return 1, "Stub label: misogynistic"

    messages = [{
        "role": "user",
        "content": (
            "Classify the following sentence. Start your answer with either "
            "'misogynistic:' or 'non-misogynistic:' followed by a brief explanation.\n\n"
            f"Sentence: \"{input_text}\""
        )
    }]
    response = _call_llm_chat(messages).strip()
    lower_resp = response.lower()

    if "non-misogynistic" in lower_resp:
        return 0, response
    elif "misogynistic" in lower_resp:
        return 1, response
    return 0, response + " [Uncertain classification, defaulted to non-misogynistic]"


def generate_prompt_variants(input_text: str, max_variants: int = 2) -> list[str]:
    """
    Uses LLM to generate paraphrased variants of the input sentence.
    Strips quotes/annotations and ensures clean output.
    """
    if not USE_LLM:
        return [f"{input_text} (variant {i+1})" for i in range(max_variants)]

    messages = [{
        "role": "user",
        "content": (
            f"Rephrase the following sentence in exactly {max_variants} different ways. "
            f"Each variant should be a clean and natural sentence. Do not add quotes, annotations, or explanations. "
            f"Only return the rephrased sentences, numbered as 1. and 2.\n\n"
            f"Sentence: {input_text}\n\nVariants:"
        )
    }]

    response = _call_llm_chat(messages)
    variants = []

    for line in response.splitlines():
        match = re.match(r'^\s*\d+\.\s*(.*)', line)
        if match:
            variant = match.group(1).strip()
            variant = re.sub(r"[\"“”]+", "", variant)  # Remove quotes
            variant = re.sub(r"\(.*?\)", "", variant)  # Remove parentheticals
            if variant and len(variant.split()) > 2:
                variants.append(variant.strip())

    return variants if variants else [input_text + " (fallback paraphrase)"]
