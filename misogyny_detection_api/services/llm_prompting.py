# File: services/llm_prompting.py

import requests
import re

LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
USE_LLM = True  # Toggle False to run without LLM

def _call_llm_chat(messages: list[dict], max_tokens: int = 300, temperature: float = 0.7) -> str:
    payload = {
        "model": "mistral",  # Match with LM Studio model name
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(LMSTUDIO_API_URL, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return ""

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
    Uses LLM to generate clean 2 variants of the given sentence.
    Each output is stripped of quotes or annotations.
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
