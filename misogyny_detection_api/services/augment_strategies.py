# File: services/augment_strategies.py

import random
from typing import List
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')

def apply_synonym_replacement(text: str, max_replacements: int = 2) -> List[str]:
    words = text.split()
    variants, attempts = [], 0

    while len(variants) < 2 and attempts < 10:
        new_words = words[:]
        replaced = 0
        for i, word in enumerate(words):
            syns = wordnet.synsets(word)
            if syns:
                lemmas = [l.name().replace("_", " ") for l in syns[0].lemmas() if l.name().lower() != word.lower()]
                if lemmas:
                    new_words[i] = random.choice(lemmas)
                    replaced += 1
            if replaced >= max_replacements:
                break
        variant = " ".join(new_words)
        if variant != text and variant not in variants:
            variants.append(variant)
        attempts += 1

    return variants

def apply_typo_noise(text: str, num_typos: int = 2) -> List[str]:
    def inject_typo(s):
        chars = list(s)
        for _ in range(num_typos):
            if len(chars) < 2:
                break
            idx = random.randint(0, len(chars) - 2)
            action = random.choice(["swap", "delete", "repeat"])
            if action == "swap":
                chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
            elif action == "delete":
                del chars[idx]
            elif action == "repeat":
                chars.insert(idx, chars[idx])
        return "".join(chars)

    return list({inject_typo(text) for _ in range(3)})
