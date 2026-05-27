#paraphrases generation

#queries loading
import numpy as np
control_prompt = [str(x) for x in np.load("control_prompts.npy", allow_pickle=True)]
from adv_queries import prompts
prompts_only = [entry['prompt'] for entry in prompts]

#Synonyms paraphrases
import random
import re
from functools import lru_cache
import nltk
from nltk.corpus import wordnet as wn

_WORD_RE = re.compile(r"[A-Za-z]+|[^A-Za-z\s]+|\s+")  # keep punctuation + spaces

@lru_cache(maxsize=200_000)
def _synonyms(word, pos_tag):
    tag_map = {"N": wn.NOUN, "V": wn.VERB, "J": wn.ADJ, "R": wn.ADV}
    wn_pos = tag_map.get(pos_tag[:1], None)
    if wn_pos is None:
        return ()

    syns = set()
    for syn in wn.synsets(word, pos=wn_pos):
        for lemma in syn.lemmas():
            w = lemma.name().replace("_", " ")
            if w.lower() != word.lower():
                syns.add(w)
    return tuple(sorted(syns))

def _compile_wordnet_prompt(text: str, min_len=4):
    """
    One-time preprocessing per prompt.
    Returns:
      pieces: list[str] (tokens + punctuation + spaces, to preserve formatting)
      word_positions: indices into pieces that correspond to alphabetic words
      eligible_word_positions: subset of word_positions eligible for synonym swap
      syn_choices: dict[pos_in_pieces] -> tuple[str] synonyms
    """
    pieces = _WORD_RE.findall(text)
    # word tokens are alphabetic pieces; keep their positions
    word_positions = [i for i, s in enumerate(pieces) if s.isalpha()]

    # POS tag the *word tokens only* (fewer tokens, faster)
    words = [pieces[i] for i in word_positions]
    tagged = nltk.pos_tag(words)  # list[(word, tag)]

    eligible_word_positions = []
    syn_choices = {}

    for wp, (w, tag) in zip(word_positions, tagged):
        if len(w) < min_len:
            continue
        syns = _synonyms(w.lower(), tag)
        if syns:
            eligible_word_positions.append(wp)
            syn_choices[wp] = syns

    return pieces, eligible_word_positions, syn_choices

def sample_wordnet_variant(compiled, *, p=0.25, seed=None):
    """
    Fast generation once compiled.
    """
    pieces, eligible_pos, syn_choices = compiled
    if not eligible_pos:
        return "".join(pieces)

    rng = random.Random(seed)

    m = max(1, int(p * len(eligible_pos)))
    m = min(m, len(eligible_pos))

    chosen = rng.sample(eligible_pos, m)
    out = list(pieces)
    for pos in chosen:
        out[pos] = rng.choice(syn_choices[pos])
    return "".join(out)

def wordnet_paraphrases(text: str, *, n=50, p=0.25, min_len=4, base_seed=42):
    compiled = _compile_wordnet_prompt(text, min_len=min_len)
    variants = []
    seen = set()
    j = 0
    # capped attempts to avoid slow “hunt for uniqueness”
    max_attempts = 10 * n
    while len(variants) < n and j < max_attempts:
        v = sample_wordnet_variant(compiled, p=p, seed=base_seed + j)
        j += 1
        if v == text:
            continue
        if v in seen:
            continue
        seen.add(v)
        variants.append(v)
    # pad if not enough unique exist
    while len(variants) < n:
        variants.append(variants[-1] if variants else text)
    return variants
    
para_synonyms_control = []

for idx, text in enumerate(control_prompt):

    variants = wordnet_paraphrases(
        text,
        n=50,          # number of paraphrases
        p=0.5,         # proportion of eligible words swapped
        min_len=4,
        base_seed=2026 + idx*1000
    )

    para_synonyms_control.append(variants)

#exact same loop for para_synonyms_adv with prompts_only jailbreak queries 

#Letter Swap paraphrases

import random
import re
from typing import Optional, Tuple, List, Union

_WORD_RE = re.compile(r"\w+")

def _permute_internal_force_change(word: str, rng: random.Random, max_tries: int = 10) -> str:
    if len(word) < 4:
        return word
    middle = list(word[1:-1])
    original = middle.copy()
    for _ in range(max_tries):
        rng.shuffle(middle)
        if middle != original:
            return word[0] + "".join(middle) + word[-1]
    return word

def internal_permutation_sentence(
    text: str,
    n_words: Union[int, float],
    *,
    min_len: int = 4,
    seed: Optional[int] = None,
    max_tries_per_word: int = 10,
) -> Tuple[str, List[Tuple[str, str]]]:
    rng = random.Random(seed)

    matches = list(_WORD_RE.finditer(text))
    eligible_idx = [idx for idx, m in enumerate(matches) if len(m.group(0)) >= min_len]
    if not eligible_idx:
        return text, []

    if isinstance(n_words, float):
        if not (0 < n_words <= 1):
            raise ValueError("If n_words is float, it must be in (0,1].")
        k = int(round(n_words * len(eligible_idx)))
    else:
        k = int(n_words)

    k = max(0, min(k, len(eligible_idx)))
    if k == 0:
        return text, []

    chosen = set(rng.sample(eligible_idx, k))

    out_parts = []
    last_end = 0
    changes: List[Tuple[str, str]] = []

    for idx, m in enumerate(matches):
        start, end = m.span()
        out_parts.append(text[last_end:start])

        w = m.group(0)
        if idx in chosen and len(w) >= min_len:
            w_new = _permute_internal_force_change(w, rng, max_tries=max_tries_per_word)
            out_parts.append(w_new)
            if w_new != w:
                changes.append((w, w_new))
        else:
            out_parts.append(w)

        last_end = end

    out_parts.append(text[last_end:])
    return "".join(out_parts), changes

def generate_internal_perm_variants_pct(
    text: str,
    pct_words: float,
    *,
    n_variants: int = 50,
    min_len: int = 4,
    base_seed: int = 42,
    max_attempts_factor: int = 40,
    max_tries_per_word: int = 10,
):
    if not (0 < pct_words <= 1):
        raise ValueError("pct_words must be in (0, 1].")

    seen = set()
    variants = []
    changes_list = []

    max_attempts = max_attempts_factor * n_variants
    attempts = 0
    j = 0

    while len(variants) < n_variants and attempts < max_attempts:
        t_new, changes = internal_permutation_sentence(
            text,
            pct_words,
            min_len=min_len,
            seed=base_seed + j,
            max_tries_per_word=max_tries_per_word,
        )
        attempts += 1
        j += 1

        if not changes:
            continue
        if t_new in seen:
            continue

        seen.add(t_new)
        variants.append(t_new)
        changes_list.append(changes)

    # pad to fixed length
    if len(variants) < n_variants:
        pad_text = variants[-1] if variants else text
        while len(variants) < n_variants:
            variants.append(pad_text)
            changes_list.append([])

    return variants, changes_list
    
    
n_variants = 50
pct_words = 0.5
base_seed = 42

para_swap_control = []          # (96, 50) strings
para_swap_control_changes = []  # (96, 50) list of (old,new)

for idx, text in enumerate(control_prompt):
    variants, changes_list = generate_internal_perm_variants_pct(
        text,
        pct_words=pct_words,
        n_variants=n_variants,
        min_len=4,
        base_seed=base_seed + idx * 10000,  # separate RNG streams per prompt
        max_attempts_factor=40,
        max_tries_per_word=10,
    )
    para_swap_control.append(variants)
    para_swap_control_changes.append(changes_list)
    
    
#Symbol paraphrases (renamed Numbers in the paper)

import random
from typing import Optional, Tuple, List, Union

def inject_char_noise(
    text: str,
    k: Union[int, float],
    symbol: str = "8",
    *,
    seed: Optional[int] = None,
    avoid_whitespace: bool = True,
) -> Tuple[str, List[int]]:
    """
    Replace k random characters with `symbol`.

    k:
      - int  → exact number of replacements
      - float in (0,1] → percentage of eligible characters
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not isinstance(symbol, str) or len(symbol) != 1:
        raise ValueError("symbol must be a single character")

    rng = random.Random(seed)

    if avoid_whitespace:
        candidates = [i for i, ch in enumerate(text) if not ch.isspace()]
    else:
        candidates = list(range(len(text)))

    n = len(candidates)
    if n == 0:
        return text, []

    # Convert k
    if isinstance(k, float):
        if not (0 < k <= 1):
            raise ValueError("If k is float, it must be in (0,1]")
        k_eff = int(round(k * n))
    else:
        k_eff = int(k)

    k_eff = min(k_eff, n)

    idxs = rng.sample(candidates, k_eff)

    chars = list(text)
    for i in idxs:
        chars[i] = symbol

    return "".join(chars), sorted(idxs)

para_symbol_adv = []

for idx, text in enumerate(prompts_only):
    prompt_variants = []
    for j in range(50):
        t_i, pos = inject_char_noise(
            text,
            k=int(0.25 * len(text)),
            symbol=str(j % 10),
            seed=42 + idx*100 + j
        )
        prompt_variants.append(t_i)

    para_symbol_adv.append(prompt_variants)
    
#Leet Speak paraphrases

####LeetSpeak
DEFAULT_LEET_MAP_MULTI = {
    "a": ["4", "@", "/\\", "α"],
    "b": ["8", "ß", "|3"],
    "c": ["(", "<", "¢"],
    "d": ["|)", "Ð"],
    "e": ["3", "€"],
    "f": ["ƒ"],
    "g": ["9", "6", "&"],
    "h": ["#", "|-|"],
    "i": ["1", "!", "|"],
    "j": ["_|"],
    "k": ["|<"],
    "l": ["1", "|_", "|"],
    "m": ["/\\/\\", "|\\/|"],
    "n": ["|\\|"],
    "o": ["0", "()", "°"],
    "p": ["|>", "ρ"],
    "q": ["0_", "(,)"],
    "r": ["|2"],
    "s": ["5", "$"],
    "t": ["7", "+", "†"],
    "u": ["(_)", "µ"],
    "v": ["\\/"],
    "w": ["\\/\\/", "vv"],
    "x": ["><", "}{"],
    "y": ["`/"],
    "z": ["2", "~/_"],
}

import random


def leetspeak_digits_k(text: str, mapping=DEFAULT_LEET_MAP_MULTI, k=None, seed=None):
    if seed is not None:
        random.seed(seed)

    text_list = list(text)

    eligible_indices = [
        i for i, ch in enumerate(text)
        if ch.lower() in mapping
    ]

    if not eligible_indices:
        return text

    if k is None:
        k = len(eligible_indices)

    k = min(k, len(eligible_indices))

    chosen = random.sample(eligible_indices, k)

    for idx in chosen:
        ch = text_list[idx].lower()
        replacement = random.choice(mapping[ch])
        text_list[idx] = replacement

    return "".join(text_list)
    
eps = 0.5
base_seed = 42
num_variants = 50

para_leetspeak_control = []

for idx, text in enumerate(control_prompt):

    eligible_indices = [
        i for i, ch in enumerate(text)
        if ch.lower() in DEFAULT_LEET_MAP_MULTI
    ]

    eligible_count = len(eligible_indices)
    k = int(eps * eligible_count)

    prompt_variants = [
        leetspeak_digits_k(
            text,
            mapping=DEFAULT_LEET_MAP_MULTI,
            k=k,
            seed=base_seed + idx*1000 + j
        )
        for j in range(num_variants)
    ]

    para_leetspeak_control.append(prompt_variants)