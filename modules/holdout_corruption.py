"""
Korupsi HELD-OUT untuk menguji kanonikalisasi, bukan hafalan augmenter.

Menjawab keberatan: Progressive Corruption Recall yang memakai augmenter yang
sama dengan pelatihan tidak bisa membedakan dua hal -- apakah model benar-benar
mengkanonikalisasi, atau sekadar mempelajari distribusi augmenter itu sendiri.

Berkas ini SENGAJA tidak memakai `data/augmenter.py`. Seluruh jenis korupsi di
bawah TIDAK ADA di augmenter pelatihan:

    homoglif Unicode    : а (Cyrillic) menggantikan a (Latin)
    kebingungan OCR     : rn -> m, cl -> d, 0 -> O
    leetspeak           : e -> 3, o -> 0, i -> 1
    substitusi fonetik  : ph -> f, ck -> k, tion -> shun
    pemanjangan vokal   : soo, sooo (gaya media sosial)
    penghapusan vokal   : txt spk

Selisih performa antara korupsi held-out dan korupsi augmenter adalah ukuran
langsung seberapa besar hasil PCR merupakan overfitting terhadap augmenter.
"""

import random

_HOMOGLYPH = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440",
              "c": "\u0441", "x": "\u0445", "y": "\u0443"}
_OCR = [("rn", "m"), ("m", "rn"), ("cl", "d"), ("d", "cl"),
        ("O", "0"), ("o", "0"), ("l", "1"), ("I", "l")]
_LEET = {"e": "3", "o": "0", "i": "1", "a": "4", "s": "5", "t": "7"}
_PHONETIC = [("ph", "f"), ("ck", "k"), ("tion", "shun"), ("ough", "uf"),
             ("kn", "n"), ("wr", "r"), ("qu", "kw")]
_VOWELS = "aeiou"


def homoglyph(text, rate=0.15, rng=None):
    rng = rng or random
    return "".join(_HOMOGLYPH.get(c, c) if c in _HOMOGLYPH and rng.random() < rate else c
                   for c in text)


def ocr_confusion(text, rate=0.3, rng=None):
    rng = rng or random
    out = text
    for a, b in _OCR:
        if a in out and rng.random() < rate:
            out = out.replace(a, b, 1)
    return out


def leetspeak(text, rate=0.25, rng=None):
    rng = rng or random
    return "".join(_LEET.get(c, c) if c in _LEET and rng.random() < rate else c
                   for c in text)


def phonetic_respell(text, rate=0.5, rng=None):
    rng = rng or random
    out = text
    for a, b in _PHONETIC:
        if a in out and rng.random() < rate:
            out = out.replace(a, b, 1)
    return out


def vowel_stretch(text, rate=0.2, max_rep=3, rng=None):
    rng = rng or random
    out = []
    for c in text:
        out.append(c)
        if c in _VOWELS and rng.random() < rate:
            out.append(c * rng.randint(1, max_rep))
    return "".join(out)


def vowel_drop(text, rate=0.4, rng=None):
    rng = rng or random
    words = text.split()
    res = []
    for w in words:
        if len(w) > 3 and rng.random() < rate:
            # Huruf pertama dipertahankan supaya kata tetap dapat dikenali.
            w = w[0] + "".join(c for c in w[1:] if c not in _VOWELS or rng.random() > 0.7)
        res.append(w)
    return " ".join(res)


ALL = {
    "homoglyph": homoglyph,
    "ocr": ocr_confusion,
    "leet": leetspeak,
    "phonetic": phonetic_respell,
    "stretch": vowel_stretch,
    "vowel_drop": vowel_drop,
}


def corrupt(text, kind, seed=0):
    rng = random.Random(seed)
    return ALL[kind](text, rng=rng)


def corrupt_mixed(text, n=2, seed=0):
    rng = random.Random(seed)
    kinds = rng.sample(list(ALL), min(n, len(ALL)))
    out = text
    for k in kinds:
        out = ALL[k](out, rng=rng)
    return out
