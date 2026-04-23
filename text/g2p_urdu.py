import re
import epitran

epi = epitran.Epitran('urd-Arab')

DIACRITICS = re.compile(r'[\u064B-\u065F]')

# ✅ ONLY ALLOWED SYMBOLS (must match symbols.py)
VALID_SYMBOLS = {
    "m","n","ŋ","p","b","t","d","ʈ","ɖ","k","ɡ",
    "s","z","ʃ","ʒ","f","x","ɣ","h",
    "l","r","ɾ","j","ʋ",
    "ə","i","e","o","u","ɪ","ʊ","ɛ","ɔ","ʌ",
    "aː","iː","uː","eː","oː","ɑː",
    "sil"
}

def remove_diacritics(text):
    return re.sub(DIACRITICS, '', text)

def g2p_urdu(text):
    text = remove_diacritics(text)

    ipa = epi.transliterate(text)
    ipa = re.sub(r'\s+', ' ', ipa).strip()

    tokens = []
    i = 0

    while i < len(ipa):
        if i + 1 < len(ipa) and ipa[i+1] == "ː":
            symbol = ipa[i] + "ː"
            i += 2
        elif ipa[i] != " ":
            symbol = ipa[i]
            i += 1
        else:
            i += 1
            continue

        # ✅ FILTER BAD SYMBOLS
        if symbol in VALID_SYMBOLS:
            tokens.append(symbol)
        else:
            continue  # DROP garbage

    tokens = ["sil"] + tokens + ["sil"]

    print("\nG2P DEBUG")
    print("IPA:", ipa)
    print("TOKENS:", tokens)

    return tokens
