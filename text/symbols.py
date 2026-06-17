%%writefile /content/FastSpeech2/text/symbols.py
# text/symbols.py — Urdu CV Dictionary v2.0.0 (MFA + eSpeak/Epitran)
# ====================================================================
# Covers phones emitted by MFA aligned with Urdu CV dict v2.0.0.
# Affricates use tie bar (t͡ʃ, d͡ʒ, d͡z) — each is ONE symbol.
# Aspirates are atomic: kʰ ≠ k + ʰ. Never split them.
# Long vowels: ɑː iː uː eː oː (NOT aː).
# ====================================================================

__pad = '_'
__unk = '<unk>'

# ── Silence ─────────────────────────────────────────────────────────
_silence = ['sil', 'sp', 'spn']

# ── Stops — plain ───────────────────────────────────────────────────
# ── Stops — plain ───────────────────────────────────────────────────
_stops_plain = [
    'p',    # voiceless bilabial
    'b',    # voiced bilabial
    't̪',   # voiceless dental
    'd̪',   # voiced dental
    'd',    # voiced alveolar (loanwords / eSpeak neutralisation)  ← ADD
    'ʈ',    # voiceless retroflex
    'ɖ',    # voiced retroflex
    'k',    # voiceless velar
    'ɡ',    # voiced velar
    'q',    # voiceless uvular
    'ʔ',    # glottal stop
    't',
    'g',
]

# ── Stops — aspirated / breathy ─────────────────────────────────────
_stops_aspirated = [
    'pʰ',   # aspirated bilabial
    'bʱ',   # breathy bilabial
    't̪ʰ',  # aspirated dental
    'd̪ʱ',  # breathy dental
    'ʈʰ',   # aspirated retroflex
    'ɖʱ',   # breathy retroflex
    'kʰ',   # aspirated velar
    'ɡʱ',   # breathy velar
]

# ── Affricates ──────────────────────────────────────────────────────
# Tie bar is part of the symbol. t͡ʃ ≠ t + ʃ.
_affricates = [
    't͡ʃ',   # voiceless palatal
    't͡ʃʰ',  # aspirated voiceless palatal
    'd͡ʒ',   # voiced palatal
    'd͡ʒʱ',  # breathy voiced palatal
    'd͡z',   # voiced alveolar
]

# ── Nasals ──────────────────────────────────────────────────────────
_nasals = [
    'm',    # bilabial
    'n',    # alveolar
    'ɳ',    # retroflex
    'ŋ',    # velar
    'nː',   # geminate alveolar
]

# ── Fricatives ──────────────────────────────────────────────────────
_fricatives = [
    'f',    # labiodental voiceless
    's',    # alveolar voiceless
    'z',    # alveolar voiced
    'ʃ',    # postalveolar voiceless
    'ʒ',    # postalveolar voiced
    'x',    # velar voiceless
    'ɣ',    # velar voiced
    'h',    # glottal voiceless
    'ɦ',    # glottal voiced
    'ʕ',    # pharyngeal voiced
]

# ── Approximants ────────────────────────────────────────────────────
_approximants = [
    'j',    # palatal
    'w',    # labio-velar
    'ʋ',    # labiodental
]

# ── Laterals ────────────────────────────────────────────────────────
_laterals = [
    'l',    # alveolar
    'lː',   # geminate alveolar
    'l̪',   # dental
]

# ── Taps / Trills ───────────────────────────────────────────────────
_taps = [
    'r',    # trill
    'ɾ',    # alveolar tap
    'ˈɾ',   # stressed alveolar tap
    'ɽ',    # retroflex tap
    'ɽʱ',   # breathy retroflex tap
]

# ── Long consonants ─────────────────────────────────────────────────
_long_consonants = [
    'pː',   # geminate bilabial
    't̪ː',  # geminate dental
    # lː and nː already in their lists — deduplicated below
]

# ── Vowels — short oral ─────────────────────────────────────────────
# ── Vowels — short oral ─────────────────────────────────────────────
_vowels_short = [
    'a',    # short open front/central (eSpeak short /a/)           ← ADD
    'i',    # close front unrounded
    'u',    # short close back rounded                              ← ADD
    'e',    # mid front unrounded
    'eʱ',   # breathy mid front
    'ə',    # schwa
    'ə̯',   # non-syllabic schwa
    'ɛ',    # open-mid front unrounded
    'ɪ',    # near-close front unrounded
    'ʊ',    # near-close back rounded
    'o',    # mid back rounded
    'ɔ',    # open-mid back rounded
    'æ',    # near-open front unrounded
]
_urdu_script = [
    'ی',   # U+06CC  count: 12
    'ہ',   # U+06C1  count:  6
    'ک',   # U+06A9  count:  5
    'ھ',   # U+06BE  count:  1
]

# ── Vowels — long oral ──────────────────────────────────────────────
_vowels_long = [
    'iː',   # long close front
    'eː',   # long mid front
    'ɛː',   # long open-mid front
    'ɑː',   # long open back      ← ɑː NOT aː
    'uː',   # long close back
    'oː',   # long mid back
    'ɔː',   # long open-mid back
]

# ── Vowels — nasal ──────────────────────────────────────────────────
_vowels_nasal = [
    'ə̃',   # nasalised schwa
    'ʊ̃',   # nasalised near-close back
    'ɑ̃ː',  # long nasalised open back
    'ɛ̃ː',  # long nasalised open-mid front
    'ẽː',   # long nasalised mid front
    'ĩː',   # long nasalised close front
    'õː',   # long nasalised mid back
    'ũː',   # long nasalised close back
]
# ── Approximants ────────────────────────────────────────────────────
_approximants = [
    'j',    # palatal
    'w',    # labio-velar
    'ʋ',    # labiodental approximant
    'v',    # labiodental fricative (loanword phones, e.g. "video")

]

# ── Assemble ────────────────────────────────────────────────────────
_all = (
      _silence
    + _stops_plain
    + _stops_aspirated
    + _affricates
    + _nasals
    + _fricatives
    + _approximants
    + _laterals
    + _taps
    + _long_consonants
    + _vowels_short
    + _vowels_long
    + _vowels_nasal
    + _urdu_script
    +_approximants
)

# Deduplicate preserving insertion order
_seen, _phones = set(), []
for _p in _all:
    if _p not in _seen:
        _seen.add(_p)
        _phones.append(_p)

symbols = [__pad, __unk] + _phones

# ── Sanity checks ───────────────────────────────────────────────────
assert len(symbols) == len(set(symbols)), \
    'Duplicate symbols detected! Check _all for repeated entries.'
assert symbols[0] == '_',      'Index 0 must be pad token'
assert symbols[1] == '<unk>',  'Index 1 must be unk token'

# ── Self-test ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'Total symbols : {len(symbols)}')
    for i, s in enumerate(symbols):
        print(f'  {i:3d}  {repr(s):25s}  {s}')

    # Verify full CV dict coverage
    cv_dict_phones = {
        'b','bʱ','d̪','d̪ʱ','d͡z','d͡ʒ','d͡ʒʱ',
        'e','eʱ','eː','f','h','i','iː',
        'j','k','kʰ','l','lː','l̪',
        'm','n','nː','o','oː',
        'p','pʰ','pː','q','r','s',
        't̪','t̪ʰ','t̪ː','t͡ʃ','t͡ʃʰ',
        'uː','w','x','z','æ',
        'õː','ĩː','ẽː','ũː','ŋ',
        'ɑː','ɑ̃ː','ɔː',
        'ɖ','ɖʱ',
        'ə','ə̃','ə̯',
        'ɛ','ɛː','ɛ̃ː',
        'ɡ','ɡʱ','ɣ','ɦ','ɪ','ɳ',
        'ɽ','ɽʱ','ɾ','ˈɾ',
        'ʃ','ʈ','ʈʰ','ʊ','ʊ̃','ʋ','ʒ','ʔ','ʕ',
         'a', 'u', 'd','g',
        'sil','sp','spn',
    }
    missing = cv_dict_phones - set(symbols)
    if missing:
        print(f'\n⚠️  Missing from symbols: {missing}')
    else:
        print('\n✅ All CV dict phones covered')


















































# # """ from https://github.com/keithito/tacotron """

# # """
# # Defines the set of symbols used in text input to the model.

# # The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

# # # from https://github.com/keithito/tacotron

# # """
# # Defines symbols for Urdu FastSpeech2 training.
# # """

# # symbols.py

# __pad = "_"
# __unk = "<unk>"

# symbols = [
#     __pad,
#     __unk,

#     # consonants
#     "m","n","ŋ","p","b","t","d","ʈ","ɖ","k","ɡ",
#     "s","z","ʃ","ʒ","f","x","ɣ","h",
#     "l","r","ɾ","j","ʋ",

#     # aspirated
#     "pʰ","bʰ","tʰ","dʰ","kʰ","ɡʰ",

#     # vowels
#     "a","e","i","o","u","ə","ɪ","ʊ","ɛ","ɔ","ʌ",

#     # long vowels
#     "aː","iː","uː","eː","oː",

#     # nasal
#     "ẽː","ũː",

#     # silence
#     "sil"
# ]

# #characetr level pipeline 
# # __pad = "_"
# # __unk = "<unk>"

# # _special = "-"

# # _punctuation = "،۔!؟،، ، . , : ؛ ؟ \" ' ( ) "
# # _letters = "اآبپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنوہھءیےں"

# # _silences = ["@sp", "@sil"]

# # symbols = (
# #     [__pad]
# #     + [__unk]
# #     + list(_special)
# #     + list(_punctuation)
# #     + list(_letters)
# #     + _silences
# # )




# # # """ from https://github.com/keithito/tacotron """

# # # """
# # # Defines the set of symbols used in text input to the model.

# # # The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

# # # # from https://github.com/keithito/tacotron

# # # """
# # # Defines symbols for Urdu FastSpeech2 training.
# # # """

# # # symbols.py

# # __pad = "_"
# # __unk = "<unk>"

# # symbols = [
# #     __pad,
# #     __unk,

# #     # consonants
# #     "m","n","ŋ","p","b","t","d","ʈ","ɖ","k","ɡ",
# #     "s","z","ʃ","ʒ","f","x","ɣ","h",
# #     "l","r","ɾ","j","ʋ",

# #     # aspirated
# #     "pʰ","bʰ","tʰ","dʰ","kʰ","ɡʰ",

# #     # vowels
# #     "a","e","i","o","u","ə","ɪ","ʊ","ɛ","ɔ","ʌ",

# #     # long vowels
# #     "aː","iː","uː","eː","oː",

# #     # nasal
# #     "ẽː","ũː",

# #     # silence
# #     "sil"
# # ]

# # #characetr level pipeline 
# # # __pad = "_"
# # # __unk = "<unk>"

# # # _special = "-"

# # # _punctuation = "،۔!؟،، ، . , : ؛ ؟ \" ' ( ) "
# # # _letters = "اآبپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنوہھءیےں"

# # # _silences = ["@sp", "@sil"]

# # # symbols = (
# # #     [__pad]
# # #     + [__unk]
# # #     + list(_special)
# # #     + list(_punctuation)
# # #     + list(_letters)
# # #     + _silences
# # # )
