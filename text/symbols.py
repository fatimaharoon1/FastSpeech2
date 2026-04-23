# """ from https://github.com/keithito/tacotron """

# """
# Defines the set of symbols used in text input to the model.

# The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

# # from https://github.com/keithito/tacotron

# """
# Defines symbols for Urdu FastSpeech2 training.
# """

# symbols.py

__pad = "_"
__unk = "<unk>"

symbols = [
    __pad,
    __unk,

    # consonants
    "m","n","ŋ","p","b","t","d","ʈ","ɖ","k","ɡ",
    "s","z","ʃ","ʒ","f","x","ɣ","h",
    "l","r","ɾ","j","ʋ",

    # aspirated
    "pʰ","bʰ","tʰ","dʰ","kʰ","ɡʰ",

    # vowels
    "a","e","i","o","u","ə","ɪ","ʊ","ɛ","ɔ","ʌ",

    # long vowels
    "aː","iː","uː","eː","oː",

    # nasal
    "ẽː","ũː",

    # silence
    "sil"
]

#characetr level pipeline 
# __pad = "_"
# __unk = "<unk>"

# _special = "-"

# _punctuation = "،۔!؟،، ، . , : ؛ ؟ \" ' ( ) "
# _letters = "اآبپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنوہھءیےں"

# _silences = ["@sp", "@sil"]

# symbols = (
#     [__pad]
#     + [__unk]
#     + list(_special)
#     + list(_punctuation)
#     + list(_letters)
#     + _silences
# )
