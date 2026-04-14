""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

# from https://github.com/keithito/tacotron

"""
Defines symbols for Urdu FastSpeech2 training.
"""
__pad = "_"
__unk = "<unk>"

_special = "-"

_punctuation = "،۔!؟،، ، . , : ؛ ؟ \" ' ( ) "
_letters = "اآبپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنوہھءیےں"

_silences = ["@sp", "@sil"]

symbols = (
    [__pad]
    + [__unk]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _silences
)
