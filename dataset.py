import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )

        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)

        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]

        # =========================
        # TEXT → ensure int64
        # =========================
        phone = np.array(
            text_to_sequence(self.text[idx], self.cleaners),
            dtype=np.int64,
        )

        # =========================
        # LOAD FEATURES
        # =========================
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path).astype(np.float32)

        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path).astype(np.float32)

        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path).astype(np.float32)

        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path).astype(np.int32)

        # ===========================================================
        # FIX: enforce phone == duration length BEFORE returning.
        #
        # text_to_sequence() may tokenize the phoneme string into a
        # different count than the duration array saved on disk.
        # If they diverge the variance adaptor crashes at:
        #   x = x + pitch_embedding   (152 vs 156 etc.)
        #
        # Strategy: truncate the longer to match the shorter, then
        # rebalance the mel / pitch / energy arrays to match the new
        # duration sum so every downstream tensor stays consistent.
        # ===========================================================
        n_phones = len(phone)
        n_durations = len(duration)

        if n_phones != n_durations:
            min_len = min(n_phones, n_durations)
            if n_phones > min_len:
                phone = phone[:min_len]
            if n_durations > min_len:
                duration = duration[:min_len]

        # Recompute the frame budget from the (possibly truncated) durations
        # and trim mel / pitch / energy to match exactly.
        frame_budget = int(duration.sum())

        if mel.shape[0] > frame_budget:
            mel = mel[:frame_budget]
        elif mel.shape[0] < frame_budget:
            # Shave the last duration bucket(s) until sum fits actual mel length
            actual_frames = mel.shape[0]
            excess = frame_budget - actual_frames
            for i in range(len(duration) - 1, -1, -1):
                reduction = min(int(duration[i]), excess)
                duration[i] -= reduction
                excess -= reduction
                if excess == 0:
                    break
            frame_budget = int(duration.sum())

        # ------------------------------------------------------------------
        # Pitch / Energy alignment
        #
        # Config says phoneme_level, so we need exactly len(phone) values.
        #
        # Case 1: already phoneme-level on disk → just truncate/pad to match
        # Case 2: frame-level on disk (len == frame_budget or close) →
        #         average over duration buckets to produce phoneme-level,
        #         which is what the preprocessor should have done.
        # ------------------------------------------------------------------

        def to_phoneme_level(arr, duration):
            """Average a frame-level array into phoneme-level using durations."""
            # First clip array to exactly frame_budget frames
            total = int(sum(duration))
            if len(arr) > total:
                arr = arr[:total]
            elif len(arr) < total:
                arr = np.pad(arr, (0, total - len(arr)))
            # Average each bucket
            averaged = []
            pos = 0
            for dur in duration:
                dur = int(dur)
                if dur > 0:
                    averaged.append(float(np.mean(arr[pos : pos + dur])))
                else:
                    averaged.append(0.0)
                pos += dur
            return np.array(averaged, dtype=np.float32)

        n_phones = len(phone)

        # Pitch
        if len(pitch) == n_phones:
            # Already phoneme-level and length matches — nothing to do
            pass
        elif len(pitch) <= frame_budget + 5:
            # Frame-level on disk → convert to phoneme-level
            pitch = to_phoneme_level(pitch, duration)
        else:
            # Unexpected length — force truncate to n_phones as last resort
            pitch = pitch[:n_phones]

        # Energy
        if len(energy) == n_phones:
            pass
        elif len(energy) <= frame_budget + 5:
            energy = to_phoneme_level(energy, duration)
        else:
            energy = energy[:n_phones]

        # Final hard guarantee — both must equal n_phones
        if len(pitch) != n_phones:
            pitch = pitch[:n_phones] if len(pitch) > n_phones else np.pad(pitch, (0, n_phones - len(pitch)))
        if len(energy) != n_phones:
            energy = energy[:n_phones] if len(energy) > n_phones else np.pad(energy, (0, n_phones - len(energy)))

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []

            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)

            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        # Sanity-check: every sample must have text_len == duration_len
        # This should always pass after the fix in __getitem__ but we keep
        # it here as a loud guard during development.
        for i, (tl, dur) in enumerate(zip(text_lens, durations)):
            if tl != len(dur):
                raise ValueError(
                    f"[reprocess] Sample '{ids[i]}' still has "
                    f"text_len={tl} but duration_len={len(dur)} "
                    f"after __getitem__ fix — investigate this sample."
                )

        speakers = np.array(speakers)

        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()

        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = []
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )

        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]

        phone = np.array(
            text_to_sequence(self.text[idx], self.cleaners),
            dtype=np.int64,
        )

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []

            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)

            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]

        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)










#import json
# import math
# import os

# import numpy as np
# from torch.utils.data import Dataset

# from text import text_to_sequence
# from utils.tools import pad_1D, pad_2D


# class Dataset(Dataset):
#     def __init__(
#         self, filename, preprocess_config, train_config, sort=False, drop_last=False
#     ):
#         self.dataset_name = preprocess_config["dataset"]
#         self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
#         self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
#         self.batch_size = train_config["optimizer"]["batch_size"]

#         self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
#             filename
#         )

#         with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
#             self.speaker_map = json.load(f)

#         self.sort = sort
#         self.drop_last = drop_last

#     def __len__(self):
#         return len(self.text)

#     def __getitem__(self, idx):
#         basename = self.basename[idx]
#         speaker = self.speaker[idx]
#         speaker_id = self.speaker_map[speaker]
#         raw_text = self.raw_text[idx]

#         # =========================
#         # TEXT → ensure int64
#         # =========================
#         phone = np.array(
#             text_to_sequence(self.text[idx], self.cleaners),
#             dtype=np.int64
#         )

#         # =========================
#         # LOAD FEATURES (FORCE float32)
#         # =========================
#         mel_path = os.path.join(
#             self.preprocessed_path,
#             "mel",
#             "{}-mel-{}.npy".format(speaker, basename),
#         )
#         mel = np.load(mel_path).astype(np.float32)

#         pitch_path = os.path.join(
#             self.preprocessed_path,
#             "pitch",
#             "{}-pitch-{}.npy".format(speaker, basename),
#         )
#         pitch = np.load(pitch_path).astype(np.float32)

#         energy_path = os.path.join(
#             self.preprocessed_path,
#             "energy",
#             "{}-energy-{}.npy".format(speaker, basename),
#         )
#         energy = np.load(energy_path).astype(np.float32)

#         duration_path = os.path.join(
#             self.preprocessed_path,
#             "duration",
#             "{}-duration-{}.npy".format(speaker, basename),
#         )
#         duration = np.load(duration_path).astype(np.float32)

#         sample = {
#             "id": basename,
#             "speaker": speaker_id,
#             "text": phone,
#             "raw_text": raw_text,
#             "mel": mel,
#             "pitch": pitch,
#             "energy": energy,
#             "duration": duration,
#         }

#         return sample

#     def process_meta(self, filename):
#         with open(
#             os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
#         ) as f:
#             name = []
#             speaker = []
#             text = []
#             raw_text = []

#             for line in f.readlines():
#                 n, s, t, r = line.strip("\n").split("|")
#                 name.append(n)
#                 speaker.append(s)
#                 text.append(t)
#                 raw_text.append(r)

#             return name, speaker, text, raw_text

#     def reprocess(self, data, idxs):
#         ids = [data[idx]["id"] for idx in idxs]
#         speakers = [data[idx]["speaker"] for idx in idxs]
#         texts = [data[idx]["text"] for idx in idxs]
#         raw_texts = [data[idx]["raw_text"] for idx in idxs]
#         mels = [data[idx]["mel"] for idx in idxs]
#         pitches = [data[idx]["pitch"] for idx in idxs]
#         energies = [data[idx]["energy"] for idx in idxs]
#         durations = [data[idx]["duration"] for idx in idxs]

#         text_lens = np.array([text.shape[0] for text in texts])
#         mel_lens = np.array([mel.shape[0] for mel in mels])

#         speakers = np.array(speakers)

#         texts = pad_1D(texts)
#         mels = pad_2D(mels)
#         pitches = pad_1D(pitches)
#         energies = pad_1D(energies)
#         durations = pad_1D(durations)

#         return (
#             ids,
#             raw_texts,
#             speakers,
#             texts,
#             text_lens,
#             max(text_lens),
#             mels,
#             mel_lens,
#             max(mel_lens),
#             pitches,
#             energies,
#             durations,
#         )

#     def collate_fn(self, data):
#         data_size = len(data)

#         if self.sort:
#             len_arr = np.array([d["text"].shape[0] for d in data])
#             idx_arr = np.argsort(-len_arr)
#         else:
#             idx_arr = np.arange(data_size)

#         tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size):]
#         idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
#         idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()

#         if not self.drop_last and len(tail) > 0:
#             idx_arr += [tail.tolist()]

#         output = []
#         for idx in idx_arr:
#             output.append(self.reprocess(data, idx))

#         return output


# class TextDataset(Dataset):
#     def __init__(self, filepath, preprocess_config):
#         self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

#         self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
#             filepath
#         )

#         with open(
#             os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")
#         ) as f:
#             self.speaker_map = json.load(f)

#     def __len__(self):
#         return len(self.text)

#     def __getitem__(self, idx):
#         basename = self.basename[idx]
#         speaker = self.speaker[idx]
#         speaker_id = self.speaker_map[speaker]
#         raw_text = self.raw_text[idx]

#         phone = np.array(
#             text_to_sequence(self.text[idx], self.cleaners),
#             dtype=np.int64
#         )

#         return (basename, speaker_id, phone, raw_text)

#     def process_meta(self, filename):
#         with open(filename, "r", encoding="utf-8") as f:
#             name = []
#             speaker = []
#             text = []
#             raw_text = []

#             for line in f.readlines():
#                 n, s, t, r = line.strip("\n").split("|")
#                 name.append(n)
#                 speaker.append(s)
#                 text.append(t)
#                 raw_text.append(r)

#             return name, speaker, text, raw_text

#     def collate_fn(self, data):
#         ids = [d[0] for d in data]
#         speakers = np.array([d[1] for d in data])
#         texts = [d[2] for d in data]
#         raw_texts = [d[3] for d in data]

#         text_lens = np.array([text.shape[0] for text in texts])

#         texts = pad_1D(texts)

#         return ids, raw_texts, speakers, texts, text_lens, max(text_lens)
