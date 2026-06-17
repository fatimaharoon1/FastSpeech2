%%writefile /content/FastSpeech2/preprocessor/preprocessor.py
# ============================================================
# preprocessor/preprocessor.py — FastSpeech2 Production v6
# ============================================================
# Changes from v5:
#   - get_alignment fills inter-word gaps with 'sp' intervals
#     so dur_sum matches T even when MFA leaves unlabelled gaps
# ============================================================

import os
import random
import json
import logging

import tgt
import librosa
import numpy as np
import pyworld as pw

from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s"
)

log = logging.getLogger("preprocessor")


# ============================================================
# CONSTANTS
# ============================================================

MAX_FRAME_DRIFT  = 50
MIN_DURATION_SEC = 0.3


# ============================================================
# PREPROCESSOR
# ============================================================

class Preprocessor:

    def __init__(self, config):

        self.config        = config
        self.in_dir        = config["path"]["raw_path"]
        self.out_dir       = config["path"]["preprocessed_path"]
        self.val_size      = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length    = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in (
            "phoneme_level", "frame_level"
        )
        assert config["preprocessing"]["energy"]["feature"] in (
            "phoneme_level", "frame_level"
        )

        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )
        self.pitch_normalization  = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

        self._min_frames = int(
            MIN_DURATION_SEC * self.sampling_rate / self.hop_length
        )

    # ========================================================
    # BUILD DATASET
    # ========================================================

    def build_from_path(self):

        os.makedirs(os.path.join(self.out_dir, "mel"),      exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "pitch"),    exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "energy"),   exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "duration"), exist_ok=True)

        log.info("Processing Data ...")
        log.info(f"  raw_path          : {self.in_dir}")
        log.info(f"  preprocessed_path : {self.out_dir}")
        log.info(f"  sampling_rate     : {self.sampling_rate}")
        log.info(f"  min duration      : {MIN_DURATION_SEC}s  |  max drift : {MAX_FRAME_DRIFT} frames")

        out           = []
        n_frames      = 0
        pitch_scaler  = StandardScaler()
        energy_scaler = StandardScaler()
        speakers      = {}
        skip_reasons  = {}
        skip_files    = {}   # reason -> [basename, ...]

        all_speakers = sorted([
            s for s in os.listdir(self.in_dir)
            if os.path.isdir(os.path.join(self.in_dir, s))
        ])

        if not all_speakers:
            raise RuntimeError(
                f"No speaker subdirectories found in {self.in_dir}"
            )

        for i, speaker in enumerate(tqdm(all_speakers)):

            speakers[speaker] = i
            spk_dir   = os.path.join(self.in_dir, speaker)
            wav_files = [f for f in os.listdir(spk_dir) if f.endswith(".wav")]

            for wav_name in wav_files:

                basename = wav_name[:-4]
                tg_path  = os.path.join(
                    self.out_dir, "TextGrid", speaker,
                    f"{basename}.TextGrid"
                )

                if not os.path.exists(tg_path):
                    skip_reasons["no_textgrid"] = (
                        skip_reasons.get("no_textgrid", 0) + 1
                    )
                    skip_files.setdefault("no_textgrid", []).append(basename)
                    continue

                ret = self.process_utterance(speaker, basename)

                if ret is None:
                    skip_reasons["unexpected_none"] = (
                        skip_reasons.get("unexpected_none", 0) + 1
                    )
                    skip_files.setdefault("unexpected_none", []).append(basename)
                    continue

                info, pitch, energy, n, skip_reason = ret

                if skip_reason is not None:
                    skip_reasons[skip_reason] = (
                        skip_reasons.get(skip_reason, 0) + 1
                    )
                    skip_files.setdefault(skip_reason, []).append(basename)
                    continue

                out.append(info)
                n_frames += n

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape(-1, 1))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape(-1, 1))

        # ── Skip report ──────────────────────────────────────
        log.info("=" * 60)
        log.info(f"Kept    : {len(out)} samples")
        total_skipped = sum(skip_reasons.values())
        log.info(f"Skipped : {total_skipped} samples")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            log.info(f"  {reason:<45s} {count:>6d}")
        log.info("=" * 60)

        # ── Write skipped filenames to txt files ──────────────
        skipped_dir = os.path.join(self.out_dir, "skipped")
        os.makedirs(skipped_dir, exist_ok=True)
        for reason, names in skip_files.items():
            out_path = os.path.join(skipped_dir, f"{reason}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(names) + "\n")
            log.info(f"  Wrote {len(names)} filenames → {out_path}")

        if len(out) == 0:
            raise RuntimeError(
                "No samples survived preprocessing.\n"
                "Check DEBUG lines above for per-file skip reasons."
            )

        # ── Statistics ───────────────────────────────────────
        log.info("Computing statistics ...")

        if self.pitch_normalization:
            pitch_mean = float(pitch_scaler.mean_[0])
            pitch_std  = float(pitch_scaler.scale_[0])
        else:
            pitch_mean, pitch_std = 0.0, 1.0

        if self.energy_normalization:
            energy_mean = float(energy_scaler.mean_[0])
            energy_std  = float(energy_scaler.scale_[0])
        else:
            energy_mean, energy_std = 0.0, 1.0

        pitch_min,  pitch_max  = self.normalize(
            os.path.join(self.out_dir, "pitch"),  pitch_mean,  pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            json.dump(speakers, f, indent=2)

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            json.dump(
                {
                    "pitch":  [float(pitch_min),  float(pitch_max),
                               float(pitch_mean), float(pitch_std)],
                    "energy": [float(energy_min), float(energy_max),
                               float(energy_mean), float(energy_std)],
                },
                f, indent=2,
            )

        total_hours = n_frames * self.hop_length / self.sampling_rate / 3600
        log.info(f"Total audio : {total_hours:.3f} hours")

        # ── Train / val split ────────────────────────────────
        random.shuffle(out)
        effective_val = min(self.val_size, max(1, len(out) // 10))

        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[effective_val:]:
                f.write(m + "\n")

        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[:effective_val]:
                f.write(m + "\n")

        log.info(f"train.txt : {len(out) - effective_val} samples")
        log.info(f"val.txt   : {effective_val} samples")

        return out

    # ========================================================
    # PROCESS UTTERANCE
    # ========================================================

    def process_utterance(self, speaker, basename):

        def skip(reason, detail=""):
            if detail:
                log.debug(f"SKIP {basename} [{reason}] {detail}")
            return None, None, None, None, reason

        wav_path  = os.path.join(self.in_dir,  speaker, f"{basename}.wav")
        text_path = os.path.join(self.in_dir,  speaker, f"{basename}.lab")
        tg_path   = os.path.join(
            self.out_dir, "TextGrid", speaker, f"{basename}.TextGrid"
        )

        # ── 1. ALIGNMENT ─────────────────────────────────────
        try:
            textgrid = tgt.io.read_textgrid(tg_path)
            phone, duration, start, end = self.get_alignment(
                textgrid.get_tier_by_name("phones")
            )
        except Exception as e:
            return skip("textgrid_parse_error", str(e))

        if len(phone) != len(duration):
            return skip("alignment_length_mismatch",
                        f"phones={len(phone)} durations={len(duration)}")
        if len(phone) == 0:
            return skip("empty_alignment")
        if start >= end:
            return skip("zero_duration_utterance",
                        f"start={start:.4f} end={end:.4f}")

        # ── 2. LOAD AUDIO ─────────────────────────────────────
        try:
            wav, sr_native = librosa.load(wav_path, sr=None)
        except Exception as e:
            return skip("wav_load_error", str(e))

        if sr_native != self.sampling_rate:
            log.warning(
                f"{basename}: sr={sr_native} != {self.sampling_rate}. Resampling."
            )
            wav = librosa.resample(
                wav, orig_sr=sr_native, target_sr=self.sampling_rate
            )

        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float64)

        if len(wav) // self.hop_length < self._min_frames:
            return skip("utterance_too_short",
                        f"frames={len(wav) // self.hop_length}")

        # ── 3. READ TEXT ──────────────────────────────────────
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                raw_text = f.readline().strip()
        except Exception as e:
            return skip("lab_read_error", str(e))

        if not raw_text.strip():
            return skip("empty_lab_file")

        # ── 4. PITCH ──────────────────────────────────────────
        frame_period_ms = self.hop_length / self.sampling_rate * 1000.0
        try:
            _f0, t = pw.dio(
                wav, self.sampling_rate, frame_period=frame_period_ms
            )
            pitch = pw.stonemask(
                wav, _f0, t, self.sampling_rate
            ).astype(np.float32)
        except Exception as e:
            return skip("pitch_extraction_error", str(e))

        # ── 5. MEL + ENERGY ───────────────────────────────────
        try:
            mel_spectrogram, energy = Audio.tools.get_mel_from_wav(
                wav.astype(np.float32), self.STFT
            )
        except Exception as e:
            return skip("mel_extraction_error", str(e))

        # ── 6. VERIFY FRAME COUNTS ────────────────────────────
        mel_frames    = mel_spectrogram.shape[1]
        pitch_frames  = len(pitch)
        energy_frames = len(energy)
        duration_sum  = int(sum(duration))

        fc_max = max(mel_frames, pitch_frames, energy_frames)
        fc_min = min(mel_frames, pitch_frames, energy_frames)

        if fc_max - fc_min > MAX_FRAME_DRIFT:
            return skip("frame_count_divergence",
                        f"mel={mel_frames} pitch={pitch_frames} energy={energy_frames}")

        T = fc_min
        mel_spectrogram = mel_spectrogram[:, :T]
        pitch           = pitch[:T]
        energy          = energy[:T]

        drift = abs(T - duration_sum)
        if drift > MAX_FRAME_DRIFT:
            log.info(f"DEBUG {basename}: T={T} dur_sum={duration_sum} drift={drift}")
            return skip("alignment_frame_drift",
                        f"T={T} sum(dur)={duration_sum} drift={drift}")

        # ── 7. FIX ROUNDING ───────────────────────────────────
        if T != duration_sum:
            diff = T - duration_sum
            for idx in range(len(duration) - 1, -1, -1):
                new_val = duration[idx] + diff
                if new_val >= 1:
                    duration[idx] = new_val
                    break
            else:
                return skip("rounding_correction_failed", f"diff={diff}")

        if int(sum(duration)) != T:
            return skip("duration_sum_wrong_after_fix",
                        f"sum={sum(duration)} T={T}")

        # ── 8. QUALITY GATE ───────────────────────────────────
        if np.sum(pitch != 0) <= 1:
            return skip("insufficient_voiced_frames",
                        f"voiced={int(np.sum(pitch != 0))}")

        # ── 9. PHONEME-LEVEL PITCH AVERAGING ─────────────────
        if self.pitch_phoneme_averaging:
            nonzero_ids = np.where(pitch != 0)[0]
            if len(nonzero_ids) < 2:
                return skip("too_few_nonzero_pitch_frames")

            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch_interp   = interp_fn(np.arange(T))
            averaged_pitch = np.zeros(len(duration), dtype=np.float32)
            pos = 0
            for k, dur in enumerate(duration):
                dur = int(dur)
                if dur > 0:
                    averaged_pitch[k] = np.mean(pitch_interp[pos : pos + dur])
                pos += dur
            pitch = averaged_pitch

        # ── 10. PHONEME-LEVEL ENERGY AVERAGING ───────────────
        if self.energy_phoneme_averaging:
            averaged_energy = np.zeros(len(duration), dtype=np.float32)
            pos = 0
            for k, dur in enumerate(duration):
                dur = int(dur)
                if dur > 0:
                    averaged_energy[k] = np.mean(energy[pos : pos + dur])
                pos += dur
            energy = averaged_energy

        # ── 11. HARD SHAPE ASSERTIONS ─────────────────────────
        n_phones = len(duration)
        if self.pitch_phoneme_averaging:
            assert len(pitch) == n_phones, \
                f"[BUG] {basename}: pitch={len(pitch)} n_phones={n_phones}"
        else:
            assert len(pitch) == T, \
                f"[BUG] {basename}: pitch={len(pitch)} T={T}"
        if self.energy_phoneme_averaging:
            assert len(energy) == n_phones, \
                f"[BUG] {basename}: energy={len(energy)} n_phones={n_phones}"
        else:
            assert len(energy) == T, \
                f"[BUG] {basename}: energy={len(energy)} T={T}"
        assert mel_spectrogram.shape[1] == T, \
            f"[BUG] {basename}: mel={mel_spectrogram.shape[1]} T={T}"
        assert int(sum(duration)) == T, \
            f"[BUG] {basename}: sum(dur)={sum(duration)} T={T}"

        # ── 12. SAVE ──────────────────────────────────────────
        np.save(
            os.path.join(self.out_dir, "duration",
                         f"{speaker}-duration-{basename}.npy"),
            np.array(duration, dtype=np.int32)
        )
        np.save(
            os.path.join(self.out_dir, "pitch",
                         f"{speaker}-pitch-{basename}.npy"),
            pitch
        )
        np.save(
            os.path.join(self.out_dir, "energy",
                         f"{speaker}-energy-{basename}.npy"),
            energy
        )
        np.save(
            os.path.join(self.out_dir, "mel",
                         f"{speaker}-mel-{basename}.npy"),
            mel_spectrogram.T
        )

        text = "{" + " ".join(phone) + "}"
        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            T,
            None,
        )

    # ========================================================
    # ALIGNMENT  (fills inter-word gaps with 'sp')
    # ========================================================

    def get_alignment(self, tier):
        sil_phones = {"sil", "sp", "spn"}
        phones, durations = [], []
        start_time = end_time = 0.0
        end_idx = 0

        # --------------------------------------------------
        # Step 1: fill gaps between MFA intervals with 'sp'
        # --------------------------------------------------
        filled   = []
        prev_end = None

        for interval in tier._objects:
            s = interval.start_time
            e = interval.end_time
            p = interval.text

            # if there is a gap before this interval, insert sp
            if prev_end is not None and s - prev_end > 0.001:
                filled.append((prev_end, s, "sp"))

            filled.append((s, e, p))
            prev_end = e

        # --------------------------------------------------
        # Step 2: process filled intervals as normal
        # --------------------------------------------------
        for s, e, p in filled:

            if len(phones) == 0 and p in sil_phones:
                continue
            if len(phones) == 0:
                start_time = s

            dur = int(
                np.round(e * self.sampling_rate / self.hop_length)
                - np.round(s * self.sampling_rate / self.hop_length)
            )

            phones.append(p)
            durations.append(max(0, dur))

            if p not in sil_phones:
                end_time = e
                end_idx  = len(phones)

        return phones[:end_idx], durations[:end_idx], start_time, end_time

    # ========================================================
    # REMOVE OUTLIERS
    # ========================================================

    def remove_outlier(self, values):
        values = np.array(values, dtype=np.float32)
        if len(values) == 0:
            return values
        p25, p75 = np.percentile(values, 25), np.percentile(values, 75)
        iqr = p75 - p25
        return values[(values > p25 - 1.5 * iqr) & (values < p75 + 1.5 * iqr)]

    # ========================================================
    # NORMALIZE
    # ========================================================

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            if not filename.endswith(".npy"):
                continue
            filepath = os.path.join(in_dir, filename)
            values   = (np.load(filepath) - mean) / std
            np.save(filepath, values)
            if len(values) > 0:
                max_value = max(max_value, float(np.max(values)))
                min_value = min(min_value, float(np.min(values)))
        return min_value, max_value
























































# # ============================
# # preprocessor.py (FIXED v2)
# # ============================

# import os
# import random
# import json

# import tgt
# import librosa
# import numpy as np
# import pyworld as pw
# from scipy.interpolate import interp1d
# from sklearn.preprocessing import StandardScaler
# from tqdm import tqdm

# import audio as Audio


# class Preprocessor:
#     def __init__(self, config):

#         self.config = config
#         self.in_dir = config["path"]["raw_path"]
#         self.out_dir = config["path"]["preprocessed_path"]
#         self.val_size = config["preprocessing"]["val_size"]

#         self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
#         self.hop_length = config["preprocessing"]["stft"]["hop_length"]

#         assert config["preprocessing"]["pitch"]["feature"] in [
#             "phoneme_level",
#             "frame_level",
#         ]

#         assert config["preprocessing"]["energy"]["feature"] in [
#             "phoneme_level",
#             "frame_level",
#         ]

#         self.pitch_phoneme_averaging = (
#             config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
#         )

#         self.energy_phoneme_averaging = (
#             config["preprocessing"]["energy"]["feature"] == "phoneme_level"
#         )

#         self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
#         self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

#         self.STFT = Audio.stft.TacotronSTFT(
#             config["preprocessing"]["stft"]["filter_length"],
#             config["preprocessing"]["stft"]["hop_length"],
#             config["preprocessing"]["stft"]["win_length"],
#             config["preprocessing"]["mel"]["n_mel_channels"],
#             config["preprocessing"]["audio"]["sampling_rate"],
#             config["preprocessing"]["mel"]["mel_fmin"],
#             config["preprocessing"]["mel"]["mel_fmax"],
#         )

#     def build_from_path(self):

#         os.makedirs(os.path.join(self.out_dir, "mel"), exist_ok=True)
#         os.makedirs(os.path.join(self.out_dir, "pitch"), exist_ok=True)
#         os.makedirs(os.path.join(self.out_dir, "energy"), exist_ok=True)
#         os.makedirs(os.path.join(self.out_dir, "duration"), exist_ok=True)

#         print("Processing Data ...")

#         out = list()
#         n_frames = 0

#         pitch_scaler = StandardScaler()
#         energy_scaler = StandardScaler()

#         speakers = {}

#         for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):

#             speakers[speaker] = i

#             for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):

#                 if ".wav" not in wav_name:
#                     continue

#                 basename = wav_name.split(".")[0]

#                 tg_path = os.path.join(
#                     self.out_dir,
#                     "TextGrid",
#                     speaker,
#                     "{}.TextGrid".format(basename),
#                 )

#                 if os.path.exists(tg_path):

#                     ret = self.process_utterance(speaker, basename)

#                     if ret is None:
#                         continue

#                     info, pitch, energy, n = ret

#                     out.append(info)

#                     if len(pitch) > 0:
#                         pitch_scaler.partial_fit(pitch.reshape((-1, 1)))

#                     if len(energy) > 0:
#                         energy_scaler.partial_fit(energy.reshape((-1, 1)))

#                     n_frames += n

#         print("Computing statistic quantities ...")

#         if self.pitch_normalization:
#             pitch_mean = pitch_scaler.mean_[0]
#             pitch_std = pitch_scaler.scale_[0]
#         else:
#             pitch_mean = 0
#             pitch_std = 1

#         if self.energy_normalization:
#             energy_mean = energy_scaler.mean_[0]
#             energy_std = energy_scaler.scale_[0]
#         else:
#             energy_mean = 0
#             energy_std = 1

#         pitch_min, pitch_max = self.normalize(
#             os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
#         )

#         energy_min, energy_max = self.normalize(
#             os.path.join(self.out_dir, "energy"), energy_mean, energy_std
#         )

#         with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
#             f.write(json.dumps(speakers))

#         with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
#             stats = {
#                 "pitch": [
#                     float(pitch_min),
#                     float(pitch_max),
#                     float(pitch_mean),
#                     float(pitch_std),
#                 ],
#                 "energy": [
#                     float(energy_min),
#                     float(energy_max),
#                     float(energy_mean),
#                     float(energy_std),
#                 ],
#             }
#             f.write(json.dumps(stats))

#         print(
#             "Total time: {} hours".format(
#                 n_frames * self.hop_length / self.sampling_rate / 3600
#             )
#         )

#         random.shuffle(out)
#         out = [r for r in out if r is not None]

#         with open(
#             os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8"
#         ) as f:
#             for m in out[self.val_size :]:
#                 f.write(m + "\n")

#         with open(
#             os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8"
#         ) as f:
#             for m in out[: self.val_size]:
#                 f.write(m + "\n")

#         return out

#     def process_utterance(self, speaker, basename):

#         wav_path = os.path.join(
#             self.in_dir, speaker, "{}.wav".format(basename)
#         )
#         text_path = os.path.join(
#             self.in_dir, speaker, "{}.lab".format(basename)
#         )
#         tg_path = os.path.join(
#             self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
#         )

#         # =========================
#         # ALIGNMENT
#         # =========================

#         textgrid = tgt.io.read_textgrid(tg_path)

#         phone, duration, start, end = self.get_alignment(
#             textgrid.get_tier_by_name("phones")
#         )

#         # ---- FIX 1: hard check immediately after alignment ----
#         # phones and durations must be 1-to-1 or the whole pipeline is broken.
#         # Truncate the longer one to match the shorter one so we never carry
#         # a mismatched pair any further downstream.
#         if len(phone) != len(duration):
#             min_pd = min(len(phone), len(duration))
#             print(
#                 f"[WARN] Alignment length mismatch in {basename}: "
#                 f"phones={len(phone)}, durations={len(duration)}. "
#                 f"Truncating both to {min_pd}."
#             )
#             phone = phone[:min_pd]
#             duration = duration[:min_pd]

#         text = "{" + " ".join(phone) + "}"

#         if start >= end:
#             return None

#         # =========================
#         # LOAD AUDIO
#         # =========================

#         wav, _ = librosa.load(wav_path)
#         wav = wav[
#             int(self.sampling_rate * start) : int(self.sampling_rate * end)
#         ].astype(np.float32)

#         with open(text_path, "r") as f:
#             raw_text = f.readline().strip("\n")

#         # =========================
#         # PITCH EXTRACTION
#         # =========================

#         pitch, t = pw.dio(
#             wav.astype(np.float64),
#             self.sampling_rate,
#             frame_period=(self.hop_length / self.sampling_rate * 1000),
#         )

#         pitch = pw.stonemask(
#             wav.astype(np.float64), pitch, t, self.sampling_rate
#         )

#         # =========================
#         # MEL + ENERGY
#         # =========================

#         mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)

#         # =========================
#         # SYNCHRONIZE FRAME-LEVEL LENGTHS
#         # =========================

#         # ---- FIX 2: also cap against sum(duration) so frame arrays and
#         #             duration array always agree before averaging ----
#         duration_sum = sum(duration)

#         min_len = min(
#             mel_spectrogram.shape[1],
#             len(pitch),
#             len(energy),
#             duration_sum,
#         )

#         mel_spectrogram = mel_spectrogram[:, :min_len]
#         pitch = pitch[:min_len]
#         energy = energy[:min_len]

#         # Trim the last duration bucket if the frame arrays are shorter
#         if duration_sum > min_len:
#             excess = duration_sum - min_len
#             # Shave excess frames off the end, bucket by bucket
#             for idx in range(len(duration) - 1, -1, -1):
#                 reduction = min(duration[idx], excess)
#                 duration[idx] -= reduction
#                 excess -= reduction
#                 if excess == 0:
#                     break

#         # Drop any zero-duration phonemes that result from the trim
#         # and keep phones perfectly in sync
#         non_zero = [(p, d) for p, d in zip(phone, duration) if d > 0]
#         if len(non_zero) == 0:
#             return None
#         phone, duration = zip(*non_zero)
#         phone = list(phone)
#         duration = list(duration)

#         text = "{" + " ".join(phone) + "}"   # rebuild after possible pruning

#         # =========================
#         # REMOVE BAD SAMPLE
#         # =========================

#         if np.sum(pitch != 0) <= 1:
#             return None

#         # =========================
#         # PHONEME LEVEL PITCH
#         # =========================

#         if self.pitch_phoneme_averaging:

#             nonzero_ids = np.where(pitch != 0)[0]

#             interp_fn = interp1d(
#                 nonzero_ids,
#                 pitch[nonzero_ids],
#                 fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
#                 bounds_error=False,
#             )

#             pitch = interp_fn(np.arange(0, len(pitch)))

#             pos = 0
#             averaged_pitch = []

#             for dur in duration:
#                 dur = int(dur)
#                 if dur > 0:
#                     averaged_pitch.append(np.mean(pitch[pos : pos + dur]))
#                 else:
#                     averaged_pitch.append(0.0)
#                 pos += dur

#             pitch = np.array(averaged_pitch, dtype=np.float32)

#         # =========================
#         # PHONEME LEVEL ENERGY
#         # =========================

#         if self.energy_phoneme_averaging:

#             pos = 0
#             averaged_energy = []

#             for dur in duration:
#                 dur = int(dur)
#                 if dur > 0:
#                     averaged_energy.append(np.mean(energy[pos : pos + dur]))
#                 else:
#                     averaged_energy.append(0.0)
#                 pos += dur

#             energy = np.array(averaged_energy, dtype=np.float32)

#         # =========================
#         # FINAL SAFETY CHECK
#         # =========================

#         if self.pitch_phoneme_averaging and len(pitch) != len(duration):
#             print(
#                 f"[ERROR] Pitch mismatch after averaging: "
#                 f"{basename} {len(pitch)} vs {len(duration)}"
#             )
#             return None

#         if self.energy_phoneme_averaging and len(energy) != len(duration):
#             print(
#                 f"[ERROR] Energy mismatch after averaging: "
#                 f"{basename} {len(energy)} vs {len(duration)}"
#             )
#             return None

#         # =========================
#         # SAVE FILES
#         # =========================

#         dur_filename = "{}-duration-{}.npy".format(speaker, basename)
#         np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

#         pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
#         np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

#         energy_filename = "{}-energy-{}.npy".format(speaker, basename)
#         np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

#         mel_filename = "{}-mel-{}.npy".format(speaker, basename)
#         np.save(
#             os.path.join(self.out_dir, "mel", mel_filename),
#             mel_spectrogram.T,
#         )

#         return (
#             "|".join([basename, speaker, text, raw_text]),
#             self.remove_outlier(pitch),
#             self.remove_outlier(energy),
#             mel_spectrogram.shape[1],
#         )

#     def get_alignment(self, tier):

#         sil_phones = ["sil", "sp", "spn"]

#         phones = []
#         durations = []

#         start_time = 0
#         end_time = 0
#         end_idx = 0

#         for t in tier._objects:

#             s = t.start_time
#             e = t.end_time
#             p = t.text

#             # Trim leading silence
#             if phones == []:
#                 if p in sil_phones:
#                     continue
#                 else:
#                     start_time = s

#             # ---- FIX 3: append phone and duration together, always ----
#             # Previously silence phones in the middle were appended to phones
#             # but end_idx was not updated, creating a latent off-by-one.
#             # Now we build both lists in lock-step and trim together at the end.
#             phones.append(p)
#             durations.append(
#                 int(
#                     np.round(e * self.sampling_rate / self.hop_length)
#                     - np.round(s * self.sampling_rate / self.hop_length)
#                 )
#             )

#             if p not in sil_phones:
#                 end_time = e
#                 end_idx = len(phones)   # always points PAST the last real phone

#         # Trim trailing silence — both lists identically
#         phones = phones[:end_idx]
#         durations = durations[:end_idx]

#         # Guarantee no negative durations from rounding
#         durations = [max(0, d) for d in durations]

#         return phones, durations, start_time, end_time

#     def remove_outlier(self, values):

#         values = np.array(values)

#         p25 = np.percentile(values, 25)
#         p75 = np.percentile(values, 75)

#         lower = p25 - 1.5 * (p75 - p25)
#         upper = p75 + 1.5 * (p75 - p25)

#         normal_indices = np.logical_and(values > lower, values < upper)

#         return values[normal_indices]

#     def normalize(self, in_dir, mean, std):

#         max_value = np.finfo(np.float64).min
#         min_value = np.finfo(np.float64).max

#         for filename in os.listdir(in_dir):

#             filename = os.path.join(in_dir, filename)
#             values = (np.load(filename) - mean) / std
#             np.save(filename, values)

#             max_value = max(max_value, max(values))
#             min_value = min(min_value, min(values))

#         return min_value, max_value



























































# # import os
# # import random
# # import json

# # import tgt
# # import librosa
# # import numpy as np
# # import pyworld as pw
# # from scipy.interpolate import interp1d
# # from sklearn.preprocessing import StandardScaler
# # from tqdm import tqdm

# # import audio as Audio


# # class Preprocessor:
# #     def __init__(self, config):
# #         self.config = config
# #         self.in_dir = config["path"]["raw_path"]
# #         self.out_dir = config["path"]["preprocessed_path"]
# #         self.val_size = config["preprocessing"]["val_size"]
# #         self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
# #         self.hop_length = config["preprocessing"]["stft"]["hop_length"]

# #         assert config["preprocessing"]["pitch"]["feature"] in [
# #             "phoneme_level",
# #             "frame_level",
# #         ]
# #         assert config["preprocessing"]["energy"]["feature"] in [
# #             "phoneme_level",
# #             "frame_level",
# #         ]
# #         self.pitch_phoneme_averaging = (
# #             config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
# #         )
# #         self.energy_phoneme_averaging = (
# #             config["preprocessing"]["energy"]["feature"] == "phoneme_level"
# #         )

# #         self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
# #         self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

# #         self.STFT = Audio.stft.TacotronSTFT(
# #             config["preprocessing"]["stft"]["filter_length"],
# #             config["preprocessing"]["stft"]["hop_length"],
# #             config["preprocessing"]["stft"]["win_length"],
# #             config["preprocessing"]["mel"]["n_mel_channels"],
# #             config["preprocessing"]["audio"]["sampling_rate"],
# #             config["preprocessing"]["mel"]["mel_fmin"],
# #             config["preprocessing"]["mel"]["mel_fmax"],
# #         )

# #     def build_from_path(self):
# #         os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
# #         os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
# #         os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
# #         os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

# #         print("Processing Data ...")
# #         out = list()
# #         n_frames = 0
# #         pitch_scaler = StandardScaler()
# #         energy_scaler = StandardScaler()

# #         # Compute pitch, energy, duration, and mel-spectrogram
# #         speakers = {}
# #         for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
# #             speakers[speaker] = i
# #             for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
# #                 if ".wav" not in wav_name:
# #                     continue

# #                 basename = wav_name.split(".")[0]
# #                 tg_path = os.path.join(
# #                     self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
# #                 )
# #                 if os.path.exists(tg_path):
# #                     ret = self.process_utterance(speaker, basename)
# #                     if ret is None:
# #                         continue
# #                     else:
# #                         info, pitch, energy, n = ret
# #                     out.append(info)

# #                 if len(pitch) > 0:
# #                     pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
# #                 if len(energy) > 0:
# #                     energy_scaler.partial_fit(energy.reshape((-1, 1)))

# #                 n_frames += n

# #         print("Computing statistic quantities ...")
# #         # Perform normalization if necessary
# #         if self.pitch_normalization:
# #             pitch_mean = pitch_scaler.mean_[0]
# #             pitch_std = pitch_scaler.scale_[0]
# #         else:
# #             # A numerical trick to avoid normalization...
# #             pitch_mean = 0
# #             pitch_std = 1
# #         if self.energy_normalization:
# #             energy_mean = energy_scaler.mean_[0]
# #             energy_std = energy_scaler.scale_[0]
# #         else:
# #             energy_mean = 0
# #             energy_std = 1

# #         pitch_min, pitch_max = self.normalize(
# #             os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
# #         )
# #         energy_min, energy_max = self.normalize(
# #             os.path.join(self.out_dir, "energy"), energy_mean, energy_std
# #         )

# #         # Save files
# #         with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
# #             f.write(json.dumps(speakers))

# #         with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
# #             stats = {
# #                 "pitch": [
# #                     float(pitch_min),
# #                     float(pitch_max),
# #                     float(pitch_mean),
# #                     float(pitch_std),
# #                 ],
# #                 "energy": [
# #                     float(energy_min),
# #                     float(energy_max),
# #                     float(energy_mean),
# #                     float(energy_std),
# #                 ],
# #             }
# #             f.write(json.dumps(stats))

# #         print(
# #             "Total time: {} hours".format(
# #                 n_frames * self.hop_length / self.sampling_rate / 3600
# #             )
# #         )

# #         random.shuffle(out)
# #         out = [r for r in out if r is not None]

# #         # Write metadata
# #         with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
# #             for m in out[self.val_size :]:
# #                 f.write(m + "\n")
# #         with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
# #             for m in out[: self.val_size]:
# #                 f.write(m + "\n")

# #         return out

# #     def process_utterance(self, speaker, basename):
# #         wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
# #         text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
# #         tg_path = os.path.join(
# #             self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
# #         )

# #         # Get alignments
# #         textgrid = tgt.io.read_textgrid(tg_path)
# #         phone, duration, start, end = self.get_alignment(
# #             textgrid.get_tier_by_name("phones")
# #         )
# #         text = "{" + " ".join(phone) + "}"
# #         if start >= end:
# #             return None

# #         # Read and trim wav files
# #         wav, _ = librosa.load(wav_path)
# #         wav = wav[
# #             int(self.sampling_rate * start) : int(self.sampling_rate * end)
# #         ].astype(np.float32)

# #         # Read raw text
# #         with open(text_path, "r") as f:
# #             raw_text = f.readline().strip("\n")

# #         # Compute fundamental frequency
# #         pitch, t = pw.dio(
# #             wav.astype(np.float64),
# #             self.sampling_rate,
# #             frame_period=self.hop_length / self.sampling_rate * 1000,
# #         )
# #         pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

# #         pitch = pitch[: sum(duration)]
# #         if np.sum(pitch != 0) <= 1:
# #             return None

# #         # Compute mel-scale spectrogram and energy
# #         mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
# #         mel_spectrogram = mel_spectrogram[:, : sum(duration)]
# #         energy = energy[: sum(duration)]

# #         if self.pitch_phoneme_averaging:
# #             # perform linear interpolation
# #             nonzero_ids = np.where(pitch != 0)[0]
# #             interp_fn = interp1d(
# #                 nonzero_ids,
# #                 pitch[nonzero_ids],
# #                 fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
# #                 bounds_error=False,
# #             )
# #             pitch = interp_fn(np.arange(0, len(pitch)))

# #             # Phoneme-level average
# #             pos = 0
# #             for i, d in enumerate(duration):
# #                 if d > 0:
# #                     pitch[i] = np.mean(pitch[pos : pos + d])
# #                 else:
# #                     pitch[i] = 0
# #                 pos += d
# #             pitch = pitch[: len(duration)]

# #         if self.energy_phoneme_averaging:
# #             # Phoneme-level average
# #             pos = 0
# #             for i, d in enumerate(duration):
# #                 if d > 0:
# #                     energy[i] = np.mean(energy[pos : pos + d])
# #                 else:
# #                     energy[i] = 0
# #                 pos += d
# #             energy = energy[: len(duration)]

# #         # Save files
# #         dur_filename = "{}-duration-{}.npy".format(speaker, basename)
# #         np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

# #         pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
# #         np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

# #         energy_filename = "{}-energy-{}.npy".format(speaker, basename)
# #         np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

# #         mel_filename = "{}-mel-{}.npy".format(speaker, basename)
# #         np.save(
# #             os.path.join(self.out_dir, "mel", mel_filename),
# #             mel_spectrogram.T,
# #         )

# #         return (
# #             "|".join([basename, speaker, text, raw_text]),
# #             self.remove_outlier(pitch),
# #             self.remove_outlier(energy),
# #             mel_spectrogram.shape[1],
# #         )

# #     def get_alignment(self, tier):
# #         sil_phones = ["sil", "sp", "spn"]

# #         phones = []
# #         durations = []
# #         start_time = 0
# #         end_time = 0
# #         end_idx = 0
# #         for t in tier._objects:
# #             s, e, p = t.start_time, t.end_time, t.text

# #             # Trim leading silences
# #             if phones == []:
# #                 if p in sil_phones:
# #                     continue
# #                 else:
# #                     start_time = s

# #             if p not in sil_phones:
# #                 # For ordinary phones
# #                 phones.append(p)
# #                 end_time = e
# #                 end_idx = len(phones)
# #             else:
# #                 # For silent phones
# #                 phones.append(p)

# #             durations.append(
# #                 int(
# #                     np.round(e * self.sampling_rate / self.hop_length)
# #                     - np.round(s * self.sampling_rate / self.hop_length)
# #                 )
# #             )

# #         # Trim tailing silences
# #         phones = phones[:end_idx]
# #         durations = durations[:end_idx]

# #         return phones, durations, start_time, end_time

# #     def remove_outlier(self, values):
# #         values = np.array(values)
# #         p25 = np.percentile(values, 25)
# #         p75 = np.percentile(values, 75)
# #         lower = p25 - 1.5 * (p75 - p25)
# #         upper = p75 + 1.5 * (p75 - p25)
# #         normal_indices = np.logical_and(values > lower, values < upper)

# #         return values[normal_indices]

# #     def normalize(self, in_dir, mean, std):
# #         max_value = np.finfo(np.float64).min
# #         min_value = np.finfo(np.float64).max
# #         for filename in os.listdir(in_dir):
# #             filename = os.path.join(in_dir, filename)
# #             values = (np.load(filename) - mean) / std
# #             np.save(filename, values)

# #             max_value = max(max_value, max(values))
# #             min_value = min(min_value, min(values))

# #         return min_value, max_value
