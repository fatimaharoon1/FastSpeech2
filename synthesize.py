import argparse
import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# URDU TEXT PREPROCESSING
# =========================
def preprocess_urdu(text):
    text = text.strip()
    return text


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    model.eval()

    for batch in batchs:
        batch = to_device(batch, device)

        with torch.no_grad():
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )

            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--mode", type=str, choices=["batch", "single"], required=True)

    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)

    parser.add_argument("--speaker_id", type=int, default=0)

    parser.add_argument("-p", "--preprocess_config", type=str, required=True)
    parser.add_argument("-m", "--model_config", type=str, required=True)
    parser.add_argument("-t", "--train_config", type=str, required=True)

    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)

    args = parser.parse_args()

    # =========================
    # LOAD CONFIG
    # =========================
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)

    configs = (preprocess_config, model_config, train_config)

    # =========================
    # LOAD MODEL + VOCODER
    # =========================
    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    # =========================
    # PREPARE INPUTS
    # =========================

    if args.mode == "batch":
        dataset = TextDataset(args.source, preprocess_config)

        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )

    elif args.mode == "single":

        assert args.text is not None, "Provide --text for single mode"

        text = preprocess_urdu(args.text)

        # 🔥 URDU FIX: NO G2P, NO LEXICON
        sequence = text_to_sequence(text, ["urdu_cleaners"])

        ids = [text[:50]]
        raw_texts = [text]
        speakers = np.array([args.speaker_id])

        texts = np.array([sequence])
        text_lens = np.array([len(sequence)])

        batchs = [
            (ids, raw_texts, speakers, texts, text_lens, max(text_lens))
        ]

    else:
        raise ValueError("Invalid mode")

    # =========================
    # CONTROL VALUES
    # =========================
    control_values = (
        args.pitch_control,
        args.energy_control,
        args.duration_control
    )

    # =========================
    # RUN SYNTHESIS
    # =========================
    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
