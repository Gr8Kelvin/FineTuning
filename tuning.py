import os
import sys
import random
import logging
import torch
import numpy as np
import torchaudio
import json
import string
from datetime import datetime
from multiprocessing import freeze_support
from TTS.config import BaseDatasetConfig, BaseAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig, XttsArgs
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
import types

DATASET_PATH = r"C:\Avada\Documents\python\voice\roberta"
OUTPUT_PATH = os.path.dirname(os.path.abspath(__file__))
NUM_EPOCHS = 15
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
TEST_TEXT = "This is a test of the fine-tuned voice model. It's been a very stressful day at work"


def patched_encode(self, text, language=None):
    lang = language or self.default_language or "en"
    text = text.strip()
    if self.use_phonemes:
        text = self.phonemize(text, lang)
    return text


def patched_text_to_ids(self, text, language=None):
    text = self.encode(text, language)
    ids = self.tokenizer.encode(text)
    return ids


# Apply the patches
VoiceBpeTokenizer.encode = patched_encode
VoiceBpeTokenizer.text_to_ids = patched_text_to_ids
VoiceBpeTokenizer.default_language = "en"
VoiceBpeTokenizer.print_logs = lambda self, level=0: None
VoiceBpeTokenizer.use_phonemes = False
VoiceBpeTokenizer.text_cleaner = None
VoiceBpeTokenizer.add_blank = False
VoiceBpeTokenizer.use_eos_bos = False

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig, BaseAudioConfig])

from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import load_tts_samples
from trainer import Trainer, TrainerArgs, ConsoleLogger
from TTS.utils.manage import ModelManager
from TTS.tts.models.xtts import Xtts
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from trainer.logging.tensorboard_logger import TensorboardLogger
from TTS.tts.datasets.dataset import TTSDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fixed_collate_fn(self, batch):
    if isinstance(batch, list):
        if len(batch) == 0:
            return {}

        # Process each item in the batch
        texts = []
        token_id = []
        token_id_lengths = []
        wavs = []
        wav_lens = []
        speaker_names = []
        language_names = []
        attn_mask = []

        for item in batch:
            texts.append(item["text"])
            if "token_ids" in item:
                token_id.append(item["token_ids"])
                token_id_lengths.append(len(item["token_ids"]))
            wavs.append(item["wav"])
            wav_lens.append(len(item["wav"]))

            if "speaker_name" in item:
                speaker_names.append(item["speaker_name"])
            else:
                speaker_names.append("dummy_speaker")

            if "language_name" in item:
                language_names.append(item["language_name"])
            else:
                language_names.append("en")

        max_wav_len = max(wav_lens)
        wav_tensor = torch.zeros(len(wavs), max_wav_len)
        for i, wav in enumerate(wavs):
            wav_tensor[i, :len(wav)] = torch.tensor(wav)

        return_dict = {
            "text": texts,
            "wav": wav_tensor,
            "wav_lens": torch.tensor(wav_lens),
            "speaker_name": speaker_names,
            "speaker_names": speaker_names,
            "language_name": language_names,
            "language_names": language_names,
        }

        if hasattr(self, "language_id_mapping") and language_names:
            try:
                language_ids = []
                for ln in language_names:
                    if ln in self.language_id_mapping:
                        language_ids.append(self.language_id_mapping[ln])
                    else:
                        language_ids.append(list(self.language_id_mapping.values())[0])
                return_dict["language_ids"] = torch.tensor(language_ids).long()
            except Exception as e:
                logger.warning(f"Error setting language IDs: {str(e)}")
                return_dict["language_ids"] = torch.zeros(len(language_names)).long()

        if hasattr(self, "speaker_id_mapping") and speaker_names:
            try:
                speaker_ids = []
                for sn in speaker_names:
                    if sn in self.speaker_id_mapping:
                        speaker_ids.append(self.speaker_id_mapping[sn])
                    else:
                        speaker_ids.append(list(self.speaker_id_mapping.values())[0])
                return_dict["speaker_ids"] = torch.tensor(speaker_ids).long()
            except Exception as e:
                logger.warning(f"Error setting speaker IDs: {str(e)}")
                return_dict["speaker_ids"] = torch.zeros(len(speaker_names)).long()

        if token_id:
            return_dict["token_id"] = token_id
            return_dict["token_id_lengths"] = torch.tensor(token_id_lengths).long()

        if attn_mask:
            return_dict["attns"] = attn_mask

        print("Returning batch with keys:", list(return_dict.keys()))
        return return_dict
    else:
        logger.error(f"Expected batch to be a list, but got {type(batch)}")
        return {}


def get_unique_output_path(base_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    unique_path = os.path.join(base_path, f"xtts_finetune_{timestamp}_{random_str}")

    while os.path.exists(unique_path):
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        unique_path = os.path.join(base_path, f"xtts_finetune_{timestamp}_{random_str}")

    os.makedirs(unique_path, exist_ok=True)
    return unique_path


def process_audio_files(wav_dir, metadata_file, config):
    ap = AudioProcessor(**config.audio.to_dict())

    processed_dir = os.path.join(wav_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    required_files = []
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 1:
                filename = parts[0].strip()
                if not filename.lower().endswith(".wav"):
                    filename = filename + ".wav"
                required_files.append(filename)

    processed_files = []
    for fname in required_files:
        src_path = os.path.join(wav_dir, fname)
        dest_path = os.path.join(processed_dir, fname)

        if not os.path.exists(src_path):
            logger.warning(f"Missing audio file: {src_path} - skipping")
            continue

        try:
            waveform, sr = torchaudio.load(src_path)
            if waveform.shape[0] > 1:
                waveform = ap.to_mono(waveform)
            if sr != config.audio.sample_rate:
                waveform = ap.resample(waveform, sr, config.audio.sample_rate)

            torchaudio.save(dest_path, waveform, config.audio.sample_rate)
            processed_files.append(fname)
            logger.info(f"Processed: {fname}")
        except Exception as e:
            logger.error(f"Error processing {fname}: {str(e)}")

    return processed_dir, processed_files


def custom_formatter(root_path, meta_file, **kwargs):
    metadata = []
    with open(os.path.join(root_path, meta_file), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")

            if len(parts) >= 2:
                filename = parts[0].strip()
                if not filename.lower().endswith(".wav"):
                    filename = filename + ".wav"

                wav_file = os.path.join(root_path, "wavs", "processed", filename)
                text = parts[1].strip()

                speaker_name = parts[2].strip() if len(parts) >= 3 else "default_speaker"
                language = parts[3].strip() if len(parts) >= 4 else "en"

                if os.path.exists(wav_file):
                    metadata.append({
                        "root_path": root_path,
                        "audio_file": wav_file,
                        "text": text,
                        "speaker_name": speaker_name,
                        "language": language
                    })
                else:
                    logger.warning(f"Audio file not found: {wav_file}")

    return metadata


def patch_dataset_getitem(config):
    def patched_load_data(self, idx):
        item = self.samples[idx]
        text = item["text"]

        language = item.get("language", "en")

        wav = self.load_wav(item["audio_file"])
        wav_filename = os.path.basename(item["audio_file"])

        speaker_name = item["speaker_name"] if "speaker_name" in item else None

        token_ids = self.get_token_ids(idx, text, language)
        return {"text": text, "token_ids": token_ids, "wav": wav, "item_idx": idx,
                "wav_file_name": wav_filename, "speaker_name": speaker_name, "language_name": language}

    def patched_get_token_ids(self, idx, text, language=None):
        if self.tokenizer is not None:
            token_ids = self.tokenizer.text_to_ids(text, language)
            return token_ids
        return None

    TTSDataset.load_data = patched_load_data
    TTSDataset.get_token_ids = patched_get_token_ids
    TTSDataset.load_wav = lambda self, filename: AudioProcessor(**config.audio.to_dict()).load_wav(filename)


def main():
    print("Starting XTTS fine-tuning process...")

    TTSDataset.collate_fn = types.MethodType(fixed_collate_fn, TTSDataset)

    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_path = DATASET_PATH
    output_path = OUTPUT_PATH
    wav_dir = os.path.join(dataset_path, "wavs")
    metadata_file = os.path.join(dataset_path, "metadata.csv")

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file not found: {metadata_file}")
        return

    # Download pre-trained model
    logger.info("Downloading pre-trained XTTS v2 model...")
    model_manager = ModelManager()
    model_path, config_path, _ = model_manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")

    def save_ids_to_file(self, output_path):
        speaker_ids = {name: str(idx) for idx, name in enumerate(self.speaker_names)}
        with open(os.path.join(output_path, "speakers.json"), "w", encoding="utf-8") as f:
            json.dump(speaker_ids, f, indent=4)

    SpeakerManager.save_ids_to_file = save_ids_to_file

    logger.info("Setting up model configuration...")
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    config.model_args.speakers_file = os.path.join(output_path, "speakers.json")

    # Set up the tokenizer
    tokenizer = TTSTokenizer.init_from_config(config)

    # Set up optimizer
    if config.optimizer == "RAdam":
        config.optimizer_params.pop("lr", None)
        config.lr = LEARNING_RATE


    config.run_name = "xtts_finetune"
    config.eval_batch_size = BATCH_SIZE
    config.batch_size = BATCH_SIZE
    config.epochs = NUM_EPOCHS
    config.output_path = output_path

    logger.info("Loading pre-trained model...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, vocab_path=os.path.join(model_path, "vocab.json"))

    #model arguments
    config.model_args.use_speaker_embedding = True
    config.model_args.use_d_vector_file = False
    config.model_args.use_language_embedding = True

    config.datasets = [BaseDatasetConfig(
        formatter="custom",
        meta_file_train="metadata.csv",
        path=dataset_path,
        language="en"
    )]

    config.audio = BaseAudioConfig(
        sample_rate=22050,
        frame_length_ms=50.0,
        frame_shift_ms=12.5,
        num_mels=80,
        preemphasis=0.0,
        ref_level_db=20,
        do_trim_silence=True,
        power=1.5,
    )

    patch_dataset_getitem(config)

    logger.info("Processing audio files...")
    processed_dir, processed_files = process_audio_files(wav_dir, metadata_file, config)

    if not processed_files:
        logger.error("No valid audio files found. Check your dataset directory.")
        return

    # Load training samples
    logger.info("Loading training samples...")
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        formatter=custom_formatter,
        eval_split=True,
        eval_split_size=0.1,
    )

    if not train_samples:
        logger.error("No training samples found. Check your metadata.csv file.")
        return

    logger.info(f"Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples")

    # Set up training arguments
    trainer_args = TrainerArgs(
        continue_path="",
        restore_path=os.path.join(model_path, "model.pth"),
        use_accelerate=False,
        overfit_batch=False,
        start_with_eval=False,
        grad_accum_steps=1,
        gpu=0 if torch.cuda.is_available() else None,
    )

    # Setting up my loggers
    console_logger = ConsoleLogger()
    tensorboard_logger = TensorboardLogger(log_dir=os.path.join(output_path, "logs"), model_name=config.run_name)

    # Custom trainer class
    class XTTSTrainer(Trainer):
        def get_criterion(self, model):
            return None

        def get_model_criterion(self):
            return self.model.compute_loss

    # Initialize and run trainer
    try:
        logger.info("Starting training...")
        print("Starting training...")
        trainer = XTTSTrainer(
            args=trainer_args,
            config=config,
            output_path=output_path,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
            training_assets={"audio_processor": AudioProcessor(**config.audio.to_dict())},
            c_logger=console_logger,
            dashboard_logger=tensorboard_logger
        )

        trainer.fit()

        # Saving the fine-tuned model
        logger.info("Saving fine-tuned model...")
        model.save_checkpoint(
            config=config,
            checkpoint_dir=output_path,
            checkpoint_path=os.path.join(output_path, "finetuned_model.pth"),
            speaker_manager=model.speaker_manager,
            vocab_path=os.path.join(model_path, "vocab.json"),
        )

        # Generate a test sample
        speaker_wav = os.path.join(processed_dir, processed_files[0])
        logger.info(f"Generating test sample using {speaker_wav}")

        test_wav = model.synthesize(
            text=TEST_TEXT,
            language="en",
            speaker_wav=speaker_wav,
        )

        test_output_file = os.path.join(output_path, "test_output.wav")
        torchaudio.save(
            test_output_file,
            torch.tensor(test_wav["wav"]).unsqueeze(0),
            config.audio.sample_rate
        )

        logger.info(f"Fine-tuning complete! Test output saved to {test_output_file}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    freeze_support()
    main()