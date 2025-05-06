import os
import sys
import random
import logging
import torch
import numpy as np
import torchaudio
import json
from multiprocessing import freeze_support
from TTS.config import BaseDatasetConfig, BaseAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.configs.xtts_config import XttsConfig,XttsAudioConfig,XttsArgs
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

VoiceBpeTokenizer.use_phonemes = False
VoiceBpeTokenizer.print_logs = lambda self, level = 0: None
VoiceBpeTokenizer.text_to_ids = TTSTokenizer.text_to_ids
VoiceBpeTokenizer.ids_to_text     = TTSTokenizer.ids_to_text
VoiceBpeTokenizer.text_cleaner      = None
VoiceBpeTokenizer.add_blank         = False
VoiceBpeTokenizer.use_eos_bos       = False

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig,XttsArgs,BaseDatasetConfig, BaseAudioConfig ])

from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import load_tts_samples
from trainer import Trainer, TrainerArgs, ConsoleLogger
from TTS.utils.manage import ModelManager
from TTS.tts.models.xtts import Xtts
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from trainer.logging.tensorboard_logger import TensorboardLogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    output_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(output_path, "roberta")
    wav_dir = os.path.join(dataset_path, "wavs")

    model_manager = ModelManager()
    model_path, config_path, _ = model_manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    config_path = os.path.join(model_path, "config.json")
    vocab_path = os.path.join(model_path, "vocab.json")
    print(config_path)


    def save_ids_to_file(self, output_path):
        speaker_ids = {name: str(idx) for idx, name in enumerate(self.speaker_names)}
        with open(os.path.join(output_path, "speakers.json"), "w", encoding="utf-8") as f:
            json.dump(speaker_ids, f, indent=4)
    SpeakerManager.save_ids_to_file = save_ids_to_file


    config = XttsConfig()
    config.load_json(config_path)
    config.model_args.speakers_file = os.path.join(output_path, "speaker.json")

    tokenizer = VoiceBpeTokenizer.init_from_config(config)


    if config.optimizer == "RAdam":
        config.optimizer_params.pop("lr", None)
        config.lr = 1e-5
    config.run_name = "xtts_finetune"
    config.eval_batch_size = 4
    config.output_path = output_path

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, vocab_path=os.path.join(model_path, "vocab.json"))


    speaker_name = "roberta"
    config.model_args.use_speaker_embedding = True
    config.model_args.use_d_vector_file = False
    config.model_args.use_language_embedding = False

    config.datasets = [BaseDatasetConfig(
        formatter="custom",
        meta_file_train="metadata.csv",
        path=dataset_path,
        language="en"
    )]

    def custom_formatter(root_path, meta_file, **kwargs):
        metadata = []
        with open(os.path.join(root_path, meta_file), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    filename = parts[0].strip()
                    if not filename.lower().endswith(".wav"):
                        filename = filename + ".wav"
                    wav_file = os.path.join(root_path, "wavs", filename)
                    text = parts[1].strip()
                    metadata.append({
                        "root_path": root_path,
                        "audio_file": wav_file,
                        "text": text,
                        "speaker_name": speaker_name,
                        "language": config.datasets[0].language
                    })
        return metadata

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

    ap = AudioProcessor(**config.audio.to_dict())

    from TTS.tts.datasets.dataset import TTSDataset
    TTSDataset.load_wav = lambda self, filename: ap.load_wav(filename)

    processed_dir = os.path.join(wav_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    required_files = []
    with open(os.path.join(dataset_path, "metadata.csv"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 1:
                required_files.append(parts[0].strip() + ".wav")

    processed_files = []
    for fname in required_files:
        src_path = os.path.join(wav_dir, fname)
        dest_path = os.path.join(processed_dir, fname)

        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing audio file: {src_path}")

        waveform, sr = torchaudio.load(src_path)
        if waveform.shape[0] > 1:
            waveform = ap.to_mono(waveform)
        if sr != config.audio.sample_rate:
            waveform = ap.resample(waveform, sr, config.audio.sample_rate)

        torchaudio.save(dest_path, waveform, config.audio.sample_rate)
        processed_files.append(fname)
        logger.info(f"Processed: {fname}")

    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        formatter=custom_formatter,
        eval_split=True,
        eval_split_size=0.1,
    )
    speaker_manager = SpeakerManager()

    trainer_args = TrainerArgs(
        continue_path="",
        restore_path=os.path.join(model_path, "model.pth"),
        use_accelerate=False,
        overfit_batch=False,
        start_with_eval=False,
        grad_accum_steps=1,
        gpu=0 if torch.cuda.is_available() else None,
    )
    console_logger = ConsoleLogger()
    tensorboard_logger = TensorboardLogger(log_dir=os.path.join(output_path, "logs"), model_name=config.run_name)

    # 9. Initialize trainer
    class XTTSTrainer(Trainer):
        def get_criterion(self, model):
            return None

        def get_model_criterion(self):
            return self.model.compute_loss

    trainer = XTTSTrainer(
        args=trainer_args,
        config=config,
        output_path=output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        training_assets={"audio_processor": ap},
        c_logger=console_logger,
        dashboard_logger=tensorboard_logger
    )

    trainer.fit()

    model.save_checkpoint(
        config=config,
        checkpoint_dir=output_path,
        checkpoint_path=os.path.join(output_path, "finetuned_model.pth"),
        speaker_manager=model.speaker_manager,
        vocab_path=os.path.join(model_path, "vocab.json"),
    )

    test_text = "I like big butts and I cannot lie"
    test_wav = model.synthesize(
        text=test_text,
        language="en",
        speaker_wav=os.path.join(processed_dir, processed_files[0]),
    )
    torchaudio.save(os.path.join(output_path, "test_output.wav"),
                    torch.tensor(test_wav["wav"]).unsqueeze(0),
                    config.audio.sample_rate)

    logger.info("Fine-tuning complete! Test output saved to test_output.wav")


if __name__ == '__main__':
    freeze_support()
    main()