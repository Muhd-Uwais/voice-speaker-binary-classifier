# Voice Speaker Binary Classifier


![status: development](https://img.shields.io/badge/status-development-yellow?style=flat-square)


Status: the project in developement

A binary classifier to identify whether an audio sample belongs to a target speaker using CNNs over mel-spectrogram features.

## Data layout
- data/
  - speaker0: non-target speaker samples (False)
  - speaker1: target speaker samples (True)

## Processing pipeline (modules)
- Chunk long recordings into 3s WAVs: [`src.data.audio_chunker.AudioChunker`](src/data/audio_chunker.py)
- Load dataset to NumPy batches: [`src.data.dataset_loader.AudioDatasetLoader`](src/data/dataset_loader.py)
- Waveform-level augmentation: [`src.data.audio_augmentor.AudioAugmentor`](src/data/audio_augmentor.py)
- Waveform → mel-spectrograms: [`src.data.mel_processor.WaveformToMel`](src/data/mel_processor.py)
- Mel-level augmentation (SpecAugment, VTLP): [`src.data.mel_processor.MelAugmentor`](src/data/mel_processor.py)

## Current workflow
1. Collect audio into data/speaker0 and data/speaker1.
2. Split long recordings into 3s chunks with [`src.data.audio_chunker.AudioChunker`](src/data/audio_chunker.py), then move chunks into the root speaker folders.
3. Load and batch with [`src.data.dataset_loader.AudioDatasetLoader`](src/data/dataset_loader.py).
4. Optionally augment waveforms via [`src.data.audio_augmentor.AudioAugmentor`](src/data/audio_augmentor.py) on train only.
5. Convert to mel-spectrograms via [`src.data.mel_processor.WaveformToMel`](src/data/mel_processor.py) and normalize to [0,1].
6. Optionally apply mel augmentations via [`src.data.mel_processor.MelAugmentor`](src/data/mel_processor.py) on train only.
7. Train a small CNN over input shape (64, 188, 1). Track accuracy/F1/ROC-AUC.
