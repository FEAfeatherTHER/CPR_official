# CPR: Composer–Performer and Refiner

1. **Composer–Performer (CP)** takes prompt audio, prompt MIDI, and target MIDI,
   and writes **24 kHz mono audio**. The Composer is an autoregressive Qwen3
   Transformer; the Performer renders local Mel patches with flow matching.
   A Vocos vocoder converts the generated Mel features to audio.
2. **Refiner (R)** takes a **24 kHz audio file** and writes **48 kHz mono audio**
   using the LavaSR-based bandwidth-extension model and low-frequency fusion.

## Installation

Create a dedicated Conda environment with Python 3.10:

```bash
conda create -n cpr python=3.10 pip -y
conda activate cpr
python -m pip install -r requirements.txt
```

On the first CP run, CLAP automatically downloads and caches its RoBERTa initialization resources.

## Checkpoints

Download the inference assets from [bruceL33/CPR on Hugging Face](https://huggingface.co/bruceL33/CPR).
Run the following command from the code repository root with the `cpr` Conda environment activated.

Download the four files directly into `checkpoints/`:

```bash
hf download bruceL33/CPR \
  composer_performer.safetensors \
  composer_performer_config.json \
  vocos.safetensors \
  refiner.bin \
  --local-dir checkpoints
```

| Component | File under `checkpoints/` |
|---|---|
| Composer–Performer | `composer_performer.safetensors` |
| Model architecture and Qwen configuration | `composer_performer_config.json` |
| 24 kHz Vocos vocoder | `vocos.safetensors` |
| Refiner | `refiner.bin` |


### Download CLAP

CLAP is required for Composer–Performer inference. Download the official music checkpoint
[music_audioset_epoch_15_esc_90.14.pt](https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt) and save it as `checkpoints/clap.pt`:

```bash
mkdir -p checkpoints
curl --fail --location \
  --output checkpoints/clap.pt \
  https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt
```

## Inference

Run commands from the repository root. The included [piano example](examples/piano/README.md)
contains three inputs: `prompt.wav`, `prompt.mid`, and `target.mid`.

### 1. Composer–Performer → 24 kHz

```bash
bash scripts/infer_cp.sh
```

### 2. Refiner → 48 kHz

use any existing 24 kHz WAV:

```bash
python infer_refiner.py --input /path/to/audio_24k.wav --output outputs/refined.wav
```
