### Bark-RVC Integration Pipeline
## Project Overview

The Bark-RVC Integration Pipeline is an innovative tool designed to seamlessly merge the capabilities of two cutting-edge projects: Bark and RVC. This pipeline aims to revolutionize the field of voice cloning and text-to-speech (TTS) technology, especially for the French language.

# Bark: A Multilingual Text-to-Audio Model

Bark, developed by Suno, is a transformer-based model capable of generating highly realistic, multilingual speech, along with other audio forms such as music, background noise, and sound effects. It excels in producing natural-sounding text-to-speech outputs, including nonverbal communications like laughter and sighs. Bark's strength lies in its versatility and the realistic quality of the audio it generates.

# RVC: Enhancing Voice Cloning

RVC (Realistic Voice Cloning) works to enhance the naturalness of cloned voices. By taking an audio file generated from a TTS tool, RVC applies trained model weights to produce a voice that closely represents the original speaker's characteristics. This process elevates the authenticity of the voice cloning experience.

# Project Genesis

This project is inspired by JarodMica's "rvc-tts-pipeline", which originally used "Tortoise" as the TTS tool. However, Tortoise's limitations in handling the French language led to the integration of Bark. Bark's superior performance with French and its advanced text-to-audio capabilities make it an ideal replacement, ensuring high-quality TTS output in French.

# Objectives

Our goal is to provide a robust, efficient pipeline that combines the strengths of Bark's multilingual TTS capabilities with RVC's voice cloning technology. This integration is particularly geared towards enhancing French language TTS, addressing a significant gap in the current technology landscape.
Usage and Disclaimer

This project is still a work in progress and may contain bugs. Users should note that Bark is a fully generative text-to-audio model and may produce unexpected outputs. As with any research tool, it should be used responsibly and at your own risk.
Contributions and Feedback

We welcome contributions and feedback from the community to improve and refine this pipeline. Your insights and suggestions are invaluable in advancing this project.

**It is still a work in progress, there will be bugs and issues.**

## Installation

0.  Create and activate your python environment

```
conda create -n bark-rvc-pipeline python=3.8
conda activate bark-rvc-pipeline
```

1. Create a "bark-rvc" with the project's content and use it as a root directory.
```
git clone https://github.com/hedizekri/bark-rvc.git
```

2. Install the required version of pytorch based on your processing environment. Find the correct version on: https://pytorch.org/get-started/locally/ (choose Nightly build for Mac M1/M2).

3. Install Bark from Suno-AI Github.

```
git clone https://github.com/suno-ai/bark
cd bark && pip install .
```

4. Then, to install rvc, run the following: 

```
pip install -e git+https://github.com/JarodMica/rvc.git#egg=rvc
pip install -e git+https://github.com/JarodMica/rvc-tts-pipeline.git#egg=rvc_tts_pipe
```

5. If you're using a Mac, you can install libsndfile using Homebrew. Open your terminal and run the following command:

```
brew install libsndfile
```

Find the installation path running:
```
brew --prefix libsndfile
```

Add this new path in your $PATH variable using this command replacing "path_to_libsndfile" with the path you got on the previous command. Then reload the configuration file.

```
export PATH="/path_to_libsndfile/bin:$PATH"
```

6. Install required packages.

```
pip install -r requirements.txt
```

7. Download both ```hubert_base.pt``` and ```rmvpe.pt``` files from HuggingFace and move them into the parent directory. Find them on https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main.

8. Move the pretrained model (.pth file) of your choice in the "models" folder.

## Basic usage

To use the ```bark-rvc-pipeline.py``` script in your way, please manipulate these three values.

```SCRIPT``` : Write the text you would like to generate speech on in this value.
```SPEAKER``` : Choose the Bark voice model of your choice according to the documentation (https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c)
```RVC_MODEL``` : Download or generate the cloning voice model of your choice and use it to make the Bark output even more realistic. (If your voice model filename is ```winnie.pth```, declare this value as ```winnie```)

The core ```rvc_convert``` function requires two parameters:

```model_path = path to the model```

```input_path = path to the audio file to convert (or tts output audio file)```

Then, it can simply be called like the following:

```
from rvc_infer import rvc_convert

rvc_convert(model_path="your_model_path_here", input_path="your_audio_path_here")
```

The docstrings of rvc_convert details out other values you may want to play around with, probably the most important being pitch and f0method.

## Acknowledgements
Huge thanks to JarodMica and the RVC creators as none of this would be possible without them. This takes and uses a lot of their code in order to make this possible.