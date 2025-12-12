# CineBrain: A Large-Scale Multi-Modal Brain Dataset for Naturalistic Audiovisual Narrative Processing

[Jianxiong Gao](https://jianxgao.github.io/), Yichang Liu, Baofeng Yang, [Jianfeng Feng](https://www.dcs.warwick.ac.uk/~feng/), [Yanwei Fu†](http://yanweifu.github.io/)

[![ArXiv](https://img.shields.io/badge/ArXiv-2503.06940-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2503.06940)
[![Dataset](https://img.shields.io/badge/Dataset-CineBrain-faa035.svg?logo=Huggingface)](https://huggingface.co/datasets/Fudan-fMRI/CineBrain)



## 📘 Overview

**CineBrain** is a large-scale multimodal brain dataset containing **fMRI, EEG, and ECG** recordings collected while participants watched episodes of *The Big Bang Theory*.

It supports research on:

- Video decoding from multimodal neural signals
- Auditory decoding from naturalistic narrative listening
- Cross-modal EEG-to-fMRI translation
- Stimulus-to-brain modeling under real-world audiovisual viewing




## 🧩 Codebase

This codebase is built upon **CogVideo (v1.0, not CogVideo 1.5)**:  
👉 https://github.com/zai-org/CogVideo/releases/tag/v1.0



## 🚀 Quick Start

### 1️⃣ Install CogVideo Environment

Follow the official [CogVideo v1.0 setup guide](https://github.com/zai-org/CogVideo/blob/v1.0/sat/README.md).

### 2️⃣ Install Additional Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare Models

1. Download the [CogVideo-5B](https://github.com/zai-org/CogVideo/releases/tag/v1.0) base model
2. Download CineBrain-related weights from [HuggingFace](https://huggingface.co/datasets/Fudan-fMRI/CineBrain) *(link to be updated)*



## 📦 Dataset

Full dataset, documentation, and metadata are available on HuggingFace:

👉 [CineBrain Dataset](https://huggingface.co/datasets/Fudan-fMRI/CineBrain)

**Includes:**
- fMRI recordings
- EEG & ECG signals
- Synchronized video, audio, and subtitles
- Time-aligned captions & event annotations




## 📄 Citation

If you find this work useful, please cite:
```bibtex
@article{gao2025cinebrain,
  title={CineBrain: A Large-Scale Multi-Modal Brain Dataset During Naturalistic Audiovisual Narrative Processing},
  author={Gao, Jianxiong and Liu, Yichang and Yang, Baofeng and Feng, Jianfeng and Fu, Yanwei},
  journal={arXiv preprint arXiv:2503.06940},
  year={2025}
}
```

