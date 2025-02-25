# Method_for_MADUV_Challenge_2025

This Github repo is a dispaly of our proposed system for The 1st INTERSPEECH Mice Autism Detection via Ultrasound Vocalisation (MADUV) Challenge (#[1]).

The code demonstrates signal preprocessing, feature extraction, pre-trained model fine-tuning, validation, and prediction.

## Introduction of the system

The system is designed for the classification of ultrasonic vocalisations (USVs) for mice autism. The system first uses a novel Uniform-spaced Filter Bank method for data preprocessing and feature extraction. It then fine-tunes pre-trained BEATs model [2](#2) on the MADUV 2025 Challenge dataset [1](#1), which contains of recordings from 84 mouse subjects, including 44 wild-type and 40 ASD model type.

## Instructions (Reproduction)
1. First, install the required dependencies included in `requirements.txt`.
2. Clone BEATs repository and download pre-trained checkpoints from [/microsoft/unilm/tree/master/beats](https://github.com/microsoft/unilm/tree/master/beats) and put it in `\train_beats`.
2. Download our fine-tuned BEATs from [huggingface.co](), and save it to `\train_beats`.
3. USV

## Acknowledgements

## References
<a id="1"></a>[1] Yang, M. Song, X. Jing, H. Zhang, K. Qian, B. Hu, K. Tamada, T. Takumi, B. W. Schuller, and Y. Yamamoto, “Mad-uv: The 1st interspeech mice autism detection via ultrasound vocalization challenge,” arXiv preprint arXiv:2501.04292, 2025.
<a id="2"></a>[2] S. Chen, Y. Wu, C. Wang, S. Liu, D. Tompkins, Z. Chen, W. Che, X. Yu, and F. Wei, “Beats: Audio pre-training with acoustic tokenizers,” in International Conference on Machine Learning. PMLR, 2023, pp. 5178–5193.
