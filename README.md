# Method for MADUV Challenge 2025

This Github repo is a dispaly of our proposed system for The 1st INTERSPEECH Mice Autism Detection via Ultrasound Vocalisation (MADUV) Challenge [ [1] ](#1).

The code demonstrates signal preprocessing, feature extraction, pre-trained model fine-tuning, validation, and prediction.

## Introduction of the system

The system is designed for the classification of ultrasonic vocalisations (USVs) for mice autism. The system first uses a novel Uniform-spaced Filter Bank method for data preprocessing and feature extraction. It then fine-tunes pre-trained BEATs model [ [2] ](#2) on the MADUV 2025 Challenge dataset [ [1] ](#1), which contains of recordings from 84 mouse subjects, including 44 wild-type and 40 ASD model type.

## Instructions (Reproduction)
1. First, install the required dependencies included in `requirements.txt`.
2. Clone BEATs repository and download pre-trained checkpoints from [microsoft/unilm/tree/master/beats](https://github.com/microsoft/unilm/tree/master/beats) and put it in `\train_beats`.
3. Download MADUV dataset, and save it to `\train_beats\dataset`.
4. **Data Trimming**: Run the following command to trim the audio files into segments of 30s with a 15s overlap:

   ```bash
   python trim.py --input_path <path_to_dataset> --output_path <path_to_output_dir> --chunk 30000 --overlap 15000
5. **Feature Extraction** and **Training**: Run the following command to extract features from the audio files and load the features to fine-tuning the pre-trained BEATs model:

   ``` bash
   python train_beats.py
   
6. **Prediction** Download our fine-tuned BEATs from [huggingface.co](), and save it to `\train_beats`, or train from scratch. Then run the following command to predict the class of audio samples in the test set:

    ``` bash
    python test.py

## Acknowledgements
- [microsoft/unilm/tree/master/beats](https://github.com/microsoft/unilm/tree/master/beats)
- [KamijouMikoto/MADUV_2025](https://github.com/KamijouMikoto/MADUV_2025)

## Citation

## References
<a id="1"></a>[1] Yang, M. Song, X. Jing, H. Zhang, K. Qian, B. Hu, K. Tamada, T. Takumi, B. W. Schuller, and Y. Yamamoto, “Mad-uv: The 1st interspeech mice autism detection via ultrasound vocalization challenge,” arXiv preprint arXiv:2501.04292, 2025.
[https://arxiv.org/abs/2501.04292](https://arxiv.org/abs/2501.04292)

<a id="2"></a>[2] S. Chen, Y. Wu, C. Wang, S. Liu, D. Tompkins, Z. Chen, W. Che, X. Yu, and F. Wei, “Beats: Audio pre-training with acoustic tokenizers,” in International Conference on Machine Learning. PMLR, 2023, pp. 5178–5193.
[https://dl.acm.org/doi/abs/10.5555/3618408.3618611](https://dl.acm.org/doi/abs/10.5555/3618408.3618611)
