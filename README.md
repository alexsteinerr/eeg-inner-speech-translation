## Summary

This repository contains the code accompanying the paper [*A Novel Machine Learning Model for Non-Invasive EEG-Based Inner-Speech Translation in ALS*](https://ieeexplore.ieee.org/document/11435085), presented and published both as a poster and part of the conference proceedings at the IEEE 14th International Winter Conference on Brain-Computer Interface (BCI 2026) in South Korea.

## Dataset

The EEG model is trained on the **Chisco dataset**, which contains **30,000 sentences across 5 patients**. This is a pre-existing dataset that is separate from the sEMG data collection pipeline.

> Zhang, Z., Ding, X., Bao, Y. et al. Chisco: An EEG-based BCI dataset for decoding of imagined speech. Sci Data 11, 1265 (2024). https://doi.org/10.1038/s41597-024-04114-1

### Model Architecture

The EEG-to-text generation architecture employs a region-aware encoding strategy that processes cortical regions independently before fusion:

<h1 align="center">
  <img src="figures/ml_architecture.png" alt="EEG Model Architecture" style="height: 80%; width:80%; align:center">
</h1>

**Fig. 1.** Overview of the proposed EEG-to-text generation architecture. Each of the cortical regions (frontal, temporal, central, and parietal) is processed using a dedicated encoder composed of three 1D convolutional sublayers followed by multi-head attention and normalization stacked sublayers. The produced embeddings are input to region-specific MLP fusion sublayers activated using GELU and dropout. A cross-region fusion module with anti-collapse regularization integrates distributed cortical representations into a unified latent embedding, which is projected linearly to k learned prefix tokens. These prefix tokens precondition a LoRA and PEFT-trained transformer decoder for EEG-conditioned text generation.

**Key Components:**

1. **Region-Specific Encoders**: 
   - Each cortical region (frontal, temporal, central, parietal) has a dedicated encoder
   - Three 1D convolutional sublayers extract temporal features
   - Multi-head attention and normalization layers capture region-specific patterns

2. **Region-Specific MLP Fusion**:
   - GELU activation and dropout for regularization
   - Combines features within each cortical region

3. **Cross-Region Fusion Module**:
   - Integrates distributed cortical representations
   - Anti-collapse regularization prevents mode collapse
   - Produces unified latent embedding

4. **Prefix Projection**:
   - Linear projection to k learned prefix tokens
   - Conditions the transformer decoder on EEG features

5. **LoRA-PEFT Transformer Decoder**:
   - DeepSeek-R1-Distill-Qwen-1.5B as the base model
   - LoRA (Low-Rank Adaptation) and PEFT (Parameter-Efficient Fine-Tuning) for efficient training
   - Generates text conditioned on the EEG prefix tokens

> **Detailed Explanation**: For comprehensive model architecture details, training methodology, and experimental results, see [`Steiner (2026)`](https://ieeexplore.ieee.org/document/11435085).

## Results

#### Main Results

Our model was evaluated over three random seeds. Results are reported as mean ± standard deviation, with 95% confidence intervals (CI). The model achieved a BLEU-1 score of **0.512 ± 0.012** and a ROUGE-L F1 score of **0.396 ± 0.009**. Statistical comparisons against the Transformer baseline indicated significant improvements in both metrics (*p* < 0.01), with Cohen's *d* ≥ 1.0 indicating a large effect size.

| Metric | Value | 95% CI |
|:---|:---:|:---:|
| BLEU-1 | 0.512 ± 0.012 | [0.482, 0.542] |
| ROUGE-L F1 | 0.396 ± 0.009 | [0.374, 0.418] |
| *t*-test vs. Transformer (BLEU-1) | *t* = 4.62 | *p* < 0.01 |
| *t*-test vs. Transformer (ROUGE-L) | *t* = 4.21 | *p* < 0.01 |
| Effect size (Cohen's *d*) | ≥ 1.0 | — |

#### Comparison with Existing Baselines

The following table compares our model with previously reported non-invasive EEG decoding baselines evaluated on the ZuCo corpus. Baseline values are illustrative approximations drawn from the cited works and should not be interpreted as results from a controlled, like-for-like evaluation.

| Method and Corpus | BLEU-1 | ROUGE-L F1 |
|:---|:---:|:---:|
| Amrani et al. (2023) — ZuCo | ≈ 0.428 | ≈ 0.332 |
| CET-MAE (Wang et al., 2024) — ZuCo | ≈ 0.421 | ≈ 0.326 |
| EEG2TEXT (Multi-View) — ZuCo | ≈ 0.452 | ≈ 0.342 |
| **Our Model** | **0.512** | **0.396** |

#### Ablation Study

To evaluate the contribution of individual architectural components, we conducted ablation experiments in which selected components were removed. Performance changes are reported relative to the full model; negative values indicate a reduction in performance.

| Removed Component | ΔBLEU-1 | ΔBLEU-4 | ΔROUGE-L |
|:---|---:|---:|---:|
| Cross-regional attention | −0.058 | −0.047 | −0.053 |
| Uniform region weighting | −0.035 | −0.020 | −0.036 |
| Positional embeddings | −0.021 | −0.014 | −0.025 |
| Per-region CNN only (no self-attention) | −0.114 | −0.076 | −0.118 |
| No cross-modal fusion transformer | −0.074 | −0.060 | −0.080 |
| Pure CNN baseline (no attention) | −0.097 | −0.102 | −0.139 |

The ablation results suggest that **cross-regional attention and cross-modal fusion are important contributors to performance**. Removing cross-regional attention reduced BLEU-1 by 0.058 and ROUGE-L F1 by 0.053, while removing the cross-modal fusion transformer reduced these metrics by 0.074 and 0.080, respectively. The per-region CNN-only configuration also produced substantial performance degradation, highlighting the potential value of modelling interactions across EEG regions and integrating their representations.

Learned region weighting and positional embeddings provided additional improvements, suggesting that both region-specific contributions and positional information are useful for EEG-based inner-speech decoding.

## Quickstart

This section is intentionally high‑level; see the code in each subfolder for experiment‑specific details.

### 1. Clone and Base Dependencies

```bash
git clone https://github.com/alexsteiner/eeg-inner-speech-translation.git
cd eeg-inner-speech-translation
```

### 2. Repository Structure

- **Configuration:** `config/training_config.py`  
  - Points to EEG data from the Chisco dataset (`CONFIG['data_dir']`, montage file, save directory).  
  - Sets model hyperparameters (e.g. `hidden_dim`, `decoder_prefix_len`, LoRA settings, diversity penalties).
- **Training script:** `scripts/train.py`
  - Builds an `EEGDataset` from `src/data/dataset.py` and electrode regions from `src/data/utils.py`.
  - Loads preprocessed Chisco dataset files (30,000 sentences, 5 patients).
  - Splits into train/validation and uses `DataLoader` with configurable workers.
  - Creates `EEGModel` with region‑aware encoders and DeepSeek LLM.
  - Uses separate optimizers and learning rates for:
    - encoder, LLM, projection, and LoRA parameters (see `get_optimizer_groups` in config).
  - Logs training and validation losses, saves `best_model.pt` and `last_model.pt` to a timestamped run directory.
  - Periodically calls `generate_samples` to decode qualitative examples from EEG and log target vs prediction.

- **Python:** create a virtualenv and install at least the requirements: ```pip install -r requirements.txt ```

### 3. Train Model

```bash
cd eeg-inner-speech-translation
python scripts/train.py
```

### Acknowledgments

A special thanks to **Dr. Melissa Zavaglia** for her guidance and support throughout the research process.