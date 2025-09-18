# Train Once for All: A Transitional Approach for Efficient Aspect Sentiment Triplet Extraction

This repository contains the implementation of a transition-based model for Aspect-Opinion Pair Extraction (AOPE) and Aspect Sentiment Triplet Extraction (ASTE) tasks, achieving state-of-the-art performance with linear time complexity O(n).

## 📋 Overview

Our transition-based approach addresses two key challenges in aspect-based sentiment analysis:
- **Disconnected Aspect-Opinion Extraction**: Unlike methods that extract aspects and opinions independently, our model performs joint extraction through transition actions
- **Linear Time Complexity**: While existing methods face O(n²) complexity, our transition-based parser operates in O(n) time

### Key Features

✅ First transition-based model for AOPE/ASTE tasks  
✅ Linear time complexity O(n) vs quadratic O(n²) for existing methods  
✅ Contrastive-augmented optimization for improved performance  
✅ Superior cross-domain generalization when trained on multiple datasets  
✅ State-of-the-art results on 4 benchmark datasets  

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 💻 Quick Start

### Training
```bash
python train.py
```

## 📊 Datasets

The model is evaluated on four SemEval ABSA datasets:

| Dataset | Domain | Sentences | Triplets |
|---------|--------|-----------|----------|
| 14lap | Laptop reviews | 1,453 | 2,349 |
| 14res | Restaurant reviews | 2,068 | 3,909 |
| 15res | Restaurant reviews | 1,075 | 1,747 |
| 16res | Restaurant reviews | 1,393 | 2,247 |

## 🏆 Performance Results

### AOPE Task (F1 Scores)

| Dataset | In-domain | Combined Training | Improvement |
|---------|-----------|------------------|-------------|
| 14res   | 71.86     | **88.02**       | +16.16      |
| 14lap   | 60.43     | **85.94**       | +25.51      |
| 15res   | 89.08     | **93.48**       | +4.40       |
| 16res   | 79.91     | **89.06**       | +9.15       |

### ASTE Task (F1 Scores)

| Dataset | In-domain | Combined Training | Improvement |
|---------|-----------|------------------|-------------|
| 14res   | 65.92     | **85.20**       | +19.28      |
| 14lap   | 53.36     | **81.26**       | +27.90      |
| 15res   | 86.98     | **92.42**       | +5.44       |
| 16res   | 77.95     | **86.55**       | +8.60       |

## 🏗️ Model Architecture

### Components
- **Encoder**: RoBERTa for text encoding
- **Transition Parser**: BiLSTM for state representation  
- **Optimization**: Cross-entropy loss + Contrastive loss (1:1 ratio)

### Transition Actions
1. **Shift (SF)**: Move token from stack to buffer
2. **Stop (ST)**: Halt when only one token remains
3. **Merge (M)**: Combine multiple tokens
4. **Left Constituent Removal (Ln)**: Remove left constituent
5. **Right Constituent Removal (Rn)**: Remove right constituent
6. **Left-Relation Formation (LR)**: Create right→left relation
7. **Right-Relation Formation (RR)**: Create left→right relation

## 🔬 Baseline Comparisons

### 1. MiniConGTS SOTA Reproduction
- [Colab Notebook](https://colab.research.google.com/drive/1Bfy4fwz5_Gh7JP-R3z0iXowfowXCwlIL?usp=sharing)
- Re-implementation of MiniConGTS on fused datasets

### 2. BARTABSA (2021 Version)
- Original repository: [https://github.com/yhcc/BARTABSA/tree/main](https://github.com/yhcc/BARTABSA/tree/main)
- **Setup Instructions**:
  1. Merge train data from `peng/data`
  2. Modify training set in `peng/train.py`
  3. Run on respective test sets of the four datasets

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{hou2024train,
  title={Train Once for All: A Transitional Approach for Efficient Aspect Sentiment Triplet Extraction},
  author={Hou, Xinmeng and Fu, Lingyue and Meng, Chenhao and Du, Kounianhua and Wang, Wuqi and Hu, Hai},
  journal={arXiv preprint},
  year={2024}
}
```

## 📧 Contact

- **Xinmeng Hou**: fh2450@tc.columbia.edu  
- **Hai Hu** (Corresponding Author): hu.hai@cityu.edu.hk

## 🙏 Acknowledgments

This project is funded by Shanghai Pujiang Program (22PJC063) awarded to Hai Hu.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Code Availability

Code is available at: https://anonymous.4open.science/r/trans_aste-8FCF
