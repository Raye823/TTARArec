# ReAd: Retrieve-then-Adapt

## Introduction

This is the code for our www 2026 paper: **ReAd: Retrieve-then-Adapt: Retrieval-Augmented Test-Time Adaptationfor Sequential Recommendation**.

> 📄 **Paper**: [Coming Soon]

## Environment Dependencies

This project uses the following versions of Python and PyTorch:

- **Python**: >3
- **PyTorch**: torch==2.5.1+cu124

For a more detailed list of dependencies, please refer to the `requirements.txt` file.

### Installation

```bash
pip install -r requirements.txt
```

---

## Dataset

### Dataset Format

All datasets should be placed in the `recbole/dataset/` directory. Each dataset must contain at least the following three columns:

```
user_id:token    item_id:token    timestamp:float
```

The project includes the **Amazon Beauty** dataset by default. Other datasets need to be imported manually.

### Dataset Sources

#### MovieLens Datasets
- **Official Site**: https://grouplens.org/datasets/movielens/
- **Recommended**: ml-1m, ml-10m

#### Amazon Review Datasets
- **Official Site**: http://jmcauley.ucsd.edu/data/amazon/
- **Categories**: Beauty, Sports and Outdoors, Office Products, Home and Kitchen, etc.

#### Preprocessed Datasets (Recommended)
For your convenience, we provide preprocessed datasets compatible with RecBole:
- **Google Drive**: https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj

Simply download and extract the datasets to the `recbole/dataset/` directory.

---

## How to Run

### 1. Evaluate Pre-trained ReAd Models

To evaluate a pre-trained ReAd checkpoint:

```bash
# Amazon Beauty
python run_read.py -e -mp pretrainmodel/TTARARec/Amazon_Beauty/model.pth

# Amazon Sports and Outdoors
python run_read.py -e -mp pretrainmodel/TTARARec/Amazon_Sports_and_Outdoors/model.pth

# Amazon Office Products
python run_read.py -e -mp pretrainmodel/TTARARec/Amazon_Office/model.pth

# Amazon Home and Kitchen
python run_read.py -e -mp pretrainmodel/TTARARec/Amazon_Home/model.pth

# MovieLens-1M
python run_read.py -e -mp pretrainmodel/TTARARec/ml-1m/model.pth
```

### 2. Train ReAd on Pre-trained Backbone Models

#### 2.1 Using DuoRec as Backbone

```bash
# Amazon Beauty
python run_read.py -d Amazon_Beauty -pt duorec -pp pretrainmodel/DuoRec/Amazon_Beauty/model.pth

# Amazon Sports and Outdoors
python run_read.py -d Amazon_Sports_and_Outdoors -pt duorec -pp pretrainmodel/DuoRec/Amazon_Sports_and_Outdoors/model.pth

# Amazon Office Products
python run_read.py -d Amazon_Office -pt duorec -pp pretrainmodel/DuoRec/Amazon_Office/model.pth

# Amazon Home and Kitchen
python run_read.py -d Amazon_Home -pt duorec -pp pretrainmodel/DuoRec/Amazon_Home/model.pth

# MovieLens-1M
python run_read.py -d ml-1m -pt duorec -pp pretrainmodel/DuoRec/ml-1m/model.pth
```

#### 2.2 Using SASRec as Backbone

```bash
# Amazon Beauty
python run_read.py -d Amazon_Beauty -pt sasrec -pp pretrainmodel/SASRec/Amazon_Beauty/model.pth

# Amazon Sports and Outdoors
python run_read.py -d Amazon_Sports_and_Outdoors -pt sasrec -pp pretrainmodel/SASRec/Amazon_Sports_and_Outdoors/model.pth

# Amazon Office Products
python run_read.py -d Amazon_Office -pt sasrec -pp pretrainmodel/SASRec/Amazon_Office/model.pth

# Amazon Home and Kitchen
python run_read.py -d Amazon_Home -pt sasrec -pp pretrainmodel/SASRec/Amazon_Home/model.pth

# MovieLens-1M
python run_read.py -d ml-1m -pt sasrec -pp pretrainmodel/SASRec/ml-1m/model.pth
```

### 3. Train Backbone Models from Scratch

#### 3.1 Train DuoRec Example

```bash
python run_basic.py --dataset Amazon_Beauty --config_file duorec.yaml --model DuoRec
```

#### 3.2 Train SASRec Example

```bash
python run_basic.py --dataset Amazon_Beauty --config_file sasrec.yaml --model SASRec
```

---

## Notes

- **Randomness**: Due to Faiss indexing randomness, results may vary slightly with different random seeds. Try adjusting the seed in config files to optimize performance.

- **Backbone Quality**: Better backbone models lead to better ReAd performance. Consider fine-tuning hyperparameters or training longer for improved results.

---

## Project Structure

```
./
├── recbole/                    # RecBole framework
│   ├── model/
│   │   └── sequential_recommender/
│   │       ├── ttararec.py    # ReAd model implementation
│   │       ├── duorec.py      # DuoRec baseline
│   │       └── sasrec.py      # SASRec baseline
│   ├── dataset/               # Dataset directory
│   │   └── Amazon_Beauty/
│   └── utils/
│       └── ttararec_utils.py  # ReAd evaluation utilities
├── run_read.py                # Main script for ReAd
├── run_basic.py              # Script for training baseline
├── ReAd.yaml                 # ReAd configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{read2025,
  title={ReAd: Retrieval-Augmented Recommendation for Sequential Recommendation},
  author={[Authors]},
  booktitle={[Conference]},
  year={2025}
}
```

---

## Acknowledgments

This project is built upon the [RecBole](https://github.com/RUCAIBox/RecBole) framework. We thank the RecBole team for their excellent work.

---

## License


---

## Contact

For any questions or issues, please open an issue on GitHub or contact:

- 📧 Email: 

