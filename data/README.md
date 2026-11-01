# Data

This directory is for local dataset files used by the imbalance selective-classification study.

Raw files are intentionally ignored by git. The current local paths expected by the configs are:

- `data/raw/baf/Base.parquet`
- `data/interim/baf_split_assignments.parquet`
- `data/raw/online_shoppers/online_shoppers_intention.csv`
- `data/raw/secom/secom.data`
- `data/raw/secom/secom_labels.data`

BAF Base can be installed with:

```bash
.venv/bin/python scripts/download_baf.py --config configs/imbalance_selective/baf.yaml
```

If Kaggle access fails but a local raw `Base.csv` or `Base.parquet` exists, install from that local source:

```bash
.venv/bin/python scripts/download_baf.py \
  --config configs/imbalance_selective/baf.yaml \
  --source-file /path/to/Base.parquet
```

## Sources

- Online Shoppers Purchasing Intention Dataset: <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset>
- SECOM: <https://archive.ics.uci.edu/dataset/179/secom>
- BAF: <https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data>
