# Dataset location

Large dataset files are intentionally not included in this repository.

Download or attach the Kaggle dataset, then arrange it like this:

```text
data/
├── train_labels.v2.csv
├── train_sequences.xlsx        # or train_sequences.csv
└── MSA/
    ├── <target_id>.MSA.fasta
    └── ...
```

Dataset link: https://www.kaggle.com/competitions/stanford-rna-3d-folding/data

On Kaggle, the dataset path may look like:

```text
/kaggle/input/stanford-rna-3d-folding/stanford-rna-3d-folding
```

In that case, either copy files into `data/` locally or pass the path directly:

```bash
python -m rna3d_folding.train --data-dir /kaggle/input/stanford-rna-3d-folding/stanford-rna-3d-folding
```
