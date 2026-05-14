# Project summary for resume/GitHub

**RNA 3D Folding with MSA-Enhanced DistTransformer**

Built a PyTorch Transformer pipeline to predict local RNA 3D pairwise distance matrices from sequence windows and MSA profile features. The model encodes 21-residue RNA windows using nucleotide identity and evolutionary profile frequencies, then predicts normalized 21×21 pairwise distance matrices through a Transformer encoder and pairwise MLP head.

Reported notebook results on an 8,000-window sampled subset:

- Test RMSE: 0.0612
- Test MAE: 0.0394
- Pearson correlation: 0.8248
- Per-window RMSE mean: 0.0569

Resume bullet:

> Developed an MSA-enhanced Transformer pipeline for RNA 3D local structure modeling, encoding 21-residue sequence windows into one-hot and MSA-profile features to predict pairwise distance matrices with 0.0612 test RMSE and 0.8248 Pearson correlation.
