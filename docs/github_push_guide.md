# Push this project to an empty GitHub repository

After unzipping this folder, open a terminal inside the folder and run:

```bash
git init
git branch -M main
git add .
git commit -m "Initial commit: add RNA 3D folding DistTransformer pipeline"
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

If the remote already exists:

```bash
git remote set-url origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

Recommended repository description:

```text
MSA-enhanced Transformer pipeline for RNA 3D local pairwise distance prediction using the Stanford RNA 3D Folding dataset.
```

Suggested GitHub topics:

```text
rna-3d-folding deep-learning pytorch transformer bioinformatics computational-biology kaggle
```
