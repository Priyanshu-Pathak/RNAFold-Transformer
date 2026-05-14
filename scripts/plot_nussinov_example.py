from pathlib import Path

from rna3d_folding.nussinov import nussinov
from rna3d_folding.plotting import plot_nussinov_arc

sequence = "GGGAAACCCUUUAAAGGGCCC"
dot_bracket, pairs = nussinov(sequence)
print(sequence)
print(dot_bracket)
plot_nussinov_arc(sequence, pairs, Path("outputs/figures/nussinov_example.png"))
