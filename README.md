# APARENT1-poly-A-cleavage-usage-scoring-pipeline
Scores how much masking a short target site (for example the binding site of an antisense oligo) changes the poly(A) signal predicted by the original **APARENT1** model
([johli/aparent](https://github.com/johli/aparent)) in a gene region of interest. Three step
scripts plus a shared `common.py` (keep them in one folder); everything project-specific lives
in one JSON config.

| Script | Covers | Output |
|---|---|---|
| `01_build_sequences.py` | fetch ROI + flank, convert to mRNA sense, locate each oligo target | `sequences/<mode>/ref_mrna.txt`, `meta.json`, `oligo_targets.tsv` |
| `02_score_aparent1.py hexamer` | canonical-hexamer windows; reference vs. N-masked (`--mask n`) or scrambled (`--mask scramble`) | `scores/hexamer_scores_<mask>.csv` |
| `02_score_aparent1.py peaks` | PAS peak detection on the reference, log-odds scoring of every peak, reference vs. N-masked | `scores/peak_scores_Nmask_all.csv`, `..._score_regions.csv` |
| `03_plot_bars.py` | bar plots per oligo | `figures/*.png` |

```
reference genome ─01─▶ mRNA-sense ROI ─02 hexamer/peaks (APARENT1)─▶ delta table ─03─▶ bar plots
oligos.csv ─────────┘ (target = reverse complement of the oligo)
```

## 1. Requirements

- Python 3.6 (the original model code needs TensorFlow 1.13 / Keras 2.2.4); CPU is enough.
- Environment: `conda env create -f environment.yml && conda activate aparent1`
- The `aparent` package and weights are **not** in conda/pip and are not copied here. Clone
  the upstream repository and point `aparent_dir` at it (the scripts add it to `sys.path`):
  ```bash
  git clone https://github.com/johli/aparent.git Path_to_aparent_checkout
  git -C Path_to_aparent_checkout checkout 69ad29791709b48689ff5d9e3a3daefc568de9ce   # version used originally
  ```
  The weights `saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5` are
  tracked in that repository (about 25 MB). The functions used are
  `aparent.predictor.get_aparent_encoder`, `find_polya_peaks`, `score_polya_peaks`.
  `aparent-resnet` (APARENT2-style models) is not needed here.
- A reference genome FASTA (a `.fai` index is created automatically if missing). Without
  one, set `"ucsc_genome": "hg38"` instead (original behaviour, needs internet and `requests`).
  Use the same assembly as your coordinates.
- A CSV/TSV of oligos with header `name,sequence` (5'->3', antisense to the mRNA).

## 2. Configure

Copy `config.example.json` to `config.json` and replace every `Path_to_...` and
`chrN:START-END`.

| Key | Meaning |
|---|---|
| `work_dir` | All outputs are written here |
| `reference_fasta` / `ucsc_genome` | Genome source (FASTA, or UCSC REST fallback) |
| `aparent_dir`, `model_h5` | Upstream checkout and the APARENT1 weights file |
| `oligos_csv` | Oligo table (`name,sequence`) |
| `locus.roi` | Region scanned for hexamers (0-based, half-open `chr:START-END`) |
| `locus.strand` | `+` or `-`; for `-` the mRNA-sense sequence is the reverse complement |
| `locus.regions` | Named sub-regions (exons/introns), inclusive `chr:START-END`; used to label sites |
| `locus.score_regions` | Names of the regions whose hexamers are scored (hexamer mode) and which are kept in the `*_score_regions.csv` (peak mode) |
| `hexamer` | `flank` (nt added each side, default 300), `hexamer_pos` (default 70), `hexamers` list (10 canonical + near-canonical PAS) |
| `masking.scramble_seed` | Seed for `--mask scramble` (default 42) |
| `peaks` | Peak finding/scoring parameters, defaults as in the original runs (see below) |
| `slurm` | Used by `submit` |

Analysing a different region of the same gene (e.g. the last exon cluster) needs only a
different `roi`, `regions` and `score_regions`; no code change.

## 3. Run

Add `--dry-run` before the subcommand to print the plan without touching the model or writing results.

```bash
python 01_build_sequences.py --config config.json
python 02_score_aparent1.py  --config config.json hexamer --mask n
python 02_score_aparent1.py  --config config.json hexamer --mask scramble
python 02_score_aparent1.py  --config config.json peaks
python 03_plot_bars.py --config config.json --input scores/hexamer_scores_n.csv
python 03_plot_bars.py --config config.json --input scores/hexamer_scores_n.csv --metric delta_iso
python 03_plot_bars.py --config config.json --input scores/hexamer_scores_n.csv --style grouped
python 03_plot_bars.py --config config.json --input scores/peak_scores_Nmask_score_regions.csv
```

`python 02_score_aparent1.py --config config.json submit hexamer --mask n` writes
`slurm/<step>.sbatch` and calls `sbatch`.

**GPU / cluster note.** The original runs used 4 CPUs, 16 GB, 1-4 h and no GPU; set
`slurm.gpu: true` only if you really need one. Always submit with `sbatch` (the `submit`
command does this); never hold a GPU with a blocking `srun`. Use only your own conda environments.

## 4. How scoring is defined

APARENT1 takes a **205 nt** one-hot window (mRNA sense) and returns an isoform probability
(`iso[0, 0]`, usage of the proximal site) and a 206-long cleavage vector (positions 0-204
plus a no-cut class at index 205). Any character outside A/C/G/T (`N`, `X`) is all-zero
one-hot, which is how masking and padding work.

**Hexamer mode.** Each hexamer found inside `score_regions` is placed at index 70 of the
window. For reference (WT) and masked sequence:
`iso` = isoform probability; `peak_cut` = max of cleavage positions 0-204;
`composite` = `iso x peak_cut`. Masked sequence = the first occurrence of
`reverse_complement(oligo)` replaced by `N` (or by a seeded shuffle of the same bases). A
hexamer/oligo pair is only reported when the masked window differs from the reference window.

**Peak mode.** The mRNA-sense ROI (+ flank) is padded with 500 `X` on both sides, peaks are
found with `find_polya_peaks` on the reference (stride 5, min height 0.05, min distance 100,
prominence >= 0.05), and every peak is scored by `score_polya_peaks` (window offsets 0-29,
`max` aggregation, mean of isoform and cut-sum scores, log-odds). The *same peak positions*
are scored again on the N-masked sequence.

Output columns, hexamer CSV (sorted by `abs_delta_composite`):

| Column | Meaning |
|---|---|
| `oligo`, `polyA_hexamer` | Oligo name; hexamer sequence |
| `hexamer_genomic_pos`, `hexamer_region` | 0-based position of the hexamer's first mRNA-sense base; region label |
| `WT_iso_prob`, `masked_iso_prob` | Isoform probability, reference / masked |
| `WT_peak_cut_prob`, `masked_peak_cut_prob` | Maximum cleavage probability, reference / masked |
| `WT_composite`, `masked_composite` | `iso x peak_cut` |
| `delta_iso`, `abs_delta_iso` | masked - WT isoform probability (and absolute value) |
| `delta_composite`, `abs_delta_composite` | masked - WT composite (and absolute value); main ranking metric |

Peak CSV: `oligo`, `oligo_target_pos`/`oligo_target_region` (where the mask was placed),
`peak_genomic_pos`/`peak_region`, `WT_log_odds`, `masked_log_odds`, `delta` (masked - WT,
log-odds), `abs_delta`, `in_score_regions`. Negative delta = masking lowers poly(A) signal.

Plots: one bar per oligo from the site/peak with the largest |delta| (not the best positive
one), green up / red down.

## 5. Things that commonly go wrong

- **Wrong Python.** TF 1.13 / Keras 2.2.4 need Python 3.6; a newer interpreter fails at import.
- **Where the code lives.** `find_polya_peaks`/`score_polya_peaks` are in the `aparent`
  checkout (`aparent.predictor`); the resnet repository does not provide them.
- **Assembly / chromosome naming.** Coordinates, FASTA (or `ucsc_genome`) must match;
  `chr1` vs `1` gives "not in FASTA".
- **Coordinate convention.** Everything is compared against 0-based positions; `roi` is
  half-open, `regions` are inclusive. Annotation coordinates taken 1-based can be 1 nt off at
  region borders.
- **Oligo not found.** The target is searched only as an exact reverse complement and only the
  first occurrence is masked. Check `sequences/<mode>/oligo_targets.tsv` (`found=no`).
- **Oligos without effect.** A hexamer whose window does not contain the target is silently
  not listed; with a scramble mask the result can be 0 if the shuffle equals the original.
- **Windows out of bounds.** A hexamer within 70 nt of the fetched start (or 135 nt of the
  end) is skipped; increase `hexamer.flank`.
- **Peaks depend on the flank/padding.** Changing `peaks.flank` or `x_pad` changes which peaks
  are detected; keep them fixed when comparing oligos. A peak closer than about 75 nt to the
  sequence start gets the model's sentinel score -10, which makes `delta` meaningless.
- **Stale plot inputs.** Plot a CSV from the same run; the plotting step needs the column
  names above.
