# Moosic — Playlist Clustering

Automating playlist creation for a music startup using unsupervised machine learning — a
case-study project for the "Unsupervised ML" module of a data science course. Full write-up:
**[project wrap-up page](https://iliya-radulov.github.io/projects/active/moosic-clustering-wrapup/moosic-clustering-wrapup.html)** · **[presentation deck](/reports/moosic_presentation.html)**.

## The question

Can Spotify's audio features (danceability, energy, tempo, etc.) group songs the way a human
listener actually hears "similar" — and which clustering algorithm handles that best?

## The short answer

There is no single correct number of playlists sitting in the data waiting to be found.
"Good" has to be defined explicitly, as a business decision, before any algorithm can be
judged against it. Once that's done: **K-Means, with a business-constrained recursive
split/merge pipeline, is the practical recommendation** — DBSCAN and Hierarchical Clustering
were run in full, not as a formality, and both independently confirmed the same underlying
structure rather than offering a better alternative.

## What's actually in this repo

```
notebooks/          8 Jupyter notebooks, numbered in build order (01 → 08)
outputs/             Generated figures, CSVs, and intermediate results
reports/             Full session report (every bug, decision, and correction — kept, not
                     edited out) and the final presentation deck (HTML slides)
data/                Not included — see "Getting the data" below
```

Each notebook is self-contained (loads fresh from the CSV, doesn't depend on another
notebook's live memory) and follows one convention throughout: main pipeline steps are
numbered `1, 2, 3...`; experimental variants on a step are labelled `5a`, `5b`, etc., each
using its own uniquely-named variables so an exploratory cell can never silently corrupt the
main result.

| Notebook | Covers |
|---|---|
| `01_baseline_clustering.ipynb` | Feature selection, correlation analysis, free K-Means baseline |
| `02_recursive_pipeline.ipynb` | Business-constrained recursive split / reassign / merge |
| `03_cluster_deepdive.ipynb` | Inspecting and re-splitting a specific cluster |
| `04_dbscan.ipynb` | DBSCAN parameter sweep, noise-bucket investigation |
| `05_agglomerative.ipynb` | Hierarchical clustering, linkage comparison |
| `06_hybrid_pipeline.ipynb` | Agglomerative + K-Means hybrid, and its correction |
| `07_dbscan_on_subset.ipynb` | Re-running DBSCAN on its own mega-cluster |
| `08_cross_algorithm_agreement.ipynb` | Quantified agreement (ARI/NMI) across all three methods |

## Key findings

- **Feature selection wasn't arbitrary.** Started from 13 Spotify audio features, dropped
  `loudness` (redundant with energy/acousticness via correlation analysis), kept 7 —
  including `speechiness` and `instrumentalness`, missing from the course's own example code,
  which turned out to carry real signal.
- **The size-target question has no algorithmic answer.** Elbow/silhouette methods alone
  suggested 3–5 clusters — mathematically defensible, practically useless (1,000+ songs per
  "playlist"). The actual playlist count (~130, sized 20–60) came from an explicit business
  constraint, not from the data.
- **DBSCAN never found a workable middle ground** between fragmentation and one dominant
  mega-cluster, across a full parameter sweep, two different scalers, and a targeted re-run
  on its own mega-cluster subset. But its "noise" bucket wasn't empty — it contained a real,
  identifiable rap/hip-hop pocket the other methods hadn't isolated yet.
- **A hybrid pipeline (Agglomerative superclusters, then K-Means within each) looked like a
  clear improvement — until a specific example was checked by ear**, not just by the numbers.
  The "worst offending" single-stage playlist turned out to be a completely coherent jazz &
  bossa nova set the hybrid's rigid category boundaries would have split apart. Documented as
  a genuine tradeoff, not papered over as a clean win.
- **Cross-algorithm agreement was measured, not eyeballed:** K-Means and Agglomerative agree
  strongly (ARI 0.555) despite entirely different assumptions. On the one distinction DBSCAN
  committed to — the classical/instrumental cluster — agreement with K-Means is **perfect
  (ARI = 1.000)**.

## Getting the data

The song dataset (`5000_songs.csv`, Spotify audio features) is course material and isn't
redistributed here. Any similarly-structured Spotify audio-features dataset will work with
the notebooks as-is — the feature list is defined explicitly at the top of each notebook.

## Stack

Python · pandas · scikit-learn (KMeans, DBSCAN, AgglomerativeClustering) · scipy (hierarchical
linkage) · matplotlib / seaborn · Jupyter

## Author's note

Individual project, built in close collaboration with Claude (Anthropic) throughout — from
first confusion about whether clustering was even the right tool, through nine documented
bugs and one self-corrected finding, to a final, quantitatively cross-validated
recommendation. The session report keeps every wrong turn in, on purpose.
