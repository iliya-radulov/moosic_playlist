# Moosic Clustering Project — Session Report

**Case study:** Moosic — automating playlist creation from Spotify audio features
**Dataset:** 5000 songs, features: danceability, energy, acousticness, tempo, valence
**Algorithm:** K-Means (baseline for comparison against DBSCAN/Agglomerative, planned as future work)

---

## 1. Starting Question

The project began from a genuine point of confusion: with 5000 songs, a "fine" clustering
(many clusters) produces too many small, hard-to-use groups, while a coarse clustering
(few clusters) produces oversized, musically meaningless groups. This raised the question
of whether K-means-based playlist generation is a realistic technique or just a teaching
exercise.

**Conclusion reached:** it's a real technique, but it's a stepping stone, not a complete
solution on its own. Inertia and the silhouette score measure geometric structure — how
tight and how separated clusters are — but neither has any concept of what makes a good
*playlist*. This is precisely the tension the case study's "Aligning Clustering with
Business Goals" section addresses: mathematically clean clusters and musically meaningful
ones are not guaranteed to be the same thing. The rest of this session was spent building
a pipeline that treats the business constraint (a usable playlist size) as an explicit
input to the algorithm, rather than hoping it emerges from the metrics alone.

---

## 2. Auditing the Existing Sandbox

Before building anything new, the existing scripts (`moosic.py`, `scale.py`, `test.py`)
were reviewed and several concrete issues were found:

| Script | Issue | Cause |
|---|---|---|
| `moosic.py` | `KeyError` risk | Feature name typo: `'acoustiness'` instead of `'acousticness'` |
| `test.py` | Inconsistent path | Loaded `../data/songs.csv` while other scripts used `songs.csv` |
| `moosic.py` | k chosen inconsistently | Computed inertia/silhouette across k=2–9, then hardcoded `best_k=2` regardless of the analysis |
| `scale.py` | k chosen automatically | Picked k via `argmax(silhouette)` with no human judgement — exactly the anti-pattern the business-alignment lesson warns against |
| `test.py` | k chosen arbitrarily | Hardcoded `k=3`, no metric computed at all |

Each script used a different, uncoordinated philosophy for choosing k — this was the root
cause of inconsistent, hard-to-compare results across the sandbox.

---

## 3. Methodology Built

### 3.1 Baseline ("free") clustering

Following the elbow/silhouette method from the course notebooks, a baseline clustering was
run on the full 5000-song dataset with no target size imposed. The silhouette curve showed
a knee around **k=8**.

**Bug found and fixed — radar chart axis mismatch.** The first radar chart of the 8
clusters showed all curves nearly overlapping. Root cause: cluster centroids were
inverse-transformed back into original units (e.g. tempo in BPM) but plotted on an axis
hardcoded to `(0, 1)` — meant for scaled data. This silently clipped/distorted every value.
**Fix:** plot the *scaled* centroids (already 0–1 under `MinMaxScaler`) directly, and keep
the inverse-transformed values only for the printed reference table.

**Two indentation bugs found and fixed**, both of the same shape: a `for` loop meant to run
"once per cluster" was mistakenly un-indented to sit *outside* its parent loop, so it only
ran once total, using leftover data from the last cluster processed. This affected both the
terminal playlist printout and the generated Markdown report.

At k=8 (5000 songs), cluster sizes ranged roughly 380–1180 songs each — confirming the
"free" clustering, left alone, does not produce anything close to playlist-sized groups.

### 3.2 Business constraint: target playlist size

**Decision: target playlist size = 20–60 songs.**

This number is the single most important design choice in the whole pipeline — see the
parameter sensitivity results in Section 5.

### 3.3 Recursive splitting

Rather than forcing a single global k, clusters are split **recursively**: any cluster
larger than `TARGET_MAX` is re-clustered on just its own members (fresh local KMeans, small
k), and this repeats until every leaf is under the target size or hits a floor
(`MIN_SONGS_TO_SPLIT`).

Key design decision: all branches share **one global scaler**, fit once on the full
dataset — not a fresh scaler per branch — so centroids from different branches remain
comparable to each other later.

### 3.4 Global reassignment pass

Recursive splitting is *locally* greedy: a song can end up in a branch only because it was
the least-wrong option available at that point in the tree, even if a completely different,
already-finished branch would now be a better fit. **Fix:** after all leaves are
determined, take every leaf centroid together and run one flat nearest-centroid pass over
all 5000 songs, reassigning anything closer to a different leaf's centroid.

### 3.5 Merging undersized leaves

Because the stopping rule only checks the *upper* bound (`n <= TARGET_MAX`), tiny leaves
(sometimes 1–2 songs) are a near-guaranteed side effect of uneven splits — not a data
problem, a structural one. A merge step folds any leaf below `TARGET_MIN` into a
neighboring cluster.

This step went through **three bug-fix iterations**, documented here because the debugging
process itself is part of the learning record:

1. **Order-of-operations bug:** merging was originally done *before* the global
   reassignment pass, so reassignment silently discarded the merge and reintroduced tiny
   clusters. **Fix:** reassign first, then merge the *result* of reassignment, and do not
   reassign again afterward.
2. **Unbounded merge target bug:** the merge function had no upper-size check, so
   undersized leaves could snowball a single cluster to 100+ songs. **Fix:** prefer merge
   targets that stay under `TARGET_MAX`, falling back only if nothing fits.
3. **Fake "nearest neighbor" bug:** `pairwise_distances_argmin` was called but its result
   was never used — the merge target was actually just the first entry in an unsorted
   candidate list, not the true nearest cluster. This caused one specific cluster to
   repeatedly "win" merges by list position, producing a single cluster of ~2200 songs in
   one test run. **Fix:** compute real Euclidean distances between centroids, sort
   nearest-to-farthest, and walk that order for a valid merge target.

---

## 4. Experiment Sweeps — Parameter Sensitivity

Multiple sweeps were run varying `target_min`/`target_max`, `min_songs_to_split`, and
`max_k_per_split`, logged to a running CSV.

### Finding 1 — target range dominates every other parameter

| Range | Width | % in range |
|---|---|---|
| 12–20 | 8 | 61–70% |
| 15–25 | 10 | 65% |
| 30–40 | 10 | 24–35% |
| 15–30 | 15 | 79–85% |
| 20–40 | 20 | 81–87% |
| 30–60 | 30 | 84–87% |
| 20–60 | 40 | 96–98% |

Wider windows perform better, as expected — but **width alone doesn't fully explain the
results.** 15–25 and 30–40 are both 10 wide, yet 15–25 outperforms 30–40 by more than 2×.
This shows that the *position* of the target window relative to the data's natural leaf-size
distribution matters as much as its width — the recursive splitter naturally produces a lot
of leaves in the low-20s range, so windows that include that region perform much better than
windows that don't, independent of width.

### Finding 2 — `max_k_per_split` sensitivity scales with window narrowness

| Window width | Spread in % across `max_k_per_split` values |
|---|---|
| 8 | ~9 points |
| 15 | ~5.5 points |
| 20 | ~6 points |
| 40 | ~1.5 points |

On a wide target window there's enough slack that any reasonable branching factor works. On
a narrow window, there's much less room for error, so how coarsely each split branches
matters far more.

### Finding 3 — `min_songs_to_split` has **zero effect**, structurally

Confirmed identically across every sweep, including a direct retest at 5, 10, 15, 20 on the
same window: results were identical to the decimal. Root cause: the stopping condition is
`if n <= TARGET_MAX or n < MIN_SONGS_TO_SPLIT`. Since `TARGET_MAX` was larger than every
`MIN_SONGS_TO_SPLIT` value tested, the first condition always fires first as `n` shrinks —
the second condition is mathematically unreachable under these settings. This is not a
tuning result, it's a logical guarantee of the current code.

---

## 5. Case Study: Investigating the Largest Remaining Cluster

Even after the merge fix, one cluster consistently remained oversized relative to the
20–60 (and stricter) target windows. Rather than assuming this was still a bug, it was
investigated directly:

- **Feature comparison vs. dataset average:** danceability +0.21, energy +0.12, valence
  +0.14, tempo −9 BPM — a real, upbeat/danceable profile, not a "sits near the average of
  everything" generic bucket.
- **Within-cluster standard deviation:** tight across danceability, energy, and valence —
  confirming this is an internally coherent group, not a merge-artifact grab-bag.
- **Manual listen/inspection:** largely consistent with dance-pop/reggaeton
  ("Tusa," "Perfect Strangers," "Don't You Worry Child") — but **"Teenage Dirtbag"**, an
  alt-rock track, was also present, almost certainly grouped in on tempo/energy alone. This
  is a concrete, specific example of the case study's opening question — audio features
  cannot fully capture genre or "feel" the way a human listener does.
- **Language mixing:** English, Spanish, German, and Italian titles all appeared in the
  same cluster — expected, since none of the five chosen features encode language, a
  limitation the case study explicitly names.

**Conclusion:** this cluster is genuinely coherent by the numbers, just oversized relative
to a usable playlist length. Structural soundness and business-appropriate size are two
different properties, and this cluster is a clean, documented example of a case where they
diverge.

---

## 6. Recursive Re-Split of the Largest Cluster

The 97-song cluster above was tested for further splitting.

| k | Silhouette | Sizes |
|---|---|---|
| 2 | 0.413 | 21, 76 |
| 3 | 0.335 | 26, 20, 51 |
| 4 | 0.300 | 20, 21, 30, 26 |

Silhouette score decreases monotonically as k increases — **k=2 is the best-supported
split, not merely the simplest option.**

**Result at k=2:**
- Sub-cluster 0 (21 songs): tempo ~130 BPM, energy 0.79 — faster, driving tracks
- Sub-cluster 1 (76 songs): tempo ~104 BPM, valence 0.625, higher acousticness — slower,
  warmer tracks

**Notable correction:** based on the song list alone, this split initially looked like it
might follow a genre/language line (Latin/reggaeton vs. EDM/dance-pop). The actual feature
data shows the real separating axis is **tempo and energy** — language and apparent genre
cut across both sub-clusters. This is a useful, honest finding: even direct human inspection
of song titles reached for the wrong explanation until the underlying numbers were checked.

---

## 7. Summary of Bugs Found and Fixed

| # | Bug | Symptom | Fix |
|---|---|---|---|
| 1 | Feature name typo | `KeyError` on `'acoustiness'` | Corrected spelling |
| 2 | Inconsistent file paths | Script only worked from one specific working directory | Standardized relative paths |
| 3 | k chosen inconsistently across scripts | Non-comparable results | Adopted one documented target-range-driven method |
| 4 | Radar chart axis mismatch | All clusters appeared to overlap | Plot scaled centroids, not inverse-transformed ones |
| 5 | Indentation bug (×2) | Loops ran once instead of per-cluster | Correct indentation |
| 6 | Merge before reassignment | Merge silently undone, tiny clusters returned | Reassign first, merge the result |
| 7 | Unbounded merge target | One cluster could balloon past target size | Prefer merge targets under `TARGET_MAX` |
| 8 | Fake nearest-neighbor merge | `pairwise_distances_argmin` computed but unused; one cluster ballooned to ~2200 songs | Use real sorted distances for merge target selection |
| 9 | Log schema drift | Old and new CSV rows had different column counts, causing misleading duplicate-looking rows | Recommend a version/notes tag on future log rows |

---

## 8. Reflections — Aligning Clustering with Business Goals

The core lesson of this session, tied directly back to the course material: **inertia and
silhouette narrow down candidates, they don't make the final decision.** The actual
business constraint (a usable playlist length) had to be fed into the pipeline explicitly,
as a target range driving a recursive splitting/merging process — it never emerged on its
own from the metrics. Even after building that pipeline correctly, the "is this cluster
actually good" question still required direct inspection: feature-average comparisons,
within-cluster spread, and reading actual song titles. The metrics guided *where to look*;
they never replaced *looking*.

---

## 9. Known Limitations / Honest Gaps

- The re-split heuristic (`k = n // TARGET_MAX`, capped) is a simple rule, not an
  elbow/silhouette search at every branch — a deliberate simplification to avoid
  over-engineering, not a rigorously optimized choice.
- `MIN_SONGS_TO_SPLIT` is currently dead code under the tested configurations — a candidate
  for either removal or a redesigned stopping rule if pursued further.
- No listening/human-evaluation pass has been done at scale — only a handful of clusters
  have been spot-checked by title and feature averages, per the case study's own guidance
  that structure and meaningfulness are not the same thing.
- Only K-Means has been tested. DBSCAN and Agglomerative Clustering — explicitly asked for
  by the case study — have not yet been implemented or compared.

---

## 10. Next Steps

1. Compare K-Means against **DBSCAN** and **Agglomerative Clustering** on the same
   evaluation lens (size distribution, coherence, business-fit) — the case study's second
   open question, not yet addressed.
2. Decide whether to invest in a full parameter-sensitivity re-sweep now that the merge
   logic is fixed, versus treating the current findings as sufficient for the report.
3. Optional: a proper human-listening pass on a larger sample of final playlists, beyond
   the single deep-dive case study in Section 5–6.

---

## 11. Restructuring into Notebooks

The pipeline was reorganized into a numbered notebook series (`01_baseline_clustering.ipynb`,
`02_recursive_pipeline.ipynb`, `03_cluster_deepdive.ipynb`, `04_dbscan.ipynb`, ...), each
independently runnable from a fresh CSV load rather than depending on another script's live
memory state. Convention: main pipeline steps numbered `1, 2, 3...`; experimental variants
on a step labelled `5a`, `5b`, etc., using distinct variable names so they can never silently
corrupt the main pipeline's state (a real bug hit once — an experiment cell reused the shared
`scaler` variable and broke a later step; fixed by giving every experiment cell its own
uniquely-named objects).

## 12. Feature Selection, Revisited

The original 5-feature set (`danceability, energy, acousticness, tempo, valence`) was
copied directly from the case study's own example code — never independently justified.
A correlation matrix across a wider candidate set of 9 features found:

- **`loudness` is redundant** — correlates ~0.8+ with `energy` and `acousticness`. Including
  it would let "intensity" count roughly twice in distance calculations. Dropped.
- **`speechiness` and `instrumentalness` carry genuinely new signal** — low correlation with
  everything else in the set. Added.
- **`key` (cyclic, 0–11) and `mode` (binary)** need special handling to be usable at all
  (key's numeric adjacency doesn't match musical adjacency); excluded for now, noted as
  future work rather than silently ignored.
- **`time_signature` and `duration_ms`** carry little information for mood/style; excluded.

**Final feature set:** `danceability, energy, acousticness, tempo, valence, speechiness,
instrumentalness` (7 features).

### PCA on the trimmed feature set

PCA was run on the 7-feature set to check for further, subtler redundancy beyond what the
correlation matrix already caught.

| Component | Variance explained |
|---|---|
| PC1 | 46.6% |
| PC2 | 31.8% |
| PC3 | 10.6% |
| PC4 | 4.3% |
| PC5 | 3.1% |
| PC6 | 2.5% |
| PC7 | 1.0% |

Just 2 components already capture 78.4% of total variance; 5 components are needed to
reach 95%. **Important clarification, since this was initially misread:** PCA does not
select a subset of the original 7 features — it builds 7 entirely new components, each a
mixture of all 7 inputs. "5 components needed for 95%" is not validation that "5 was the
right original feature count" (a coincidence of numbers, not a causal link) — it shows that
the 7 original features only carry about 2–3 truly independent directions of information,
consistent with the redundancy already flagged in the correlation matrix (energy and
acousticness in particular pull strongly in opposite directions).

Clustering silhouette scores were compared directly at k=8 across raw 7-feature space vs.
PCA-reduced (2 and 3 component) space, to test whether dimensionality reduction helps or
hurts clustering quality specifically — see notebook `04` follow-up work for the numeric
comparison once run.

**Caveat for the report:** silhouette scores computed in a 2D/3D PCA space aren't strictly
comparable in magnitude to scores computed in 7D space — distance behaves differently as
dimensionality changes. Treat this as a directional signal (did quality hold up, roughly?),
not a precise before/after percentage.

## 13. DBSCAN — Success Criteria, Defined Explicitly

Before comparing any algorithms further, a real ambiguity surfaced: "good playlist" has no
single, universal, agreed definition — confirmed directly with the course instructor, who
noted that dropping a substantial fraction of songs as noise is an acceptable DBSCAN outcome
if the resulting clusters are coherent. This is not a project team disagreement to resolve —
it reflects a genuine, unstated business-priority choice the case study itself never makes
explicit.

**Two axes are now defined explicitly, to be scored for every algorithm from here on:**

1. **Coverage** — % of the 5000 songs actually placed into some cluster.
   - K-Means (with the recursive split/merge pipeline): ~100% by construction.
   - DBSCAN: 100% minus whatever % is labelled noise (`-1`).
2. **Coherence** — how tight and musically defensible the resulting clusters are (silhouette
   score on non-noise points, plus manual spot-checks).

**Framing for the final report:** K-Means (as built here) and DBSCAN embody two different,
equally legitimate business philosophies rather than one being objectively better. K-Means
optimizes for coverage — every song gets used, matching a "use the full catalog" priority.
DBSCAN optimizes for coherence over coverage — it will leave a song unplaced rather than
force a poor fit, matching a "never ship an awkward playlist" priority. The report's
conclusion should state which philosophy fits Moosic's actual priorities rather than declare
one algorithm an outright winner.

**Methodology notes carried into the DBSCAN notebook:**
- `MinMaxScaler` used (not the course's `StandardScaler`) for consistency with the K-Means
  notebooks — confirmed with the instructor that this specific choice doesn't undermine
  comparability, since silhouette score is a normalized ratio, not a raw distance measure.
- `n_neighbors=7` used for the k-distance graph (matching the feature count), rather than a
  generic "2× dimensionality" heuristic — simpler and sufficient for this assignment's scope,
  per instructor guidance.
- DBSCAN's silhouette score is computed excluding noise points; K-Means's is not. Any
  comparison table must report both the silhouette score **and** the noise percentage
  together — silhouette alone is misleading, since a higher DBSCAN score can simply mean it
  discarded its hardest cases rather than solved them.

## 14. Next Steps (Updated)

1. Finish running the DBSCAN notebook against the actual 5000-song data (k-distance graphs,
   parameter tuning, noise inspection, coverage/coherence scoring against the K-Means
   baseline).
2. Agglomerative Clustering — same evaluation lens, same coverage/coherence framing.
3. Final report section translating the coverage-vs-coherence framing into a concrete
   recommendation for Moosic, rather than a bare "algorithm X scored higher" statement.

---

## 15. DBSCAN — Results

### Parameter sweep (min_samples=7, eps 0.03–0.50)

| eps | clusters | noise | coverage % | silhouette |
|---|---|---|---|---|
| 0.03 | 0 | 5235 | 0.0 | N/A |
| 0.05 | 2 | 5218 | 0.3 | 0.970 |
| 0.07 | 14 | 4965 | 5.2 | 0.547 |
| 0.09 | 18 | 4103 | 21.6 | 0.113 |
| 0.11 | 14 | 2980 | 43.1 | -0.043 |
| 0.13 | 16 | 2005 | 61.7 | -0.252 |
| 0.15 | 10 | 1192 | 77.2 | -0.218 |
| 0.17 | 4 | 767 | 85.3 | -0.157 |
| 0.20 | 2 | 417 | 92.0 | 0.456 |
| 0.22 | 2 | 307 | 94.1 | 0.452 |
| 0.25+ | 1 | <165 | 96.8-100 | N/A (single cluster) |

**Core finding: no `eps` value in the tested range produces many reasonably-sized clusters,
good coherence, and high coverage simultaneously.** The mid-range (0.09-0.20) is where
coverage is acceptable, but silhouette is at its worst (deeply negative in places) —
fragmented, overlapping density regions, not clean separation. Coverage and silhouette only
both look good at 0.20-0.22, but this is misleading at the cluster-size level (see below).
Past ~0.22, the entire dataset collapses into a single cluster.

**Investigated directly — the eps=0.20 result is not what the silhouette score implies.**
Cluster breakdown: one cluster of 4286 songs (82% of the dataset), one cluster of 532 songs,
417 noise. The "good" 0.456 silhouette describes one enormous, undifferentiated majority
cluster plus one smaller distinct pocket — not a usable set of playlists. This mirrors the
project's very first finding (k=2 in early K-Means testing produces the same kind of useless
mega-cluster) — now confirmed independently on a structurally different algorithm. This
consistency is itself a meaningful result: the tension between "few coarse groups" and "many
fine groups with no usable middle ground" appears to be a property of this dataset/feature
space, not an artifact of K-Means specifically.

### Case study: the 532-song minority cluster (eps=0.20)

Feature comparison vs. dataset average:

| Feature | Overall avg | Cluster avg |
|---|---|---|
| danceability | 0.510 | 0.308 |
| energy | 0.654 | 0.113 |
| acousticness | 0.290 | 0.951 |
| tempo | 118.7 | 103.4 |
| valence | 0.444 | 0.148 |
| speechiness | 0.083 | 0.045 |
| instrumentalness | 0.257 | 0.877 |

A genuinely coherent, musically real cluster — reads clearly as classical/instrumental
music by the numbers, not a merge artifact. Two notable exceptions in the sample:

- **"If I Ain't Got You" (Alicia Keys)** — explainable, not an error: a sparse piano
  ballad that genuinely resembles classical pieces on these specific audio axes (low
  energy, high acousticness), despite being a different genre by human classification.
  Reinforces a limitation already flagged in the K-Means deep-dive: these features track
  *sound*, not genre label.
- **"Moves Like Jagger" (Maroon 5)** — investigated and resolved (see Section 17). Not a
  data error: a second, genuine recording exists under the same title, with a
  low-energy/high-acousticness/high-instrumentalness profile consistent with an unlabeled
  instrumental/piano cover.

### DBSCAN methodology confirmed with instructor

- "Good playlist" has no single agreed definition; discarding a large fraction of songs as
  noise is an acceptable DBSCAN outcome if remaining clusters are coherent — this is a
  genuine, unstated business-priority choice, not a project ambiguity to resolve unilaterally.
- Scaler choice (`MinMaxScaler` vs. course's `StandardScaler`) does not undermine
  comparability here, since silhouette score is a normalized ratio, not a raw distance value.
- `n_neighbors=7` (matching feature count) is a sufficient, simpler choice than a generic
  "2x dimensionality" heuristic for this assignment's scope.

## 16. Next Steps (Updated Again)

1. Resolve the "Moves Like Jagger" data check.
2. Agglomerative Clustering — same evaluation lens, same coverage/coherence framing.
3. Final report section translating the coverage-vs-coherence framing into a concrete
   recommendation for Moosic, incorporating the DBSCAN "no viable middle ground" finding as
   a cross-algorithm-confirmed result, not a K-Means-specific quirk.

---

## 17. Resolving "Moves Like Jagger" — Unlabelled Cover Versions, Not a Data Bug

Direct investigation (checking the exact row via substring search, since sample printouts
truncate long titles) found **two separate rows** sharing the title "Moves Like Jagger":
one with a genuine upbeat pop profile (danceability 0.719, energy 0.736, acousticness
0.009 — matching the real Maroon 5 track), and one with a near-silent, fully-acoustic,
fully-instrumental profile (energy 0.036, acousticness 0.989, instrumentalness 0.924).

Checking the surrounding rows in the dataset (index 2036-2042) showed the second version
sits inside a genuine cluster of classical/piano pieces (Fauré's "Pavane," a Chopin
Nocturne, an Interstellar soundtrack piano-cello arrangement) — not scattered, isolated
corruption. **"Shape of You" (Ed Sheeran) shows the identical pattern**: one row matching
the real released track (danceability 0.825, energy 0.652, tempo ~96 BPM — closely matching
the actual song), and a second row with a classical-style profile (energy 0.136,
acousticness 0.990, instrumentalness 0.946, tempo 153.878 — likely a tempo-detection
half/double-time artifact on a slower arrangement).

**Conclusion: this is not a data-quality problem.** The dataset contains a small number of
genuine instrumental/piano cover versions of well-known pop songs, sharing exact titles
with the originals, with no arrangement tag distinguishing them (unlike, e.g., "Cornfield
Chase - Piano-Cello Version," which kept its tag). This is a positive finding for the
project: the audio features correctly separated the pop original from its instrumental
cover in both confirmed cases, despite the title giving no indication whatsoever — a
concrete, specific answer to the case study's opening question about whether these features
capture similarity the way a listener would.

**Methodology note, worth keeping as its own record:** the first attempt at a systematic
duplicate-title check computed feature spread using **unscaled** raw values. Since `tempo`
(~60-200 BPM) is on a completely different numeric scale than the other six features
(~0-1), averaging raw standard deviations across features silently reduced the whole
metric to "how different is the tempo" — the other six features barely contributed. This
produced a misleading top-10 list dominated by ordinary tempo differences between
legitimately different recordings, not genuine multi-feature divergence. **Fixed by
recomputing the check on scaled data**, giving every feature equal footing — the corrected
list surfaced a coherent, plausible set of well-known songs (Hello, Wake Me Up, Chandelier,
Halo, and others) likely to have real alternate-arrangement covers in a dataset this size.
Caught and corrected mid-investigation rather than reported on the flawed version.

## 18. Next Steps (Updated Again)

1. Agglomerative Clustering — same evaluation lens, same coverage/coherence framing.
2. Final report section translating the coverage-vs-coherence framing into a concrete
   recommendation for Moosic, incorporating the DBSCAN "no viable middle ground" finding and
   the confirmed cover-version case study as cross-checked, resolved evidence.

---

## 19. DBSCAN Robustness — Confirmed Under a Second Scaler

The core DBSCAN finding (no `eps` gives many reasonably-sized clusters + good coherence +
high coverage simultaneously) was independently re-tested under `StandardScaler`, not just
`MinMaxScaler`. Different absolute `eps` range (0.1-2.0, vs. 0.03-0.5 under MinMax — expected,
since the two scalers don't share distance units), but the identical shape of result:
fragmentation at low `eps`, a brief misleading "recovery" right before collapse (here,
`eps=0.9-1.0`, silhouette ~0.33-0.34), confirmed via direct cluster-size inspection to be the
same single-mega-cluster-plus-scraps pattern (4759-4928 songs, ~93-94% of the dataset, in one
cluster both times). Total collapse to one cluster shortly after.

**Separately, very low `min_samples` (1-3) was tested** and produces a different-looking but
equally invalid result: near-100% coverage with only 2-3 clusters, negative silhouette. This
is DBSCAN's "chaining" failure mode — a lenient `min_samples` lets clusters merge through thin
bridges of sparse connecting points rather than requiring genuine density gaps. Same
underlying conclusion, reached from the opposite direction (too permissive rather than too
strict).

**This robustness across two scalers and multiple `min_samples` regimes is itself the
strongest part of the DBSCAN finding** — it rules out "this is just an artifact of one
preprocessing choice" as an objection, before the report ever needs to field it.

## 20. Inside DBSCAN's Noise Bucket — a Hidden Rap/Hip-Hop Pocket

Direct comparison of feature averages (mainstream cluster vs. classical cluster vs. noise,
canonical run: `eps=0.20, min_samples=7`) found noise sitting *between* the two clusters on
5 of 7 features (danceability, energy, acousticness, valence, instrumentalness) — consistent
with these being genuine boundary/blend cases (e.g. acoustic ballads, stripped-down pop) that
DBSCAN correctly declined to force into either group.

**Tempo and speechiness broke this pattern** — noise scored *higher* than both clusters on
both, not between them. Checked directly against Spotify's own speechiness bands
(above ~0.33 indicates spoken word, described as "possibly rap"): a real cluster of noise
points cleared that threshold. Song titles confirmed it directly — "This Is Why I'm Hot,"
"Candy Shop," "Bad Boy for Life," "I Need A Girl Pt. II" — genuine, recognizable hip-hop/rap
tracks.

**Sub-clustering the noise set (canonical, 417 songs) at k=2-5 found k=3 best-supported**
(silhouette 0.238, sizes [118, 129, 170]); the highest-speechiness, highest-tempo sub-group
is the confirmed rap/hip-hop pocket. **Independently replicated** via a second, differently-
configured noise set (`min_samples=3`, 274 songs; sub-clustering at k=3 gave sizes
[115, 74, 85], again with one sub-group showing clearly elevated speechiness (0.196) and
tempo (136.6) relative to its own group average) — the same signal, found twice, under two
different DBSCAN parameterizations. That second run also suggested a third distinct pocket
worth noting: high energy + high instrumentalness + low acousticness, plausibly
electronic/dance instrumental tracks — not yet investigated further.

**Conclusion:** DBSCAN's noise bucket was not one homogeneous leftover pile. It contained at
least one, likely two, genuinely coherent musical groups — too small or too oddly-shaped
relative to the dominant mainstream mass to form their own dense region under any tested
`eps`/`min_samples`. DBSCAN's real value in this project turned out to be diagnostic
(surfacing real structure worth a closer look) rather than as a usable playlist generator on
its own.

## 21. Notebook Restructuring — a Real Shared-State Bug, Caught and Fixed

While investigating the rap-pocket question, a mix-up occurred: a side-experiment cell
(`eps=0.20, min_samples=7`) computed its own cluster labels but never wrote them into the
shared `df['dbscan_cluster']` column, so a later "noise investigation" step kept silently
reading from a different, earlier run (`min_samples=3`) instead of the intended one. Same
root cause as the `scaler` mix-up in notebook `01`'s Step 5a, caught earlier — an
experimental variant touching shared state.

`04_dbscan.ipynb` was rebuilt around one rule: only one designated step (Step 5, "Chosen
Configuration") writes to the shared `df['dbscan_cluster']` column that every later step
depends on; every side-experiment gets its own uniquely-named variables. The mix-up itself
wasn't wasted, however — the sub-clustering it accidentally ran became the independent
replication described in Section 20.

## 22. Framing: "Blind Cinderella"

A working analogy developed for the final report/presentation, worth keeping close to this
wording:

*Unsupervised clustering on audio features is like asking Cinderella to sort a mixed pile of
small objects by touch alone, in a dark room — using only shape, weight, and texture. If she
sorts beans, peas, and lentils together because they're similar enough in shape and weight,
that's not her failing — we chose to put gloves on her (deliberately selecting which
features/"senses" she gets) rather than turn the lights on. Add more senses (more features,
like speechiness and instrumentalness) and she can distinguish more. But some distinctions —
color, in the beans example; genre or language, in Moosic's — are not recoverable by touch at
all, no matter how skilled she becomes, because the information was never available to the
sense she's using. And "good sorting" is itself subjective: some people like their corn and
beans mixed together. There's no universal correct sort for Cinderella to find — only a sort
that matches what a specific person asked for.*

**Two concrete project examples anchor both halves of this analogy:**
- **"Teenage Dirtbag" landing in a dance-pop cluster** (Section 5 of this report) — touch
  correctly matching two objects that are the same shape/weight (tempo/energy) while being
  genuinely different in a way touch can't sense (genre). The blindfold failing.
- **The unlabelled piano covers of "Moves Like Jagger" and "Shape of You"** (Section 17) —
  touch correctly telling apart two objects with an identical label (title) by feeling a real
  difference the label didn't reveal. The blindfold succeeding, better than reading the label
  would have.

**Why shape and color aren't fully independent, and what that implies:** real beans, peas,
and lentils differ in shape as well as color — shape imperfectly predicts color. This is why
clustering *sometimes* approximates genre correctly without ever measuring it directly:
tempo/energy/acousticness correlate with genre often enough for "feel" to substitute for
"label" some of the time — and the misfits (Teenage Dirtbag) are exactly the cases where
that correlation breaks down.

## 23. Next Steps (Updated)

1. Agglomerative Clustering — same evaluation lens, same coverage/coherence framing, built
   with the same "one canonical run, uniquely-named side-experiments" notebook convention
   from the start.
2. Final report / presentation section built around the coverage-vs-coherence framing and
   the Cinderella analogy, translating the technical findings into a concrete recommendation
   for Moosic.

---

## 24. Considered and Skipped — Sub-Clustering DBSCAN's Mega-Cluster

Before moving to Agglomerative Clustering, sub-clustering the DBSCAN mainstream mega-cluster
(4286-4928 songs depending on run, ~82-94% of the full dataset) with K-Means was considered,
by analogy with the noise-bucket sub-clustering in Section 20.

**Decision: not pursued.** Since the mega-cluster already comprises the large majority of the
dataset, sub-clustering it would functionally reproduce something close to the existing
recursive K-Means pipeline (notebooks `01`-`03`) already run on the full dataset. A genuinely
different question is buried in there — whether removing DBSCAN's classical cluster and noise
points *before* running K-Means measurably improves the result on what remains, i.e. whether
DBSCAN's first pass acts as a useful pre-cleaning step — but the expected effect size is small
(the mega-cluster is still 90%+ of the data either way), and not worth the remaining time
budget with two days left before the final report/presentation is due. Documented here as a
deliberate scope decision, not an oversight.

## 25. Next Steps (Updated)

1. Agglomerative Clustering.
2. Final report / presentation.



---

## 26. Agglomerative Clustering — Results

Ward linkage silhouette peaked at k=2-3 (0.385-0.391), but size inspection revealed the
same misleading pattern seen throughout this project: k=2/k=3's "better" score comes from
one dominant cluster (3759 songs, 72% of the dataset) that k=3 doesn't actually break up —
it only further splits the *other*, smaller group. Silhouette rewarding fewer, broader
groups over genuinely useful ones, confirmed a fourth time across a fourth distinct
mechanism.

**k=8, by contrast, gives 8 genuinely balanced clusters (133-1111 songs)** — no mega-cluster,
unlike every DBSCAN configuration tested. Interpreted by feature averages:

| Cluster | Size | Character |
|---|---|---|
| 4 | 1111 | Mainstream pop/dance (high danceability/energy/valence) — largest, but only 21% of the dataset |
| 6 | 698 | Highest valence (0.817) — upbeat "feel-good" pop |
| 1 | 734 | High energy, low valence/danceability — aggressive/dark, plausibly rock. Not clearly surfaced by K-Means or DBSCAN before |
| 0 | 562 | Moderate/acoustic-leaning pop |
| 2 | 521 | Low energy, high acousticness — mellow acoustic/folk |
| 3 | 601 | Very high acousticness (0.926) + instrumentalness (0.867) — **replicates DBSCAN's classical/instrumental cluster independently** |
| 5 | 875 | High energy + high instrumentalness (0.771) + near-zero acousticness — **replicates the electronic/dance-instrumental pocket hypothesized but not confirmed from DBSCAN's noise sub-clustering (Section 20)** |
| 7 | 133 | Similar to 5 but faster, brighter (positive valence) — festival/EDM-leaning variant, smallest, likely genuine niche |

Two independent cross-algorithm replications (classical/instrumental, electronic-instrumental)
are strong corroborating evidence — found via two structurally different mechanisms
(density-based noise analysis vs. hierarchical merging), not an artifact of one method.

## 26. Hybrid Pipeline (Agglomerative + Recursive K-Means)

Built and run: Agglomerative (Ward, k=8) as a coarse first stage, with the existing
recursive split/reassign/merge pipeline run separately within each of the 8 superclusters
rather than on the whole dataset at once. Both pipelines re-run in the same notebook on
identical data for a guaranteed apples-to-apples comparison.

**Headline size-target metrics were nearly identical** (single-stage: 129 playlists, 98.4%
in [20,60]; hybrid: 133 playlists, 99.2%) — expected, since both pipelines are specifically
tuned to hit that target regardless of starting structure; too close to the metric's ceiling
to meaningfully distinguish the two approaches.

**A corrected coherence check** (mean within-playlist feature spread, on properly-scaled
data — the first attempt had the same tempo-scale bug caught earlier in the duplicate-title
investigation, re-fixed here) found the hybrid marginally tighter (0.0652 vs. 0.0656, ~0.6%)
— real, but modest, not the headline finding.

**The more decisive-looking check: 76 of 129 single-stage playlists (59%) span more than one
Agglomerative supercluster** (53 stay within one, 47 span two, 27 span three, 2 span all
four tested) — initially read as evidence the hybrid's structural constraint (never crossing
a supercluster boundary) prevents a real coherence failure in the single-stage pipeline.

### Correction, on inspection of the most extreme case

The playlist spanning all 4 superclusters was inspected directly. Nearly every song is a
recognizable jazz or bossa nova standard — confirmed for "Água de Beber" and "Corcovado"
(both genuine Antônio Carlos Jobim compositions), consistent by title for "Autumn Leaves,"
"St. Thomas," "Moanin'," "Blue Bossa," "Lover Man," "Comin' Home Baby," "Almost Like Being
In Love," plus several Portuguese-language bossa nova/samba tracks. **This is not a random
cross-genre mixture — it is one coherent genre, correctly held together as a single playlist
by the single-stage pipeline.**

**Why this genre spans 4 Agglomerative superclusters:** jazz/bossa nova has real, wide sonic
variance on these 7 features that has nothing to do with genre identity — a slow
instrumental combo piece and an upbeat vocal samba are genuinely far apart on
tempo/energy/instrumentalness despite both being unambiguously "jazz" to a listener.
Agglomerative's superclusters correctly detect that sonic diversity; that detection is
simply narrower than what "genre" means to a human — the same audio-feature limitation
identified via the Teenage Dirtbag and cover-version findings, now from the opposite
direction (features splitting apart what a human unifies, rather than uniting what a human
would split).

### Revised conclusion — a genuine tradeoff, not a clean win

- **Single-stage K-Means** can correctly hold a sonically-diverse-but-genre-coherent group
  together (demonstrated here) — but nothing structurally prevents it from also
  accidentally blending genuinely unrelated material.
- **The hybrid approach** structurally guarantees no cross-supercluster blending — at the
  cost of potentially fracturing a stylistically unified genre that happens to be sonically
  diverse, exactly as seen in this case.

Neither approach is unambiguously better; each fails in a different, identifiable way. This
is a stronger, more honest finding for the presentation than either "hybrid wins" or "hybrid
loses" — the recommendation should present this as a real, demonstrated tradeoff, illustrated
with this specific playlist as the concrete example, rather than asserting one pipeline is
superior.

## 27. Next Steps

1. Final report / presentation — synthesize all findings (K-Means, DBSCAN, Agglomerative,
   hybrid) into the 5-minute Moosic brief, using the coverage-vs-coherence framing, the
   Cinderella analogy, and this hybrid tradeoff as the central technical narrative.

---

## 28. DBSCAN Run on Its Own Mega-Cluster (Per Instructor Suggestion)

Re-ran DBSCAN, with a freshly-fit `eps`, on just the 4286-song mega-cluster subset from the
canonical run — distinct from the earlier "considered and skipped" idea (Section 24), which
was about K-Means, not DBSCAN. The question: does the mega-cluster hide internal density
structure a single global `eps` was too coarse to see?

**Sweep result: same failure pattern, confirmed a fourth independent time.** Fragmented at
low eps, negative silhouette through the middle range (down to -0.331), full collapse to one
cluster by eps=0.18-0.22. No eps threads the needle within the subset either — strengthens
rather than contradicts the standing "no viable middle ground" conclusion.

**However, at eps=0.09, three real, sizable sub-clusters survived alongside heavy noise
(78%) and many tiny fragments** — and their profiles are the strongest cross-validation
evidence in the whole project:

| Sub-cluster | Size | Profile | Matches |
|---|---|---|---|
| 0 | 463 | High danceability/energy, near-zero acousticness/instrumentalness | Clean "mainstream pop/dance" core |
| 6 | 118 | Energy 0.964, valence 0.142, danceability 0.276 | **Replicates Agglomerative cluster 1** (aggressive/dark, rock-leaning) — 3rd independent finding |
| 13 | 262 | Energy 0.937, instrumentalness 0.886, near-zero acousticness | **Replicates Agglomerative cluster 5** (electronic/dance-instrumental) — 3rd independent finding |

**Closing synthesis:** across four structurally different techniques (original K-Means, DBSCAN's
noise sub-clustering, Agglomerative's k=8, and DBSCAN re-run on its own mega-cluster), the same
small set of real musical categories independently re-emerges each time — mainstream pop,
classical/instrumental, electronic-instrumental, aggressive/dark rock, rap/hip-hop, and jazz/bossa
nova. None of these methods know about each other's results; the consistency is the strongest
single piece of evidence in the project that these are genuine underlying categories in the
data, not artifacts of any one algorithm's assumptions.

## 29. Project Status

All technical work complete: K-Means (chosen recommendation), DBSCAN (diagnostic, two real
discoveries), Agglomerative (confirmed both discoveries independently, informed the hybrid
experiment), hybrid pipeline (genuine tradeoff finding via the jazz/bossa nova correction),
and this closing cross-validation experiment. Remaining work is presentation build and
rehearsal only — no further clustering experiments planned before submission.
