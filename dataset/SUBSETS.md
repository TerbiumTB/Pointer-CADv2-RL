# Reproducible PointerCAD subsets

`preprocessing/sample_pointercad_subset.py` samples complete CAD models from
the authoritative step-level split. It never places different steps of one
model in different splits. In this tooling:

- a **record** is one `<chunk>_<model_id>_<part_id>` step ID;
- an **object/model** is one `(chunk, model_id)` and can contain many records;
- a **shard** is one source `chunk`;
- target record counts may differ slightly because models are indivisible.

The default config contains two profiles:

- `smoke`: 3 chunks × 4 models, including multi-step, fillet, chamfer, and
  models containing neither fillet nor chamfer;
- `train_small`: approximately 15,000 records split 90/5/5 between train,
  validation, and test.

Edit paths in `config/dataset_subsets.yaml`, then preview deterministic
selection without writing:

```bash
python preprocessing/sample_pointercad_subset.py \
  --config config/dataset_subsets.yaml \
  --profile smoke \
  --dry-run
```

Build and validate one profile:

```bash
python preprocessing/sample_pointercad_subset.py \
  --config config/dataset_subsets.yaml \
  --profile smoke

python preprocessing/validate_pointercad_subset.py \
  /path/to/subsets/pointercad-smoke
```

Omit `--profile` to build all profiles. Existing output is protected by
default; pass `--overwrite` explicitly to replace it.

Both commands show `tqdm` progress bars while reading/grouping split records,
sampling smoke candidates, materializing files, and validating the result.
Smoke sampling opens CAD JSON only for the current candidate subset and caches
those results; it resamples when the requested operation mix is not satisfied.
Major phase boundaries and final counts are logged at `INFO`; use
`--log-level WARNING` for quieter output.

## Materialization modes

- `copy`: independent and portable, but duplicates every selected model.
- `hardlink`: uses little additional space and is fast, but source and subset
  files share the same inode. Treat both trees as read-only.
- `symlink`: creates a lightweight dataset view tied to the source paths.
- `none`: writes only the sampled split and manifest. The trainer should use
  the original `dataset_dir`; the validator reads that path from the manifest.

Every materialized subset has the loader-compatible layout:

```text
<subset>/
├── dataset/<chunk>/<model_id>/...
├── train_val_test.json
└── subset_manifest.json
```

The validator independently rebuilds model statistics from the subset split,
detects fillet/chamfer from CAD JSON (with a plan fallback), checks split
ownership and duplicate IDs, verifies required prompt/plan/parameter/vector/
graph/JSON files, and exits with a non-zero status if a condition fails.
