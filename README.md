# DASE7506 MP2

The frozen method selects **395/467 test answers correctly: 84.58244111349036%**. Validation is 758/798 (94.9874686716792%). Read the English report for the method, course connections, comparisons and limitations.

## Contents

| Path | Why it is retained |
|---|---|
| `report.docx` | English experimental report, within the 10-page limit |
| `code/submission_stack.py` and its five local dependency modules | Exact frozen inference implementation |
| `code/model_bundle_stack/` | All nine matching inference assets; 3,754,806 bytes uncompressed |
| `code/model_bundle_stack_manifest.json` | Original freeze time, source hashes, asset hashes and selection |
| Official runner, scorer, parser and data checker in `code/` | Reproduce the published metric without changing the protocol |
| `code/evaluate_stack.py`, `code/verify_stack.py` | Complete-process timing and frozen integration checks |
| Base-training sources, `code/prepare_stack_retraining.py`, `code/stack_v2.py` | Training implementation and documented OOF/combiner refitting |
| Two pinned requirements files | Separate inference and training environments |
| `code/results_stack/` | Official test JSON, resource record, validation score and validation predictions required by verification |
| `code/experiment_records.json`, `code/improvement_history.json` | Compact experiment, seed, selection and development-history evidence |
| `code/data/manifest.json`, `LICENSE.txt`, `NOTICE.txt` | Original data checksums, licensing and attribution |
| `AI_DISCLOSURE.md`, `SUBMISSION_MANIFEST.json` | AI/reuse disclosure and complete submission file hashes |

Old checkpoints, prediction caches for inference, virtual environments, full training caches, temporary images, duplicate reports and original datasets are excluded. `submission.py` and `submission_v6.py` are retained because the final entry imports helper functions from them. Their historical names do not make them redundant. Use `submission_stack.py` as the prediction entry.

## Obtain the original course data

Download the original course repository from https://github.com/xudongwu-0/DASE7506-MP2 and copy its complete `code/data/` directory into this repository's `code/data/`, preserving files exactly. This is the only separately required benchmark input; no external training data or model download is used. Copy candidates and references for all three splits, their embeddings, and the manifest/license/notice. The data checker verifies the supplied checksums and alignment.

The protocol is `dase7506-mp2-open-hard-verifier-v5`. The data directory is intentionally ignored for large candidate files and reference JSONL files so they are not duplicated on GitHub. The three small metadata/license files are included.

## Install and reproduce without training

Python 3.12 is recommended. From the repository root in PowerShell:

```powershell
cd code
py -3.12 -m venv .venv-improved
$mp2py='.\.venv-improved\Scripts\python.exe'
& $mp2py -m pip install -r requirements-improved-inference.txt
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:NUMEXPR_NUM_THREADS='4'
& $mp2py check_data.py
& $mp2py verify_stack.py
& $mp2py evaluate_stack.py --split test --output-dir reproduced_stack
```

Expected `reproduced_stack/test-score.json`: `n=467`, `correct=395`, `accuracy=84.58244111349036`. For validation, replace `test` by `validation`: 758/798. Inference requires NumPy 1.26.4, SciPy 1.13.1 and CatBoost 1.2.8, runs offline after installation, and does not require PyTorch or a GPU.

The helper times the following two official commands in fresh processes:

```powershell
& $mp2py predict.py --method submission_stack.py --checkpoint model_bundle_stack --split test --output reproduced_stack/test-predictions.json
& $mp2py score_predictions.py --split test --predictions reproduced_stack/test-predictions.json --output reproduced_stack/test-score.json
```

The original record is 1.471292 seconds for prediction plus scoring, with a four-thread limit on Windows/Python 3.12.7/AMD Ryzen 7 7800X3D. It includes imports, loading, preprocessing and every model. The reference Xeon Platinum 8457C has not been measured. Local runtime and timing metadata can vary; the frozen candidate choices should agree.

For an instructor-supplied audit bank, use the same entry and assets:

```powershell
& $mp2py predict.py --method submission_stack.py --checkpoint model_bundle_stack --split audit --data-dir C:\path\to\audit_inputs --output reproduced_stack/audit-predictions.json
```

## Training process and scope

The training sources are retained because the assignment requests training code, not just a prediction file. Run from `code/` in a clean working copy:

```powershell
& $mp2py -m pip install -r requirements-improved-training.txt
& $mp2py prepare_stack_retraining.py
& $mp2py stack_v2.py
```

This prepares train/validation features, uses the supplied training-fitted PCA, trains four-fold base models and refits the meta-model. It never reads test inputs or references. Existing `runs/improve_v14/oof_*.npy` files are reused; use a clean copy to recompute. Results are saved under `runs/improve_v14/` and do not automatically overwrite the frozen release assets. It requires PyTorch 2.4.0 and scikit-learn 1.7.2 in addition to inference dependencies.

**Scope:** this refits OOF predictors and the combiner while retaining the supplied PCA and full-training base weights for final validation/inference. It is not one-command, end-to-end reconstruction of every final asset or historical selection. Exact score reproduction must use the provided frozen checkpoint. Base-training implementations are `binary_v2.py`, `group_v2.py`, `robust_v2.py`, and `boost_v2.py`; `train.py` provides baseline training and shared label/export helpers, while `neural_v2.py` contains training-only PCA fitting and early neural comparisons. To explore the base-family searches after feature preparation, run:

```powershell
& $mp2py binary_v2.py
& $mp2py group_v2.py
& $mp2py robust_v2.py
& $mp2py boost_v2.py
```

These create new family-search outputs under `runs/`; they do not rebuild the frozen directory automatically. Original source hyperparameters and compact records identify the selected states:

| Component | Seed and selected state |
|---|---|
| PCA | 314; 256 components stored, first 128 used by relevant branches |
| A binary classifier | 43; epoch 15 |
| B answer group width 128 | 71; epoch 3 |
| C robust scorer | 103; 100 epochs, nine snapshots averaged from epochs 60–100 |
| D neural projection and ranker | 29; projection 4 epochs, 300 depth-4 trees |
| E answer group width 64 | 43; epoch 7 |
| OOF split | 6701; four stratified question-level folds |
| Meta-model | 1337; depth 3, 100 trees, 0.5 mixture weight |

Training used CPU, with one PyTorch thread and a BLAS cap of four. Per-fit costs are retained where recorded; an exact total covering all development attempts was not recorded. PCA is pre-fitted on the entire unlabeled training set, so OOF is not fully nested preprocessing cross-validation. Retraining can differ across numerical libraries.

## Integrity and code version

`SUBMISSION_MANIFEST.json` hashes every submitted file other than itself. `code/model_bundle_stack_manifest.json` preserves the original frozen inference source/asset hashes. `verify_stack.py` checks those hashes, the 16 MiB asset limit, output types, arbitrary input sizes, changed IDs, question/candidate permutations and selected reference-isolation behavior.

The final inference entry is `code/submission_stack.py`; matching assets are `code/model_bundle_stack/`. 
