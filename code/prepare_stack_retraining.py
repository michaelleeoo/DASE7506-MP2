"""Prepare training-only feature caches and provided base assets for OOF stack training."""
from pathlib import Path
import json,shutil
import numpy as np
from common import load_inputs
from features_v3 import extract

root=Path(__file__).resolve().parent
features=root/'runs/improve_v3';features.mkdir(parents=True,exist_ok=True)
for split in ('train','validation'):
    cache=features/f'{split}_features.npz'
    if not cache.exists():
        bank,arrays=load_inputs(split);np.savez(cache,**extract(bank,arrays))
        print('Prepared',cache)
if not (features/'pca.npz').exists():shutil.copy2(root/'model_bundle_stack/pca_v3.npz',features/'pca.npz')
base=root/'model_bundle_v6'
if not base.exists():
    base.mkdir()
    config=json.loads((root/'model_bundle_stack/config.json').read_text())
    names={'config.json','pca_v3.npz'}
    for m in config['members']:
        names.add(m['file'])
        if 'projection_file' in m:names.add(m['projection_file'])
    for name in names:shutil.copy2(root/'model_bundle_stack'/name,base/name)
print('Ready: python stack_v2.py (refits training-only OOF predictions and the meta-model)')
