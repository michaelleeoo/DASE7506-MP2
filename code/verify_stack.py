"""Final integration checks for the exact improved artifact; validation inputs only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import copy,hashlib,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from common import load_inputs
from submission_stack import predict,reward_scores

def main():
    root=Path(__file__).resolve().parent
    checkpoint=root/'model_bundle_stack'
    manifest=json.loads((root/'model_bundle_stack_manifest.json').read_text())
    for file,m in manifest['assets'].items():assert hashlib.sha256((checkpoint/file).read_bytes()).hexdigest()==m['sha256'],file
    for file,digest in manifest['source_hashes'].items():assert hashlib.sha256((root/file).read_bytes()).hexdigest()==digest,file
    assert manifest['asset_bytes']==sum(p.stat().st_size for p in checkpoint.iterdir() if p.is_file())<16*2**20
    bank,arrays=load_inputs('validation');bank=copy.deepcopy(bank[:11]);arrays={'embeddings':arrays['embeddings'][:11]}
    scores=reward_scores(bank,arrays,checkpoint);choices=predict(bank,arrays,checkpoint)
    assert all(type(x) is int and 0<=x<=3 for x in choices)
    for i,b in enumerate(bank):b['id']=f'new-audit-{i}'
    assert predict(bank,arrays,checkpoint)==choices
    assert predict(bank[:1],{'embeddings':arrays['embeddings'][:1]},checkpoint)==choices[:1]
    assert predict([],{'embeddings':arrays['embeddings'][:0]},checkpoint)==[]
    order=[2,0,3,1];swapped=copy.deepcopy(bank)
    for b in swapped:
        b['candidates']=[b['candidates'][i] for i in order]
        for i,c in enumerate(b['candidates']):c['index']=i
    other=reward_scores(swapped,{'embeddings':arrays['embeddings'][:,order]},checkpoint)
    np.testing.assert_allclose(other,scores[:,order],atol=3e-4,rtol=3e-4)
    reverse=list(reversed(range(len(bank))))
    assert predict([bank[i] for i in reverse],{'embeddings':arrays['embeddings'][reverse]},checkpoint)==[choices[i] for i in reverse]
    original=Path.open
    def guarded(self,*args,**kwargs):
        if self.suffix=='.jsonl':raise AssertionError('Dataset/reference read inside prediction')
        return original(self,*args,**kwargs)
    with patch.object(Path,'open',guarded):assert predict(bank,arrays,checkpoint)==choices
    # Match the official recorded choices for the same validation inputs.
    recorded=json.loads((root/'results_stack/validation-predictions.json').read_text())
    assert recorded['selected'][:11]==choices
    print('PASS: frozen hashes, asset limit, Python int outputs, unseen IDs, arbitrary sizes, row/candidate permutations, reference isolation, official choice consistency')

if __name__=='__main__':main()
