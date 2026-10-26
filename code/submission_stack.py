"""Out-of-fold-trained stacking verifier; offline inference, no reference reads."""
import json
from pathlib import Path
import numpy as np
from catboost import CatBoost
from common import parse_answer
from features_v3 import extract
from features_v2 import matrix
from submission_v6 import forward_numpy,relational_forward,group_mask

def probability(scores,valid):
    s=scores-scores.mean(1,keepdims=True)
    s/=np.maximum(np.sqrt(np.mean(s*s,1,keepdims=True)),.01)
    s=np.where(valid,s,-1e9);ex=np.exp(s-s.max(1,keepdims=True))
    return ex/ex.sum(1,keepdims=True)

def all_scores(bank,arrays,checkpoint):
    root=Path(checkpoint);config=json.loads((root/'config.json').read_text())
    f=extract(bank,arrays)
    with np.load(root/'pca_v3.npz') as z:pm,projection=z['mean'],z['projection']
    e=(f['embedding']-pm)@projection[:,:128]
    group_x=np.concatenate([e,e-e.mean(1,keepdims=True),f['aux'],f['center_aux']],-1)
    keep=[i for i in range(f['aux'].shape[-1]) if i not in (22,23,24)]
    binary_x=np.concatenate([f['embedding'],f['aux'][...,keep]],-1)
    mask=group_mask(bank);valid=np.array([[parse_answer(c['text']) is not None for c in b['candidates']] for b in bank])
    logits=[]
    for m in config['members']:
        kind=m['kind']
        if kind=='boost':
            x=matrix(f,'context')
            with np.load(root/m['projection_file']) as z:
                xx=(x-z['mean'])/z['scale'];h=np.maximum(xx@z['w0']+z['b0'],0);ns=forward_numpy(xx,z)[:,:,None]
                bx=np.concatenate([h,h-h.mean(1,keepdims=True),ns,ns-ns.mean(1,keepdims=True),f['aux'],f['center_aux']],-1)
            model=CatBoost();model.load_model(str(root/m['file']))
            scores=model.predict(bx.reshape(-1,bx.shape[-1]),prediction_type='RawFormulaVal',thread_count=4).reshape(-1,4)
        else:
            x=binary_x if kind=='binary' else group_x
            with np.load(root/m['file']) as z:
                xx=(x-z['mean'])/z['scale']
                scores=relational_forward(xx,z,'group',mask) if kind=='group' else forward_numpy(xx,z)
        logits.append(scores)
    return logits,f,valid,config

def reward_scores(bank,arrays,checkpoint):
    root=Path(checkpoint);logits,f,valid,config=all_scores(bank,arrays,root)
    parts=[];probabilities=[]
    for s in logits:
        p=probability(s,valid);probabilities.append(p)
        entropy=-(p*np.log(np.maximum(p,1e-10))).sum(1)
        parts.extend([p[:,:,None],(p-p.max(1,keepdims=True))[:,:,None],np.broadcast_to(entropy[:,None,None],(*p.shape,1))])
    average=np.mean(probabilities,axis=0)
    parts.extend([average[:,:,None],f['aux'],f['center_aux']])
    x=np.concatenate(parts,-1).astype(np.float32)
    model=CatBoost();model.load_model(str(root/config['stack']['file']))
    raw=model.predict(x.reshape(-1,x.shape[-1]),prediction_type='RawFormulaVal',thread_count=4).reshape(-1,4)
    p=probability(raw,valid);weight=config['stack']['stack_weight']
    return np.where(valid,weight*p+(1-weight)*average,-1e9)

def predict(bank,arrays,checkpoint):
    if not bank:return []
    return reward_scores(bank,arrays,checkpoint).argmax(1).astype(int).tolist()
