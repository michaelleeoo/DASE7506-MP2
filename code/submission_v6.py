"""Improved offline verifier. All assets loaded from checkpoint, no reference access."""
import json
from pathlib import Path
from collections import Counter
import re
import numpy as np
from submission import text_features,make_features,forward_numpy
from features_v2 import extract as extract_v2,matrix
from features_v3 import extract as extract_v3

TOKEN=re.compile(r'(?u)\b\w\w+\b')

def group_mask(bank):
    from common import parse_answer
    mask=np.zeros((len(bank),4,4),np.float32)
    for i,row in enumerate(bank):
        answers=[parse_answer(c['text']) for c in row['candidates']]
        for j,a in enumerate(answers):
            for k,b in enumerate(answers):mask[i,j,k]=float(j==k or (a is not None and a==b))
    return mask

def relational_forward(x,z,kind,mask=None):
    h=np.maximum(x@z['enc_w']+z['enc_b'],0)
    mean=h.mean(1,keepdims=True);maximum=np.broadcast_to(h.max(1,keepdims=True),h.shape)
    if kind=='group':
        count=mask.sum(2,keepdims=True);group=(mask/count)@h
        groupmax=np.where(mask[:,:,:,None]>0,h[:,None,:,:],-1e9).max(2)
        combined=np.concatenate([group,groupmax,group-mean,group*mean,maximum,count/4],-1)
    else:combined=np.concatenate([h,h-mean,h*mean,maximum],-1)
    return (np.maximum(combined@z['mid_w']+z['mid_b'],0)@z['out_w']+z['out_b'])[...,0]

def lexical_encode(texts,vocab,idf,components):
    result=np.zeros((len(texts),components.shape[1]),np.float32)
    for i,text in enumerate(texts):
        tokens=TOKEN.findall(text.lower())
        counts=Counter(tokens+[' '.join(p) for p in zip(tokens,tokens[1:])])
        known=[(vocab[w],c) for w,c in counts.items() if w in vocab]
        if not known:continue
        inds=np.array([k for k,c in known]);tf=np.array([1+np.log(c) for k,c in known],np.float32)*idf[inds]
        tf/=max(np.linalg.norm(tf),1e-12)
        result[i]=tf@components[inds]
    return result

def build_pca(f,mean,projection,dim):
    e=(f['embedding']-mean)@projection[:,:dim]
    return np.concatenate([e,e-e.mean(1,keepdims=True),f['aux'],f['center_aux']],axis=-1)

def reward_scores(bank,arrays,checkpoint):
    root=Path(checkpoint)
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
    features={};pcas={};cache={};lex=None
    result=np.zeros((len(bank),4),np.float32)
    def get_features(version):
        if version not in features:
            features[version]=(extract_v2 if version=='v2' else extract_v3)(bank,arrays)
        return features[version]
    def pca(version):
        if version not in pcas:
            with np.load(root/f'pca_{version}.npz',allow_pickle=False) as z:pcas[version]=(z['mean'],z['projection'])
        return pcas[version]
    for member in config['members']:
        kind=member['kind'];mode=member.get('mode');version=member.get('version','v3')
        if kind in ('old','original'):
            if 'old_features' not in cache:cache['old_features']=text_features(bank)
            key=('original',mode)
            if key not in cache:cache[key]=make_features(arrays['embeddings'],*cache['old_features'],mode)
            x=cache[key]
        else:
            f=get_features(version)
            if kind=='binary':
                dim=member['dim'];mode=member['mode']
                if dim==0:e=f['embedding']
                else:
                    pm,projection=pca('v3');e=(f['embedding']-pm)@projection[:,:dim]
                aux=f['aux']
                if mode=='embedding':x=e
                elif mode in ('local','raw'):
                    keep=[i for i in range(aux.shape[-1]) if i not in (22,23,24)]
                    x=np.concatenate([e,aux[...,keep]],-1)
                else:x=np.concatenate([e,e-e.mean(1,keepdims=True),aux,f['center_aux']],-1)
            elif kind=='formula':
                from features_v8 import formula_features
                if 'formulas' not in cache:cache['formulas']=formula_features(bank)
                if 'question_lexical' not in cache:
                    vocab=json.loads((root/'vocabulary.json').read_text(encoding='utf-8'))
                    with np.load(root/'lexical.npz',allow_pickle=False) as z:idf,components=z['idf'],z['components']
                    cache['question_lexical']=np.repeat(lexical_encode([b['question'] for b in bank],vocab,idf,components)[:,None,:],4,axis=1)
                parts=[f['aux'],f['center_aux'],cache['formulas'],cache['question_lexical']]
                dim=member['dim']
                if dim:
                    pm,projection=pca('v3');e=(f['embedding']-pm)@projection[:,:dim]
                    parts=[e,e-e.mean(1,keepdims=True)]+parts
                x=np.concatenate(parts,-1)
            elif kind in ('lexical','set','group','robust'):
                pm,projection=pca('v3');dim=member['dim'];feature_kind=member['kind_features']
                if feature_kind in ('joint','lexical') and lex is None:
                    vocab=json.loads((root/'vocabulary.json').read_text(encoding='utf-8'))
                    with np.load(root/'lexical.npz',allow_pickle=False) as z:idf,components=z['idf'],z['components']
                    q=np.repeat(lexical_encode([b['question'] for b in bank],vocab,idf,components)[:,None,:],4,axis=1)
                    c=lexical_encode([c['text'] for b in bank for c in b['candidates']],vocab,idf,components).reshape(len(bank),4,-1)
                    lex=(q,c)
                if feature_kind=='lexical':
                    q,c=lex;x=np.concatenate([q,c,c-q,c*q,f['aux'],f['center_aux']],-1)
                else:
                    x=build_pca(f,pm,projection,dim)
                    if feature_kind=='joint':
                        q,c=lex;x=np.concatenate([x,q,c,c-q,c*q],-1)
            elif kind=='kernel':
                pm,projection=pca(version);e=(f['embedding']-pm)@projection[:,:member['dim']]
                parts=[e,f['aux']]
                if member['context']:parts.extend([e-e.mean(1,keepdims=True),f['center_aux']])
                x=np.concatenate(parts,-1)
            elif kind=='rank' or mode=='pca':
                pm,projection=pca(version)
                x=build_pca(f,pm,projection,128 if kind=='rank' else 256)
            else:x=matrix(f,mode)
        if kind=='boost':
            with np.load(root/member['projection_file'],allow_pickle=False) as z:
                xx=(x-z['mean'])/z['scale'];h=np.maximum(xx@z['w0']+z['b0'],0)
                ns=forward_numpy(xx,z)[:,:,None]
                x=np.concatenate([h,h-h.mean(1,keepdims=True),ns,ns-ns.mean(1,keepdims=True),f['aux'],f['center_aux']],-1)
        if kind in ('tree','rank','boost'):
            from catboost import CatBoost
            model=CatBoost();model.load_model(str(root/member['file']))
            scores=model.predict(x.reshape(-1,x.shape[-1]),prediction_type='RawFormulaVal',thread_count=4).reshape(len(bank),4)
        else:
            with np.load(root/member['file'],allow_pickle=False) as z:
                x=(x-z['mean'])/z['scale']
                if kind=='kernel':
                    flat=x.reshape(-1,x.shape[-1]);sv=z['support'];out=[];svnorm=np.sum(sv*sv,axis=1)
                    for start in range(0,len(flat),256):
                        q=flat[start:start+256]
                        distance=np.maximum(np.sum(q*q,axis=1)[:,None]+svnorm[None,:]-2*q@sv.T,0)
                        out.append(np.exp(-float(z['gamma'])*distance)@z['dual']+z['intercept'][0])
                    scores=np.concatenate(out).reshape(len(bank),4)
                elif kind in ('set','group'):scores=relational_forward(x,z,kind,group_mask(bank) if kind=='group' else None)
                else:scores=forward_numpy(x,z)
        scores=scores-scores.mean(1,keepdims=True)
        normalization=config['normalization']
        if normalization=='global':scores/=member['scale']
        else:scores/=np.maximum(np.sqrt(np.mean(scores*scores,axis=1,keepdims=True)),.01)
        if 'softmax' in normalization:
            from common import parse_answer
            valid=np.array([[parse_answer(c['text']) is not None for c in b['candidates']] for b in bank])
            logits=np.where(valid,scores/float(normalization[-1]),-1e9)
            ex=np.exp(logits-logits.max(1,keepdims=True));scores=ex/ex.sum(1,keepdims=True)
            if normalization.startswith('answer_'):
                mask=group_mask(bank)*valid[:,:,None]*valid[:,None,:]
                count=np.maximum(mask.sum(2),1)
                scores=(mask@scores[:,:,None])[...,0]/np.sqrt(count)
        elif normalization=='rank':scores=np.argsort(np.argsort(scores,axis=1),axis=1).astype(np.float32)/3
        result+=member['weight']*scores
    return result

def predict(bank,arrays,checkpoint):
    if not bank:return []
    if checkpoint is None:raise ValueError('Provide the frozen improved checkpoint with --checkpoint')
    from common import parse_answer
    scores=reward_scores(bank,arrays,checkpoint)
    valid=np.array([[parse_answer(c['text']) is not None for c in b['candidates']] for b in bank])
    scores=np.where(valid,scores,-1e9)
    return np.argmax(scores,axis=1).astype(int).tolist()
