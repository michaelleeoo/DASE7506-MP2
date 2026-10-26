"""CPU training on train only, selection on validation only; no test access."""
import os
os.environ['MKL_CBWR'] = 'COMPATIBLE'
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import DATA, load_inputs, parse_answer, read_jsonl
from submission import make_features, text_features, majority_choices
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

def labels(split, bank):
    refs = read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    assert len(refs) == len(bank)
    assert all(r['id']==b['id'] and r['question']==b['question'] for r,b in zip(refs,bank))
    return np.array([[parse_answer(c['text']) is not None and parse_answer(c['text'])==parse_answer(r['answer'])
                      for c in b['candidates']] for b,r in zip(bank,refs)], dtype=np.float32)

class RewardModel(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = (nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Dropout(.15),nn.Linear(hidden,1))
                    if hidden else nn.Sequential(nn.Linear(dim,1)))
    def forward(self,x): return self.net(x).squeeze(-1)

def export_model(model, mean, scale, path):
    weights = {'mean':mean,'scale':scale}
    layers = [m for m in model.net if isinstance(m,nn.Linear)]
    weights['depth'] = np.array(len(layers))
    for i,layer in enumerate(layers):
        weights[f'w{i}'] = layer.weight.detach().numpy().T.copy()
        weights[f'b{i}'] = layer.bias.detach().numpy().copy()
    np.savez(path,**weights)

def fit(name,mode,hidden,loss_kind,seed,train_x,val_x,y,vy,output,teacher=None,epochs=36):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean = train_x.mean((0,1)).astype(np.float32)
    scale = np.maximum(train_x.std((0,1)),.1).astype(np.float32)
    x = torch.from_numpy((train_x-mean)/scale)
    vx = torch.from_numpy((val_x-mean)/scale)
    ty = torch.from_numpy(y)
    eligible = np.flatnonzero((y.sum(1)>0)&(y.sum(1)<4))
    model = RewardModel(x.shape[-1],hidden)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001 if hidden else .003,weight_decay=.02)
    best_acc,best_state,best_epoch,history = -1,None,0,[]
    start = time.perf_counter()
    soft = torch.from_numpy(teacher) if teacher is not None else None
    for epoch in range(1,epochs+1):
        model.train()
        total = 0.
        for ids in np.array_split(rng.permutation(eligible),max(1,len(eligible)//128)):
            scores,target = model(x[ids]),ty[ids]
            if loss_kind=='pair':
                mask = target[:,:,None]*(1-target[:,None,:])
                losses = F.softplus(-(scores[:,:,None]-scores[:,None,:]))
                loss = ((losses*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)).mean()
            else:
                loss = -((target/target.sum(1,keepdim=True))*F.log_softmax(scores,dim=1)).sum(1).mean()
            if soft is not None:
                kd = F.kl_div(F.log_softmax(scores/2,dim=1),soft[ids],reduction='batchmean')*4
                loss = .5*loss+.5*kd
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            total += float(loss)
        model.eval()
        with torch.no_grad(): vs = model(vx).numpy()
        acc = float(vy[np.arange(len(vy)),vs.argmax(1)].mean())
        history.append({'epoch':epoch,'validation_accuracy':100*acc,'loss_sum':total})
        if acc>best_acc:
            best_acc,best_epoch = acc,epoch
            best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): vs,ts = model(vx).numpy(),model(x).numpy()
    file = name+'.npz'
    export_model(model,mean,scale,output/file)
    rec = dict(name=name,mode=mode,hidden=hidden,loss=loss_kind,seed=seed,epochs_searched=epochs,
               selected_epoch=best_epoch,validation_accuracy=100*best_acc,
               training_seconds=time.perf_counter()-start,file=file,
               asset_bytes=(output/file).stat().st_size,history=history)
    print(json.dumps({k:v for k,v in rec.items() if k!='history'}),flush=True)
    return rec,vs,ts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('runs/development'))
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    banks,arrays,ys,extras = {},{},{},{}
    for split in ('train','validation'):
        banks[split],arrays[split] = load_inputs(split)
        ys[split] = labels(split,banks[split])
        extras[split] = text_features(banks[split])
    records,val_scores,train_scores = [],{},{}
    baseline = []
    for name,choices in [('first',np.zeros(len(banks['validation']),dtype=int)),
                         ('majority',majority_choices(banks['validation']))]:
        baseline.append(dict(name=name,validation_accuracy=float(100*ys['validation'][np.arange(len(choices)),choices].mean())))
    specs = [('style_linear','style',0,'pair',17),('answer_mlp','answer',32,'pair',17),
             ('embedding_linear','embedding',0,'pair',17),('embedding_mlp','embedding',64,'pair',17),
             ('no_style_mlp','context_no_style',64,'pair',17),
             ('context_pair_17','context',64,'pair',17),('context_pair_29','context',64,'pair',29),
             ('context_pair_43','context',64,'pair',43),('context_list_17','context',64,'list',17),
             ('individual_list_17','individual',64,'list',17),
             ('no_style_list_17','context_no_style',64,'list',17),
             ('small_hard_71','context',32,'list',71)]
    for name,mode,hidden,loss,seed in specs:
        xx = [make_features(arrays[s]['embeddings'],*extras[s],mode) for s in ('train','validation')]
        rec,vs,ts = fit(name,mode,hidden,loss,seed,*xx,ys['train'],ys['validation'],args.output)
        records.append(rec)
        val_scores[name],train_scores[name] = vs,ts
    teacher_names = ['context_pair_17','context_pair_29','context_pair_43']
    ts = np.mean([train_scores[n] for n in teacher_names],axis=0)
    soft = torch.softmax(torch.from_numpy(ts/2),dim=1).numpy()
    xx = [make_features(arrays[s]['embeddings'],*extras[s],'context') for s in ('train','validation')]
    rec,vs,_ = fit('distilled_32','context',32,'list',71,*xx,ys['train'],ys['validation'],args.output,teacher=soft)
    records.append(rec)
    val_scores[rec['name']] = vs
    ensemble_scores = np.mean([val_scores[n] for n in teacher_names],axis=0)
    ensemble_accuracy = float(100*ys['validation'][np.arange(len(vs)),ensemble_scores.argmax(1)].mean())
    candidates = [(r['validation_accuracy'],-r['asset_bytes'],[r['name']]) for r in records]
    candidates.append((ensemble_accuracy,-sum(r['asset_bytes'] for r in records if r['name'] in teacher_names),teacher_names))
    chosen = max(candidates)[2]
    lookup = {r['name']:r for r in records}
    selected = np.mean([val_scores[n] for n in chosen],axis=0)
    all_answers = [[parse_answer(c['text']) for c in b['candidates']] for b in banks['validation']]
    def confidence(temp):
        logits = selected/temp
        prob = np.exp(logits-logits.max(1,keepdims=True))
        prob /= prob.sum(1,keepdims=True)
        choice = selected.argmax(1)
        return np.clip(np.array([sum(prob[i,j] for j in range(4) if a[j]==a[choice[i]] and a[j] is not None)
                                 for i,a in enumerate(all_answers)]),0.,1.)
    correct = ys['validation'][np.arange(len(vs)),selected.argmax(1)]
    calibration = [dict(temperature=t,brier=float(np.mean((confidence(t)-correct)**2))) for t in [.5,1.,1.5,2.,3.,5.,8.]]
    temperature = min(calibration,key=lambda r:r['brier'])['temperature']
    config = dict(schema=1,members=[dict(file=lookup[n]['file'],mode=lookup[n]['mode'],weight=1/len(chosen)) for n in chosen],
                  temperature=temperature,selection_split='validation',training_split='train',tie_break='lowest candidate index')
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    np.savez(args.output/'validation_analysis.npz',labels=ys['validation'],**val_scores,
             ensemble=ensemble_scores,selected=selected,confidence=confidence(temperature))
    result = dict(baselines=baseline,models=records,ensemble_validation_accuracy=ensemble_accuracy,
                  selected_members=chosen,selected_validation_accuracy=max(candidates)[0],
                  calibration_development_only=calibration,total_seconds=time.perf_counter()-start,
                  device='CPU',threads=1,blas_thread_cap=4,torch=torch.__version__,numpy=np.__version__,
                  search_policy='13 runs x 36 epochs; validation accuracy then asset size; three matched controls added before test evaluation')
    (args.output/'experiments.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','calibration_development_only')}))

if __name__=='__main__': main()
