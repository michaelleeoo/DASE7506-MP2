"""Conservative equation verification: do not mistake algebra fragments for arithmetic."""
import re
import numpy as np
from features_v2 import extract as base_extract,normalize_math,EXPR,evaluate

BINARY=re.compile(r'(?:\d|\))\s*[-+*/]\s*(?:\d|\()')
def strict_checks(text):
    errors=[]
    normalized=normalize_math(text)
    for line in normalized.splitlines():
        line=re.sub(r'^\s*[-•]\s+','',line)
        parts=line.split('=')
        for left,right in zip(parts[:-1],parts[1:]):
            lm=list(EXPR.finditer(left));rm=list(EXPR.finditer(right))
            if not lm or not rm:continue
            l,r=lm[-1],rm[0]
            # Left must be a complete arithmetic expression, not the tail of 3x+12.
            if left[l.end():].strip(' .:;\\') or not BINARY.search(l.group()):continue
            prefix=left[:l.start()].rstrip()
            if prefix and (prefix[-1].isalnum() or prefix[-1] in ')+-*/') and l.group().lstrip().startswith(('+','-')):continue
            # 2x and 2.5y are symbolic terms, not the numbers 2 and 2.5.
            if r.end()<len(right) and (right[r.end()].isalnum() or right[r.end()]=='_'):continue
            lv,rv=evaluate(l.group()),evaluate(r.group())
            if lv is None or rv is None:continue
            error=abs(lv-rv)/max(1.,abs(lv),abs(rv))
            errors.append(error)
    e=np.array(errors)
    good=int((e<1e-5).sum());bad=int((e>1e-3).sum());large=int((e>.05).sum())
    return [np.log1p(len(e)),np.log1p(good),np.log1p(bad),np.log1p(large),
            bad/max(1,len(e)),large/max(1,len(e)),float(e.max()) if len(e) else 0.,
            float(e.mean()) if len(e) else 0.,float(e[-1]) if len(e) else 0.,
            float(bad>0),float(large>0),float(good>0 and bad==0)]

def extract(bank,arrays):
    f=base_extract(bank,arrays)
    checks=np.array([[strict_checks(c['text']) for c in row['candidates']] for row in bank],np.float32)
    f['aux']=np.concatenate([f['aux'],checks],axis=-1)
    f['center_aux']=f['aux']-f['aux'].mean(1,keepdims=True)
    return f
