"""Reference-free arithmetic and numeric-coverage features for arbitrary banks."""
import ast
from functools import lru_cache
import operator
import re
import numpy as np
from common import parse_answer
from submission import text_features

NUM=re.compile(r'(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?|(?<![\w.])\.\d+')
FRAC=re.compile(r'\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}')
TEXT=re.compile(r'\\(?:text|mathrm|mathbf|operatorname)\s*\{[^{}]*\}')
BOX=re.compile(r'\\boxed\{([^{}]*)\}')
EXPR=re.compile(r'(?<![\w.])[-+]?\s*(?:\d+(?:\.\d+)?|\.\d+|\()[\d.\s+*/()^-]*')
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}

@lru_cache(maxsize=60000)
def evaluate(expr):
    """Restricted AST evaluator: arithmetic only; never eval/execute candidate text."""
    try:
        expr=expr.strip()
        if len(expr)>150 or not expr: return None
        tree=ast.parse(expr,mode='eval')
        def walk(node):
            if isinstance(node,ast.Expression): return walk(node.body)
            if isinstance(node,ast.Constant) and type(node.value) in (int,float): return float(node.value)
            if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
                return walk(node.operand)*(1 if isinstance(node.op,ast.UAdd) else -1)
            if isinstance(node,ast.BinOp) and type(node.op) in OPS:
                value=OPS[type(node.op)](walk(node.left),walk(node.right))
                if not np.isfinite(value) or abs(value)>1e18: raise ValueError()
                return value
            raise ValueError()
        return walk(tree)
    except (SyntaxError,ValueError,ZeroDivisionError,OverflowError,RecursionError,TypeError): return None

def normalize_math(text):
    t=text.lower().replace(',','').replace('−','-').replace('×','*').replace('÷','/')
    t=TEXT.sub(' ',t)
    for _ in range(5):
        changed=FRAC.sub(r'((\1)/(\2))',t)
        if changed==t: break
        t=changed
    t=BOX.sub(r'\1',t)
    for a,b in [(r'\times','*'),(r'\cdot','*'),(r'\div','/'),(r'\left',''),(r'\right',''),
                (r'\,',' '),(r'\!',''),(r'\approx','='),(r'\rightarrow','='),(r'\Rightarrow','='),
                ('**',''),('$',''),('%','/100'),('{','('),('}',')')]:
        t=t.replace(a,b)
    return t

def numbers(text):
    result=[]
    for n in NUM.findall(text):
        try: result.append(round(float(n.replace(',','')),7))
        except ValueError: pass
    return result

def arithmetic_features(question,text):
    normalized=normalize_math(text)
    checks=[]
    last_value=None
    for line in normalized.splitlines():
        parts=line.split('=')
        for left,right in zip(parts[:-1],parts[1:]):
            ls=[m.group().strip() for m in EXPR.finditer(left)]
            rs=[m.group().strip() for m in EXPR.finditer(right)]
            if not ls or not rs: continue
            l,r=ls[-1],rs[0]
            # Ignore decorative step markers and prose immediately adjoining numbers.
            if not any(c in l+r for c in '+-*/'): continue
            lv,rv=evaluate(l),evaluate(r)
            if lv is None or rv is None: continue
            discrepancy=abs(lv-rv)/max(1.,abs(lv),abs(rv))
            checks.append(discrepancy)
            last_value=rv
    qnums=set(numbers(question))
    tnums=set(numbers(text))
    final=parse_answer(text)
    from fractions import Fraction
    value=float(Fraction(final)) if final is not None else 0.
    discrepancy=np.array(checks)
    correct=int((discrepancy<1e-5).sum())
    wrong=int((discrepancy>1e-3).sum())
    # General numeric relationships, not question-ID/template answer lookups.
    qlist=sorted(qnums)
    derivable=0
    for x in qlist:
        for y in qlist:
            options=[x+y,x-y,x*y]
            if y!=0: options.append(x/y)
            if any(abs(v-value)<1e-5*max(1,abs(value)) for v in options): derivable+=1
    return [np.log1p(len(checks)),np.log1p(correct),np.log1p(wrong),wrong/max(1,len(checks)),
            float(discrepancy.max()) if len(checks) else 0.,
            float(discrepancy.mean()) if len(checks) else 0.,
            float(np.median(discrepancy)) if len(checks) else 0.,
            float(last_value is not None and abs(last_value-value)<1e-5*max(1,abs(value))),
            np.log1p(len(qnums-tnums)),len(qnums&tnums)/max(1,len(qnums)),
            np.log1p(len(tnums-qnums)),np.log1p(derivable),
            np.log1p(len(tnums)),np.log1p(abs(value)),
            float('round' in text.lower()),float('remain' in question.lower()),
            float('profit' in question.lower()),float('percent' in question.lower() or '%' in question),
            float('how many' in question.lower()),float('how much' in question.lower()),
            float('total' in question.lower()),float('each' in question.lower()),
            float('more' in question.lower()),float('less' in question.lower())]

def extract(bank,arrays):
    style,answer=text_features(bank)
    math=np.array([[arithmetic_features(b['question'],c['text']) for c in b['candidates']] for b in bank],dtype=np.float32)
    raw=np.concatenate([style,answer,math],axis=-1)
    e=arrays['embeddings'].astype(np.float32)
    return dict(embedding=e,aux=raw,center_aux=raw-raw.mean(1,keepdims=True),
                center_embedding=e-e.mean(1,keepdims=True))

def matrix(features,mode):
    if mode=='raw': return np.concatenate([features['embedding'],features['aux']],axis=-1)
    if mode=='context': return np.concatenate([features['embedding'],features['center_embedding'],features['aux'],features['center_aux']],axis=-1)
    if mode=='delta': return np.concatenate([features['center_embedding'],features['aux'],features['center_aux']],axis=-1)
    if mode=='aux': return np.concatenate([features['aux'],features['center_aux']],axis=-1)
    raise ValueError(mode)
