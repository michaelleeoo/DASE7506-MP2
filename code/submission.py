"""Offline reward model. NumPy inference; never opens dataset references."""
import json
from pathlib import Path
import re
from fractions import Fraction
import numpy as np
from common import parse_answer

NUMBER = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?')
WORDS = re.compile(r'[a-z]+')
STYLE_NAMES = ['log_chars', 'log_words', 'log_lines', 'log_numbers', 'log_equals',
               'log_steps', 'log_bold', 'log_headings', 'log_boxed', 'log_think',
               'log_questions', 'log_exclamations', 'log_hesitation', 'log_confidence',
               'log_reconsideration', 'log_math_ops']
ANSWER_NAMES = ['valid', 'negative', 'zero', 'integer', 'log_magnitude', 'in_question',
                'agreement', 'distinct_answers', 'max_agreement', 'question_numbers',
                'question_log_chars', 'question_fraction']

def text_features(bank):
    style = np.zeros((len(bank), 4, len(STYLE_NAMES)), dtype=np.float32)
    answer = np.zeros((len(bank), 4, len(ANSWER_NAMES)), dtype=np.float32)
    for i, row in enumerate(bank):
        q = row['question'].lower()
        qnums = set(NUMBER.findall(q.replace(',', '')))
        parsed = [parse_answer(c['text']) for c in row['candidates']]
        counts = [sum(v is not None and v == a for v in parsed) for a in parsed]
        distinct = len(set(a for a in parsed if a is not None))
        for j, candidate in enumerate(row['candidates']):
            t = candidate['text'].lower()
            vals = [len(t), len(WORDS.findall(t)), t.count('\n'), len(NUMBER.findall(t)),
                    t.count('='), t.count('step'), t.count('**'), t.count('###'),
                    t.count('boxed'), t.count('think'), t.count('?'), t.count('!'),
                    sum(t.count(s) for s in ('perhaps', 'maybe', 'assume', 'assuming')),
                    sum(t.count(s) for s in ('clearly', 'therefore', 'certainly', 'final answer')),
                    sum(t.count(s) for s in ('however', 'wait', 'instead', 'mistake', 'alternatively')),
                    sum(t.count(s) for s in ('+', '-', '*', '/', 'frac', 'times', 'div'))]
            style[i, j] = np.log1p(vals)
            a = parsed[j]
            v = Fraction(a) if a is not None else Fraction(0)
            answer[i, j] = [a is not None, v < 0, v == 0, v.denominator == 1,
                            np.log1p(min(abs(float(v)), 1e15)), a in qnums,
                            counts[j]/4, distinct/4, max(counts)/4, np.log1p(len(qnums)),
                            np.log1p(len(q)), 'fraction' in q or '%' in q]
    return style, answer

def make_features(embeddings, style, answer, mode):
    if mode == 'style': parts = [style, style-style.mean(1, keepdims=True)]
    elif mode == 'answer': parts = [answer, answer-answer.mean(1, keepdims=True)]
    elif mode == 'embedding': parts = [embeddings]
    elif mode == 'individual': parts = [embeddings, style, answer[..., :6]]
    elif mode in ('context', 'context_no_style'):
        parts = [embeddings, embeddings-embeddings.mean(1, keepdims=True)]
        if mode == 'context': parts += [style, style-style.mean(1, keepdims=True)]
        parts += [answer, answer-answer.mean(1, keepdims=True)]
    else: raise ValueError(mode)
    return np.concatenate(parts, axis=-1)

def majority_choices(bank):
    result = []
    for row in bank:
        answers = [parse_answer(c['text']) for c in row['candidates']]
        result.append(int(np.argmax([sum(a is not None and a == b for b in answers) for a in answers])))
    return result

def forward_numpy(x, weights):
    depth = int(weights['depth'])
    for i in range(depth):
        x = x @ weights[f'w{i}'] + weights[f'b{i}']
        if i < depth-1: x = np.maximum(x, 0)
    return x[..., 0]

def reward_scores(bank, arrays, checkpoint):
    root = Path(checkpoint)
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    style, answer = text_features(bank)
    cache = {}
    result = np.zeros((len(bank), 4), dtype=np.float32)
    for member in config['members']:
        mode = member['mode']
        if mode not in cache: cache[mode] = make_features(arrays['embeddings'], style, answer, mode)
        with np.load(root/member['file'], allow_pickle=False) as data:
            scores = forward_numpy((cache[mode]-data['mean'])/data['scale'], data)
        result += member['weight'] * scores
    return result, config

def predict(bank, arrays, checkpoint):
    """Return Python integers; question IDs are never used as features."""
    if not bank: return []
    if checkpoint is None: return majority_choices(bank)
    scores, _ = reward_scores(bank, arrays, checkpoint)
    return np.argmax(scores, axis=1).astype(int).tolist()
