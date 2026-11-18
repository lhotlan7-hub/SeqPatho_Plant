"""Exploratory paired subgroup analysis for Table S16; see README.md and DATA_FORMAT.md."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import check_subgroup_inputs  # performs and writes the input audit

from _paths import ARCHIVE as ROOT, OUTPUT
N_BOOT, RNG_SEED = 5000, 20261117
p = pd.read_csv(ROOT/'matched_comparison/by_split/taxid_evidence_1116_oof_predictions.csv')
m = pd.read_csv(ROOT/'manifest/cohort_manifest_audited.csv')
h = pd.read_csv(ROOT/'dependency_audit/homology_best_train_match_taxid_evidence_1116.csv')
d = p.pivot(index=['record_id','fold','seed','label','taxid'],columns='model',values='score').reset_index()
assert d[['dual','protein']].notna().all().all() and len(d)==2232
d = d.merge(m[['record_id','dna_is_exact_cds']],on='record_id',validate='many_to_one')
d = d.merge(h[['test_record_id','fold','n_train_30_80']],left_on=['record_id','fold'],right_on=['test_record_id','fold'],validate='many_to_one')
groups = {'low_homology':d.n_train_30_80.eq(0),'exact_cds':d.dna_is_exact_cds.eq(True),
          'other_dna':d.dna_is_exact_cds.eq(False),'without_318829':d.taxid.ne(318829),'full_budget':d.label.notna()}

def setup(x, taxa):
    x=x.sort_values('record_id'); y=x.label.to_numpy(dtype=int)
    codes=pd.Categorical(x.taxid,categories=taxa).codes; n=len(taxa)
    pos=np.bincount(codes,weights=y,minlength=n); neg=np.bincount(codes,weights=1-y,minlength=n)
    a={}
    for model in ['dual','protein']:
        score=x[model].to_numpy(); pi=np.flatnonzero(y==1); ni=np.flatnonzero(y==0)
        cmp=(score[pi,None]>score[None,ni]).astype(float)+0.5*(score[pi,None]==score[None,ni])
        mat=np.zeros((n,n)); np.add.at(mat,(codes[pi,None],codes[None,ni]),cmp)
        order=np.lexsort((x.record_id.to_numpy(),-score))
        a[model]=(mat,codes[order],y[order])
    return pos,neg,a

def calc(obj, weights, budget):
    pos,neg,a=obj; npos=pos@weights; nneg=neg@weights
    if npos==0 or (not budget and nneg==0): return None
    out={}
    for model,(mat,codes,labels) in a.items():
        if budget:
            w=weights[codes]; k=int(math.ceil(.1*(npos+nneg))); cum=np.cumsum(w); idx=int(np.searchsorted(cum,k,side='left'))
            hits=float(np.dot(w[:idx],labels[:idx])+(k-(cum[idx-1] if idx else 0))*labels[idx]); precision=hits/k
            out[model]=(precision,precision/(npos/(npos+nneg)))
        else:
            out[model]=(float(weights@mat@weights/(npos*nneg)),)
    return np.array(out['dual']),np.array(out['protein'])

rng=np.random.default_rng(RNG_SEED); summary=[]; foldrows=[]
for name,mask in groups.items():
    z=d[mask].copy(); records=z.drop_duplicates('record_id'); budget=name=='full_budget'
    configs=[]
    for fold,part in z.groupby('fold',sort=True):
        taxa=sorted(part.taxid.unique()); seed_objs=[]
        for seed,x in part.groupby('seed',sort=True):
            o=setup(x,taxa); obs=calc(o,np.ones(len(taxa),dtype=int),budget)
            if obs is not None:
                seed_objs.append((seed,o,obs))
                for idx,metric in enumerate(['precision_at_10pct','enrichment'] if budget else ['auroc']):
                    foldrows.append(dict(group=name,fold=fold,seed=seed,metric=metric,dual=obs[0][idx],protein=obs[1][idx],delta=obs[0][idx]-obs[1][idx],n=len(x),positive=int(x.label.sum()),taxa=len(taxa)))
        configs.append((fold,taxa,seed_objs))
    folds=pd.DataFrame([r for r in foldrows if r['group']==name]); metrics=['precision_at_10pct','enrichment'] if budget else ['auroc']; boot=[[] for _ in metrics]; rejected=0
    for _ in range(N_BOOT):
        values=[[] for _ in metrics]; invalid=False
        for fold,taxa,seed_objs in configs:
            if not seed_objs: continue
            w=np.bincount(rng.integers(0,len(taxa),len(taxa)),minlength=len(taxa))
            for seed,obj,_ in seed_objs:
                v=calc(obj,w,budget)
                if v is None: invalid=True; break
                for k in range(len(metrics)): values[k].append(v[0][k]-v[1][k])
            if invalid: break
        if invalid: rejected+=1; continue
        for k in range(len(metrics)): boot[k].append(float(np.mean(values[k])))
    for k,metric in enumerate(metrics):
        f=folds[folds.metric.eq(metric)]; vals=np.asarray(boot[k]); lo,hi=np.quantile(vals,[.025,.975]) if len(vals) else (np.nan,np.nan)
        summary.append(dict(group=name,metric=metric,n_records=len(records),n_positive=int(records.label.sum()),n_taxa=records.taxid.nunique(),eligible_folds=f.fold.nunique(),eligible_fold_seed_runs=len(f),dual_mean=f.dual.mean(),protein_mean=f.protein.mean(),delta_mean=f.delta.mean(),ci_low=lo,ci_high=hi,bootstrap_valid=len(vals),bootstrap_rejected=rejected,bootstrap_total=N_BOOT))
result=pd.DataFrame(summary); result.to_csv(OUTPUT /'subgroup_analysis.csv',index=False); pd.DataFrame(foldrows).to_csv(OUTPUT /'subgroup_fold_metrics.csv',index=False)
print(result.to_string(index=False))
