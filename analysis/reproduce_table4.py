"""Recompute selected Table 4 methods from the saved features and split manifest."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _paths import ARCHIVE as R, OUTPUT
A=R/'table4_assets'
index=pd.read_csv(A/'source_index.csv').sort_values('manifest_row').reset_index(drop=True)
assert index.manifest_row.tolist()==list(range(1116)) and not index.record_id.duplicated().any()
roles=pd.read_csv(R/'manifest/split_manifest.csv')
roles=roles[roles.split_name.eq('taxid_evidence_1116')]
methods={
 'plm4vf_esm2_mean':('esm2_mean_1022.npy','lr',5000),
 'funvfpred_aac':('feat_aac.npy','rf',None),
 'funvfpred_aac_dde':('feat_aac_dde.npy','rf',None),
 'ntv2_dna_linear':('ntv2_mean.npy','lr',3000),
}
ref=pd.read_csv(A/'per_fold_seedavg.csv')
rows=[]; predictions=[]
for name,(filename,typ,iters) in methods.items():
    X=np.load(A/filename); assert len(X)==len(index)
    for fold in range(5):
        r=roles[roles.fold.eq(fold)].set_index('record_id').loc[index.record_id]
        assert (r.label.to_numpy()==index.label.to_numpy()).all()
        train=(r.inner_role=='inner_train').to_numpy(); test=(r.outer_role=='outer_test').to_numpy()
        scores=[]
        for seed in (42,43):
            if typ=='rf': clf=RandomForestClassifier(n_estimators=100,max_depth=None,random_state=seed,n_jobs=1)
            else: clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=iters,C=1.0,class_weight='balanced'))
            clf.fit(X[train],index.label.to_numpy()[train]); score=clf.predict_proba(X[test])[:,1]; scores.append(score)
            predictions.extend(dict(method=name,fold=fold,seed=seed,record_id=rid,label=int(y),score=float(s)) for rid,y,s in zip(index.record_id[test],index.label[test],score))
        auc=roc_auc_score(index.label[test],np.mean(scores,axis=0))
        reference=ref[(ref.method==name)&(ref.fold==fold)]; assert len(reference)==1
        old=float(reference.auroc.iloc[0]); rows.append(dict(method=name,fold=fold,auroc=auc,archived_table_auroc=old,delta=auc-old,passes_tolerance=abs(auc-old)<=0.0015))
out=pd.DataFrame(rows); out.to_csv(OUTPUT /'table4_recomputed.csv',index=False)
pd.DataFrame(predictions).to_csv(OUTPUT /'table4_recomputed_oof.csv',index=False)
for name,x in out.groupby('method',sort=False):
    print(f'{name}: mean {x.auroc.mean():.4f}, archive {x.archived_table_auroc.mean():.4f}, max abs fold delta {x.delta.abs().max():.4f}; pass={x.passes_tolerance.all()}')
