from pathlib import Path
import pandas as pd

from _paths import ARCHIVE as ROOT, OUTPUT
p = pd.read_csv(ROOT / 'matched_comparison/by_split/taxid_evidence_1116_oof_predictions.csv')
m = pd.read_csv(ROOT / 'manifest/cohort_manifest_audited.csv')
h = pd.read_csv(ROOT / 'dependency_audit/homology_best_train_match_taxid_evidence_1116.csv')
a = pd.read_csv(ROOT / 'sequences/sequence_pair_audit.csv')
s = pd.read_csv(ROOT / 'manifest/split_manifest.csv')
assert not p.duplicated(['record_id', 'model', 'seed']).any()
assert not m.record_id.duplicated().any() and not h.test_record_id.duplicated().any() and not a.record_id.duplicated().any()
assert set(p.model) == {'dual','protein'} and len(p) == 4464
base = p[p.model.eq('dual') & p.seed.eq(42)][['record_id','fold','label','taxid']]
assert len(base)==1116 and not base.record_id.duplicated().any()
for seed in p.seed.unique():
    for model in p.model.unique():
        t=p[(p.seed==seed)&(p.model==model)][['record_id','fold','label','taxid']]
        assert base.sort_values('record_id').reset_index(drop=True).equals(t.sort_values('record_id').reset_index(drop=True))
base=base.merge(m[['record_id','pathogen_taxid','dna_is_exact_cds','label']], on='record_id',validate='one_to_one',suffixes=('','_manifest'))
base=base.merge(h[['test_record_id','fold','test_label','n_train_30_80']],left_on=['record_id','fold'],right_on=['test_record_id','fold'],validate='one_to_one')
base=base.merge(a[['record_id','dna_is_exact_cds']],on='record_id',validate='one_to_one',suffixes=('','_audit'))
test=s[(s.split_name=='taxid_evidence_1116') & (s.outer_role=='outer_test')][['record_id','fold']]
assert len(test)==1116 and not test.record_id.duplicated().any()
base=base.merge(test,on=['record_id','fold'],validate='one_to_one')
assert (base.label==base.label_manifest).all() and (base.label==base.test_label).all()
assert (base.taxid.astype(str)==base.pathogen_taxid.astype(str)).all()
assert base.dna_is_exact_cds.notna().all() and (base.dna_is_exact_cds==base.dna_is_exact_cds_audit).all()
assert base.n_train_30_80.notna().all()
groups={'low_homology':base.n_train_30_80.eq(0),'exact_cds':base.dna_is_exact_cds.eq(True),'other_dna':base.dna_is_exact_cds.eq(False),'without_318829':base.taxid.ne(318829),'all':base.label.notna()}
rows=[]
for name, mask in groups.items():
    for fold,x in base[mask].groupby('fold'):
        rows.append(dict(group=name,fold=fold,n=len(x),positive=int(x.label.sum()),taxa=x.taxid.nunique(),taxa_positive=x.loc[x.label.eq(1),'taxid'].nunique(),taxa_negative=x.loc[x.label.eq(0),'taxid'].nunique()))
out=pd.DataFrame(rows)
out.to_csv(OUTPUT /'subgroup_input_audit.csv',index=False)
print(out.to_string(index=False))
