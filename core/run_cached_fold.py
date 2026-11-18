"""Internal, hash-checked single-fold runner. Use run_experiment.py for the CLI."""
from pathlib import Path
import json,sys,hashlib,time
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score,matthews_corrcoef,balanced_accuracy_score
from models import DualHead,ProteinOnlyHead,train_fold
root=Path(__file__).resolve().parent.parent
config_path=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else root/'config.json'
c=json.loads(config_path.read_text(encoding='utf-8'))
def path(x):
 p=Path(x);return p if p.is_absolute() else root/p
def digest_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
expected_inputs=c.get('expected_input_sha256')
if not expected_inputs or set(expected_inputs)!=set(['manifest','splits','training_config']): raise ValueError('Manifest, split and training configuration hashes are required')
for key,val in expected_inputs.items():
 if digest_file(path(c[key]))!=val: raise ValueError('Input SHA-256 mismatch: '+key)
expected=c.get('expected_cache_sha256')
if not expected or set(expected)!=set(c['cache']): raise ValueError('Cache hashes required for each configured file; inspect provenance before updating')
for key,val in c['cache'].items():
 if digest_file(path(val))!=expected[key]: raise ValueError('Cache SHA-256 mismatch: '+key)
m=pd.read_csv(path(c['manifest']),dtype={'record_id':str},low_memory=False);m=m[m.include_evidence_linked==True].reset_index(drop=True)
idx=pd.read_csv(path(c['cache']['protein_index']),dtype={'record_id':str});assert len(m)==1116 and m.record_id.is_unique and idx.record_id.tolist()==m.record_id.tolist();assert m.cache_row.tolist()==list(range(1116))
s=pd.read_csv(path(c['splits']),dtype={'record_id':str});s=s[(s.split_name==c['split'])&(s.fold==c['fold'])].copy();assert set(s.record_id)==set(m.record_id) and s.record_id.is_unique
s=s.set_index('record_id').loc[m.record_id].reset_index();assert np.array_equal(s.label.to_numpy(),m.label.to_numpy());assert np.array_equal(s.row_index.to_numpy(),m.cache_row.to_numpy())
roles={x:np.where(s.inner_role==x)[0] for x in ['inner_train','inner_validation','outer_test']};assert all(len(x)>0 and m.label.iloc[x].nunique()==2 for x in roles.values());assert not(set(m.iloc[roles['inner_train']].pathogen_taxid)&set(m.iloc[roles['outer_test']].pathogen_taxid));assert not(set(m.iloc[roles['inner_validation']].pathogen_taxid)&set(m.iloc[roles['outer_test']].pathogen_taxid));assert not(set(m.iloc[roles['inner_train']].pathogen_taxid)&set(m.iloc[roles['inner_validation']].pathogen_taxid))
a={key:np.load(path(c['cache'][src]),mmap_mode='r')[:1116].astype(dtype) for key,src,dtype in [('dna','dna',np.float32),('dm','dna_mask',bool),('prot','protein',np.float32),('pm','protein_mask',bool)]};assert a['dna'].shape==(1116,2,4,768) and a['prot'].shape==(1116,3,640) and a['dm'].any(1).all() and a['pm'].any(1).all()
cfg=json.loads(path(c['training_config']).read_text(encoding='utf-8'));cfg['model']['dna_hidden_size']=768
if '--check' in sys.argv:
 print('Validated manifest, split, cache hashes, shapes and disjoint TaxIDs for fold',c['fold']);sys.exit(0)
requested=c.get('device','auto')
device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested=='auto' else requested)
out=path(c['output']);out.mkdir(exist_ok=True,parents=True);runs=[];pred=[]
for seed in c['seeds']:
 for model,cls in [('dual',DualHead),('protein',ProteinOnlyHead)]:
  t=time.time();y,score,thr,ep,auc=train_fold(lambda:cls(cfg),a,m.label.to_numpy(np.float32),roles['inner_train'],roles['inner_validation'],roles['outer_test'],seed*100+c['fold'],cfg,device,out/('{}_seed{}_fold{}.pt'.format(model,seed,c['fold'])))
  v=(score>=thr).astype(int);te=m.iloc[roles['outer_test']];assert np.array_equal(y,te.label.to_numpy()); runs.append(dict(model=model,seed=seed,fold=c['fold'],auroc=roc_auc_score(y,score),ap=average_precision_score(y,score),mcc=matthews_corrcoef(y,v),balanced_accuracy=balanced_accuracy_score(y,v),threshold=thr,best_epoch=ep,val_auc=auc,seconds=time.time()-t));pred.extend(dict(model=model,seed=seed,fold=c['fold'],record_id=rid,label=int(label),score=float(sc),threshold=thr,pred=int(pr)) for rid,label,sc,pr in zip(te.record_id,y,score,v));print(runs[-1],flush=True)
pd.DataFrame(runs).to_csv(out/'fold_metrics.csv',index=False);pd.DataFrame(pred).to_csv(out/'oof_predictions.csv',index=False)
files=[path(c['manifest']),path(c['splits']),path(c['training_config'])]+[path(v) for v in c['cache'].values()];hashes={str(x):digest_file(x) for x in files};(out/'input_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');(out/'run_meta.json').write_text(json.dumps({'device':str(device),'torch':torch.__version__,'cuda':torch.version.cuda,'fold':c['fold'],'seed':c['seeds'],'protocol':'TaxID-grouped cached representation retrain','caveat':'No claim of raw-sequence end-to-end reproduction'},indent=2),encoding='utf-8')
