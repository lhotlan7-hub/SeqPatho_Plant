"""Run matched models on frozen cached representations; never replace an existing run."""
from pathlib import Path
import argparse,json,subprocess,sys,pandas as pd
root=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(root/'runtime_config.json'));ap.add_argument('--folds',default='0');ap.add_argument('--name',required=True);ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');ap.add_argument('--check',action='store_true');args=ap.parse_args()
if not args.name.replace('_','').replace('-','').isalnum():raise SystemExit('name must be alphanumeric, underscore or hyphen')
folds=[int(x) for x in args.folds.split(',')];assert len(set(folds))==len(folds) and all(0<=f<=4 for f in folds)
c=json.loads(Path(args.config).read_text(encoding='utf-8-sig'));c['device']=args.device;base=root/'runs'/args.name
if base.exists():raise SystemExit('Run directory already exists: '+str(base))
base.mkdir(parents=True)
for fold in folds:
 cc=dict(c);cc['fold']=fold;cc['output']=str(base/('fold_'+str(fold)));f=base/('config_fold_{}.json'.format(fold));f.write_text(json.dumps(cc,indent=2),encoding='utf-8');subprocess.run([sys.executable,str(root/'core/run_cached_fold.py'),str(f)]+(['--check'] if args.check else []),check=True)
if args.check: print('All requested folds validated');sys.exit(0)
allm=pd.concat([pd.read_csv(base/('fold_'+str(f))/'fold_metrics.csv') for f in folds]);allm.to_csv(base/'metrics.csv',index=False)
allp=pd.concat([pd.read_csv(base/('fold_'+str(f))/'oof_predictions.csv') for f in folds]);assert not allp.duplicated(['model','seed','fold','record_id']).any();allp.to_csv(base/'oof.csv',index=False)
summary=allm.groupby('model')[['auroc','ap','mcc','balanced_accuracy']].agg(['mean','std']);summary.to_csv(base/'summary.csv');print(summary.to_string());print('folds',folds,'results',base)
