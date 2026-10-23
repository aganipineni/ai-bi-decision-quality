#!/usr/bin/env python3
"""Capacity-constrained screening benchmark on three public, observed datasets.

python research.py --self-test
python research.py --run                 # fit, cache predictions, build reports
python research.py --report              # regenerate results without refitting
python research.py --verify-refit bank   # independent refit of one complete fold
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, platform, time, urllib.request, zipfile
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; OUT=ROOT/'results'
SEED=802; FITS=3; MASKS=3; COST=.1; CAP=.2
MODELS=['BI frequency','Logistic','Random forest','Gradient boosting']
MAPS=['Raw','Logit-Platt','Beta','Isotonic']
POLICIES=MAPS+['Brier-selected','Utility-selected','Direct threshold']
COSTS=[.05,.1,.2,.4]; CAPS=[.05,.1,.2,.3,.4,.5]
STRESS=[('clean',0.)]+[(s,r) for s in ['cell','record'] for r in [.15,.30,.50]]
SOURCES={
 'bank':('https://archive.ics.uci.edu/static/public/222/bank+marketing.zip','bank_marketing.zip','e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4'),
 'shop':('https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip','online_shoppers.zip','2972e6184d3ad7beaaa831d9fc2b059dc3ee29df69d1ec593c466a5cd8485d14'),
 'churn':('https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip','iranian_churn.zip','696c3a1812267980751f30d19193a1b430ac4ad76172bb089e664c96813ead66')}
NAMES={'bank':'Bank marketing','shop':'Online shopping','churn':'Telecom churn'}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def read_data(name):
    DATA.mkdir(exist_ok=True)
    url,filename,checksum=SOURCES[name]; path=DATA/filename
    if not path.exists():
        with urllib.request.urlopen(url,timeout=90) as response: path.write_bytes(response.read())
    assert sha(path)==checksum, 'Source checksum mismatch'
    z=zipfile.ZipFile(path)
    if name=='bank':
        zz=zipfile.ZipFile(io.BytesIO(z.read('bank-additional.zip')))
        df=pd.read_csv(io.BytesIO(zz.read('bank-additional/bank-additional-full.csv')),sep=';')
        excluded=['duration']; y=df.y.eq('yes').astype(int); x=df.drop(columns=['y']+excluded)
        cats=x.select_dtypes(include='object').columns.tolist(); bic=['poutcome','job']
    elif name=='shop':
        df=pd.read_csv(io.BytesIO(z.read('online_shoppers_intention.csv')))
        excluded=['PageValues']; y=df.Revenue.astype(int); x=df.drop(columns=['Revenue']+excluded)
        cats=['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend']; bic=['VisitorType','TrafficType']
    else:
        df=pd.read_csv(io.BytesIO(z.read('Customer Churn.csv')))
        df.columns=[' '.join(c.split()) for c in df.columns]
        excluded=['Status','Customer Value']; y=df.Churn.astype(int); x=df.drop(columns=['Churn']+excluded)
        cats=['Complains','Age Group','Tariff Plan']; bic=['Complains','Tariff Plan']
    keep=~df.duplicated(); ids=np.flatnonzero(keep.to_numpy())
    x=x.loc[keep].reset_index(drop=True); y=y.loc[keep].to_numpy()
    for c in cats: x[c]=x[c].astype(str)
    info=dict(raw_n=len(df),n=len(y),duplicates_removed=int((~keep).sum()),positives=int(y.sum()),
              features=x.columns.tolist(),categorical=cats,excluded=excluded,source_url=url,source_sha256=checksum)
    return x,y,ids,cats,bic,info

def outer_splits(name,y):
    n=len(y)
    if name=='bank':
        for f in range(5):
            a=int(n*(.4+.1*f)); b=int(n*(.5+.1*f)); c=int(n*(.6+.1*f)) if f<4 else n
            mid=a+(b-a)//2
            yield np.arange(a),np.arange(a,mid),np.arange(mid,b),np.arange(b,c)
    else:
        for f,(dev,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(np.zeros(n),y)):
            tr,ca=train_test_split(dev,test_size=.2,stratify=y[dev],random_state=SEED+f)
            cf,cs=train_test_split(ca,test_size=.5,stratify=y[ca],random_state=SEED+50+f)
            yield tuple(np.sort(v) for v in [tr,cf,cs,te])

def inner_split(name,tr,y,seed):
    if name=='bank': return tr[:int(.8*len(tr))],tr[int(.8*len(tr)):]
    a,b=train_test_split(tr,test_size=.2,stratify=y[tr],random_state=seed)
    return np.sort(a),np.sort(b)

def resample(index,name,rng):
    if name=='bank':
        block=min(100,len(index)); starts=rng.integers(0,len(index),int(np.ceil(len(index)/block)))
        pos=((starts[:,None]+np.arange(block))%len(index)).ravel()[:len(index)]
    else: pos=rng.integers(0,len(index),len(index))
    return index[np.sort(pos)]

def preprocess(x,cats):
    nums=[c for c in x if c not in cats]
    return ColumnTransformer([
        ('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nums),
        ('cat',make_pipeline(SimpleImputer(strategy='constant',fill_value='__MISSING__'),
                             OneHotEncoder(handle_unknown='ignore',sparse_output=False)),cats)],sparse_threshold=0)

class Frequency:
    def __init__(self,columns): self.columns=columns
    def keys(self,x): return list(map(tuple,x[self.columns].fillna('__MISSING__').astype(str).to_numpy()))
    def fit(self,x,y):
        self.prior=float(np.mean(y)); totals={}
        for k,v in zip(self.keys(x),y):
            n,s=totals.get(k,(0,0)); totals[k]=(n+1,s+int(v))
        self.rates={k:(s+20*self.prior)/(n+20) for k,(n,s) in totals.items()}; return self
    def predict_proba(self,x):
        p=np.array([self.rates.get(k,self.prior) for k in self.keys(x)])
        return np.column_stack([1-p,p])

def estimator(m,value,x,cats,bic,seed):
    if m==0: return Frequency(bic)
    if m==1: model=LogisticRegression(C=value,max_iter=1500,random_state=seed)
    elif m==2: model=RandomForestClassifier(n_estimators=150,min_samples_leaf=value,max_features='sqrt',n_jobs=2,random_state=seed)
    else: model=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=value,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=seed)
    return make_pipeline(preprocess(x,cats),model)

CANDIDATES=[[20],[.1,1.,10.],[5,20,50],[7,15,31]]

def fit_maps(p,y):
    p=np.clip(p,1e-6,1-1e-6); out=[{}]
    for kind in ['platt','beta']:
        z=logit(p)[:,None] if kind=='platt' else np.column_stack([np.log(p),-np.log1p(-p)])
        z=np.column_stack([z,np.ones(len(p))]); start=np.r_[np.ones(z.shape[1]-1),0.]
        bounds=[(1e-3,50)]*(z.shape[1]-1)+[(-25,25)]
        def objective(theta):
            eta=z@theta
            return np.mean(np.logaddexp(0,eta)-y*eta),z.T@(expit(eta)-y)/len(y)
        opt=minimize(objective,start,jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':1000,'ftol':1e-12})
        if not opt.success: raise RuntimeError(str(opt.message))
        out.append({'kind':kind,'parameters':opt.x.tolist(),'at_lower_bound':bool(np.any(opt.x[:-1]<.00101))})
    iso=IsotonicRegression(out_of_bounds='clip').fit(p,y)
    out.append({'kind':'iso','x':iso.X_thresholds_.tolist(),'y':iso.y_thresholds_.tolist()})
    return out

def transform(p,mapping):
    if not mapping: return p.copy()
    p=np.clip(p,1e-6,1-1e-6)
    if mapping['kind']=='iso': return np.interp(p,mapping['x'],mapping['y'])
    t=mapping['parameters']
    if mapping['kind']=='platt': return expit(t[0]*logit(p)+t[1])
    return expit(t[0]*np.log(p)-t[1]*np.log1p(-p)+t[2])

def choose(p,rank,ids,cost=COST,capacity=CAP):
    order=np.lexsort((ids,-rank)); eligible=order[p[order]>cost]
    action=np.zeros(len(p),dtype=np.int8); action[eligible[:int(len(p)*capacity)]]=1
    return action

def direct_threshold(p,y,ids,cost,capacity):
    order=np.lexsort((ids,-p)); s=p[order]; k=int(len(p)*capacity)
    if k==0: return 1.0
    # Feasible strict thresholds: no action, complete score groups, or a full capacity prefix.
    ends=np.flatnonzero(s[:-1]>s[1:])+1
    lengths=np.unique(np.r_[0,ends[ends<k],k])
    utility=np.r_[0.,np.cumsum(y[order]-cost)]
    m=int(lengths[np.argmax(utility[lengths])])
    if m==0: return 1.0
    if m==k: return -1.0
    return float((s[m-1]+s[m])/2)

def scenario_key(c,q): return f'{c:g}/{q:g}'

def evaluate_selection(pc,yc,ic,ps,ys,ids,maps):
    calibrated=np.array([transform(ps,a) for a in maps])
    losses=[float(np.mean((p-ys)**2)) for p in calibrated]
    choices={}
    for c in COSTS:
        for q in CAPS:
            scores=[float(np.mean(choose(p,ps,ids,c,q)*(ys-c))) for p in calibrated]
            choices[scenario_key(c,q)]={'utility_map':int(np.argmax(scores)),
                'threshold':direct_threshold(pc,yc,ic,c,q)}
    return {'brier_map':int(np.argmin(losses)),'selection_brier':losses,'scenarios':choices}

def make_stress(x,te,cats,name,f):
    blocks=[]
    for s,rate in STRESS:
        for r in range(MASKS):
            z=x.iloc[te].copy().astype({c:'float64' for c in x if c not in cats})
            if rate:
                rng=np.random.default_rng(SEED+list(SOURCES).index(name)*10000+100*f+r)
                shape=(len(te),x.shape[1]) if s=='cell' else (len(te),1)
                mask=np.broadcast_to(rng.random(shape)<rate,z.shape)
                z=z.mask(mask)
            blocks.append(z)
    return pd.concat(blocks,ignore_index=True)

def fit_fold(name,f,rep,data):
    x,y,ids,cats,bic,info=data; tr,cf,cs,te=list(outer_splits(name,y))[f]
    groups=[tr,cf,cs,te]
    assert len(np.unique(np.concatenate(groups)))==sum(map(len,groups))
    if name=='bank': assert tr.max()<cf.min()<=cf.max()<cs.min()<=cs.max()<te.min()
    seed=SEED+list(SOURCES).index(name)*10000+100*f+rep
    it,iv=inner_split(name,tr,y,SEED+f)
    if rep:
        rng=np.random.default_rng(seed)
        it=resample(it,name,rng); tr=resample(tr,name,rng); cf=resample(cf,name,rng)
    ztest=make_stress(x,te,cats,name,f)
    scores=np.zeros((len(MODELS),len(STRESS),MASKS,len(te)))
    record={'fold':f,'fit_replicate':rep,'train_n':len(tr),'calibration_fit_n':len(cf),'selection_n':len(cs),'test_n':len(te),
            'train_prevalence':float(y[tr].mean()),'calibration_prevalence':float(y[cf].mean()),
            'selection_prevalence':float(y[cs].mean()),'test_prevalence':float(y[te].mean()),'models':[]}
    for m in range(len(MODELS)):
        t0=time.perf_counter(); losses=[]
        with threadpool_limits(limits=2):
            for candidate in CANDIDATES[m]:
                model=estimator(m,candidate,x,cats,bic,seed).fit(x.iloc[it],y[it])
                losses.append(float(brier_score_loss(y[iv],model.predict_proba(x.iloc[iv])[:,1])))
            best=CANDIDATES[m][int(np.argmin(losses))]
            model=estimator(m,best,x,cats,bic,seed).fit(x.iloc[tr],y[tr])
            pc=model.predict_proba(x.iloc[cf])[:,1]; ps=model.predict_proba(x.iloc[cs])[:,1]
            maps=fit_maps(pc,y[cf]); selection=evaluate_selection(pc,y[cf],ids[cf],ps,y[cs],ids[cs],maps)
            scores[m]=model.predict_proba(ztest)[:,1].reshape(len(STRESS),MASKS,len(te))
        record['models'].append({'name':MODELS[m],'candidate_losses':losses,'chosen_parameter':best,'maps':maps,**selection})
        print(f'{name} fold {f+1}/5 refit {rep+1}/{FITS} {MODELS[m]}: {time.perf_counter()-t0:.1f}s',flush=True)
    return scores,record,te

def run():
    OUT.mkdir(exist_ok=True); cache=OUT/'.cache'; cache.mkdir(exist_ok=True)
    fingerprint=sha(__file__)+sha(ROOT/'protocol.json')
    manifest={'revision':2,'seed':SEED,'fit_replicates':FITS,'mask_replicates':MASKS,'stress':STRESS,'costs':COSTS,'capacities':CAPS,
              'models':MODELS,'policies':POLICIES,'protocol_sha256':sha(ROOT/'protocol.json'),'code_sha256':sha(__file__),
              'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'datasets':{}}
    for name in SOURCES:
        data=read_data(name); x,y,ids,cats,bic,info=data; rows=[]; all_scores=[]
        for f in range(5):
            reps=[]; records=[]
            for rep in range(FITS):
                cp=cache/f'{name}_{f}_{rep}.npz'; jp=cp.with_suffix('.json')
                valid=cp.exists() and jp.exists() and json.loads(jp.read_text()).get('fingerprint')==fingerprint
                if valid:
                    zz=np.load(cp); score=zz['raw']; te=zz['test_index']; rec=json.loads(jp.read_text())['record']
                    print(f'{name} fold {f+1} refit {rep+1}: verified cache',flush=True)
                else:
                    score,rec,te=fit_fold(name,f,rep,data)
                    np.savez_compressed(cp,raw=score,test_index=te); dump(jp,{'fingerprint':fingerprint,'record':rec})
                reps.append(score); records.append(rec)
            all_scores.append(np.stack(reps)); rows.append((te,np.full(len(te),f))); info.setdefault('folds',[]).append(records)
        ti=np.concatenate([r[0] for r in rows]); fold=np.concatenate([r[1] for r in rows])
        assert len(np.unique(ti))==len(ti)
        np.savez_compressed(OUT/f'{name}_predictions.npz',raw=np.concatenate(all_scores,axis=-1),y=y[ti],row_id=ids[ti],fold=fold)
        info.update(test_n=len(ti),test_positive=int(y[ti].sum()))
        manifest['datasets'][name]=info; dump(OUT/'manifest.json',manifest)
    report()

def actions(raw,ids,fold,records,rep,m,s,r,c=COST,q=CAP):
    pred=raw[rep,m,s,r]; result=np.zeros((len(POLICIES),len(pred)),dtype=np.int8)
    for f in np.unique(fold):
        ii=np.flatnonzero(fold==f); rec=records[int(f)][rep]['models'][m]; p=pred[ii]
        mapped=[transform(p,a) for a in rec['maps']]
        config=rec['scenarios'][scenario_key(c,q)]
        use=list(range(4))+[rec['brier_map'],config['utility_map']]
        for j,k in enumerate(use): result[j,ii]=choose(mapped[k],p,ids[ii],c,q)
        result[6,ii]=choose(p,p,ids[ii],config['threshold'],q)
    return result

def bootstrap(values,fold,name,n_boot=2000,block=100):
    # Rows are paired across policies; actions and fitted models remain fixed.
    if values.ndim==1: values=values[:,None]
    rng=np.random.default_rng(SEED+block); means=np.zeros((n_boot,values.shape[1]))
    groups=[np.flatnonzero(fold==f) for f in np.unique(fold)]
    for b in range(n_boot):
        total=np.zeros(values.shape[1])
        for ii in groups:
            if name=='bank':
                starts=rng.integers(0,len(ii),int(np.ceil(len(ii)/block)))
                jj=ii[((starts[:,None]+np.arange(block))%len(ii)).ravel()[:len(ii)]]
            else: jj=ii[rng.integers(0,len(ii),len(ii))]
            total+=values[jj].sum(axis=0)
        means[b]=total/len(values)
    return np.quantile(means,[.025,.975],axis=0).T

def report():
    man=json.loads((OUT/'manifest.json').read_text()); dataset_rows=[]; model_rows=[]; policy_rows=[]; stress_rows=[]; sensitivity=[]; diagnostic=[]; window_rows=[]; checks=[]
    for name,info in man['datasets'].items():
        z=np.load(OUT/f'{name}_predictions.npz'); raw=z['raw']; y=z['y']; ids=z['row_id']; fold=z['fold']; recs=info['folds']
        dataset_rows.append({'dataset':name,'raw_n':info['raw_n'],'retained_n':info['n'],'removed_duplicates':info['duplicates_removed'],
            'predictors':len(info['features']),'test_n':len(y),'test_prevalence':y.mean(),'split':'chronological' if name=='bank' else 'stratified'})
        utility=np.zeros((FITS,4,len(STRESS),MASKS,7)); contributions=np.zeros((4,7,len(y)))
        for rep in range(FITS):
            for m in range(4):
                for s in range(len(STRESS)):
                    for r in range(MASKS):
                        a=actions(raw,ids,fold,recs,rep,m,s,r)
                        utility[rep,m,s,r]=1000*np.mean(a*(y-COST),axis=1)
                        if rep==0 and s==0 and r==0: contributions[m]=1000.0*a*(y-COST)
                for f in range(5):
                    ii=np.flatnonzero(fold==f); modelrec=recs[f][rep]['models'][m]
                    ac=actions(raw,ids,fold,recs,rep,m,0,0)
                    window_rows.append({'dataset':name,'fold':f+1,'fit_replicate':rep,'model':MODELS[m],
                        'test_prevalence':y[ii].mean(),'calibration_prevalence':recs[f][rep]['calibration_prevalence'],
                        'selection_prevalence':recs[f][rep]['selection_prevalence'],
                        'chosen_parameter':modelrec['chosen_parameter'],'brier_map':MAPS[modelrec['brier_map']],
                        'raw_utility':1000*np.mean(ac[0,ii]*(y[ii]-COST)),
                        'selected_utility':1000*np.mean(ac[4,ii]*(y[ii]-COST)),
                        'selected_brier':float(np.mean((transform(raw[rep,m,0,0,ii],modelrec['maps'][modelrec['brier_map']])-y[ii])**2))})
        # Per-model main table, original fit, with uncertainty reported separately.
        for m in range(4):
            p=raw[0,m,0,0]; selected=np.empty(len(y))
            for f in range(5):
                ii=np.flatnonzero(fold==f); modelrec=recs[f][0]['models'][m]
                selected[ii]=transform(p[ii],modelrec['maps'][modelrec['brier_map']])
            model_rows.append({'dataset':name,'model':MODELS[m],'auc':np.average([roc_auc_score(y[fold==f],p[fold==f]) for f in range(5)],weights=[np.sum(fold==f) for f in range(5)]),'average_precision':np.average([average_precision_score(y[fold==f],p[fold==f]) for f in range(5)],weights=[np.sum(fold==f) for f in range(5)]),'pooled_auc':roc_auc_score(y,p),'pooled_average_precision':average_precision_score(y,p),
                'brier_raw':brier_score_loss(y,p),'brier_selected':brier_score_loss(y,selected),'utility_raw':utility[0,m,0,0,0],
                'utility_brier_selected':utility[0,m,0,0,4],
                'delta_refit_min':float(np.min(utility[:,m,0,0,4]-utility[:,m,0,0,0])),
                'delta_refit_max':float(np.max(utility[:,m,0,0,4]-utility[:,m,0,0,0]))})
        avg_contrib=contributions.mean(axis=0).T
        ci=bootstrap(avg_contrib-avg_contrib[:,[0]],fold,name)
        dif=utility[:,:,0,0,:].mean(axis=1)
        dif=dif-dif[:,[0]]
        for j,pol in enumerate(POLICIES):
            policy_rows.append({'dataset':name,'policy':pol,'utility':avg_contrib[:,j].mean(),'delta_vs_raw':dif[0,j],
                'ci_low':ci[j,0],'ci_high':ci[j,1],'refit_min':dif[:,j].min(),'refit_max':dif[:,j].max(),
                'cell50_utility':utility[0,:,3,:,j].mean(),'record50_utility':utility[0,:,6,:,j].mean()})
        for s,(kind,rate) in enumerate(STRESS):
            for j,pol in enumerate(POLICIES):
                val=utility[:,:,s,:,j].mean(axis=(1,2))
                stress_rows.append({'dataset':name,'stress':kind,'rate':rate,'policy':pol,'utility':val[0],
                    'refit_min':val.min(),'refit_max':val.max()})
        if name=='bank':
            for block in [25,100,250]:
                ci2=bootstrap(avg_contrib-avg_contrib[:,[0]],fold,name,block=block)
                checks.append({'dataset':name,'bootstrap_block':block,'delta_ci':{p:ci2[j].tolist() for j,p in enumerate(POLICIES)}})
        # All fixed models, all cost/capacity pairs, original fit; no favorable-model selection.
        for c in COSTS:
            for q in CAPS:
                utilities=[]
                for m in range(4):
                    a=actions(raw,ids,fold,recs,0,m,0,0,c,q)
                    utilities.append(1000*np.mean(a*(y-c),axis=1))
                    for f in range(5):
                        ii=np.flatnonzero(fold==f); p=raw[0,m,0,0,ii]; base=a[0,ii]; order=np.lexsort((ids[ii],-p))
                        for k in range(1,4):
                            mapped=transform(p,recs[f][0]['models'][m]['maps'][k]); b=a[k,ii]
                            n0=int(base.sum()); n1=int(b.sum()); length=abs(n1-n0)
                            analytic=1000*np.sign(n1-n0)*np.sum(y[ii][order[min(n0,n1):max(n0,n1)]]-c)/len(ii)
                            actual=1000*np.mean((b-base)*(y[ii]-c))
                            assert np.isclose(analytic,actual,atol=1e-10), 'Prefix identity failed'
                            saturated=n0==n1==int(len(ii)*q)
                            if saturated: assert np.array_equal(base,b)
                            diagnostic.append({'dataset':name,'model':MODELS[m],'fold':f+1,'map':MAPS[k],'cost':c,'capacity':q,
                                'brier_improvement':float(np.mean((p-y[ii])**2)-np.mean((mapped-y[ii])**2)),
                                'utility_delta':actual,'changed_fraction':float(np.mean(base!=b)),
                                'raw_selected':n0,'mapped_selected':n1,'both_capacity_bound':saturated,'prefix_difference':length})
                for j,p in enumerate(POLICIES):
                    v=np.mean(utilities,axis=0)
                    sensitivity.append({'dataset':name,'cost':c,'capacity':q,'policy':p,'utility':v[j],'delta_vs_raw':v[j]-v[0]})
    frames={ 'table1':dataset_rows,'table2':model_rows,'table3':policy_rows,'stress_results':stress_rows,
             'sensitivity_results':sensitivity,'decision_diagnostics':diagnostic,'fold_results':window_rows}
    for filename,rows in frames.items(): pd.DataFrame(rows).to_csv(OUT/f'{filename}.csv',index=False,float_format='%.10f')
    diag=pd.DataFrame(diagnostic)
    stats={'prefix_identity_checks':len(diag),'all_prefix_checks_passed':True,
           'capacity_bound_checks':int(diag.both_capacity_bound.sum()),
           'unchanged_decision_fraction':float((diag.changed_fraction==0).mean()),
           'brier_better_utility_worse_count':int(((diag.brier_improvement>1e-10)&(diag.utility_delta<-1e-10)).sum()),
           'brier_better_count':int((diag.brier_improvement>1e-10).sum()),'bootstrap_sensitivity':checks}
    dump(OUT/'analysis.json',stats); plots(frames,stats)
    print(json.dumps(stats,indent=2),flush=True)

def plots(frames,stats):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.dpi':600,'axes.titleweight':'bold','axes.labelsize':9})
    colors={'Raw':'#222222','Brier-selected':'#0072B2','Utility-selected':'#D55E00','Direct threshold':'#009E73',
            'Logit-Platt':'#0072B2','Beta':'#CC79A7','Isotonic':'#D55E00'}
    def save(fig,name):
        fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');fig.savefig(OUT/f'{name}.png',bbox_inches='tight');plt.close(fig)
    df=pd.DataFrame(frames['sensitivity_results']); selected=df[df.policy=='Brier-selected']
    limit=max(1.,np.abs(selected.delta_vs_raw).max())
    fig,axes=plt.subplots(1,3,figsize=(7.5,3.0),layout='constrained')
    for ax,name,label in zip(axes,SOURCES,['a','b','c']):
        arr=selected[selected.dataset==name].pivot(index='cost',columns='capacity',values='delta_vs_raw').to_numpy()
        im=ax.imshow(arr,cmap='RdBu',norm=TwoSlopeNorm(vcenter=0,vmin=-limit,vmax=limit),aspect='auto')
        ax.set(xticks=range(len(CAPS)),xticklabels=[f'{100*x:g}' for x in CAPS],yticks=range(len(COSTS)),yticklabels=COSTS,
               xlabel='Action capacity (%)',ylabel='Assumed cost',title=f'({label}) {NAMES[name]}')
        for i in range(len(COSTS)):
            for j in range(len(CAPS)):
                ax.text(j,i,('0.0' if abs(arr[i,j])<.05 else f'{arr[i,j]:.1f}'),ha='center',va='center',fontsize=8,color='white' if abs(arr[i,j])>.65*limit else 'black')
    fig.colorbar(im,ax=axes,label='Brier-selected minus raw utility / 1,000',shrink=.85);save(fig,'figure1')
    df=pd.DataFrame(frames['stress_results']); fig,axes=plt.subplots(2,3,figsize=(7.5,5.6),layout='constrained')
    for row,kind in enumerate(['cell','record']):
        for col,name in enumerate(SOURCES):
            ax=axes[row,col]
            for pol in ['Raw','Brier-selected','Utility-selected','Direct threshold']:
                sub=df[(df.dataset==name)&(df.policy==pol)&df.stress.isin(['clean',kind])].sort_values('rate')
                ax.plot(sub.rate,sub.utility,'o-',ms=3,label=pol,color=colors[pol],lw=1.5)
                ax.fill_between(sub.rate,sub.refit_min,sub.refit_max,color=colors[pol],alpha=.12,linewidth=0)
            ax.set(title=f'({chr(97+row*3+col)}) {NAMES[name]}',xlabel='Missingness rate',ylabel=('Cell missingness' if row==0 else 'Record outage')+'\nUtility / 1,000',xticks=[0,.15,.3,.5]);ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    save(fig,'figure3')
    df=pd.DataFrame(frames['decision_diagnostics']); primary=df[(df.cost==COST)&(df.capacity==CAP)]
    fig,axes=plt.subplots(1,3,figsize=(7.5,3.3),layout='constrained')
    for ax,name,label in zip(axes,SOURCES,['a','b','c']):
        sub=primary[primary.dataset==name]
        for mapname,marker in zip(MAPS[1:],['o','^','s']):
            ss=sub[sub['map']==mapname];ax.scatter(ss.brier_improvement,ss.utility_delta,c=colors[mapname],marker=marker,s=24,alpha=.8,label=mapname)
        ax.axhline(0,c='gray',lw=.8);ax.axvline(0,c='gray',lw=.8)
        ax.set(title=f'({label}) {NAMES[name]}',xlabel='Brier loss improvement\n(raw − calibrated)',ylabel='Calibrated − raw utility / 1,000');ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False);save(fig,'figure2')
    df=pd.DataFrame(frames['fold_results']);fig,axes=plt.subplots(1,2,figsize=(7.5,3.6),layout='constrained')
    bank=df[(df.dataset=='bank')&(df.fit_replicate==0)].groupby('fold').mean(numeric_only=True)
    ax=axes[0]
    for col,label,color in [('calibration_prevalence','Calibration fit','#0072B2'),('selection_prevalence','Calibration selection','#009E73'),('test_prevalence','Test','#D55E00')]:
        ax.plot(bank.index,bank[col]*100,'o-',label=label,color=color)
    ax.set(title='(a) Bank: prevalence by window',xlabel='Chronological test window',ylabel='Positive outcomes (%)',xticks=range(1,6));ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.2)
    ax=axes[1]
    for j,(name,color) in enumerate(zip(SOURCES,['#0072B2','#D55E00','#009E73'])):
        sub=df[df.dataset==name].copy();sub['delta']=sub.selected_utility-sub.raw_utility
        # Weighted fold means preserve the pooled test-sample estimand.
        info=json.loads((OUT/'manifest.json').read_text())['datasets'][name]
        weights={f+1:info['folds'][f][0]['test_n'] for f in range(5)}
        vals=[np.average(g.delta,weights=g.fold.map(weights)) for _,g in sub.groupby('fit_replicate')]
        ax.scatter([j-.10,j,j+.10],vals,color=color,s=40)
        ax.vlines(j,min(vals),max(vals),color=color,alpha=.4,lw=4)
    ax.axhline(0,c='gray',lw=.8);ax.set(title='(b) Refitting stability',xticks=range(3),xticklabels=['Bank','Shopping','Churn'],ylabel='Selected − raw utility / 1,000');ax.grid(axis='y',alpha=.2);save(fig,'figure4')
    fig,ax=plt.subplots(figsize=(10,3.2),layout='constrained');ax.axis('off')
    boxes=[(.02,.30,.22,.50,'Observed business data\nThree public datasets\nTemporal + stratified tests'),(.30,.30,.27,.50,'Separate fit / selection / test\nFour predictors × four maps\nCost and capacity knowledge'),(.65,.30,.32,.50,'Decision audit\nWhen are actions unchanged?\nWhen does calibration harm utility?')]
    from matplotlib.patches import FancyBboxPatch
    for xx,yy,ww,hh,txt in boxes:
        ax.add_patch(FancyBboxPatch((xx,yy),ww,hh,boxstyle='round,pad=.012',facecolor='#EFF4F8',edgecolor='#32627B',lw=1.3,transform=ax.transAxes));ax.text(xx+ww/2,yy+hh/2,txt,ha='center',va='center',fontsize=10,linespacing=1.6,transform=ax.transAxes)
    for a,b in [(.25,.29),(.59,.64)]:ax.annotate('',xy=(b,.55),xytext=(a,.55),arrowprops={'arrowstyle':'->','lw':1.8},xycoords='axes fraction')
    ax.text(.5,.12,'Better probability estimates do not necessarily produce different or better screening decisions.',ha='center',fontsize=10,transform=ax.transAxes)
    save(fig,'graphical_abstract')

def self_test():
    from itertools import product
    p=np.array([.8,.8,.3,.1]); ids=np.arange(4); y=np.array([1,0,1,0])
    for c in [.1,.4,.9]:
        for q in [.25,.5,1.]:
            a=choose(p,p,ids,c,q); feasible=[np.array(v) for v in product([0,1],repeat=4) if sum(v)<=int(4*q)]
            assert np.isclose(np.sum(a*(p-c)),max(np.sum(v*(p-c)) for v in feasible))
            threshold=direct_threshold(p,y,ids,c,q)
            actual=np.sum(choose(p,p,ids,threshold,q)*(y-c))
            possibilities=[np.sum(choose(p,p,ids,t,q)*(y-c)) for t in [-1,.1,.3,.8,1]]
            assert actual==max(possibilities)
    grid=np.linspace(0,1,1001); maps=fit_maps(np.linspace(.01,.99,100),np.array([0,1]*50))
    for m in maps: assert np.all(np.diff(transform(grid,m))>=-1e-12)
    p=np.array([.9,.8,.7,.6]); t=transform(p,{'kind':'iso','x':[0.,1.],'y':[.6,.6]})
    assert np.array_equal(choose(p,p,ids,.1,.5),choose(t,p,ids,.1,.5))
    assert not choose(p,p,ids,.1,0).any()
    for name in SOURCES:
        x,y,ids,cats,bic,info=read_data(name)
        tests=[]
        for tr,cf,cs,te in outer_splits(name,y):
            groups=[tr,cf,cs,te];assert len(np.unique(np.concatenate(groups)))==sum(map(len,groups));tests.extend(te)
            if name=='bank':assert tr.max()<cf.min()<=cf.max()<cs.min()<=cs.max()<te.min()
        assert len(set(tests))==len(tests)
        assert not set(info['excluded'])&set(x.columns)
    print('PASS: decision optimality, threshold optimization, monotonicity, plateau ties, disjoint splits, source checksums and exclusions.')

def verify_refit(name):
    man=json.loads((OUT/'manifest.json').read_text()); assert man['code_sha256']==sha(__file__)
    score,record,te=fit_fold(name,0,0,read_data(name)); z=np.load(OUT/f'{name}_predictions.npz')
    # Boolean indexing is applied after selecting the fit axis to preserve dimensions.
    old=z['raw'][0][...,z['fold']==0]
    delta=float(np.max(np.abs(score-old))); assert np.allclose(score,old,atol=1e-12,rtol=1e-12)
    metadata_differences=[]
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for key in a: compare(a[key],b[key])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b): compare(x,y)
        elif isinstance(a,float):
            metadata_differences.append(abs(a-b)); assert np.isclose(a,b,atol=1e-10,rtol=1e-10)
        else: assert a==b
    compare(record,man['datasets'][name]['folds'][0][0])
    result={'dataset':name,'fold':0,'fit_replicate':0,'max_absolute_prediction_difference':delta,'metadata_equivalent':True,'metadata_max_absolute_difference':max(metadata_differences,default=0.),'metadata_tolerance':1e-10,'tolerance':1e-12}
    dump(OUT/f'verification_{name}.json',result);print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run',action='store_true');parser.add_argument('--report',action='store_true');parser.add_argument('--self-test',action='store_true');parser.add_argument('--verify-refit',choices=list(SOURCES))
    args=parser.parse_args()
    if args.self_test:self_test()
    if args.run:run()
    elif args.report:report()
    if args.verify_refit:verify_refit(args.verify_refit)
    if not any(vars(args).values()):parser.print_help()
