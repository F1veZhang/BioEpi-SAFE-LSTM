#!/usr/bin/env python3
# BioEpi-SAFE-LSTM reproducibility code
# Maintainer: Jianyi Zhang

"""BioEpi-SAFE-LSTM v7: common-scale, surveillance-anchored adaptive digital fusion.

This script fixes a key comparability issue in earlier experiments: expert models used
slightly different target scalers. All expert quantiles are now converted from raw
outcome units to the *same no-digital SGQ-LSTM training scaler* before fusion and WIS
calculation. The adaptive layer begins with the surveillance-only SGQ-LSTM and activates
auxiliary digital experts only after they have beaten the backbone on resolved forecasts.
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

QCOLS=['q0.025','q0.1','q0.25','q0.5','q0.75','q0.9','q0.975']
QRAW=[c+'_raw' for c in QCOLS]
QLEVELS=np.array([.025,.1,.25,.5,.75,.9,.975])
BASE='bioepi_sgq_lstm_no_digital'
FULL='bioepi_sgq_lstm_full_raw'
ADAPT='bioepi_safe_lstm_common_scale'
STATIC='bioepi_safe_lstm_static_blend'
EXPERTS={
 'CHN':[BASE,FULL,'ridge_raw_cn_search','gb_full_raw'],
 'JPN':[BASE,FULL,'ridge_google_wiki','gb_full_raw'],
 'USA':[BASE,FULL,'ridge_google_wiki','ridge_social_raw','ridge_global_open','gb_full_raw'],
}
LABEL={ADAPT:'BioEpi-SAFE-LSTM',STATIC:'Static equal digital blend',BASE:'Surveillance SGQ-LSTM',FULL:'Full digital SGQ-LSTM','ridge_raw_cn_search':'China local-search expert','ridge_google_wiki':'Google/Wikipedia expert','ridge_social_raw':'Social-media expert','ridge_global_open':'Global open-data expert','gb_full_raw':'Digital gradient-boosting expert','persistence':'Persistence','seasonal_naive':'Seasonal naive'}
SHORTLABEL={BASE:'Surveillance LSTM',FULL:'Full-digital LSTM','ridge_raw_cn_search':'China-search Ridge','ridge_google_wiki':'Google/Wiki Ridge','ridge_social_raw':'Social Ridge','ridge_global_open':'Open-data Ridge','gb_full_raw':'Digital GB'}

@dataclass(frozen=True)
class Params:
 eta:float=1.0
 decay:float=.95
 margin:float=0.0
 max_digital:float=.50
 min_resolved:int=4
 loss_clip:float=5.0

def interval_score(y,l,u,a): return (u-l)+(2/a)*(l-y)*(y<l)+(2/a)*(y-u)*(y>u)
def wis(y,q):
 y=np.asarray(y,float);q=np.asarray(q,float);s=.5*np.abs(y-q[...,3])
 for a,l,u in [(.5,2,4),(.2,1,5),(.05,0,6)]: s+=(a/2)*interval_score(y,q[...,l],q[...,u],a)
 return s/3.5

def add_scores(d):
 d=d.copy();q=d[QCOLS].to_numpy(float);y=d.y_true.to_numpy(float)
 d['y_pred']=q[:,3];d['WIS']=wis(y,q);d['abs_error']=np.abs(y-q[:,3]);d['sq_error']=(y-q[:,3])**2
 d['covered_50']=((y>=q[:,2])&(y<=q[:,4])).astype(int);d['covered_80']=((y>=q[:,1])&(y<=q[:,5])).astype(int);d['covered_95']=((y>=q[:,0])&(y<=q[:,6])).astype(int)
 d['width_50']=q[:,4]-q[:,2];d['width_80']=q[:,5]-q[:,1];d['width_95']=q[:,6]-q[:,0]
 return d

def corr(y,p,kind):
 m=np.isfinite(y)&np.isfinite(p)
 if m.sum()<3 or np.std(y[m])==0 or np.std(p[m])==0:return np.nan
 return float(pearsonr(y[m],p[m])[0] if kind=='pearson' else spearmanr(y[m],p[m]).correlation)
def summary(g):
 y=g.y_true.to_numpy(float);p=g.y_pred.to_numpy(float)
 return pd.Series({'n':len(g),'WIS':g.WIS.mean(),'MAE':np.mean(np.abs(y-p)),'RMSE':math.sqrt(np.mean((y-p)**2)),'Pearson':corr(y,p,'pearson'),'Spearman':corr(y,p,'spearman'),'coverage_50':g.covered_50.mean(),'coverage_80':g.covered_80.mean(),'coverage_95':g.covered_95.mean(),'width_50':g.width_50.mean(),'width_80':g.width_80.mean(),'width_95':g.width_95.mean()})

def expert_list(c,variant='full'):
 x=list(EXPERTS[c])
 if variant=='base_only':return [BASE]
 if variant=='no_full_lstm_expert':return [m for m in x if m!=FULL]
 if variant=='no_gradient_boosting':return [m for m in x if m!='gb_full_raw']
 if variant=='no_source_specific_ridge':return [m for m in x if m in {BASE,FULL,'gb_full_raw'}]
 if variant=='no_china_local_search':return [m for m in x if m!='ridge_raw_cn_search']
 if variant=='no_google_wiki':return [m for m in x if m!='ridge_google_wiki']
 if variant=='no_social':return [m for m in x if m not in {'ridge_social_raw','ridge_global_open'}]
 return x

class Store:
 def __init__(self,d):
  d=d.copy();d.origin_week_start=pd.to_datetime(d.origin_week_start);d.target_week_start=pd.to_datetime(d.target_week_start)
  test=d[d['sample']=='test'].copy();keys=['evaluation','fold','official_delay_weeks','country','region','horizon','origin_week_start','target_week_start']
  b=test[test.model==BASE][keys+['scale_mu','scale_sd','event_threshold_z_train_q80']].drop_duplicates(keys).rename(columns={'scale_mu':'common_mu','scale_sd':'common_sd','event_threshold_z_train_q80':'common_event_threshold'})
  test=test.merge(b,on=keys,how='inner',validate='many_to_one')
  # Preserve the model-specific standardized target for audit, but use one common
  # no-digital SGQ-LSTM scaler for all cross-model scoring and fusion.
  test=test.rename(columns={'y_true':'y_true_model_specific','event_threshold_z_train_q80':'event_threshold_model_specific'})
  test['y_true']=(test.target_raw-test.common_mu)/test.common_sd
  test['event_threshold_z_train_q80']=test.common_event_threshold
  for c,cr in zip(QCOLS,QRAW):test[c]=(test[cr]-test.common_mu)/test.common_sd
  self.df=test;self.groups={k:g.copy() for k,g in test.groupby(['evaluation','fold','official_delay_weeks','country','region','horizon'],sort=False)};self.cache={}
 def panel(self,ev,c,r,h,delay,models,fold=1):
  ck=(ev,fold,delay,c,r,h,tuple(models))
  if ck in self.cache:return self.cache[ck]
  x=self.groups[(ev,fold,delay,c,r,h)];x=x[x.model.isin(models)].copy();ms=[m for m in models if m in set(x.model)]
  key=['origin_week_start','target_week_start'];cnt=x.groupby(key).model.nunique();keep=cnt[cnt==len(ms)].index;x=x.set_index(key).loc[keep].reset_index()
  meta=x[x.model==BASE].set_index(key).loc[keep].reset_index().sort_values('origin_week_start').reset_index(drop=True)
  meta=meta[key+['y_true','target_raw','common_mu','common_sd','event_threshold_z_train_q80','period']]
  idx=pd.MultiIndex.from_frame(meta[key]);cube=np.stack([x[x.model==m].set_index(key).loc[idx][QCOLS].to_numpy(float) for m in ms],1)
  self.cache[ck]=(ms,meta,cube);return ms,meta,cube
 def model_rows(self,keys,models):
  k=['evaluation','fold','official_delay_weeks','country','region','horizon','origin_week_start','target_week_start']
  x=self.df[self.df.model.isin(models)].merge(keys[k].drop_duplicates(),on=k,how='inner')
  # y_true and all quantiles are already on the common backbone scale.
  return add_scores(x)
 def scale_audit(self):
  z=self.df.copy()
  z['abs_target_z_difference']=np.abs(z.y_true_model_specific-z.y_true)
  z['abs_mu_difference']=np.abs(z.scale_mu-z.common_mu)
  z['abs_sd_difference']=np.abs(z.scale_sd-z.common_sd)
  qdiff=[]
  for c,cr in zip(QCOLS,QRAW):
   reconstructed=(z[cr]-z.scale_mu)/z.scale_sd
   qdiff.append(np.abs(reconstructed-z[c]).to_numpy())
  z['mean_abs_quantile_z_difference']=np.mean(np.column_stack(qdiff),axis=1)
  g=z.groupby(['evaluation','fold','official_delay_weeks','country','region','horizon','model'],dropna=False)
  return g.agg(n=('y_true','size'),mean_abs_target_z_difference=('abs_target_z_difference','mean'),max_abs_target_z_difference=('abs_target_z_difference','max'),mean_abs_mu_difference=('abs_mu_difference','mean'),mean_abs_sd_difference=('abs_sd_difference','mean'),mean_abs_quantile_z_difference=('mean_abs_quantile_z_difference','mean')).reset_index()

def safe_predict(ms,meta,cube,p,delay,update=True,static_blend=False):
 n,M,NQ=cube.shape;b=ms.index(BASE);out=np.empty((n,NQ));wh=np.zeros((n,M));pending=[];regret=np.zeros(M);resolved_n=0;loss=wis(meta.y_true.to_numpy()[:,None],cube);O=meta.origin_week_start.to_numpy();T=(meta.target_week_start+pd.to_timedelta(delay*7,unit='D')).to_numpy()
 for i in range(n):
  if update:
   resolved=[j for j in pending if T[j]<=O[i]]
   for j in resolved:regret=p.decay*regret+np.clip(loss[j]-loss[j,b],-p.loss_clip,p.loss_clip);resolved_n+=1
   pending=[j for j in pending if T[j]>O[i]]
  if static_blend:
   w=np.ones(M)*(p.max_digital/max(1,M-1));w[b]=1-p.max_digital
  else:
   active=np.zeros(M,bool);active[b]=True
   if update and resolved_n>=p.min_resolved:active|=(regret < -p.margin)
   score=np.zeros(M);score[active]=np.exp(-p.eta*np.clip(regret[active],-5,5));w=score/score.sum()
   if 1-w[b]>p.max_digital:
    o=np.arange(M)!=b;s=w[o].sum();w[o]=w[o]/s*p.max_digital;w[b]=1-p.max_digital
  out[i]=np.einsum('mq,m->q',cube[i],w);wh[i]=w;pending.append(i)
 return np.maximum.accumulate(out,axis=1),wh

def frame(meta,q,ev,fold,delay,c,r,h,model,variant):
 z=meta.copy();z['evaluation']=ev;z['fold']=fold;z['official_delay_weeks']=delay;z['country']=c;z['region']=r;z['horizon']=h;z['model']=model;z['variant']=variant
 for j,x in enumerate(QCOLS):z[x]=q[:,j]
 return add_scores(z)

def generate(store,p,ev,fold,delay,regions,variant='full',update=True,static_blend=False,weights=None):
 fs=[];weights=weights if weights is not None else []
 name=ADAPT if variant=='full' and not static_blend else (STATIC if static_blend else ADAPT+'__'+variant)
 for c,rs in regions.items():
  for r in rs:
   for h in [1,2,3,4]:
    ms,m,cube=store.panel(ev,c,r,h,delay,expert_list(c,variant),fold);q,w=safe_predict(ms,m,cube,p,delay,update,static_blend);fs.append(frame(m,q,ev,fold,delay,c,r,h,name,variant))
    for j,e in enumerate(ms):
     zz=m[['origin_week_start','target_week_start']].copy();zz['evaluation']=ev;zz['fold']=fold;zz['official_delay_weeks']=delay;zz['country']=c;zz['region']=r;zz['horizon']=h;zz['variant']=variant;zz['expert']=e;zz['expert_label']=LABEL.get(e,e);zz['weight']=w[:,j];weights.append(zz)
 return pd.concat(fs,ignore_index=True)

def tune_fold2(store):
 # Pre-specified safety grid. At least four resolved outcomes are required before activation.
 grid=[Params(e,d,m,md,4) for e in [.5,1,2] for d in [.9,.95,.98] for m in [0,.025,.05] for md in [.3,.5]]
 rows=[]
 for p in grid:
  vals=[]
  for c in ['CHN','JPN','USA']:
   for h in [1,2,3,4]:
    ms,me,cu=store.panel('rolling_window_cv',c,'national',h,0,expert_list(c),2);q,_=safe_predict(ms,me,cu,p,0);vals.append(wis(me.y_true.to_numpy(),q).mean())
  rows.append({**p.__dict__,'fold2_country_horizon_balanced_WIS':np.mean(vals)})
 tab=pd.DataFrame(rows).sort_values('fold2_country_horizon_balanced_WIS').reset_index(drop=True);r=tab.iloc[0];return Params(float(r.eta),float(r.decay),float(r.margin),float(r.max_digital),int(r.min_resolved),float(r.loss_clip)),tab

def metrics(pred,prefix,out):
 bc=pred.groupby(['evaluation','official_delay_weeks','model','horizon','country'],dropna=False).apply(summary,include_groups=False).reset_index();br=pred.groupby(['evaluation','official_delay_weeks','model','horizon','country','region'],dropna=False).apply(summary,include_groups=False).reset_index();cols=['WIS','MAE','RMSE','Pearson','Spearman','coverage_50','coverage_80','coverage_95','width_50','width_80','width_95'];mac=bc.groupby(['evaluation','official_delay_weeks','model','horizon'])[cols].mean().reset_index();mac['n_total']=bc.groupby(['evaluation','official_delay_weeks','model','horizon']).n.sum().to_numpy();mac['n_countries']=bc.groupby(['evaluation','official_delay_weeks','model','horizon']).country.nunique().to_numpy();bc.to_csv(out/f'{prefix}_by_country.csv',index=False);br.to_csv(out/f'{prefix}_by_region.csv',index=False);mac.to_csv(out/f'{prefix}_macro.csv',index=False);return bc,br,mac

def boot_pair(a,b,block,reps,seed):
 k=['country','origin_week_start','target_week_start'];x=a[k+['WIS']].rename(columns={'WIS':'a'}).merge(b[k+['WIS']].rename(columns={'WIS':'b'}),on=k)
 obs=float(x.groupby('country').apply(lambda z:(z.a-z.b).mean(),include_groups=False).mean());rng=np.random.default_rng(seed);country_boot=[]
 for _,z in x.groupby('country'):
  d=(z.sort_values('origin_week_start').a-z.sort_values('origin_week_start').b).to_numpy();n=len(d);n_blocks=int(np.ceil(n/block));max_start=max(1,n-block+1)
  starts=rng.integers(0,max_start,size=(reps,n_blocks));idx=(starts[:,:,None]+np.arange(block)[None,None,:]).reshape(reps,-1)[:,:n];idx=np.minimum(idx,n-1)
  country_boot.append(d[idx].mean(axis=1))
 bs=np.mean(np.column_stack(country_boot),axis=1)
 return {'mean_diff_A_minus_B':obs,'ci_low':float(np.quantile(bs,.025)),'ci_high':float(np.quantile(bs,.975)),'prob_A_better':float(np.mean(bs<0)),'n_pairs':len(x)}
def bootstrap(pred,ev,delay,A,Bs,reps):
 z=pred[(pred.evaluation==ev)&(pred.official_delay_weeks==delay)&(pred.region=='national')];rows=[]
 for h in [1,2,3,4]:
  a=z[(z.horizon==h)&(z.model==A)]
  for B in Bs:
   b=z[(z.horizon==h)&(z.model==B)]
   if a.empty or b.empty:continue
   for block in [4,8,13]:rows.append({'evaluation':ev,'delay':delay,'horizon':h,'model_A':A,'model_B':B,'block_weeks':block,**boot_pair(a,b,block,reps,7100+h*100+block)})
 return pd.DataFrame(rows)

def high_activity(pred):
 x=pred[pred.region=='national'].copy();x['activity']=np.where(x.y_true>=x.event_threshold_z_train_q80,'high','non_high');rows=[]
 for a in ['all','high','non_high']:
  z=x if a=='all' else x[x.activity==a];g=z.groupby(['evaluation','official_delay_weeks','model','horizon','country']).WIS.mean().reset_index();m=g.groupby(['evaluation','official_delay_weeks','model','horizon']).WIS.mean().reset_index();m['activity']=a;rows.append(m)
 return pd.concat(rows)

def p_exceed(q,t):
 out=[]
 for a,b in zip(q,t):
  if b<=a[0]:cdf=QLEVELS[0]
  elif b>=a[-1]:cdf=QLEVELS[-1]
  else:
   j=np.searchsorted(a,b)-1;f=(b-a[j])/max(1e-9,a[j+1]-a[j]);cdf=QLEVELS[j]+f*(QLEVELS[j+1]-QLEVELS[j])
  out.append(1-cdf)
 return np.array(out)
def event_one(g,cut):
 g=g.sort_values('origin_week_start').copy();g['prob']=p_exceed(g[QCOLS].to_numpy(),g.event_threshold_z_train_q80.to_numpy());alarms=[]
 for d in g.loc[g.prob>=cut,'origin_week_start']:
  if not alarms or (d-alarms[-1]).days>21:alarms.append(d)
 t=g[['target_week_start','y_true','event_threshold_z_train_q80']].drop_duplicates('target_week_start').sort_values('target_week_start');pi,_=find_peaks(t.y_true.to_numpy(),height=float(t.event_threshold_z_train_q80.median()),distance=4);peaks=list(t.iloc[pi].target_week_start);used=set();leads=[]
 for peak in peaks:
  e=[(i,d) for i,d in enumerate(alarms) if i not in used and d<=peak and (peak-d).days<=42]
  if e:i,d=e[0];used.add(i);leads.append((peak-d).days/7)
 tp=len(leads);fp=len(alarms)-tp;fn=len(peaks)-tp;sen=tp/(tp+fn) if tp+fn else np.nan;ppv=tp/(tp+fp) if tp+fp else np.nan;f1=2*sen*ppv/(sen+ppv) if sen+ppv else np.nan
 return {'n_events':len(peaks),'n_alarms':len(alarms),'matched':tp,'sensitivity':sen,'PPV':ppv,'F1':f1,'mean_lead_weeks':np.mean(leads) if leads else np.nan}
def events(dev,ext):
 ts=[];rs=[]
 for c in ['CHN','JPN','USA']:
  for h in [1,2,3,4]:
   a=dev[(dev.country==c)&(dev.horizon==h)&(dev.model==ADAPT)];b=ext[(ext.country==c)&(ext.horizon==h)&(ext.model==ADAPT)]
   cand=[]
   for cut in np.arange(.1,.91,.05):
    m=event_one(a,float(cut));cand.append((not(np.isfinite(m['PPV']) and m['PPV']>=.4),-np.nan_to_num(m['F1'],nan=-1),-np.nan_to_num(m['PPV'],nan=-1),cut,m))
   best=sorted(cand,key=lambda x:x[:4])[0];ts.append({'country':c,'horizon':h,'probability_threshold':best[3],**{'dev_'+k:v for k,v in best[4].items()}});rs.append({'country':c,'horizon':h,'probability_threshold':best[3],**event_one(b,best[3])})
 return pd.DataFrame(ts),pd.DataFrame(rs)

def plots(figs,extmac,delaymac,abl,w,high):
 models=[ADAPT,BASE,FULL,'persistence','gb_full_raw'];z=extmac[(extmac.official_delay_weeks==0)&extmac.model.isin(models)]
 fig,ax=plt.subplots(figsize=(9,5))
 for m in models:
  s=z[z.model==m].sort_values('horizon');ax.plot(s.horizon,s.WIS,marker='o',label=LABEL.get(m,m))
 ax.set(xlabel='Forecast horizon (weeks)',ylabel='Country-balanced WIS',xticks=[1,2,3,4],title='Common-scale temporal external evaluation');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figs/'fig1_common_scale_wis.png',dpi=220);plt.close(fig)
 a=z[z.model==ADAPT].set_index('horizon').WIS;b=z[z.model==BASE].set_index('horizon').WIS;rel=a/b;fig,ax=plt.subplots(figsize=(7,4));ax.bar(rel.index,rel.values);ax.axhline(1,ls='--');ax.set(xlabel='Horizon',ylabel='Relative WIS vs no-digital SGQ-LSTM',xticks=[1,2,3,4],title='Relative WIS versus the surveillance-only SGQ-LSTM');fig.tight_layout();fig.savefig(figs/'fig2_adaptive_vs_no_digital.png',dpi=220);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5))
 for d in [0,1,2]:
  aa=delaymac[(delaymac.official_delay_weeks==d)&(delaymac.model==ADAPT)].set_index('horizon').WIS;bb=delaymac[(delaymac.official_delay_weeks==d)&(delaymac.model==BASE)].set_index('horizon').WIS;ax.plot([1,2,3,4],(aa/bb).reindex([1,2,3,4]),marker='o',label=f'Delay {d}')
 ax.axhline(1,ls='--');ax.set(xlabel='Horizon',ylabel='Relative WIS',xticks=[1,2,3,4],title='Relative WIS under simulated official-reporting delay');ax.legend();fig.tight_layout();fig.savefig(figs/'fig3_delay.png',dpi=220);plt.close(fig)
 ws=w[(w.evaluation=='temporal_external_2025_2026')&(w.official_delay_weeks==0)&(w.variant=='full')].groupby(['country','horizon','expert_label']).weight.mean().reset_index();fig,axs=plt.subplots(1,3,figsize=(16,5),sharey=False,constrained_layout=True)
 for ax,c in zip(axs,['CHN','JPN','USA']):
  zc=ws[ws.country==c].copy();order=list(dict.fromkeys([SHORTLABEL.get(e,LABEL.get(e,e)) for e in EXPERTS[c]]));long_to_short={LABEL.get(e,e):SHORTLABEL.get(e,LABEL.get(e,e)) for e in EXPERTS[c]};zc['expert_label']=zc['expert_label'].map(lambda e:long_to_short.get(e,e));p=zc.pivot(index='expert_label',columns='horizon',values='weight').reindex(order);im=ax.imshow(p.fillna(0),aspect='auto',vmin=0,vmax=.8);ax.set_title(c);ax.set_xticks(range(4),[1,2,3,4]);ax.set_yticks(range(len(order)),order,fontsize=8);ax.set_xlabel('Forecast horizon (weeks)')
 fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.72,label='Mean fusion weight');fig.suptitle('Evidence-activated expert weights');fig.savefig(figs/'fig4_weights.png',dpi=220,bbox_inches='tight');plt.close(fig)
 p=abl.pivot(index='variant',columns='horizon',values='delta_WIS_vs_full').sort_index();fig,ax=plt.subplots(figsize=(10,5));im=ax.imshow(p,aspect='auto');ax.set_xticks(range(4),[1,2,3,4]);ax.set_yticks(range(len(p)),p.index);ax.set_title('Adaptive-fusion ablation (positive = worse)');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figs/'fig5_ablation.png',dpi=220);plt.close(fig)
 hh=high[(high.evaluation=='temporal_external_2025_2026')&(high.official_delay_weeks==0)&(high.activity=='high')&high.model.isin([ADAPT,BASE,FULL])];fig,ax=plt.subplots(figsize=(8,5))
 for m in [ADAPT,BASE,FULL]:s=hh[hh.model==m].sort_values('horizon');ax.plot(s.horizon,s.WIS,marker='o',label=LABEL.get(m,m))
 ax.set(xlabel='Horizon',ylabel='High-activity WIS',xticks=[1,2,3,4],title='Peak-sensitive evaluation');ax.legend();fig.tight_layout();fig.savefig(figs/'fig6_high_activity.png',dpi=220);plt.close(fig)

def regional_plot(figs,regional_by_region):
 z=regional_by_region[(regional_by_region.evaluation=='temporal_external_2025_2026')&(regional_by_region.official_delay_weeks==0)]
 a=z[z.model==ADAPT][['country','region','horizon','WIS']].rename(columns={'WIS':'adaptive'})
 b=z[z.model==BASE][['country','region','horizon','WIS']].rename(columns={'WIS':'base'})
 x=a.merge(b,on=['country','region','horizon']);x['relative_WIS']=x.adaptive/x.base
 usa=x[x.country=='USA'].pivot(index='region',columns='horizon',values='relative_WIS').reindex([f'hhs{i}' for i in range(1,11)])
 if len(usa):
  fig,ax=plt.subplots(figsize=(8,6));im=ax.imshow(usa,aspect='auto',vmin=min(.8,float(np.nanmin(usa))),vmax=max(1.05,float(np.nanmax(usa))));ax.set_xticks(range(4),[1,2,3,4]);ax.set_yticks(range(len(usa)),usa.index);ax.set_xlabel('Forecast horizon (weeks)');ax.set_title('USA HHS regional relative WIS: SAFE-LSTM / no-digital LSTM');fig.colorbar(im,ax=ax,label='Relative WIS (<1 favors SAFE-LSTM)');fig.tight_layout();fig.savefig(figs/'fig7_usa_hhs_relative_wis.png',dpi=220);plt.close(fig)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[1]/'data'/'expert_predictions_v6_common_scale_test.pkl.gz');ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--bootstrap-reps',type=int,default=2000);a=ap.parse_args();root=a.root;tab=root/'results'/'tables';fig=root/'results'/'figures';meta=root/'results'/'metadata';[x.mkdir(parents=True,exist_ok=True) for x in [tab,fig,meta]]
 store=Store(pd.read_pickle(a.input));store.scale_audit().to_csv(meta/'v7_common_scale_audit.csv',index=False)
 p,grid=tune_fold2(store);grid.to_csv(meta/'v7_common_scale_tuning_fold2.csv',index=False);(meta/'v7_common_scale_selected_params.json').write_text(json.dumps(p.__dict__,indent=2))
 weights=[];national={'CHN':['national'],'JPN':['national'],'USA':['national']};regional={'CHN':['northern_provinces','southern_provinces'],'USA':[f'hhs{i}' for i in range(1,11)]}
 # Primary prequential evaluations: fold 2 is used only for meta-parameter selection,
 # fold 3 is locked internal validation, and the later period is temporal evaluation.
 dev=generate(store,p,'rolling_window_cv',3,0,national,weights=weights)
 ext=pd.concat([generate(store,p,'temporal_external_2025_2026',1,d,national,weights=weights) for d in [0,1,2]],ignore_index=True)
 # Regional robustness is kept separate from the national-only headline analysis.
 reg_weights=[];dev_reg=generate(store,p,'rolling_window_cv',3,0,regional,weights=reg_weights);ext_reg=generate(store,p,'temporal_external_2025_2026',1,0,regional,weights=reg_weights)
 static=generate(store,p,'temporal_external_2025_2026',1,0,national,update=False,static_blend=True,weights=weights)
 ab=[]
 for v in ['base_only','no_full_lstm_expert','no_gradient_boosting','no_source_specific_ridge','no_china_local_search','no_google_wiki','no_social']:ab.append(generate(store,p,'temporal_external_2025_2026',1,0,national,v,weights=weights))
 ab=pd.concat(ab+[static],ignore_index=True)
 comps=[BASE,FULL,'persistence','seasonal_naive','gb_full_raw','ridge_raw_cn_search','ridge_google_wiki','ridge_social_raw','ridge_global_open','ridge_no_digital']
 devall=pd.concat([dev,store.model_rows(dev,comps)],ignore_index=True);extall=pd.concat([ext,store.model_rows(ext,comps)],ignore_index=True)
 devregall=pd.concat([dev_reg,store.model_rows(dev_reg,comps)],ignore_index=True);extregall=pd.concat([ext_reg,store.model_rows(ext_reg,comps)],ignore_index=True)
 pd.concat([dev,ext,ab],ignore_index=True).to_csv(tab/'v7_common_scale_predictions.csv.gz',index=False,compression='gzip');w=pd.concat(weights,ignore_index=True);w.to_csv(tab/'v7_common_scale_weights.csv.gz',index=False,compression='gzip')
 pd.concat([dev_reg,ext_reg],ignore_index=True).to_csv(tab/'v7_regional_predictions.csv.gz',index=False,compression='gzip');rw=pd.concat(reg_weights,ignore_index=True);rw.to_csv(tab/'v7_regional_weights.csv.gz',index=False,compression='gzip')
 dbc,dbr,dmac=metrics(devall,'v7_heldout_fold3',tab);ebc,ebr,emac=metrics(extall,'v7_external_national',tab);abc,abr,amac=metrics(pd.concat([ext[ext.official_delay_weeks==0],ab]),'v7_ablation',tab)
 rdbc,rdbr,rdmac=metrics(devregall,'v7_heldout_fold3_regional',tab);rebc,rebr,remac=metrics(extregall,'v7_external_regional',tab)
 headmods=[ADAPT,BASE,FULL,'persistence','seasonal_naive','gb_full_raw'];head=emac[(emac.official_delay_weeks==0)&emac.model.isin(headmods)].copy();bw=head[head.model==BASE].set_index('horizon').WIS;head['relative_WIS_vs_no_digital']=[r.WIS/bw.loc[r.horizon] for r in head.itertuples()];head.to_csv(tab/'v7_headline_external_common_scale.csv',index=False);dh=dmac[dmac.model.isin(headmods)].copy();db=dh[dh.model==BASE].set_index('horizon').WIS;dh['relative_WIS_vs_no_digital']=[r.WIS/db.loc[r.horizon] for r in dh.itertuples()];dh.to_csv(tab/'v7_headline_heldout_fold3_common_scale.csv',index=False)
 # Paired country-time block bootstrap on the common scale.
 boots=[]
 for dly in [0,1,2]:boots.append(bootstrap(extall,'temporal_external_2025_2026',dly,ADAPT,[BASE,FULL,'persistence','gb_full_raw'],a.bootstrap_reps))
 boots.append(bootstrap(devall,'rolling_window_cv',0,ADAPT,[BASE,FULL,'persistence','gb_full_raw'],a.bootstrap_reps));boot=pd.concat(boots,ignore_index=True);boot.to_csv(tab/'v7_common_scale_bootstrap.csv',index=False)
 fw=amac[amac.model==ADAPT].set_index('horizon').WIS;aa=amac[amac.model!=ADAPT].copy();aa['variant']=aa.model.str.replace(ADAPT+'__','',regex=False).replace({STATIC:'static_equal_digital_blend'});aa['delta_WIS_vs_full']=[r.WIS-fw.loc[r.horizon] for r in aa.itertuples()];aa.to_csv(tab/'v7_common_scale_ablation.csv',index=False)
 # Strict adaptive-fusion ablation uncertainty.
 abcomp=pd.concat([ext[ext.official_delay_weeks==0],ab],ignore_index=True);abmodels=sorted([m for m in abcomp.model.unique() if m!=ADAPT]);abboot=bootstrap(abcomp,'temporal_external_2025_2026',0,ADAPT,abmodels,a.bootstrap_reps);abboot.to_csv(tab/'v7_ablation_bootstrap.csv',index=False)
 delay=emac[emac.model.isin([ADAPT,BASE,FULL,'persistence','ridge_raw_cn_search','ridge_no_digital'])];delay.to_csv(tab/'v7_common_scale_delay_macro.csv',index=False);ebc[ebc.model.isin([ADAPT,BASE,FULL,'persistence','ridge_raw_cn_search','ridge_no_digital'])].to_csv(tab/'v7_common_scale_delay_by_country.csv',index=False)
 ch=extall[(extall.country=='CHN')&(extall.official_delay_weeks==2)];rr=[]
 for h in [1,2,3,4]:
  for A,B in [(ADAPT,BASE),('ridge_raw_cn_search','ridge_no_digital')]:
   for block in [4,8,13]:rr.append({'horizon':h,'model_A':A,'model_B':B,'block_weeks':block,**boot_pair(ch[(ch.horizon==h)&(ch.model==A)],ch[(ch.horizon==h)&(ch.model==B)],block,a.bootstrap_reps,9000+h*100+block)})
 pd.DataFrame(rr).to_csv(tab/'v7_china_delay2_bootstrap.csv',index=False)
 hi=high_activity(pd.concat([devall,extall]));hi.to_csv(tab/'v7_high_activity.csv',index=False);th,er=events(dev,ext[ext.official_delay_weeks==0]);th.to_csv(tab/'v7_event_thresholds.csv',index=False);er.to_csv(tab/'v7_event_metrics.csv',index=False)
 ws=w.groupby(['evaluation','official_delay_weeks','country','region','horizon','variant','expert','expert_label']).weight.agg(['mean','std','min','max','count']).reset_index();ws.to_csv(tab/'v7_weight_summary.csv',index=False);plots(fig,emac,delay,aa,w,hi);regional_plot(fig,rebr)
 manifest={'model':ADAPT,'display_name':'BioEpi-SAFE-LSTM','common_scale_reference':BASE,'common_scale_note':'All raw expert quantiles and targets are transformed with the no-digital SGQ-LSTM training scaler before WIS comparison and fusion.','selected_params':p.__dict__,'selection':'predefined safety grid on rolling fold 2; minimum four resolved outcomes before digital activation','internal_validation':'locked held-out rolling fold 3','external_evaluation':'2025-2026 temporal evaluation; not called untouched','online_update_rule':'Weights are updated only after prior targets become observable, including the simulated reporting delay. No future outcomes are used.','bootstrap_reps':a.bootstrap_reps,'bootstrap_blocks':[4,8,13],'headline_scope':'national-only, country-balanced','regional_scope':'China North/South and USA HHS1-HHS10 reported separately','experts':EXPERTS};(meta/'manifest_v7_common_scale.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False));print(json.dumps(manifest,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
