#!/usr/bin/env python3
# BioEpi-SAFE-LSTM reproducibility code
# Maintainer: Jianyi Zhang

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ADAPT="bioepi_safe_lstm_common_scale"
BASE="bioepi_sgq_lstm_no_digital"
FULL="bioepi_sgq_lstm_full_raw"
PERSIST="persistence"
GB="gb_full_raw"


def one_headline(path: Path, output: Path) -> None:
    d=pd.read_csv(path)
    metrics={}
    for m in [ADAPT,BASE,FULL,PERSIST,GB]:
        metrics[m]=d[d.model.eq(m)].set_index('horizon')
    rows=[]
    for h in [1,2,3,4]:
        a=metrics[ADAPT].loc[h];b=metrics[BASE].loc[h];f=metrics[FULL].loc[h];p=metrics[PERSIST].loc[h];g=metrics[GB].loc[h]
        rows.append({
            'horizon_weeks':h,
            'SAFE_LSTM_WIS':a.WIS,
            'no_digital_LSTM_WIS':b.WIS,
            'static_full_digital_LSTM_WIS':f.WIS,
            'persistence_WIS':p.WIS,
            'gradient_boosting_WIS':g.WIS,
            'SAFE_improvement_vs_no_digital_percent':100*(b.WIS-a.WIS)/b.WIS,
            'SAFE_improvement_vs_static_full_digital_percent':100*(f.WIS-a.WIS)/f.WIS,
            'SAFE_improvement_vs_persistence_percent':100*(p.WIS-a.WIS)/p.WIS,
            'SAFE_Pearson':a.Pearson,
            'SAFE_Spearman':a.Spearman,
            'SAFE_coverage_50':a.coverage_50,
            'SAFE_coverage_80':a.coverage_80,
            'SAFE_coverage_95':a.coverage_95,
        })
    pd.DataFrame(rows).to_csv(output,index=False)


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=ap.parse_args()
    t=a.root/'results'/'tables';m=t/'manuscript_ready';m.mkdir(parents=True,exist_ok=True)
    one_headline(t/'v7_headline_external_common_scale.csv',m/'Table_1_temporal_external_headline.csv')
    one_headline(t/'v7_headline_heldout_fold3_common_scale.csv',m/'Table_2_locked_internal_fold3_headline.csv')

    d=pd.read_csv(t/'v7_common_scale_delay_macro.csv')
    rows=[]
    for delay in [0,1,2]:
        for h in [1,2,3,4]:
            z=d[(d.official_delay_weeks.eq(delay))&(d.horizon.eq(h))].set_index('model')
            if ADAPT not in z.index or BASE not in z.index: continue
            aa=z.loc[ADAPT];bb=z.loc[BASE]
            rows.append({'official_delay_weeks':delay,'horizon_weeks':h,'SAFE_LSTM_WIS':aa.WIS,'no_digital_LSTM_WIS':bb.WIS,'improvement_percent':100*(bb.WIS-aa.WIS)/bb.WIS,'SAFE_coverage_80':aa.coverage_80})
    pd.DataFrame(rows).to_csv(m/'Table_3_reporting_delay_scenarios.csv',index=False)

    ab=pd.read_csv(t/'v7_common_scale_ablation.csv')
    ab[['variant','horizon','WIS','delta_WIS_vs_full','coverage_80','coverage_95']].rename(columns={'horizon':'horizon_weeks'}).to_csv(m/'Table_4_adaptive_fusion_ablation.csv',index=False)

    boot=pd.read_csv(t/'v7_common_scale_bootstrap.csv')
    boot=boot[(boot.model_A.eq(ADAPT))&boot.model_B.isin([BASE,FULL,PERSIST,GB])]
    boot.to_csv(m/'Table_S1_paired_block_bootstrap.csv',index=False)

    cn=pd.read_csv(t/'v7_china_delay2_bootstrap.csv')
    cn.to_csv(m/'Table_S2_china_delay2_bootstrap.csv',index=False)

    byc=pd.read_csv(t/'v7_external_national_by_country.csv')
    byc=byc[(byc.official_delay_weeks.eq(0))&byc.model.isin([ADAPT,BASE,FULL,PERSIST])]
    byc.to_csv(m/'Table_S3_external_results_by_country.csv',index=False)

    regr=pd.read_csv(t/'v7_external_regional_by_region.csv')
    regr=regr[(regr.official_delay_weeks.eq(0))&regr.model.isin([ADAPT,BASE,FULL,PERSIST])]
    regr.to_csv(m/'Table_S4_regional_robustness.csv',index=False)

    ev=pd.read_csv(t/'v7_event_metrics.csv');ev.to_csv(m/'Table_S5_event_warning.csv',index=False)
    print(f'Wrote manuscript-ready tables to {m}')

if __name__=='__main__':main()
