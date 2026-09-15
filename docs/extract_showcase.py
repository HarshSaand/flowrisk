from pathlib import Path
import argparse, json, hashlib, sys
p=argparse.ArgumentParser();p.add_argument('--source', type=Path, required=True);a=p.parse_args()
S=a.source.resolve(); D=Path(__file__).resolve().parent; R=D.parent
D.mkdir(exist_ok=True)
import numpy as np
import pandas as pd
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def source(rel):return {'path':rel,'sha256':digest(S/rel)}
def write(data):
 import subprocess
 data['dataset_url']='https://github.com/binance/binance-public-data'
 data['source_code_commit']=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
 data['extractor_sha256']=digest(Path(__file__))
 data['sources']=sources
 data['source_repository']='https://github.com/HarshSaand/'+R.name
 data['extraction']='python docs/extract_showcase.py --source /path/to/reproduced/project'
 (D/'output-example.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str)+'\n')
f='data/test-predictions.csv';df=pd.read_csv(S/f);x=df.iloc[:2];sources=[source(f),source('outputs/provenance.json')]
rows=[]
for _,r in x.iterrows():rows.append([r.symbol,f"{r.volume_learned:,.0f}",f"{r.volume_q05:,.0f} – {r.volume_q95:,.0f}",f"{r.target_volume:,.0f}"])
write(dict(title='FlowRisk',subtitle='A saved five-minute trade-volume forecast',eyebrow='ACTUAL HELD-OUT MODEL OUTPUT',context='Forecast origin: '+x.iloc[0].timestamp+' · BTC / ETH spot aggregate trades',columns=['Symbol','Forecast · USDT','90% model interval','Observed · USDT'],rows=rows,raw=x.to_dict('records'),note='Forecasts made after the minute closes. Observed outcomes are shown retrospectively, not used as inputs. This example retains the model’s overprediction; it is not a trading instruction.',input='Closed-minute trade-flow features',output='Next-five-minute quote-volume forecast and interval'))
