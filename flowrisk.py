"""Trade-only volatility and volume forecasting with calendar-held-out evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
SYMBOLS=['BTCUSDT','ETHUSDT']
SEED=42
FEATURES=['rv_1','rv_5','rv_15','rv_60','volume_1','volume_5','volume_15','volume_60',
          'events_5','imbalance_1','imbalance_5','rv_ewma','volume_ewma','hour_sin','hour_cos','symbol_eth']

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def fetch(days=7,max_mb=250):
    import requests
    folder=ROOT/'data';folder.mkdir(exist_ok=True)
    requests_list=[]
    for symbol in SYMBOLS:
        for month in [6,7,8]:
            for day in range(1,days+1):
                date=f'2026-{month:02}-{day:02}'
                filename=f'{symbol}-aggTrades-{date}.zip'
                url=f'https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{filename}'
                requests_list.append({'symbol':symbol,'date':date,'url':url,'filename':filename})
    def head(item):
        response=requests.head(item['url'],timeout=45);response.raise_for_status()
        item['bytes']=int(response.headers['content-length']);return item
    items=list(ThreadPoolExecutor(max_workers=4).map(head,requests_list))
    total=sum(item['bytes'] for item in items)
    if total>max_mb*1024**2:raise ValueError(f'Preflight {total/1024**2:.1f}MB exceeds {max_mb}MB; choose fewer days before training')
    print(f'Preflight {len(items)} archives, {total/1024**2:.1f} MiB',flush=True)
    def download(item):
        path=folder/item['filename']
        checksum=requests.get(item['url']+'.CHECKSUM',timeout=45);checksum.raise_for_status()
        expected=checksum.text.split()[0]
        if not path.exists() or sha(path)!=expected:
            partial=path.with_suffix('.zip.partial')
            with requests.get(item['url'],stream=True,timeout=90) as response:
                response.raise_for_status()
                with partial.open('wb') as file:
                    for chunk in response.iter_content(1024*1024):file.write(chunk)
            if sha(partial)!=expected:raise ValueError('Archive checksum mismatch')
            partial.replace(path)
        item['sha256']=expected
        print('fetched',item['filename'],flush=True)
        return item
    items=list(ThreadPoolExecutor(max_workers=4).map(download,items))
    dump(folder/'provenance.json',{'source':'https://github.com/binance/binance-public-data',
        'period':f'June–August 2026, first {days} contiguous days of each month only',
        'days_per_month':days,'download_cap_mb':max_mb,'compressed_bytes':total,'files':items,
        'timestamp_unit':'microseconds (official SPOT contract from 2025 onward)'})

def aggregate_archive(path,symbol):
    """Bounded-memory CSV aggregation; preserve return continuity across chunks."""
    names=['aggregate_id','price','quantity','first_id','last_id','timestamp','buyer_maker','best_match']
    columns=['price','quantity','timestamp','buyer_maker']
    summaries=[];previous=None;previous_stamp=None;count=0
    with zipfile.ZipFile(path) as archive:
        members=[x for x in archive.namelist() if x.endswith('.csv')]
        if len(members)!=1:raise ValueError('Expected one CSV archive member')
        with archive.open(members[0]) as stream:
            for chunk in pd.read_csv(stream,header=None,names=names,usecols=columns,chunksize=250000):
                price=chunk.price.to_numpy(dtype=float)
                quantity=chunk.quantity.to_numpy(dtype=float)
                stamp=chunk.timestamp.to_numpy(dtype=np.int64)
                if (not np.isfinite(price).all() or not np.isfinite(quantity).all() or
                    (price<=0).any() or (quantity<0).any() or (np.diff(stamp)<0).any() or
                    (previous_stamp is not None and stamp[0]<previous_stamp)):
                    raise ValueError('Invalid trade order or value')
                previous_stamp=stamp[-1]
                # First event in each daily file has no preceding price and contributes zero return.
                previous_price=price[0] if previous is None else previous
                returns=np.diff(np.log(np.r_[previous_price,price]));previous=price[-1]
                maker=chunk.buyer_maker.astype(str).str.lower().map({'true':True,'false':False})
                if maker.isna().any():raise ValueError('Unknown maker flag')
                quote=price*quantity
                frame=pd.DataFrame({'minute':stamp//60_000_000,'variance':returns**2,'quote_volume':quote,
                                    'signed_quote':np.where(maker,-quote,quote),'events':1})
                summaries.append(frame.groupby('minute').sum());count+=len(chunk)
    result=pd.concat(summaries).groupby(level=0).sum()
    result.index=pd.to_datetime(result.index*60,unit='s',utc=True)
    expected=pd.date_range(result.index.min().floor('D'),periods=1440,freq='min')
    if not result.index.isin(expected).all():raise ValueError('Archive crosses expected UTC date')
    result=result.reindex(expected,fill_value=0)
    result.index.name='timestamp';result=result.reset_index();result['symbol']=symbol
    return result,count

def aggregate():
    manifest=json.loads((ROOT/'data/provenance.json').read_text())
    frames=[];stats=[]
    for item in manifest['files']:
        path=ROOT/'data'/item['filename']
        if sha(path)!=item['sha256']:raise ValueError('Changed archive')
        frame,n=aggregate_archive(path,item['symbol']);frames.append(frame)
        stats.append({'symbol':item['symbol'],'date':item['date'],'aggregate_trade_events':n,'minutes':len(frame)})
        print('aggregated',item['filename'],n,flush=True)
    pd.concat(frames).to_pickle(ROOT/'data/minutes.pkl')
    dump(ROOT/'outputs/provenance.json',manifest)
    dump(ROOT/'outputs/aggregation.json',{'daily':stats,'total_aggregate_trade_events':sum(x['aggregate_trade_events'] for x in stats),
         'total_minutes':sum(x['minutes'] for x in stats),'chunk_size':250000,
         'rv_definition':'sum of squared log aggregate-trade-price returns within each UTC minute; first event per file omitted'})

def features_from_minutes(minutes):
    chunks=[]
    for symbol,part in minutes.groupby('symbol'):
        part=part.sort_values('timestamp').copy()
        breaks=(part.timestamp.diff()!=pd.Timedelta(minutes=1)).cumsum()
        for _,block in part.groupby(breaks):
            block=block.reset_index(drop=True).copy()
            for k in [1,5,15,60]:
                block[f'rv_{k}']=block.variance.rolling(k,min_periods=k).mean()
                block[f'volume_{k}']=block.quote_volume.rolling(k,min_periods=k).mean()
            block['events_5']=block.events.rolling(5).mean()
            block['imbalance_1']=block.signed_quote/block.quote_volume.clip(lower=1)
            block['imbalance_5']=block.signed_quote.rolling(5).sum()/block.quote_volume.rolling(5).sum().clip(lower=1)
            block['rv_ewma']=block.variance.ewm(span=15,adjust=False).mean()
            block['volume_ewma']=block.quote_volume.ewm(span=15,adjust=False).mean()
            hour=block.timestamp.dt.hour+block.timestamp.dt.minute/60
            block['hour_sin']=np.sin(2*np.pi*hour/24);block['hour_cos']=np.cos(2*np.pi*hour/24)
            block['symbol_eth']=int(symbol=='ETHUSDT')
            block['target_variance']=block.variance.shift(-1)
            block['target_volume']=sum(block.quote_volume.shift(-i) for i in range(1,6))
            # timestamp is the origin minute's OPEN; the last target minute closes at t+6.
            block['label_end']=block.timestamp+pd.Timedelta(minutes=6)
            chunks.append(block.dropna(subset=FEATURES+['target_variance','target_volume']))
    frame=pd.concat(chunks).sort_values(['timestamp','symbol']).reset_index(drop=True)
    return frame

def split(frame):
    june=pd.Timestamp('2026-07-01',tz='UTC');aug=pd.Timestamp('2026-08-01',tz='UTC')
    train=frame[(frame.timestamp<june)&(frame.label_end<june)]
    dev=frame[(frame.timestamp>=june)&(frame.timestamp<aug)&(frame.label_end<aug)]
    test=frame[frame.timestamp>=aug]
    if train.empty or dev.empty or test.empty:raise ValueError('All three calendar partitions required')
    if not train.label_end.max()<dev.timestamp.min() or not dev.label_end.max()<test.timestamp.min():raise ValueError('Overlapping label horizons')
    return {'train':train,'development':dev,'test':test}

def design(frame):
    x=frame[FEATURES].copy()
    for col in FEATURES:
        if col.startswith('rv_'):x[col]=np.log1p(x[col]*1e8)
        elif col.startswith('volume_'):x[col]=np.log1p(x[col]/1e6)
        elif col=='events_5':x[col]=np.log1p(x[col])
    return x

def point_metrics(y,pred,variance=False):
    y=np.asarray(y);pred=np.maximum(np.asarray(pred),1e-15)
    out={'mae':float(mean_absolute_error(y,pred)),'rmse':float(np.sqrt(np.mean((y-pred)**2))),
         'n':len(y)}
    if variance:
        ratio=np.maximum(y,1e-15)/pred
        out['qlike']=float(np.mean(ratio-np.log(ratio)-1))
    return out

def block_bootstrap(frame,y,a,b,variance):
    if variance:
        loss=lambda p: y/np.maximum(p,1e-15)-np.log(np.maximum(y,1e-15)/np.maximum(p,1e-15))-1
    else:loss=lambda p:np.abs(y-p)
    difference=loss(a)-loss(b)
    daily=pd.DataFrame({'day':frame.timestamp.dt.strftime('%Y-%m-%d').to_numpy(),'delta':difference}).groupby('day').delta.mean()
    idx=np.random.default_rng(SEED).integers(0,len(daily),size=(2000,len(daily)))
    means=daily.to_numpy()[idx].mean(1)
    return {'loss':'QLIKE' if variance else 'MAE','learned_minus_ewma':float(difference.mean()),
            'ci95_day_block':np.quantile(means,[.025,.975]).tolist(),'unique_days':len(daily),
            'replicates':2000,'interpretation':'negative favours learned; paired days jointly retain both symbols; few blocks limit reliability'}

def train():
    import joblib
    frame=features_from_minutes(pd.read_pickle(ROOT/'data/minutes.pkl'))
    parts=split(frame);x={k:design(v) for k,v in parts.items()}
    manifest={'seed':SEED,'features':FEATURES,'forecast_origin':'after minute t fully closes; predict variance t+1 and quote volume t+1..t+5',
              'split_counts':{},'provenance_sha256':sha(ROOT/'outputs/provenance.json')}
    for key,part in parts.items():
        manifest['split_counts'][key]={'rows':len(part),'min_timestamp':str(part.timestamp.min()),'max_timestamp':str(part.timestamp.max()),
                                    'label_end_max':str(part.label_end.max()),'utc_days':sorted(part.timestamp.dt.strftime('%Y-%m-%d').unique().tolist())}
    dump(ROOT/'outputs/split-manifest.json',manifest)
    predictions=parts['test'][['timestamp','symbol','target_variance','target_volume']].copy()
    results={};attempts=[]
    for task,scale in [('variance',1e8),('volume',1e-6)]:
        isvar=task=='variance';target='target_'+task
        y={k:v[target].to_numpy() for k,v in parts.items()}
        basecol='rv' if isvar else 'volume';multiplier=1 if isvar else 5
        result={'selection_metric':'QLIKE' if isvar else 'MAE','test':{},'by_symbol':{}}
        for model,col in [('historical_60min',basecol+'_60'),('ewma_15min',basecol+'_ewma')]:
            pred=parts['test'][col].to_numpy()*multiplier
            predictions[task+'_'+model]=pred
            result['test'][model]=point_metrics(y['test'],pred,isvar)
        best=None
        for leaves in [7,15]:
            start=time.perf_counter()
            model=HistGradientBoostingRegressor(loss='poisson',learning_rate=.05,max_iter=120,
                max_leaf_nodes=leaves,l2_regularization=1,early_stopping=False,random_state=SEED)
            model.fit(x['train'],y['train']*scale)
            pred=np.maximum(model.predict(x['development'])/scale,1e-15)
            scores=point_metrics(y['development'],pred,isvar)
            metric=scores['qlike' if isvar else 'mae']
            attempts.append({'task':task,'max_leaf_nodes':leaves,'development':scores,'train_seconds':time.perf_counter()-start})
            if best is None or metric<best[0]:best=(metric,model,leaves)
        _,model,leaves=best
        pred=np.maximum(model.predict(x['test'])/scale,1e-15)
        predictions[task+'_learned']=pred
        result['selected_max_leaf_nodes']=leaves
        result['test']['learned']=point_metrics(y['test'],pred,isvar)
        joblib.dump(model,ROOT/'models'/f'{task}-mean.joblib')
        quantiles={}
        for quantile in [.05,.5,.95]:
            quantile_model=HistGradientBoostingRegressor(loss='quantile',quantile=quantile,learning_rate=.05,
                max_iter=120,max_leaf_nodes=leaves,l2_regularization=1,early_stopping=False,random_state=SEED)
            quantile_model.fit(x['train'],np.log1p(y['train']*scale))
            quantiles[quantile]=np.maximum(np.expm1(quantile_model.predict(x['test']))/scale,0)
            joblib.dump(quantile_model,ROOT/'models'/f'{task}-q{quantile}.joblib')
        raw_crossing=np.mean(quantiles[.05]>quantiles[.95])
        # Rearrangement makes the three published quantiles monotonic; document crossings.
        ordered=np.sort(np.column_stack([quantiles[q] for q in [.05,.5,.95]]),axis=1)
        result['quantiles']={'raw_outer_crossing_rate':float(raw_crossing),'rearrangement':'sort q05/q50/q95 per row',
            'pinball_loss':{str(q):float(mean_pinball_loss(y['test'],ordered[:,i],alpha=q)) for i,q in enumerate([.05,.5,.95])},
            'nominal_coverage':.9,'observed_coverage':float(np.mean((y['test']>=ordered[:,0])&(y['test']<=ordered[:,2]))),
            'mean_width':float(np.mean(ordered[:,2]-ordered[:,0]))}
        predictions[task+'_q05']=ordered[:,0];predictions[task+'_q95']=ordered[:,2]
        for symbol in SYMBOLS:
            mask=parts['test'].symbol.to_numpy()==symbol
            result['by_symbol'][symbol]={m:point_metrics(y['test'][mask],predictions.loc[mask,task+'_'+m].to_numpy(),isvar)
                 for m in ['historical_60min','ewma_15min','learned']}
        result['uncertainty']=block_bootstrap(parts['test'],y['test'],pred,predictions[task+'_ewma_15min'].to_numpy(),isvar)
        results[task]=result
        print(task,json.dumps(result),flush=True)
    predictions.to_csv(ROOT/'data/test-predictions.csv',index=False)
    dump(ROOT/'outputs/results.json',results);dump(ROOT/'outputs/experiments.json',{'attempts':attempts,'selection':'development only; no refit on development'})
    plot(predictions,results)

def plot(predictions,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for ax,task,metric,title in [(axes[0],'variance','qlike','Next-minute trade-price variance'),
                                (axes[1],'volume','mae','Next-five-minute quote volume')]:
        entries=results[task]['test'];names=list(entries);values=[entries[n][metric] for n in names]
        if task=='volume':values=np.array(values)/1e6
        ax.bar(names,values,color=['#777777','#4477aa','#228833'])
        ax.set_title(title);ax.set_ylabel('QLIKE (lower is better)' if task=='variance' else 'MAE (million USDT; lower is better)')
        ax.tick_params(axis='x',labelsize=8)
    fig.suptitle(f'Held out: {predictions.timestamp.min():%d}–{predictions.timestamp.max():%d %B %Y} · BTCUSDT + ETHUSDT · trade-only pilot',fontsize=11)
    fig.tight_layout();fig.savefig(ROOT/'outputs/benchmark.png',dpi=180);plt.close(fig)
    example=predictions[(predictions.symbol=='BTCUSDT') & (predictions.timestamp<pd.Timestamp('2026-08-02',tz='UTC'))].copy()
    fig,ax=plt.subplots(figsize=(11,4))
    ax.plot(example.timestamp,example.target_volume/1e6,label='Observed next-five-minute volume',linewidth=.8)
    ax.plot(example.timestamp,example.volume_learned/1e6,label='Learned mean forecast',linewidth=.8)
    ax.fill_between(example.timestamp,example.volume_q05/1e6,example.volume_q95/1e6,alpha=.18,label='Nominal 90% quantile interval')
    ax.set(title='BTCUSDT · 1 August 2026 held-out diagnostic',ylabel='Quote volume (million USDT)')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'outputs/forecast.png',dpi=180);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['fetch','aggregate','train'])
    parser.add_argument('--days',type=int,default=7);parser.add_argument('--max-mb',type=int,default=250)
    args=parser.parse_args()
    for name in ['data','outputs','models']:(ROOT/name).mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        if args.command=='fetch':fetch(args.days,args.max_mb)
        elif args.command=='aggregate':aggregate()
        else:train()

if __name__=='__main__':main()
