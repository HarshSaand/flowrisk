import numpy as np
import pandas as pd
import pytest
from flowrisk import features_from_minutes, split, point_metrics, design

def minute_frame(start='2026-06-01',n=180):
    return pd.DataFrame({'timestamp':pd.date_range(start,periods=n,freq='min',tz='UTC'),
         'symbol':'BTCUSDT','variance':np.arange(n)/1e8,'quote_volume':np.arange(n)+1.,
         'signed_quote':np.zeros(n),'events':np.ones(n)})

def test_targets_are_strictly_future():
    raw=minute_frame();row=features_from_minutes(raw).iloc[0]
    assert row.timestamp==raw.timestamp.iloc[59]
    assert row.target_variance==raw.variance.iloc[60]
    assert row.target_volume==raw.quote_volume.iloc[60:65].sum()
    assert row.label_end==row.timestamp+pd.Timedelta(minutes=6)

def test_future_changes_do_not_change_features():
    raw=minute_frame();a=features_from_minutes(raw)
    raw.loc[120:,'quote_volume']=999999
    b=features_from_minutes(raw)
    assert np.array_equal(design(a.iloc[:60]),design(b.iloc[:60]))

def test_gap_resets_features():
    raw=pd.concat([minute_frame(),minute_frame('2026-07-01')])
    out=features_from_minutes(raw)
    assert out[out.timestamp.dt.month==7].timestamp.min()==pd.Timestamp('2026-07-01 00:59',tz='UTC')

def test_calendar_split_horizons_are_purged():
    raw=pd.concat([minute_frame('2026-06-30 22:00'),minute_frame('2026-07-31 22:00'),minute_frame('2026-08-02')])
    parts=split(features_from_minutes(raw))
    assert parts['train'].label_end.max()<parts['development'].timestamp.min()
    assert parts['development'].label_end.max()<parts['test'].timestamp.min()

def test_missing_partition_rejected():
    with pytest.raises(ValueError):split(features_from_minutes(minute_frame()))

def test_perfect_predictions():
    m=point_metrics(np.array([1.,2.]),np.array([1.,2.]),True)
    assert m['mae']==m['qlike']==0

def test_qlike_remains_finite_at_zero():
    assert np.isfinite(point_metrics(np.array([0.,1.]),np.array([0.,0.]),True)['qlike'])

def test_warmup_and_future_tail_removed():
    assert len(features_from_minutes(minute_frame(n=180)))==180-59-5

def test_design_is_finite():
    assert np.isfinite(design(features_from_minutes(minute_frame()))).all().all()
