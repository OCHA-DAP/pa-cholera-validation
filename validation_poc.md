---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.4
  kernelspec:
    display_name: pa-cholera-validation
    language: python
    name: pa-cholera-validation
---

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

```python
# To adjust
N_YEARS = 15
P0 = 0.5 # probability of step size 0
REAL_OUTBREAKS = [ 35, 100,  127] # month number of "real" outbreaks, just made this up
DETECTION_THRESH = 4 # how many units earlier a detection can be to still be TP

# Fixed
N_STEPS = 12 * N_YEARS
STEPS = [-1, 0.0, 1]
```

### Generate the fake data

```python
rs = np.random.RandomState(1234)
steps = rs.choice(a=STEPS, size=N_STEPS, p=[(1-P0)/2, P0, (1-P0)/2])
path = np.concatenate([[0], steps]).cumsum(0) 
path = (path - min(path)) / (max(path) -  min(path))
```

### Generate detection for given threshold
Working definition of a detection: the start of a continuous group

```python
def get_detections(path, thresh):
    above_thresh = path > thresh
    # Continuous group start & stop
    groups = np.where(np.diff(np.hstack(([False],path>thresh,
                                         [False]))))[0].reshape(-1,2)
    detections = groups[:,0]
    return detections
```

### Compute TP etc

```python
def get_tp_etc(detections):
    # Make a DF of detections
    df = pd.concat([
        pd.DataFrame({"event_date": detections,
                       "true_event": False}),
        pd.DataFrame({"event_date": REAL_OUTBREAKS,
                       "true_event": True}),
        ]).sort_values(by="event_date").assign(
        TP=False, FP=False, TN=False, FN=False
    )

    # For all non-true events, if the next event is a true event
    # and within DETECTION_THRESH, this is a TP
    df.loc[ ~df['true_event'] & df['true_event'].shift(-1) & (
    df['event_date'].shift(-1)-df['event_date']<=DETECTION_THRESH),
           "TP"] = True
    # All remaining non-true events are FP
    df.loc[~df['true_event'] & ~df['TP'], 'FP'] = True
    # For true events, if the previous event is not a TP, 
    # then that true event is an FN
    df.loc[df['true_event'] & ~df['TP'].shift(1, fill_value=False), 
           'FN'] = True

    return df[['TP', 'FP', 'FN']].apply(sum)

```

### For each threshold get the number of TP, FN, FP

```python
df = pd.DataFrame({'thresh': np.arange(0, 1.05, 0.05)})
df[['TP', 'FP', 'FN']] = df.apply(
    lambda x: get_tp_etc(get_detections(path, x['thresh'])),
    axis=1,
)
df['thresh'] = df['thresh'].apply(lambda x: {f"{x:.2f}"})
df.plot(kind='bar', x='thresh', stacked=True)
plt.show()

```

### Plot precision, recall, F1


```python
df['FP'] = df['FP'].replace(0, np.nan)
df['precision'] = df.apply(lambda x: x['TP']/(x['TP']+x['FP']), axis=1)
df['recall'] = df.apply(lambda x: x['TP']/ (x['TP'] + x['FN']), axis=1)

# To avoid division by 0 error
idx = (df['precision']> 0) & (df['recall']> 0)
df.loc[idx, 'f1'] = df[idx].apply(
    lambda x:  2 /(1/x['precision'] + 1/x['recall']), axis=1)

df[['thresh', 'precision', 'recall', 'f1']].plot(x='thresh')
```

### Plot the events and risk
To do: Make this interactive with threshold slider, show TP, FN, FP

```python
thresh = 0.5
detections = get_detecdtions(path, thresh) 
fig, ax = plt.subplots()
ax.plot(path)
ax.axhline(thresh, c='k')
for d in detections:
    ax.axvline(x=d, c='r')
for r in REAL_OUTBREAKS:
    ax.axvline(x=r, c='y')
```

```python

```
