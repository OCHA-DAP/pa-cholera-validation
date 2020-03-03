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
```

```python
N_STEPS = 12 * 15
STEPS = [-1, 0.0, 1]
P0 = 0.5 # probability of step size 0
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
thresh = 0.5
above_thresh = path > thresh
# Continuous group start & stop
groups = np.where(np.diff(np.hstack(([False],path>thresh,[False]))))[0].reshape(-1,2)
detections = groups[:,0]

```

```python
fig, ax = plt.subplots()
ax.plot(path)
ax.axhline(thresh, c='k')
for d in detections:
    ax.axvline(x=d, c='r')
```
