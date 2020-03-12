# pa-cholera-validation
Validation of the Global Cholera Risk Model

## Setup

* Developed using Python 3.8.2
* Install the requirements:
```bash
pip install -r requirements.txt
```

## Running
* Download "List of Admin Units.xlsx" from Google drive,
 and "zimbabwe_final.xlsx" from Moiz's email, 
 and put them in the `input` directory 
* From 
[HDX](https://data.humdata.org/dataset/zimbabwe-administrative-levels-0-3-boundaries?force_layout=desktop)
download the level 2 admin boundaries of Zimbabwe 
("zwe_admbnda_adm2_zimstat_ocha_20180911.zip")
and contents to `input/zwe_admbnda_adm2_zimstat_ocha_2018091/`
* Execute:
```bash
python main.py
```

## Misc
* Jupyter notebooks are committed to the repo as markdown (.md) files, 
so that only input cells are under version control. Notebooks and markdown 
files can be paired using [jupytext](https://github.com/mwouts/jupytext).
