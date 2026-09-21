# Metro Interstate Traffic Volume — Data Pipeline Report


## Layout

```
sample_capstone_project/part2_python
  data/Metro_Interstate_Traffic_Volume.csv
  data/Clean_Metro_Interstate_Traffic_Volume.csv
  data/Features_Metro_Interstate_Traffic_Volume.csv
  figures/
  pipeline.py
  feature_engineering.py
  visualizations.py
  app.py
  pipeline.log
  report.md
  requirements.txt
```

## Setup

```
pip install -r requirements.txt
```

## Run
Scripts must be run in this order — each depends on the previous step's output:


```
python pipeline.py             # raw CSV -> Clean_Metro_Interstate_Traffic_Volume.csv
python feature_engineering.py  # clean CSV -> Features_Metro_Interstate_Traffic_Volume.csv
python visualizations.py       # features CSV -> figures/*.png

Then query the features CSV with app.py:

python app.py high-traffic
python app.py high-traffic --top 10
python app.py query-traffic --date 2017-07-04
python app.py query-traffic --date 2017-07-04 --hour 05
python app.py weather-traffic --weather Clear
python app.py --help

```

## Logging

- `logging.getLogger(__name__)` in each file
- console + `pipeline.log`
- format: time, level, module, message
- DEBUG = thresholds / maps; INFO = load/save/shape; WARNING = drops/imputes; ERROR = failures
