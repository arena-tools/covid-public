# Arena AI - Covid19:
 
 ## How to use these models:
 This repo contains python code for 3 different COVID-19 infection forecasting models, each at a state-level, for states within the U.S. There are 3 models: SIR, Curve fitting (IHME) and a "Phase Space" model. Training data is available on S3, and updates daily. Attributions cite external public data sources that we use to create the training dataset, and external references for model methodology. 

 If you find this useful, please tell us by emailing us at team@arena-ai.com. To file an issue, use Github issues. We hope that you find this useful, and welcome your feedback!

[More info on our work at: covid.arena-ai.com](https://covid.arena-ai.com) 
 ## Models:
 1. SIR
```python
from arenacovid.models import sir

# Set total population
N = 1e6
DAYS_TO_PREDICT = 180

# Initial Conditions
S = np.zeros(DAYS_TO_PREDICT)
I = np.zeros(DAYS_TO_PREDICT)
I[0] = 1
S[0] = N - I[0]

S_t, I_t = sir.simulate(S, I, N, lam=0.3, gamma=0.1)

```
 2. Phase Space
```python
from arenacovid.models import phase_space

# cumulative cases time series for a single State
y = data.set_index('date')['cumulative_cases']
m = phase_space.PhaseFitter(tau=2, b_default=-.05).fit(y)

....
```
 3. Curve fitting
```python
from arenacovid.models import curve_fitting

# Fit multiple states + countries at the same time
m = curve_fitting.HierarchicalCurveFitter(mu_lower_bound=mu_lower_bound, mu_upper_bound=mu_upper_bound)
m.fit(data["new_deaths_per_million"].values, data["group_id"].values, data["t"].values)

```
 ## Training Data in S3:
 **Daily Death Time Series**:
-  `s3://arena-covid-public/covid_data/death_time_series_us`
-  `s3://arena-covid-public/covid_data/death_time_series_combined`

Normalized death data for each state in the US, as well as normalized death data for US + International Countries. Cumulative death rate are quoted "per million" of population.

 **Daily Death + Case Series**:
-  `s3://arena-covid-public/covid_data/nyt_cases`

A processed, state-aggregated view of the New York Times reported case and deaths data.

## Reading data

```python
import pyarrow.parquet as pq
import s3fs
fs = s3fs.S3FileSystem()
s3_uri = 's3://arena-covid-public/covid_data/death_time_series_combined'
df = pq.ParquetDataset(s3_uri, filesystem=fs).read().to_pandas()
```

## Attributions:

1. NY Times data: https://github.com/nytimes/covid-19-data
2. UW Model: http://www.healthdata.org/covid
3. Ferguson & Imperial College SIR model: https://spiral.imperial.ac.uk:8443/bitstream/10044/1/77482/5/Imperial%20College%20COVID19%20NPI%20modelling%2016-03-2020.pdf
4. COVID-tracking project: https://covidtracking.com/
5. State and County level ICU bed and 60+ population data: https://khn.org/news/as-coronavirus-spreads-widely-millions-of-older-americans-live-in-counties-with-no-icu-beds/#lookup
6. Government actions data: https://www.kff.org/health-costs/issue-brief/state-data-and-policy-actions-to-address-coronavirus/
7. Government actions data: https://en.wikipedia.org/wiki/U.S._state_and_local_government_response_to_the_2020_coronavirus_pandemic
8. Government actions data (crowdsourced by Rex Douglass): https://github.com/rexdouglass/TIGR 
9. State and city populations: https://data.nber.org/data/census-intercensal-county-population.html



