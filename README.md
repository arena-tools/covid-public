# Arena AI - Covid19:
 
 ## How to use these models:
 This repo contains python code for 3 different COVID-19 infection forecasting models, each at a state-level, for states within the U.S. There are 3 models: SIR, Curve fitting (IHME) and a Physics-based Phase Space model. Training data is available on S3, and updates daily. Attributions cite external public data sources that we use to create the training dataset, and external references for model methodology. 

 If you find this useful, please tell us by emailing us at contact@arena-ai.com. To file an issue, use Github issues. We hope that you find this useful, and welcome your feedback!

[More info on our work at: covid.arena-ai.com](https://covid.arena-ai.com) 
 ## Models:
 1. SIR
```
from arenacovid.models import sir
....
```
 2. Phase Space
```
from arenacovid.models import phase_space
....
```
 3. Curve fitting
```
from arenacovid.models import curve_fitting
....
```
 ## Training Data in S3:
 **Location**: _s3://arena-covid-public/covid_data_
- dataset 1
- dataset 2

## How to contribute:
...

## Attributions:

1. NY Times data: https://github.com/nytimes/covid-19-data
2. UW Model: http://www.healthdata.org/covid
3. COVID-tracking project: https://covidtracking.com/
4. State and County level ICU bed and 60+ population data: https://khn.org/news/as-coronavirus-spreads-widely-millions-of-older-americans-live-in-counties-with-no-icu-beds/#lookup
5. Government actions data: https://www.kff.org/health-costs/issue-brief/state-data-and-policy-actions-to-address-coronavirus/
6. Government actions data: https://en.wikipedia.org/wiki/U.S._state_and_local_government_response_to_the_2020_coronavirus_pandemic
7. Government actions data (crowdsourced by Rex Douglass): https://github.com/rexdouglass/TIGR 
8. State and city populations: https://data.nber.org/data/census-intercensal-county-population.html



