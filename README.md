# timeseriesutils

Utility functions for working with time series data in Python. Functionality is primarily oriented around converting time series data into a format suitable for prediction with methods for tabular data such as gradient boosting. This involves "featurizing" the data and creating columns with the prediction target(s).

## Development environment

We use [conda](https://conda.io/) for environment management and [PDM](https://pdm-project.org/) for dependency management. With these tools, the development environment can be set up using:

```
conda create -n timeseriesutils python=3.11
conda activate timeseriesutils
pdm sync
```
