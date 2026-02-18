# ptemcee-bilby

Plugin for using ptemcee with bilby.

This plugin exposes the `ptemcee` sampler via the `bilby.samplers` entry point.
Once installed, you can select it in `bilby.run_sampler` using `sampler='ptemcee'`.

**Note:** `ptemcee` is no longer actively maintained.

## Installation

The package can be install using pip

```
pip install ptemcee-bilby
```

or conda

```
conda install conda-forge:ptemcee-bilby
```
