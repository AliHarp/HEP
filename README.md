[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AliHarp/HEP/HEAD)
[![Python 3.8.12](https://img.shields.io/badge/python-3.8.12-blue.svg)](https://www.python.org/downloads/release/python-3812/)
[![Read the Docs](https://readthedocs.org/projects/pip/badge/?version=latest)](https://github.com/AliHarp/HEP/blob/main/HEP_notebooks/01_intro.md)
[![License: MIT](https://img.shields.io/badge/ORCID-0000--0001--5274--5037-brightgreen)](https://orcid.org/0000-0001-5274-5037)
[![License: MIT](https://img.shields.io/badge/ORCID-0000--0003--2631--4481-brightgreen)](https://orcid.org/0000-0003-2631-4481)

# HOSPITAL EFFICIENCY PROJECT

## Overview 
The materials and data in this repository support: Hospital Efficiency Project DES model for orthopaedic elective planning [MIT permissive license](https://github.com/TomMonks/des_sharing_lit_review/blob/main/LICENSE). 

## Author ORCIDs

[![ORCID: Harper](https://img.shields.io/badge/ORCID-0000--0001--5274--5037-brightgreen)](https://orcid.org/0000-0001-5274-5037)
[![ORCID: Monks](https://img.shields.io/badge/ORCID-0000--0003--2631--4481-brightgreen)](https://orcid.org/0000-0003-2631-4481)

## Write up of study

A preprint is being prepared. 

## Aim:

The aim of this web-based simulation is to support orthopaedic elective theatre and ward planning.  

The DES model enables experimentation with the theatre schedule, number of beds, lengths-of-stay for five orthopaedic elective surgery types, the proportion of patients with a delayed discharge, and the length of delay.


## Dependencies

[![Python 3.8.12](https://img.shields.io/badge/python-3.8.12-blue.svg)](https://www.python.org/downloads/release/python-3812/)

All dependencies can be found in [`binder/environment.yml`]() and are pulled from conda-forge.  To run the code locally, we recommend install [mini-conda](https://docs.conda.io/en/latest/miniconda.html); navigating your terminal (or cmd prompt) to the directory containing the repo and issuing the following command:

> `conda env create -f binder/environment.yml`

**Online Alternatives**:
* Visit our [jupyter book](https://github.com/AliHarp/HEP) for interactive code and explanatory text
* Run our Jupyter notebooks in binder 
* [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AliHarp/HEP/HEAD)

## Repo overview 

```bash
.
├── binder
│   └── environment.yml
├── _config.yml
├── HEP_notebooks
│   ├── 01_intro
│   ├── 01_model
│   └── 02_STRESS
├── pages
├── LICENSE
├── README.md
└── _toc.yml
```
* `binder` - contains the environment.yml file (hep_env) and all dependencies managed via conda
* `_config.yml` - configuration of our Jupyter Book
* `HEP_notebooks` - the notebooks and markdown arranged by introduction, model (and test data) and reporting guidelines chapters.
* pages - set up for streamlit
* LICENSE` - details of the MIT permissive license of this work.
* `README` - this document
* `_toc.yml` - the table of contents for our Jupyter Book.

