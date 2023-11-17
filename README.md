[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AliHarp/HEP/HEAD)
[![Python 3.8.12](https://img.shields.io/badge/python-3.8.12-blue.svg)](https://www.python.org/downloads/release/python-3812/)
[![Read the Docs](https://readthedocs.org/projects/pip/badge/?version=latest)](https://github.com/AliHarp/HEP/blob/main/HEP_notebooks/01_intro.md)
[![License: MIT](https://img.shields.io/badge/ORCID-0000--0001--5274--5037-brightgreen)](https://orcid.org/0000-0001-5274-5037)
[![License: MIT](https://img.shields.io/badge/ORCID-0000--0003--2631--4481-brightgreen)](https://orcid.org/0000-0003-2631-4481)
[![ORCID: Pitt](https://img.shields.io/badge/ORCID-0000--0003--4026--8346-brightgreen)](https://orcid.org/0000-0003-4026-8346)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7951080.svg)](https://doi.org/10.5281/zenodo.7951080)

# HOSPITAL EFFICIENCY PROJECT

## Overview 
The materials and data in this repository support: Hospital Efficiency Project discrete-event simulation (DES) model for orthopaedic elective planning [MIT permissive license](https://github.com/AliHarp/HEP/blob/main/LICENSE).

The model is reported here [![Read the Docs](https://readthedocs.org/projects/pip/badge/?version=latest)](https://github.com/AliHarp/HEP/blob/main/HEP_notebooks/02_STRESS/STRESS_DES.md) using STRESS-DES guidelines:
[Monks et al. 2019](https://doi.org/10.1080/17477778.2018.1442155) 

## Author ORCIDs

[![ORCID: Harper](https://img.shields.io/badge/ORCID-0000--0001--5274--5037-brightgreen)](https://orcid.org/0000-0001-5274-5037)
[![ORCID: Monks](https://img.shields.io/badge/ORCID-0000--0003--2631--4481-brightgreen)](https://orcid.org/0000-0003-2631-4481)
[![ORCID: Pitt](https://img.shields.io/badge/ORCID-0000--0003--4026--8346-brightgreen)](https://orcid.org/0000-0003-4026-8346)

## Citing the model:

If you use or adapt the HEP model for research, reporting, education or any other reason, please cite it using details on Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7951080.svg)](https://doi.org/10.5281/zenodo.7951080)

Harper, A., & Monks, T. Hospital Efficiency Project  Orthopaedic Planning Model Discrete-Event Simulation [Computer software]. https://doi.org/10.5281/zenodo.7951080

```
@software{Harper_Hospital_Efficiency_Project,
author = {Harper, Alison and Monks, Thomas},
license = {MIT},
title = {{Hospital Efficiency Project  Orthopaedic Planning Model Discrete-Event Simulation}},
url = {https://github.com/AliHarp/HEP}
} 
```

## Aim:

The aim of this web-based simulation is to support orthopaedic elective theatre and ward planning.  

The DES model enables experimentation with the theatre schedule, number of beds, lengths-of-stay for five orthopaedic elective surgery types, the proportion of patients with a delayed discharge, and the length of delay.


## Dependencies

[![Python 3.8.12](https://img.shields.io/badge/python-3.8.12-blue.svg)](https://www.python.org/downloads/release/python-3812/)

All dependencies can be found in [`binder/environment.yml`]() and are pulled from conda-forge.  To run the code locally, we recommend install [mini-conda](https://docs.conda.io/en/latest/miniconda.html); navigating your terminal (or cmd prompt) to the directory containing the repo and issuing the following command:

> `conda env create -f binder/environment.yml`

## Online Alternatives
* Visit our [interactable web app](https://hospital-efficiency-project.streamlit.app/) to experiment with simulation parameters
* Visit our [jupyter book](https://aliharp.github.io/HEP/HEP_notebooks/01_intro.html) for interactive code and explanatory text
* Run our Jupyter notebooks in binder 
* [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AliHarp/HEP/HEAD)
* [A fork of this model](https://github.com/TomMonks/hep-deploy) was used by Tom Monks to create a deployable, containerised version in [Dockerhub](https://hub.docker.com/r/tommonks01/hep-sim) 


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

## Write up of study:

[**POST-COVID ORTHOPAEDIC ELECTIVE RESOURCE PLANNING USING SIMULATION
MODELLING**](https://www.medrxiv.org/content/10.1101/2023.05.31.23290774v1.full.pdf) (pre-print)

Alison Harper
, Thomas Monks
, Rebecca Wilson
, Maria Theresa Redaniel
, Emily Eyles
, Tim Jones
,
Chris Penfold
, Andrew Elliott
, Tim Keen
, Martin Pitt
, Ashley Blom
, Michael Whitehouse
,
and Andrew Judge

An open-source, generalisable discrete-event simulation was developed, including a web-based
application. The model used anonymised patient records between 2016-2019 of elective orthopaedic
procedures from an NHS Trust in England. In this paper, it is used to investigate scenarios including resourcing
(beds and theatres) and productivity (lengths-of-stay, delayed discharges, theatre activity) to support planning
for meeting new NHS targets aimed at reducing elective orthopaedic surgical backlogs in a proposed ring
fenced orthopaedic surgical facility. The simulation is interactive and intended for use by health service
planners and clinicians to support capacity planning of orthopaedic elective services
by identifying a balance of capacity across theatres and beds and predicting the impact of productivity
measures on capacity requirements. It is applicable beyond the study site and can be adapted for other
specialties.

[**Deploying Healthcare Simulation Models Using Containerization and Continuous Integration**](https://osf.io/qez45 ) (pre-print)

Alison Harper, Thomas Monks, Sean Manzi

Methods or approaches from disciplines outside of OR Modeling and Simulation (M&S) can potentially increase the functionality of simulation models. In healthcare research, where simulation models are commonly used, we see few applications of models that can easily be deployed by other researchers or by healthcare stakeholders. Models are treated as disposable artifacts, developed to deliver a set of results for stakeholders or for publication. By utilising approaches from software engineering, M&S researchers can develop models that are intended to be deployed for re-use. We propose one potential solution to deploying free and open source simulations using containerisation with continuous integration. A container provides a self-contained environment that encapsulates the model and all its required dependencies including the operating system, software, and packages. This overcomes a significant barrier to sharing models developed in open source software, which is dependency management. Isolating the environment in a container ensures that the simulation model behaves the same way across different computing environments. It also means that other users can interact with the model without installing software and packages, supporting both use and re-use, and reproducibility of results. We illustrate the approach using a model developed for orthopaedic elective recovery planning, developed with a user-friendly interface in Python, including a clear set of steps to support M&S researchers to deploy their own models using our hybrid framework.