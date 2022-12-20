# Element selection for functional materials discovery by integrated machine learning of atomic contributions to properties

8.11.2021 Andrij Vasylenko


## Overview

At the high level of consideration, the fundamental differences between the materials lie in the differences between the constituent chemical elements. Before the differences are detailed with the stoichiometric ratios and then atomic structures, the materials can be conceptualised at the level of their phase fields – the fields of all possible configurations of the selected chemical elements.

This code, PhaseSelect, classifies the materials at the level of sets of elements with respect to the likelihood to manifest a target functional property, while being synthetically accessible.

Please cite 
A.Vasylenko et al. 'Element selection for functional materials discovery by integrated machine learning of atomic contributions to properties' arXiv:2202.01051 (2022) 

## Requirements

python-3.7 (or later)

pip (version 19.0 or later)

OS:

Ubuntu (version 18.04 or later)

MacOS (Catalina 10.15.6 or later)
(note: for Macbook with M1 chip an individual tensorflow installation is preferable)


## Dependencies

TensorFlow>=2.4.1

scikit-learn>=0.24.0

numpy>=1.19.4

pandas>=1.1.4

pymatgen-2022.9.21

## Installation

```git clone https://github.com/lrcfmd/PhaseSelect.git``` 
```pip install .```
(prefered for MacOS with M1 chip) 

OR via PyPI:

```pip install phaseselect``` 

## Run examples

### End-to-end classification, regression and ranking of the phase field:
```python run_phaseselect.py```

### To apply PhaseSelect to a generic dataset and assess phase fields w.r.t. other properties values

0. Modify template_generic.py
 
1. Prepare the training data in the format 'phase fields' - 'properties' as in e.g. DATA/mpds_magnet_CurieTc.csv
   List the phase fields of interest to classify and / or rank synthetic accessibility in the format as in e.g. DATA/magnetic_candidates.csv

2. Specify property of interest (according to the heading in the training datafile)  and a threshold value for classification

   ```prop = 'max Tc'```
   ```Tc = your_threshold_value_float_or_integer_number```

3. Change the values of the corresponding paths to the files:
   input_file
   test_data

4. Run ```python template_generic.py``` 
