# Element selection for functional materials discovery by integrated machine learning of atomic contributions to properties

8.11.2021 Andrij Vasylenko


## Overview

At the high level of consideration, the fundamental differences between the materials lie in the differences between the constituent chemical elements. Before the differences are detailed with the stoichiometric ratios and then atomic structures, the materials can be conceptualised at the level of their phase fields – the fields of all possible configurations of the selected chemical elements.

This code, PhaseSelect, classifies the materials at the level of sets of elements with respect to the likelihood to manifest a target functional property, while being synthetically accessible.

Please cite 
A.Vasylenko et al. 'Element selection for functional materials discovery by integrated machine learning of atomic contributions to properties' arXiv:2202.01051 (2022) 

## Requirements

python-3.7

pip (version 19.0 or later)

OS:

Ubuntu (version 18.04 or later)

MacOS (Catalina 10.15.6 or later)


## Dependencies

TensorFlow-2.4.1

scikit-learn-0.24.0

numpy-1.19.4

pandas-1.1.4

pymatgen-2018.11.6

## Installation

```git clone https://github.com/lrcfmd/PhaseSelect.git``` 

## Run examples (Reproduce results)

### Classification of the ternary phase fields as high-temperature magnetic candidate materials
```python _mag_class.py```

### Ranking synthetic accessibility of the candidate phase fields as high-temperature magnetic candidate materials
```python _mag_ranking.py```

### To apply PhaseSelect to a generic dataset and classify phase fields w.r.t. other properties values

0. Modify template_generic_class.py and template_generic_ranking.py as templates
 
1. Prepare the training data in the format 'phase fields' - 'properties' as in e.g. DATA/mpds_magnet_CurieTc.csv
   List the phase fields of interest to classify and / or rank synthetic accessibility in the format as in e.g. DATA/magnetic_candidates.csv

2. Specify threshold value of a property for classification

   ```Tc = your_threshold_value_float_or_integer_number```

3. Change the values of the corresponding variables:
   training_data
   test_data

4. Run ```python template_generic_class.py``` 
