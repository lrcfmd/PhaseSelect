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

## Run example
### Classification and ranking of the ternary phase fields as high-temperature magnetic candidate materials
python __main__.py
