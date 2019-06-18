# Demystifying

This repository contains code for analyzing molecular simulations data, mainly using machine learning methods.  

# Dependencies
 * Python 2.7
 * Scikit-learn with its standard dependencies (numpy, scipy etc.)
 * biopandas (only for postprocessing)
 
We are working on upgrading the project to python 3 as well as enabling installation of dependencies via package managers such as conda, pip and similar. 

# Using the code

## As a standalone library
Include the __modules__ library in your pyton path or import it directly in your python project. Below is an example.
### Example code
```python
from modules import feature_extraction as fe, visualization
... 
Load your data samples (input features) and labels (cluster indices) here 
...

# Create a feature extractor. All extractors implement the same methods, but in this demo we use a Random Forest 
extractor = fe.RandomForestFeatureExtractor(samples, labels, classifier_kwargs={'n_estimators': 1000})
extractor.extract_features()

# Do postprocessing to average the importance per feature into importance per residues
# As well as highlight important residues on a protein structure
postprocessor = extractor.postprocessing(working_dir="output/", pdb_file="input/protein.pdb")
postprocessor.average()
postprocessor.evaluate_performance()
postprocessor.persist()

# Visualize the importance per residue with standard functionality
visualization.visualize([[postprocessor]],
                        show_importance=True,
                        outfile="output/importance_per_residue.png")

```


## Analyzing biological systems
The biological systems discussed in the paper (the beta2 adrenergic receptor, the voltage sensor domain (VSD) and Calmodulin (CaM)) come with independent run files. These can be used as templates for other systems. 

Input data can be downloaded at [here](https://drive.google.com/drive/folders/19V1mXz7Yu0V_2JZQ8wtgt7aZusAKs2Bb?usp=sharing).

## Benchmarking with a toy model
Start __run_benchmarks.py__ to run the benchmarks discussed in the paper. This can be useful to test different hyperparameter setups as well as to enhance ones understanding of how different methods work.

__run_toy_model__ contains a demo on how to launch single instances of the toy model. This script is currently not maintained.

# Citing this work
Either cite the code (__doi to come__) and/or our paper (__doi to come__).

# Support
Please open an issue or contact oliver.fleetwood (at) gmail.com it you have any questions or comments about the code. 