# Evidential-Surrogate
The source code for the last chapter of my PhD dissertation "Evidential-Surrogate: An Evidential Learning-based Model for Uncertainty Quantification and Active Learning". 

## Getting Started

### Surrogate Model Training

Use the following script to train the surrogate model, which takes the simulation parameters as input and outputs the hyperparameters of the evidential distribution:

```
python main.py --root DATASET \
               --loss Evidential
```

### Prediction and Uncertainty Qualification

Use the following script to tackle the regression task and estimate both aleatoric and epistemic uncertainties:

```
python eval.py --loss Evidential
               --resume PATH_TO_TRAINED_Evidential-Surrogate \
               --id ID_OF_TEST_INSTANCE
```

### Active Learning 

Use the following script to enrich the training set:

```
python select_param.py --n-candidates NUM_CANDIDATES
                       --resume PATH_TO_TRAINED_Evidential-Surrogate
                       --lam COEFFICIENT_BALANCING_UNCERTAINTY_AND_PROXIMITY 
```

Then, use the enriched dataset to train the surrogate model:

```
python main.py --root DATASET \
               --loss Evidential
               --active
```


