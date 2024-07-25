# Evidential-Surrogate
This repository contains the source code for the last chapter of my PhD dissertation titled "Evidential-Surrogate: An Evidential Learning-based Model for Uncertainty Quantification and Active Learning."

## Getting Started

### Surrogate Model Training

To train the surrogate model, which takes simulation parameters as input and outputs the hyperparameters of the evidential distribution, use the following script:

```
python main.py --root DATASET \
               --loss Evidential
```

### Prediction and Uncertainty Qualification

To tackle the regression task and estimate both aleatoric and epistemic uncertainties, use the following script:

```
python eval.py --loss Evidential
               --resume PATH_TO_TRAINED_Evidential-Surrogate \
               --id ID_OF_TEST_INSTANCE
```

### Active Learning 

To enrich the training set, use the following script:

```
python select_param.py --n-candidates NUM_CANDIDATES
                       --resume PATH_TO_TRAINED_Evidential-Surrogate
                       --lam COEFFICIENT_BALANCING_UNCERTAINTY_AND_PROXIMITY 
```

Then, use the enriched dataset to retrain the surrogate model:

```
python main.py --root DATASET \
               --loss Evidential
               --active
```



