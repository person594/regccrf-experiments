# Constraining Linear-chain CRFs to Regular Languages

This directory contains all code for replicating the experiments in the paper. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Experiments

The synthetic data experiments are located in the `synthetic` directory.
All of these experiments can be executed directly to reproduce with no arguments.

The SRL experiments are located in the `srl` directory.
Execute `srl.py` with no arguments for a detailed description of the run options.
This script can be used for training new models, generating predictions from existing models.
The corpus provided must be the ontonotes corpus 5.0 in CoNLL format -- preparation instructions
can be found [here](https://cemantix.org/data/ontonotes.html).

## Evaluation

For evaluation of the SRL experiments, the perl script from the CoNLL-05 shared task is included.
To use this script, we need model predictions, and gold standard labels.
Model predictions can be generated in the correct format directly from `srl.py predict`.
For gold-standard labels, Ontonotes must be further processed into a format compatible
with the perl script -- this can be accomplished by the script `srl/conll05/to-props.py`:
```
python3 to-props.py path/to/conll-formatted-ontonotes-5.0/data/conll-2012-test/ > conll-2012-test.prop
```
Once both of these are prepared, the official perl script can be used for evaluation:
```
perl srl-eval.pl conll-2012-test.prop model-prediction.prop
```

## Results

The directory `srl/results/` contains the output from the evaluation script for each of our trials. 


## Logs
We also include training logs for our SRL experiments  in the `srl/logs/` directory.


