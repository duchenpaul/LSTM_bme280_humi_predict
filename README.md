# LSTM temperature predict
A test for LSTM matchine learning.
![](src/Figure_1.png)

## Data prepration
Here is the sample data
```
cpu_load,cpu_temp,log_date
11.63370,55.000,2019-06-01 00:00:04
0.24834,56.000,2019-06-01 00:05:04
0.16639,56.000,2019-06-01 00:10:05
9.96678,56.000,2019-06-01 00:15:04
4.06977,55.000,2019-06-01 00:20:04
```

## Usage
1. Use `data_prep.py` to generate training data.
2. Use `trainer.py` to train the model.
3. Run `tensorboard --logdir=logs` to analyze run log.
4. Use `data_plot.py` to verify the predict.