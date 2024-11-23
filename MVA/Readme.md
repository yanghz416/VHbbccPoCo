Note: PyTorch is needed for GNN trainings. It is not needed for LGBM and DNN, but it is imported into the script so run:
```
pip install torch
pip install alive_progress
pip install optuna
```
or use the conda equivalent.

Example usage:
```
python training.py train ../../output_VHcc_v01/Saved_columnar_arrays_ZLL/ --signal ZH_Hto2C_Zto2L_2022_preEE --background DYto2L-2Jets_MLL-50_FxFx_2022_preEE --model_type dnn
```

DNN and GNN trainings are faster on GPUs.
