# VHcc selection with PocketCoffea

# PocketCoffea VHcc setup

1. Install Miniconda in your /net/scratch_cms3a area
2. Install PocketCoffea packages in a dedicated environment, then activate it
   2a. (Temporarily): you may need to checkout the modified PocketCoffea version: https://github.com/andreypz/PocketCoffea/tree/dev-andrey
   2b. recompile
    ```
    pip install -e .
    ```
1. Checkout this repo:
    ```
    git@github.com:cms-rwth/VHccPoCo.git
    ```
1. (If you local username is different from your CERN username) Setup your CERN username variable:
    ```
    export CERN_USERNAME="YOURUSERNAME"
    ```
1. Follow [examples](https://pocketcoffea.readthedocs.io/en/latest/analysis_example.html) to create dataset input files:
    ```
    build_datasets.py --cfg samples_Run2UL_2017.json -o -ws T2_DE_RWTH T2_DE_KIT T2_DE_DESY T2_BE_IIHE T2_CH_CERN
    ```
1. Run with futures (test before large submission):
    ```
    python scripts/runner.py --cfg VHccPoCo/cfg_Zjets.py  -o output_Zjets_Test --executor futures -s 10 -lf 1
    ```
1. Run on condor with Parsl executor:
    ```
    python scripts/runner.py --cfg VHccPoCo/cfg_Zjets.py  -o output_Zjets_v01 --executor parsl/condor -s 100 -ll ERROR
    ```