# PocketCoffea VHcc setup

1. Install Miniconda
   1.1 At RWTH it should be at your `/net/scratch_cms3a/<username>` area
   1.2 At lxplus CERN, it should be in your `/eos/user/u/username` area
2. Create a dedicated environment for PocketCoffea, install the packages, and compile:
    ```
	conda create -n PocketCoffea python=3.10 -c conda-forge
	conda activate PocketCoffea
    pip install -e .
    ```
	Follow [their installation instructions](https://pocketcoffea.readthedocs.io/en/latest/installation.html) for other options.
3. Checkout *this* repo:
    ```
    git@github.com:cms-rwth/VHccPoCo.git
    ```
3. (If you local username is different from your CERN username) Setup your CERN username variable:
    ```
    export CERN_USERNAME="YOURUSERNAME"
    ```
5. Follow [examples](https://pocketcoffea.readthedocs.io/en/latest/analysis_example.html) to create dataset input files:
    ```
    build_datasets.py --cfg samples_Run2UL_2017.json -o -ws T2_DE_RWTH T2_DE_DESY T1_DE_KIT_Disk T2_CH_CERN
    ```
6. Run with futures (test before large submission):
    ```
    python scripts/runner.py --cfg VHccPoCo/cfg_VHcc_ZLL.py  -o output_VHcc_Test --executor futures -s 10 -lf 1
    ```
7. Run on condor with Parsl executor (if the previous step was successeful):
    ```
    python scripts/runner.py --cfg VHccPoCo/cfg_VHcc_ZLL.py  -o output_VHcc_v01 --executor parsl/condor -s 100 -ll ERROR
    ```
8. Make some plots:
   ```
   make_plots.py output_VHcc_v01
   ```
