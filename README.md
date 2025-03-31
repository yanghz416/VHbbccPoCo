# PocketCoffea VHbb/cc setup

1. Install Miniconda or Micromamba
   * At RWTH it should be at your `/net/scratch_cms3a/<username>` area
   * At lxplus CERN, it should be in your `/eos/user/u/username` area
2. Three way to create a dedicated environment for `PocketCoffea`:

2.1 In apptainer shell, Fllow [their installation instructions](https://github.com/De-Cristo/VHccPoCo/tree/vhbb_dev/params/skip_plot_opt_vhbb_run3) for general options.
   but when install package, use this code:
```
cd /eos/user/h/haozhong/
voms-proxy-init -voms cms -rfc --valid 168:00
apptainer shell -B /eos -B /afs -B /cvmfs/cms.cern.ch  -B /tmp -B /eos/cms/ -B /etc/sysconfig/ngbauth-submit   -B ${XDG_RUNTIME_DIR}  --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc"                /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-stable

git clone https://github.com/PocketCoffea/PocketCoffea
cd PocketCoffea

python -m venv --system-site-packages myPocket
source myPocket/bin/activate

#pip install -e .[dev]  
pip install -e .                          

#pip install lightgbm tensorflow
pip install xrootd lightgbm torch tensorflow     
pip install numpy==1.24.0
pip install setuptools==70.*

```
Next time use pocket-coffea
```
cd /eos/user/h/haozhong/
voms-proxy-init -voms cms -rfc --valid 168:00

apptainer shell -B /eos -B /afs -B /cvmfs/cms.cern.ch  -B /tmp -B /eos/cms/ -B /etc/sysconfig/ngbauth-submit   -B ${XDG_RUNTIME_DIR}  --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc"                /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-stable
cd PocketCoffea
source myPocket/bin/activate

```

2.2 In swan, Follow [their installation instructions](https://github.com/PocketCoffea/Tutorials/blob/main/Analysis_Facilities_Setup/README.md)
 If meet problem, run '''pip install''' like 2.1.


2.3  Install the packages, and compile: (Need large AFS space)
    ```
	conda create -n PocketCoffea python=3.10 -c conda-forge
	conda activate PocketCoffea
	# install PocketCoffea
	git clone git@github.com:PocketCoffea/PocketCoffea.git
	cd PocketCoffea
	pip install -e .
    ```
	Follow [their installation instructions](https://pocketcoffea.readthedocs.io/en/latest/installation.html) for other options.
	Afterwards install additional packages needed for BDT/DNN training and evaluation. Keep using conda, since using pip might
	alter the environment, leading to conflicts.
    ```
	conda install conda-forge::xrootd
	conda install conda-forge::lightgbm
	conda install conda-forge::tensorflow
    conda install pytorch::pytorch
    conda install conda-forge:alive-progress
    conda install conda-forge:optuna
    conda install conda-forge:imblearn
    ```
    For brux20 cluster at Brown, you may need `conda install conda-forge::ca-certificates`.

4. Checkout *this* repo:
    ```
	git clone 
    ```

    
5. (If your local username is different from your CERN username) Setup your CERN username variable:
    ```
    export CERN_USERNAME="YOURUSERNAME"
    ```
6. Follow [examples](https://pocketcoffea.readthedocs.io/en/latest/analysis_example.html) to create dataset input files. First activate voms proxy. Then:
    ```
	cd VHccPoCo
	mkdir datasets
	build-datasets --cfg samples_Run3.json -o -ws T2_DE_RWTH -ws T2_DE_DESY -ws T1_DE_KIT_Disk -ws T2_CH_CERN -ir
	cd ../
    ```
    Use `-p 12` with `build-datasets` to parallelizing with 12 cores.
7. Run with the `futures` executor (test before large submission):
    ```
    runner --cfg VHccPoCo/cfg_VHcc_ZLL.py -o output_VHcc_Test --executor futures -s 10 -lf 1 -lc 1
    ```
8. Run on condor with Parsl executor (only if the previous step was successeful):
    ```
    runner --cfg VHccPoCo/cfg_VHcc_ZLL.py -o output_VHcc_v01 --executor parsl-condor@RWTH -s 60
    ```
	Note: use `dask@lxplus` executor if running at CERN.

9. Make some plots:
   ```
   make-plots -inp output_VHcc_v01 -op VHccPoCo/params/plotting.yaml
   ```
   The plot parameters can be changed by editing `VHccPoCo/params/plotting.yaml`.

10. Produce shapes for limit setting with `scripts/convertToRoot.py` script.
   * See details in [scripts/README.md](scripts/README.md)
