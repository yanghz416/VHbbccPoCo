# PocketCoffea VHbb/cc setup

1.Three way to create a dedicated environment for `PocketCoffea`:

1.1 In apptainer shell, Fllow [their installation instructions](https://github.com/De-Cristo/VHccPoCo/tree/vhbb_dev/params/skip_plot_opt_vhbb_run3) for general options.
   But when install package, use this code:
```
cd /eos/user/h/haozhong/
voms-proxy-init -voms cms -rfc --valid 168:00
apptainer shell -B /eos -B /afs -B /cvmfs/cms.cern.ch  -B /tmp -B /eos/home-h/haozhong/PocketCoffea/pocket_coffea/parameters:/usr/local/lib/python3.11/site-packages/pocket_coffea/parameters -B /eos/cms/ -B /etc/sysconfig/ngbauth-submit   -B ${XDG_RUNTIME_DIR}  --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc"                /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-stable

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

apptainer shell -B /eos -B /afs -B /cvmfs/cms.cern.ch  -B /tmp -B /eos/home-h/haozhong/PocketCoffea/pocket_coffea:/usr/local/lib/python3.11/site-packages/pocket_coffea -B /eos/cms/ -B /etc/sysconfig/ngbauth-submit   -B ${XDG_RUNTIME_DIR}  --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc"        
        /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-stable



cd PocketCoffea
source myPocket/bin/activate

```

1.2 In swan, Follow [their installation instructions](https://github.com/PocketCoffea/Tutorials/blob/main/Analysis_Facilities_Setup/README.md)
Run '''pip install''' in swan /eos/user/N/Name same as 2.1.


1.3  Install the packages, and compile: (Need large AFS space)
1.3.1 Install Miniconda or Micromamba
   * At RWTH it should be at your `/net/scratch_cms3a/<username>` area
   * At lxplus CERN, it should be in your `/eos/user/u/username` area
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

2. Checkout *this* repo:
    ```
	git clone https://github.com/yanghz416/VHbbccPoCo.git
    ```

    
3. (If your local username is different from your CERN username) Setup your CERN username variable:
    ```
    export CERN_USERNAME="YOURUSERNAME"
    ```

  
4. Follow [examples](https://pocketcoffea.readthedocs.io/en/latest/analysis_example.html) to create dataset input files.
 First activate voms proxy. Then:
    ```
	voms-proxy-init -voms cms -rfc --valid 168:00
	 cd VHccPoCo
	 mkdir datasets
	 build-datasets --cfg samples_Run3.json -o -rs 'T[2]_(EU|CN|UK|US)_\w+' -ir 
	 #Add --overwrite if the outputs are already there.
         #Sometime you may can't build all datasets, you can change samples_Run3.json and  build part datasets each time.
    
    ```
    Use `-p 12` with `build-datasets` to parallelizing with 12 cores.
   
5. Run with the `futures` executor (test before large submission):
    ```
    runner --cfg VHccPoCo/cfg_VHbb_ZLL.py -o output_vhbb_zll_test_local_postBpix --executor futures -s 10 -lf 1 -lc 1 2>&1 | tee output_vhbb_zll_local.log  # local test
    ```
    
6. Run on condor with Parsl, Lxplus or Swan executor,(only if the previous step was successeful):
    ```
    runner --cfg VHccPoCo/cfg_VHcc_ZLL.py -o output_VHcc_v01 --executor parsl-condor@RWTH -s 60

    ####meet problem when run in executor

	runner --cfg VHccPoCo/cfg_VHbb_ZLL.py --executor dask@lxplus --custom-run-options VHccPoCo/params/skip_plot_opt_vhbb_run3/custom_run_options_vhbb_test.yaml  -o output_vhbb_zll_dev_preEE 2>&1 | tee dask_out.log 
	
	runner --cfg VHccPoCo/cfg_VHbb_ZLL.py --executor dask@swan --sched-url tls://10.100.99.171:31964 --custom-run-options VHccPoCo/params/skip_plot_opt_vhbb_run3/custom_run_options_vhbb_test.yaml  -o output_vhbb_zll_preBpix 2>&1 | tee dask_out.log
    ```
	Note: use `dask@lxplus` executor if running at CERN.

7. Make some plots:
```
#Make some plots:
cd output_vhbb_zll_dev_all
make-plots -i output_all.coffea --cfg parameters_dump.yaml -o plots -op ../VHccPoCo/params/skip_plot_opt_vhbb_run3/custom_plot_options_vhbb_ZLL.yaml
#make-plots -inp output_VHcc_v01 -op VHccPoCo/params/plotting.yaml


#make root plot (I modified some contents in the Zll_Shape.yaml file based on the error message.)
python /eos/home-h/haozhong/PocketCoffea/VHccPoCo/scripts/convertToRoot.py output_all.coffea -c /eos/home-h/haozhong/PocketCoffea/VHccPoCo/params/shapemaker_vhbb_run3/Zll_Shape.yaml

```

8. Produce shapes for limit setting with `scripts/convertToRoot.py` script.
   * See details in [scripts/README.md](scripts/README.md)
