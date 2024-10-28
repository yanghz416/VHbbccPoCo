# PocketCoffea VHbb setup
**This manual originates from the [VHccPoCo](https://github.com/cms-rwth/VHccPoCo)**

---
**Recommended**

More information can be found in the installation [instruction](https://pocketcoffea.readthedocs.io/en/latest/installation.html).

```
apptainer shell -B /afs -B /cvmfs/cms.cern.ch  -B /tmp  -B /eos/cms/  -B /etc/sysconfig/ngbauth-submit  -B ${XDG_RUNTIME_DIR}  --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc" /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-stable --no-env

```

For local environment setup:

```
python -m venv --system-site-packages myPocket
source myPocket/bin/activate
pip install -e .[dev]

pip install lightgbm tensorflow
pip install setuptools==70.*
```

For the later login:

```
# Activate the virtual environment
cd PocketCoffea
source myPocket/bin/activate
```

---
## For Analysis
1. (If your local username is different from your CERN username) Setup your CERN username variable:
    ```
    export CERN_USERNAME="YOURUSERNAME"
    ```

2. Follow [examples](https://pocketcoffea.readthedocs.io/en/latest/analysis_example.html) to create dataset input files (Add `--overwrite` if the outputs are already there.):

   ```
	cd VHbbPoCo
	mkdir datasets
	build-datasets --cfg samples_Run3.json -o -rs 'T[123]_(FR|IT|BE|CH|US)_\w+' -ir
    ```
    
3. Run with the `futures` executor (test before large submission) in the same dir where the VHbbCoPo is:
    ```
    runner --cfg VHbbPoCo/cfg_VHbb_ZLL_editing.py -o output_vhbb_zll_dev_local --executor futures -s 10 -lf 1 -lc 1 2>&1 | tee output_vhbb_local.log
    # or --test for iterative processor with ``--limit-chunks/-lc``(default:2) and ``--limit-files/-lf``(default:1)
    # pocket-coffea run --cfg example_config.py --test --process-separately  -o output_test
    ```

4. Run on condor (only if the previous step was successeful):
    ``` 
    pocket-coffea run --cfg VHbbPoCo/cfg_VHbb_ZLL_test.py --executor dask@lxplus --custom-run-options VHbbPoCo/cfg/custom_run_options_vhbb_test.yaml  -o output_vhbb_test_dask

    ```

5. Make some plots:
   ```
   cd output_VHbb_ZLL_test
   pocket-coffea make-plots -i output_all.coffea --cfg parameters_dump.yaml -o plots -op ../VHbbPoCo/cfg/plotting.yaml
   ```
---

## For developers (future merge with VHccPoCo)
**It is always better to create new files rather than modifying the existing ones and to avoid uploading too much changes in one commit to make everyone's life easier!!!**

Keep the branch `vhcc_dev` up to date with the VHcc repo (in which the `VHcc_sync_dev` branch is always up to date with the center repo)
```
cd VHbbPoCo
git switch vhcc_dev
git remote add vhcc git@github.com:De-Cristo/VHccPoCo.git
git fetch vhcc
```
You can check the remote status by:
```
$ git remote -v
origin	git@github.com:De-Cristo/VHbbPoCo.git (fetch)
origin	git@github.com:De-Cristo/VHbbPoCo.git (push)
vhcc	git@github.com:De-Cristo/VHccPoCo.git (fetch)
vhcc	git@github.com:De-Cristo/VHccPoCo.git (push)
```
Then track the upper-stream, and check
```
$ git branch --set-upstream-to=vhcc/VHcc_sync_dev vhcc_dev
$ git status
On branch vhcc_dev
Your branch is up to date with 'vhcc/VHcc_sync_dev'.

nothing to commit, working tree clean
```
Then you can load changings from branch `vhbb_run3_dev` to branch `vhcc_dev`, then:
```
git push vhcc vhcc_dev:VHcc_sync_dev
```
