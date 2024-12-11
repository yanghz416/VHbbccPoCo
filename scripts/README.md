## Scripts

1. `convertToRoot.py` - a script to convert .coffea output inot the ROOT histograms for input to Combine. Work in progress...
Usage:
```
python VHccPoCo/scripts/convertToRoot.py  path/to/output_all.coffea -c path/to/params/shape_config.yaml
```
To perform the bin merging with this add the follow sections to the yaml config file:

bin_merging:
  2022_preEE_Zll_SR: 
    categories: ["2022_preEE_Zll_SR"]
    signal_processes: ["ggZH_hcc", "ZH_hcc"]
    target_uncertainty: 0.3
    target_significance_loss: 0.005
    minimum_signal: 0
    epsilon: 0.05
    
Here, bin_merging is a required section label. The labels at the level of 2022_preEE_Zll_SR define a specific instance of the bin merging. The value of the lable isn't used anywhere, it just defines a region. For each instance of the bin merging the arguments are as follows:
   - categories: list of all SRs to consider for the merging. This can be one or more but must be a list.
   - signal_processes: list of all processes to use for signal. This can be one or more but must be a list.
   - target_uncertainty: (optional, defaults to 0.3), targetUncert in variableBinning.py
   - target_significance_loss: (optional, defaults to 0.005), sigLoss in variableBinning.py
   - minimum_signal: (optional, defaults to 0), minimumSignal in variableBinning.py
   - epsilon: (optional, defaults to 0.05), epsilon in variableBinning.py

2. `compute_ctagSF_calibration.py` - a script to compute normalisation corrections after c-tagger shape correction scale factors.
   -  `plot_shapes_comparison.py` - a script to make plots and validate the above corrections.
3. `variableBinning.py` - a script to determine optimal way to combine bins from an existing histogram. This runs with the name of a root file, an output root file, a list of regions to consider simulatneously, and a list of signal processes. It rebins the histograms in the list of regions and outputs a new root file. Currently it doesn't use command line arguments, so main() needs to be modified. Ideally this will be adjusted in the future. There are also a few tunable paramters:
   - targetUncert: maximum value for bkg bin uncertainty over bkg bin counts,
                   default is 0.3
   - sigLoss: amount the significance can drop with each bin merging, 
              default is 1%
   - minimumSignal: minimum number of signal events in the last bin, 
                    default is 0
   - epsilon: how much the monotonically decreasing nature can be violated
              from the left, default 5%
   0 doPlot: whether to show the final plots, default false
4. `variableBinningParquet.py` - a script to determine optimal way to combine bins from parquet files with the scores and weights. This currently needs to be adjusted through main(), but this will be updated in the future ideally. This has some tunable paramters:
   - minBinSize: starting target size for the number of signal events 
                 in each bin, default is 0.01
   - maxBinSize: maximum number of signal events in a bin prior to 
                 merging, default is 10
   - binSizeSearchIncrement: how much to increment the bin size in each
                             iteration, default is 0.01
   - sigCut: amount of significance which can be lost with each merge,
             default is 0.01
   - uncertCut: maximum stat uncertainty per initial bin, default is 0.3
   - doPlot: whether or not to show plots, default False
