'''This script is used to extract the ctagSF calibration (normalization corrections)
The script works on histograms with (sample,category) axes and it expect the following
categories to be present:
- <catname>_no_ctag: nominal shapes without Ctag SFs
- <catname>_ctag:  ctagSFs applied
- <catname>_ctag_calib: with additional norm calibration applied

N.B. No ctag/brag requirements should be applied to extract the shapes used for this calibration.
The 2D variables used to extract the SF can be speficied in the options. By default (NJets,HT) distribution is used.

The SFs are exported in correctionlib json format. 
'''

import numpy as np
import awkward as ak
import hist
from itertools import *
from coffea.util import load
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep

hep.style.use(hep.style.ROOT)

#from pocket_coffea.utils.plot_utils import plot_shapes_comparison
from plot_shapes_comparison import plot_shapes_comparison

import argparse

parser = argparse.ArgumentParser(description="Extract btagSF calibration and validate it with plots")
parser.add_argument("-i","--input", type=str, required=True, help="Input coffea files with shapes")
parser.add_argument("-o","--output", type=str, required=True, help="Output folder")
parser.add_argument("-v","--validate", action="store_true", help="Use this switch to plot validation shapes")
parser.add_argument("-c","--compute", action="store_true", help="Use this switch to compute the SF")
parser.add_argument("--sf-hist", type=str, help="Histogram to be used for SF computation", default="Njet_Ht")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
output = load(args.input)["variables"]

variables_to_plot = [
    #'jets_Ht',"jet_pt","jet_eta","nJets", "nBJets", "jet_btagDeepFlavB"
    'HT', "nJets", "nBJets",
#    "jet_pt_1", "jet_eta_1", "jet_btagDeepFlavB_1",
#    "jet_pt_2", "jet_eta_2", "jet_btagDeepFlavB_2",
#    "jet_pt_3", "jet_eta_3", "jet_btagDeepFlavB_3",
#    "jet_pt_4", "jet_eta_4", "jet_btagDeepFlavB_4",
]
catname = 'baseline_2L2J' # in ZLL
#catname = 'baseline_1L2J' # in WLNu
#catname = 'presel_Met_2J' # in ZNuNu
            

samples = list(output["nJets"].keys())
#print(samples)
#print(list(output["nJets"][samples[0]]))
#print( output["nJets"][samples[0]] )
#years = list(output["nJets"][samples[0]].axes[2])

if args.compute:
    # Plot only shapes with and without btagSF and compute the SF
    #for var, sample, year in product(variables_to_plot, samples, years):
    #    print(var, sample)
    #    shapes = [
    #        (sample, 'no_btagSF', year, "nominal", "no btag SF"),
    #        (sample, 'btagSF', year,"nominal", "btag SF"),
    #    ]
    #    #plot_shapes_comparison(output, f"{var}", shapes, ylog=True,
    #    #                       lumi_label=f"{sample} {year}",
    #    #                       outputfile=f"{args.output}/hist_btagSFeffect_{year}_{var}_{sample}.*")

    # Compute the SF in one go
    ratios = [ ]
    sample_names = []
    for sample in samples:
        for subsample in output[f"{args.sf_hist}"][sample].keys():
            print ("Subsample = ", subsample)

            print("Computing SF for sample: ", sample, subsample)

            A = output[f"{args.sf_hist}"][sample][subsample]
            #print("A = ", A)
            
            w_num, _, x = A[catname+'_no_ctag','nominal',:,:].to_numpy()
            num_var = A[catname+'_no_ctag','nominal',:,:].variances()
            
            w_denom, _, x = A[catname+'_ctag', 'nominal',:,:].to_numpy()
            denom_var = A[catname+'_ctag', 'nominal',:,:].variances()
            
            ratio= np.where( (w_denom>0)&(w_num>0), w_num/w_denom, 1.) 
            ratio_err =  np.where( (w_denom>0)&(w_num>0),
                                   np.sqrt((1/w_denom)**2 * num_var + (w_num/w_denom**2)**2 * denom_var),
                                   0.)
            ratios.append((ratio, ratio_err))
            sample_names.append(subsample)
            
    sample_axis = hist.axis.StrCategory(sample_names, name="sample_year", label="Sample")
    sfhist = hist.Hist(sample_axis,A.axes[2],A.axes[3], data=np.stack([r[0] for r in ratios]))
    sfhist_err = hist.Hist(sample_axis,A.axes[2],A.axes[3], data=np.stack([r[1] for r in ratios]))

    # Exporting it to correctionlib
    import correctionlib, rich
    import correctionlib.convert
    # without a name, the resulting object will fail validation
    sfhist.name = "ctagSF_norm_correction"
    sfhist.label = "out"
    clibcorr = correctionlib.convert.from_histogram(sfhist)
    clibcorr.description = "SF to correct the overall normalization after the application of ctagSF weights"
    
    # set overflow bins behavior (default is to raise an error when out of bounds)
    
    for sample_cat in clibcorr.data.content:
        sample_cat.value.flow = "clamp"
        #print(sample_cat)

    cset = correctionlib.schemav2.CorrectionSet(
        schema_version=2,
        description="ctagSF normalization corrections",
        corrections=[clibcorr],
    )
    rich.print(cset)

    with open(f"{args.output}/ctagSF_calibrationSF.json", "w") as fout:
        import json
        from json import encoder
        #encoder.FLOAT_REPR = lambda o: format(o, '.4f')
        #print(cset.json(exclude_unset=True))
        ugly_json = json.loads(cset.json(exclude_unset=True), parse_float=lambda x: round(float(x), 4))
        pretty_json = json.dumps(ugly_json, indent=4)
        #print("\n Pretty Json \n:", pretty_json)
        fout.write(pretty_json)


    #Plotting the scale factor for each sample/year
    for sample in sample_names:
        print(f"Plotting the SF for {sample}")
        fig,( ax,ay) = plt.subplots(1, 2, figsize=(18, 7), dpi=100)
        plt.subplots_adjust(wspace=0.3)
        
        ax.set_title(f"{sample}")
        I = hep.hist2dplot(sfhist[sample, :,:], ax=ax, cmap="cividis", cbarextend=True)
        ax.set_xlabel("N jets")
        ax.set_ylabel("Jet $H_T$")

        ay.set_title("stat. error")
        I = hep.hist2dplot(sfhist_err[sample, :,:], ax=ay, cmap="cividis", cbarextend=True)
        ay.set_xlabel("N jets")
        ay.set_ylabel("Jet $H_T$")

        fig.savefig(f"{args.output}/plot_SFoutput_{sample}.png")


######################################################
if args.validate:
    # Plot the shape with validation

    for var, sample in product(variables_to_plot, samples):
        for subsample in output[f"{args.sf_hist}"][sample].keys():
            print(f"Plotting validation for {var} {sample} {subsample}")
            shapes = [
                (sample, subsample, catname+'_no_ctag', "nominal", "no ctag SF"),
                (sample, subsample, catname+'_ctag', "nominal", "ctag SF"),
                (sample, subsample, catname+'_ctag_calib', "nominal", "ctag SF calibrated"),
            ]
            plot_shapes_comparison(output, f"{var}", shapes, ylog=True,
                                   lumi_label=f"{subsample}",
                                   outputfile=f"{args.output}/hist_btagSFcalibrated_{var}_{subsample}.*")
            
