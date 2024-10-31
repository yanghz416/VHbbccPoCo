#!/usr/bin/env python
import os
import uproot
from coffea.util import load, save
import click

@click.command()
@click.argument(
    'inputfile',
    required=True,
    type=str,
    nargs=1,
)


def convertCoffeaToRoot(inputfile):
    hists = load(inputfile)

    print("The structure of the COFFEA file:")
    print('\t top keys:\n', hists.keys())
    print('\n\t Variables:\n', hists['variables'].keys())
    print('\n\t Samples:\n', hists['variables']['dijet_pt'].keys())
    print('\n\t Subsamples (for DATA):\n', hists['variables']['dijet_pt']['DATA_DoubleMuon'].keys())
    print('\n\t Subsamples (for ZH_Hto2C):\n', hists['variables']['dijet_pt']['ZH_Hto2C_Zto2L'].keys())
    print('\n\t Subsamples (for DYJetsToLL_FxFx__DiJet_bx):\n', hists['variables']['dijet_pt']['DYJetsToLL_FxFx__DiJet_bx'].keys())

    print('\n\t A Histogram (for ZH_Hto2C:):\n', hists['variables']['nJet']['ZH_Hto2C_Zto2L']['ZH_Hto2C_Zto2L_2022_postEE'])
    print('\n\t Draw it: \n', hists['variables']['nJet']['ZH_Hto2C_Zto2L']['ZH_Hto2C_Zto2L_2022_postEE'][{'cat':'SR_ll_2J_cJ', 'variation': 'nominal'}])

    #print(hists['variables']['dijet_pt']['ZH_Hto2C_Zto2L'].keys())
    #print(hists['variables']['dijet_pt']['ZH_Hto2C_Zto2L'])

    Channel = '2L'
    # Here we decide which histograms are used for coffea -> root conversion
    # and, possibly, a NEW name of the category
    categ_to_var = {'SR_ll_2J_cJ': ['DNN', 'Zll_SR'],
                    'SR_mm_2J_cJ': ['DNN', 'Zmm_SR'],
                    'SR_ee_2J_cJ': ['DNN', 'Zee_SR'],
                    'CR_mm_2J_HF': ['dijet_pt', 'Zmm_CR_HF'],
                    'CR_mm_2J_LF': ['dijet_pt', 'Zmm_CR_LF'],
                    'CR_mm_2J_CC': ['dijet_pt', 'Zmm_CR_CC'],
                    'CR_mm_4J_TT': ['dijet_pt', 'Zmm_CR_TT'],
                    'CR_ee_2J_HF': ['dijet_pt', 'Zee_CR_HF'],
                    'CR_ee_2J_LF': ['dijet_pt', 'Zee_CR_LF'],
                    'CR_ee_2J_CC': ['dijet_pt', 'Zee_CR_CC'],
                    'CR_ee_4J_TT': ['dijet_pt', 'Zee_CR_TT']
                    }


    output_dict = {}
    # Which processes to consider:
    samples = hists['variables']['dijet_pt'].keys()

    map_sampleName_to_processName = {
        'DATA_EGamma': 'DATA_ee',
        'DATA_DoubleMuon': 'DATA_mm',
        'ZH_Hto2C_Zto2L': 'ZH_hcc',
        'ZH_Hto2B_Zto2L': 'ZH_hbb',
        'ggZH_Hto2C_Zto2L': 'ggZH_hcc',
        'ggZH_Hto2B_Zto2L': 'ggZH_hbb',
        'TTTo2L2Nu': 'TT',
        'DYJetsToLL_FxFx': 'Zj_ll',
        'DYJetsToLL_FxFx__DiJet_ll': 'Zj_ll',
        'DYJetsToLL_FxFx__DiJet_cx': 'Zj_cj',
        'DYJetsToLL_FxFx__DiJet_bx': 'Zj_bj',
        'WW':'WW',
        'WZ':'WZ',
        'ZZ':'ZZ',
        }

    # Which eras:
    eras = ['2022_postEE']

    for samp in samples:
        if 'DATA' in samp:
            # Do not process DATA for now. It needs to be delt separately somehow...
            continue
        if samp not in map_sampleName_to_processName.keys():
            continue
        proc = map_sampleName_to_processName[samp]
        print('Processing sample:', samp, ' assigned process:', proc)

        for cat, var_and_name in categ_to_var.items():
            print('Converting histogram for:', cat, var_and_name)
            variable = var_and_name[0]
            newCatName = var_and_name[1]
            for era in eras:
                subsamples = [sname for sname in hists['variables'][variable][samp].keys() if era in sname]
                #print("\t Subsamples:", subsamples)
                if len(subsamples)==1:
                    # Note: only nominal is done here.
                    # Need to loop over variations to get shape systematics (todo)
                    myHist = hists['variables'][variable][samp][subsamples[0]][{'cat':cat, 'variation': 'nominal'}]
                else:
                    # We need to add all the histograms for sub-samples
                    myHist = hists['variables'][variable][samp][subsamples[0]][{'cat':cat, 'variation': 'nominal'}]
                    for i in range(1,len(subsamples)):
                        hist_i = myHist = hists['variables'][variable][samp][subsamples[i]][{'cat':cat, 'variation': 'nominal'}]
                        myHist = myHist + hist_i
                output_dict[era+'_'+newCatName+'/'+proc+'_'+variable+'_nominal'] = myHist


    shapes_file = os.path.join('./', 'vhcc_shapes_'+Channel+'.root')
    with uproot.recreate(shapes_file) as root_file:
        for shape, histogram in output_dict.items():
            root_file[shape] = histogram


if __name__ == "__main__":

    print("Hello world")

    convertCoffeaToRoot()

    print("... and goodbye.")
