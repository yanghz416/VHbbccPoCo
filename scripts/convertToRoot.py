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
    print('\n\t Subsamples (for DATA:):\n', hists['variables']['dijet_pt']['DATA_DoubleMuon'].keys())
    print('\n\t Subsamples (for ZH_Hto2C:):\n', hists['variables']['dijet_pt']['ZH_Hto2C_Zto2L'].keys())

    print('\n\t A Histogram (for ZH_Hto2C:):\n', hists['variables']['nJet']['ZH_Hto2C_Zto2L']['ZH_Hto2C_Zto2L_2022_postEE'])
    print('\n\t Draw it: \n', hists['variables']['nJet']['ZH_Hto2C_Zto2L']['ZH_Hto2C_Zto2L_2022_postEE'][{'cat':'SR_ll_2J_cJ', 'variation': 'nominal'}])

    #print(hists['variables']['dijet_pt']['ZH_Hto2C_Zto2L'].keys())
    #print(hists['variables']['dijet_pt']['ZH_Hto2C_Zto2L'])

    Channel = '2L'
    # Here we decide which histograms are used for coffea -> root conversion
    # and, possibly, a NEW name of the category
    categ_to_var = {'SR_ll_2J_cJ': ['DNN', 'Zll_SR'],
                    'SR_mumu_2J_cJ': ['DNN', 'Zmm_SR'],
                    'SR_ee_2J_cJ': ['DNN', 'Zee_SR'],
                    'CR_ll_2J_HF': ['dijet_pt', 'Zll_CR_HF'],
                    'CR_ll_2J_LF': ['dijet_pt', 'Zll_CR_LF'],
                    'CR_ll_4J_TT': ['dijet_pt', 'Zll_CR_TT']
                    }


    output_dict = {}
    # Which processes to consider:
    samples = hists['variables']['dijet_pt'].keys()

    sample_to_process = {
        'ZH_Hto2C_Zto2L': 'ZH_hcc',
        'ZH_Hto2B_Zto2L': 'ZH_hbb',
        'ggZH_Hto2C_Zto2L': 'ggZH_hcc',
        'ggZH_Hto2B_Zto2L': 'ggZH_hbb',
        'TTTo2L2Nu': 'TT',
        'DYJetsToLL_FxFx': 'Zj_ll',
        'DYJetsToLL_FxFx__DiJet_ll': 'Zj_ll',
        'DYJetsToLL_FxFx__DiJet_cx': 'Zj_cj',
        'DYJetsToLL_FxFx__DiJet_bx': 'Zj_bj'
        }
        
    # Which eras:
    eras = ['2022_postEE']
    
    for samp in samples:
        if 'DATA' in samp:
            # Do not process DATA for now. It needs to be merged first...
            continue
        if samp not in sample_to_process.keys():
            continue
        proc = sample_to_process[samp]
        print('Processing sample:', samp, ' assigned process:', proc)
        
        for cat, var_and_name in categ_to_var.items():
            print('Conversing histogram for:', cat, var_and_name)
            variable = var_and_name[0]
            newCatName = var_and_name[1]
            for era in eras:
                # Note: only nominal is done here.
                # Need to loop over variations to get shape systematics (todo)
                myHist = hists['variables'][variable][samp][samp+'_'+era][{'cat':cat, 'variation': 'nominal'}]
                #print(myHist)
                output_dict[era+'_'+newCatName+'/'+proc+'_'+variable+'_nominal'] = myHist


    shapes_file = os.path.join('./', 'vhcc_shapes_'+Channel+'.root')
    with uproot.recreate(shapes_file) as root_file:
        for shape, histogram in output_dict.items():
            root_file[shape] = histogram


if __name__ == "__main__":

    print("Hello world")

    convertCoffeaToRoot()

    print("... and goodbye.")
