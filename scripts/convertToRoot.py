import os
import yaml
import argparse
import uproot
from coffea.util import load, save

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def convertCoffeaToRoot(coffea_file_name, config):
  
    inputfile = coffea_file_name
    shapes_file = config["output"]["shapes_file_name"]
    eras = config["input"]["eras"]
    print(eras)
    variation = config["input"]["variation"]
    
    example_variable = config["config"]["example_variable"]
    example_data = config["config"]["example_data"]
    example_MC = config["config"]["example_MC"]
    example_subsample = config["config"]["example_subsample"]
    example_category = config["config"]["example_category"]
    
    hists = load(inputfile)

    # information printing
    print("The structure of the COFFEA file:")
    print('\t top keys:\n', hists.keys())
    print('\n\t Variables:\n', hists["variables"].keys())
    print('\n\t Samples:\n', hists["variables"][example_variable].keys())
    print('\n\t Subsamples (for DATA):\n', hists["variables"][example_variable][example_data].keys())
    print(f'\n\t Subsamples (for {example_MC}):\n', hists["variables"][example_variable][example_MC].keys())
    if example_subsample in hists["variables"][example_variable].keys():
        print(f'\n\t Subsamples (for {example_subsample}):\n', hists["variables"][example_variable][example_subsample].keys())
    era0 = eras[0]
    print(f'\n\t A Histogram (for {example_MC}:):\n', hists["variables"][example_variable][example_MC][f'{example_MC}_{era0}'])
    print('\n\t Draw it: \n', hists["variables"][example_variable][example_MC][f'{example_MC}_{era0}'][{'cat':example_category, 'variation': variation}])

    Channel = config["input"]["channel"]
    
    # Here we decide which histograms are used for coffea -> root conversion
    # and, possibly, a NEW name of the category
    
    categ_to_var = {
        category: [details['observable'], details['mata_name']]
        for category, details in config["categories"].items()
    }

    output_dict = {}
    # Which processes to consider:
    samples = hists["variables"][example_variable].keys()

    map_sampleName_to_processName = config.get("sample_to_process_map", {})
    
    for samp in samples:
        if 'DATA' in samp:
            # Do not process DATA for now. It needs to be delt separately somehow...
            continue
        if samp not in map_sampleName_to_processName.keys():
            continue
        proc = map_sampleName_to_processName[samp]
        print('Processing sample:', samp, ' assigned process:', proc)

        for era in eras:
            for cat, var_and_name in categ_to_var.items():
                print('Converting histogram for:', cat, var_and_name)
                variable = var_and_name[0]
                newCatName = var_and_name[1]
                
                subsamples = [sname for sname in hists['variables'][variable][samp].keys() if era in sname]
                #print("\t Subsamples:", subsamples)
                if len(subsamples)==1:
                    # Note: only nominal is done here.
                    # Need to loop over variations to get shape systematics (todo)
                    myHist = hists['variables'][variable][samp][subsamples[0]][{'cat':cat, 'variation': variation}]
                else:
                    # We need to add all the histograms for sub-samples
                    myHist = hists['variables'][variable][samp][subsamples[0]][{'cat':cat, 'variation': variation}]
                    for i in range(1,len(subsamples)):
                        hist_i = myHist = hists['variables'][variable][samp][subsamples[i]][{'cat':cat, 'variation': variation}]
                        myHist = myHist + hist_i
                output_dict[era+'_'+newCatName+'/'+proc+'_'+variable+f'_{variation}'] = myHist

    with uproot.recreate(shapes_file) as root_file:
        for shape, histogram in output_dict.items():
            root_file[shape] = histogram

if __name__ == "__main__":
  
    print("Hello world")
    
    parser = argparse.ArgumentParser(description="Run the Coffea to ROOT converter with a specified config file.")
    parser.add_argument( "inputfile", type=str, default="", help="Path to the input .coffea file.")
    parser.add_argument( "-c", "--config", dest="config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    
    config_path = args.config
    config = load_config(config_path)
    coffea_file_name = args.inputfile
    convertCoffeaToRoot(coffea_file_name, config)

    print("... and goodbye.")
