from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import workflow
from workflow import ZjetsBaseProcessor

import diLeptonSelection
from diLeptonSelection import *

import cloudpickle
cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(diLeptonSelection)

import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection.yaml",
                                                  f"{localdir}/params/triggers.yaml",
                                                  f"{localdir}/params/plotting.yaml",
                                                  update=True)



cfg = Configurator(
    parameters = parameters,
    datasets = {
        # 2018
        "jsons": [f"{localdir}/datasets/DATA_SingleMuon_Test.json",
                  #f"{localdir}/datasets/DATA_SingleMuon.json",
                  f"{localdir}/datasets/DATA_DoubleMuon.json",
                  f"{localdir}/datasets/DATA_EGamma.json",
                  f"{localdir}/datasets/DYJetsToLL_MLM_Test.json",
                  f"{localdir}/datasets/DYJetsToLL_FxFx_Test.json",
                  #f"{localdir}/datasets/DYJetsToLL_MLM.json",
                  #f"{localdir}/datasets/DYJetsToLL_FxFx.json",
                  f"{localdir}/datasets/DYJetsToLL_MiNNLO_MuMu.json",
                  f"{localdir}/datasets/DYJetsToLL_MiNNLO_EE.json",
                  f"{localdir}/datasets/DYJetsToLL_MiNNLO_ZptWei_MuMu.json",
                  f"{localdir}/datasets/DYJetsToLL_MiNNLO_ZptWei_EE.json",
              ],

        # 2017:
        #"jsons": [f"{localdir}/datasets/DATA_SingleMuon_2017.json",
        #          f"{localdir}/datasets/DATA_DoubleMuon_2017.json",
        #          f"{localdir}/datasets/DATA_SingleElectron_2017.json",
        #          f"{localdir}/datasets/DATA_DoubleEG_2017.json",
        #          f"{localdir}/datasets/DYJetsToLL_MLM_2017.json",
        #          f"{localdir}/datasets/DYJetsToLL_FxFx_2017.json",
        #          f"{localdir}/datasets/DYJetsToLL_MiNNLO_MuMu_2017.json",
        #          f"{localdir}/datasets/DYJetsToLL_MiNNLO_EE_2017.json",
        #          f"{localdir}/datasets/DYJetsToLL_MiNNLO_ZptWei_MuMu_2017.json",
        #          f"{localdir}/datasets/DYJetsToLL_MiNNLO_ZptWei_EE_2017.json",
        #      ],
        "filter" : {
            "samples": [
                #"DATA_DoubleMuon",
                #"DATA_DoubleEG", 
                #"DATA_EGamma", # for 2018
                "DATA_SingleMuon",
                "DYJetsToLL_MLM",
                #"DYJetsToLL_FxFx",
                #"DYJetsToLL_MiNNLO_MuMu",
                #"DYJetsToLL_MiNNLO_EE",
            ],
            "samples_exclude" : [],
            "year": ['2018']
        }
    },

    workflow = ZjetsBaseProcessor,
    
    skim = [get_HLTsel(primaryDatasets=["SingleMuon","SingleEle"])], 
    #skim = [get_HLTsel(primaryDatasets=["DoubleMuon","DoubleEle"])], 
    
    preselections = [dilepton_presel],
    categories = {
        "baseline": [passthrough],
        "mumu": [mumu_channel],
        "ee": [ee_channel],
        "mumu_2j": [mumu_2j_channel],
        "ee_2j": [ee_2j_channel],
    },

    weights = {
        "common": {
            "inclusive": ["genWeight","lumi","XS",
                          "pileup",
                          "sf_mu_id","sf_mu_iso",
                          "sf_ele_reco","sf_ele_id",
                          ],
            "bycategory" : {
            }
        },
        "bysample": {
        }
    },

    variations = {
        "weights": {
            "common": {
                "inclusive": [  "pileup",
                                "sf_mu_id", "sf_mu_iso",
                                "sf_ele_reco", "sf_ele_id",
                              ],
                "bycategory" : {
                }
            },
        "bysample": {
        }    
        },
    },
    
    
    variables = {
        **lepton_hists(coll="LeptonGood", pos=0),
        **lepton_hists(coll="LeptonGood", pos=1),
        **count_hist(name="nElectronGood", coll="ElectronGood",bins=5, start=0, stop=5),
        **count_hist(name="nMuonGood", coll="MuonGood",bins=5, start=0, stop=5),
        **count_hist(name="nJets", coll="JetGood",bins=8, start=0, stop=8),
        **count_hist(name="nBJets", coll="BJetGood",bins=8, start=0, stop=8),
        **jet_hists(coll="JetGood", pos=0),
        **jet_hists(coll="JetGood", pos=1),
        "dilep_m" : HistConf( [Axis(coll="ll", field="mass", bins=100, start=0, stop=200, label=r"$M_{\ell\ell}$ [GeV]")] ),
        "dilep_m_zoom" : HistConf( [Axis(coll="ll", field="mass", bins=40, start=70, stop=110, label=r"$M_{\ell\ell}$ [GeV]")] ),
        "dilep_dr" : HistConf( [Axis(coll="ll", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{\ell\ell}$")] ),
        "dilep_pt" : HistConf( [Axis(coll="ll", field="pt", bins=100, start=0, stop=400, label=r"$p_T{\ell\ell}$ [GeV]")] ),

        "dijet_m" : HistConf( [Axis(coll="dijet", field="mass", bins=100, start=0, stop=200, label=r"$M_{jj}$ [GeV]")] ),
        "dijet_dr" : HistConf( [Axis(coll="dijet", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),
        "dijet_pt" : HistConf( [Axis(coll="dijet", field="pt", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),
        "HT":  HistConf( [Axis(field="JetGood_Ht", bins=100, start=0, stop=400, label=r"Jet HT [GeV]")] ),


    }
)


run_options = {
    "executor"       : "parsl/condor",
    "env"            : "conda",
    "workers"        : 1,
    "scaleout"       : 50,
    "worker_image"   : "NA",
    "queue"          : "microcentury",
    "walltime"       : "00:60:00",
    "mem_per_worker" : 2, # GB
    "disk_per_worker" : "1GB", # GB
    "exclusive"      : False,
    "skipbadfiles"   : False,
    "chunk"          : 400000,
    "retries"        : 20,
    "treereduction"  : 20,
    "adapt"          : False,
    "requirements": (
            '( Machine != "lx1b02.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a03.physik.rwth-aachen.de") && '
            '( Machine != "lx3a05.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a06.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a09.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a13.physik.rwth-aachen.de") && '
            '( Machine != "lx3a14.physik.rwth-aachen.de") && '
            '( Machine != "lx3a15.physik.rwth-aachen.de") && '
            '( Machine != "lx3a23.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a25.physik.rwth-aachen.de") && '
            '( Machine != "lx3a27.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a46.physik.rwth-aachen.de") && '
            '( Machine != "lx3a44.physik.rwth-aachen.de") && '
            '( Machine != "lx3a47.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a55.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a56.physik.rwth-aachen.de") && '
            '( Machine != "lx3b08.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b09.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b13.physik.rwth-aachen.de") && '
            '( Machine != "lx3b18.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b24.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b29.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b32.physik.rwth-aachen.de") && '
            '( Machine != "lx3b33.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b34.physik.rwth-aachen.de") && '
            '( Machine != "lx3b41.physik.rwth-aachen.de") && '
            '( Machine != "lx3b46.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b47.physik.rwth-aachen.de") && '
            '( Machine != "lx3b48.physik.rwth-aachen.de") && '
            '( Machine != "lx3b49.physik.rwth-aachen.de") && '
            '( Machine != "lx3b52.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b55.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b57.physik.rwth-aachen.de") && '
            '( Machine != "lx3b62.physik.rwth-aachen.de") && '
            '( Machine != "lx3b66.physik.rwth-aachen.de") && '
            '( Machine != "lx3b68.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b69.physik.rwth-aachen.de") && '
            '( Machine != "lx3b70.physik.rwth-aachen.de") && '
            '( Machine != "lx3b71.physik.rwth-aachen.de") && '
            '( Machine != "lx3b99.physik.rwth-aachen.de") && '
            '( Machine != "lxblade01.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade02.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade03.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade04.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade05.physik.rwth-aachen.de") && '
            '( Machine != "lxblade06.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade07.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade08.physik.rwth-aachen.de") && '
            '( Machine != "lxblade09.physik.rwth-aachen.de") && '
            '( Machine != "lxblade10.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade11.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade12.physik.rwth-aachen.de") && '
            '( Machine != "lxblade13.physik.rwth-aachen.de") && '
            '( Machine != "lxblade14.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade15.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade16.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade17.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade18.physik.rwth-aachen.de") && '
            '( Machine != "lxblade19.physik.rwth-aachen.de") && '
            '( Machine != "lxblade20.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade21.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade22.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade23.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade24.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade25.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade26.physik.rwth-aachen.de") && '
            '( Machine != "lxblade27.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade28.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade29.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade30.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade31.physik.rwth-aachen.de") && '
            '( Machine != "lxblade32.physik.rwth-aachen.de") && '
            '( Machine != "lxcip01.physik.rwth-aachen.de") && '
            '( Machine != "lxcip02.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip05.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip06.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip09.physik.rwth-aachen.de") && '
            '( Machine != "lxcip10.physik.rwth-aachen.de") && '
            '( Machine != "lxcip11.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip12.physik.rwth-aachen.de") && '
            '( Machine != "lxcip14.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip15.physik.rwth-aachen.de") && '
            '( Machine != "lxcip16.physik.rwth-aachen.de") && '
            '( Machine != "lxcip17.physik.rwth-aachen.de") && '
            '( Machine != "lxcip18.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip19.physik.rwth-aachen.de") && '
            '( Machine != "lxcip24.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip25.physik.rwth-aachen.de") && '
            '( Machine != "lxcip26.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip27.physik.rwth-aachen.de") && '
            '( Machine != "lxcip28.physik.rwth-aachen.de") && '
            '( Machine != "lxcip29.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip30.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip31.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip32.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip34.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip35.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip50.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip51.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip52.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip53.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip54.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip55.physik.rwth-aachen.de") && '
            '( Machine != "lxcip56.physik.rwth-aachen.de") && '
            '( Machine != "lxcip57.physik.rwth-aachen.de") && '
            '( Machine != "lxcip58.physik.rwth-aachen.de") && '
            '( Machine != "lxcip59.physik.rwth-aachen.de")'            
        ),

    }
   
