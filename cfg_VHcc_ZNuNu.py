from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import workflow
from workflow import VHccBaseProcessor

import CommonSelectors
from CommonSelectors import *

import cloudpickle
cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(CommonSelectors)

import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection.yaml",
                                                  f"{localdir}/params/triggers.yaml",
                                                  f"{localdir}/params/ctagging.yaml",
                                                  update=True)

files_2016 = [
    f"{localdir}/datasets/Run2UL2016_MC_VJets.json",
    f"{localdir}/datasets/Run2UL2016_MC_OtherBkg.json",
    f"{localdir}/datasets/Run2UL2016_DATA.json",
]
files_2017 = [
    f"{localdir}/datasets/Run2UL2017_MC_VJets.json",
    f"{localdir}/datasets/Run2UL2017_MC_OtherBkg.json",
    f"{localdir}/datasets/Run2UL2017_DATA.json",
]
files_2018 = [
    f"{localdir}/datasets/Run2UL2018_MC_VJets.json",
    f"{localdir}/datasets/Run2UL2018_MC_OtherBkg.json",
    f"{localdir}/datasets/Run2UL2018_DATA.json",
]

parameters["proc_type"] = "ZNuNu"
cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": files_2016 + files_2017 + files_2018,
        
        "filter" : {
            "samples": [
                "DATA_MET",
                "WW", "WZ", "ZZ",
                "QCD", "QCD_Mu", "QCD_EM",
                "DYJetsToLL_FxFx",
                "ZJetsToNuNu_FxFx",
                "WJetsToLNu_FxFx",
                "WJetsToQQ", "ZJetsToQQ",
                "TTToSemiLeptonic", "TTTo2L2Nu",
                "TTToHadrons"
            ],
            "samples_exclude" : [],
            "year": ['2017']
            #"year": ['2016_PreVFP', '2016_PostVFP', '2017', '2018']
        },
    },

    workflow = VHccBaseProcessor,

    skim = [get_HLTsel(primaryDatasets=["MET"]),
            get_nObj_min(2, 32., "Jet")],

    preselections = [met_2jets_0lep],
    
    categories = {
        "presel_Met_2j": [passthrough],
        "baseline_Met_2j_ptcut":  [dijet_pt_cut, jet_met_dphi_cut],
        "SR_ZNuNu_2j_cj":  [dijet_pt_cut, jet_met_dphi_cut, ctag_j1],
    },

    weights = {
        "common": {
            "inclusive": ["genWeight","lumi","XS",
                          "pileup",
                          #"sf_mu_id","sf_mu_iso",
                          #"sf_ele_reco","sf_ele_id",
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
                "inclusive": [
                    "pileup",
                    #"sf_mu_id", "sf_mu_iso",
                    #"sf_ele_reco", "sf_ele_id",
                ],
                "bycategory" : {
                }
            },
        "bysample": {
        }
        },
    },


    variables = {
        **count_hist(name="nElectronGood", coll="ElectronGood",bins=5, start=0, stop=5),
        **count_hist(name="nMuonGood", coll="MuonGood",bins=5, start=0, stop=5),
        **count_hist(name="nJets", coll="JetGood",bins=8, start=0, stop=8),
        **count_hist(name="nBJets", coll="BJetGood",bins=8, start=0, stop=8),
        **jet_hists(coll="JetGood", pos=0),
        **jet_hists(coll="JetGood", pos=1),

        **jet_hists(coll="JetsCvsL", pos=0),
        **jet_hists(coll="JetsCvsL", pos=1),

        "nJet": HistConf( [Axis(field="nJet", bins=15, start=0, stop=15, label=r"nJet direct from NanoAOD")] ),

        
        "dijet_nom_m" : HistConf( [Axis(coll="dijet", field="mass", bins=100, start=0, stop=700, label=r"$M_{jj}$ [GeV]")] ),
        "dijet_nom_dr" : HistConf( [Axis(coll="dijet", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),
        "dijet_nom_pt" : HistConf( [Axis(coll="dijet", field="pt", bins=100, start=0, stop=500, label=r"$p_T{jj}$ [GeV]")] ),
        
        "dijet_csort_m" : HistConf( [Axis(coll="dijet_csort", field="mass", bins=100, start=0, stop=700, label=r"$M_{jj}$ [GeV]")] ),
        "dijet_csort_dr" : HistConf( [Axis(coll="dijet_csort", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),
        "dijet_csort_pt" : HistConf( [Axis(coll="dijet_csort", field="pt", bins=100, start=0, stop=500, label=r"$p_T{jj}$ [GeV]")] ),
        
        "HT":  HistConf( [Axis(field="JetGood_Ht", bins=100, start=0, stop=900, label=r"Jet HT [GeV]")] ),
        "met_pt": HistConf( [Axis(coll="MET", field="pt", bins=50, start=100, stop=600, label=r"MET $p_T$ [GeV]")] ),
        "met_phi": HistConf( [Axis(coll="MET", field="phi", bins=64, start=-math.pi, stop=math.pi, label=r"MET $phi$")] ),
        
        "met_deltaPhi_j1": HistConf( [Axis(field="deltaPhi_jet1_MET", bins=64, start=0, stop=math.pi, label=r"$\Delta\phi$(MET, jet 1)")] ),
        "met_deltaPhi_j2": HistConf( [Axis(field="deltaPhi_jet2_MET", bins=64, start=0, stop=math.pi, label=r"$\Delta\phi$(MET, jet 2)")] ),
    }
)


run_options = {
    "executor"       : "parsl/condor",
    "env"            : "conda",
    "workers"        : 1,
    "scaleout"       : 10,
    "walltime"       : "00:60:00",
    "mem_per_worker" : 2, # For Parsl
    #"mem_per_worker" : "2GB", # For Dask
    "exclusive"      : False,
    "skipbadfiles"   : False,
    "chunk"          : 500000,
    "retries"        : 20,
    "treereduction"  : 20,
    "adapt"          : False,
    "requirements": (
            '( Machine != "lx3a44.physik.rwth-aachen.de")'
        ),

    }
