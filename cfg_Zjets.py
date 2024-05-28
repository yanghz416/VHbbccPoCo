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
parameters["proc_type"] = "ZLL"

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": files_2017 + files_2018,

        "filter" : {
            "samples": [
                #"DATA_DoubleMuon",
                #"DATA_DoubleEG",
                #"DATA_EGamma", # for 2018
                "DATA_SingleMuon",
                "DATA_SingleElectron",
                "DYJetsToLL_MLM",
                "DYJetsToLL_FxFx",
                #"DYJetsToLL_MiNNLO_MuMu",
                #"DYJetsToLL_MiNNLO_EE",
                "TTTo2L2Nu"
            ],
            "samples_exclude" : [],
            "year": ['2017']
        }
    },

    workflow = VHccBaseProcessor,

    skim = [get_HLTsel(primaryDatasets=["SingleMuon","SingleEle"])],
    #skim = [get_HLTsel(primaryDatasets=["DoubleMuon","DoubleEle"])],

    preselections = [dilepton],
    categories = {
        "baseline": [passthrough],
        "mumu": [mumu_channel],
        "ee": [ee_channel],
        #"mumu_1j_no_ctag": [mumu_channel,  one_jet],
        "mumu_1j_no_ctag_MZ": [mumu_channel, dilepton, one_jet],
        #"ee_1j_no_ctag": [ee_channel, one_jet],
        "ee_1j_no_ctag_MZ": [ee_channel, dilepton, one_jet],
        "ll_1j_no_ctag_MZ": [dilepton, one_jet],
        #"mumu_1j_ctag_calib": [mumu_channel, one_jet],
        "mumu_2j": [mumu_2j],
        "ee_2j": [ee_2j]
    },

    weights = {
        "common": {
            "inclusive": ["genWeight","lumi","XS",
                          "pileup",
                          "sf_mu_id","sf_mu_iso",
                          "sf_ele_reco","sf_ele_id",
                          ],
            "bycategory" : {
                #"mumu_1j_ctag" : ["sf_ctag"],
                #"mumu_1j_ctag_calib": ["sf_ctag","sf_ctag_calib"]
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

        "dijet_m" : HistConf( [Axis(coll="dijet", field="mass", bins=100, start=0, stop=600, label=r"$M_{jj}$ [GeV]")] ),
        "dijet_dr" : HistConf( [Axis(coll="dijet", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),
        "dijet_pt" : HistConf( [Axis(coll="dijet", field="pt", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),
        "HT":  HistConf( [Axis(field="JetGood_Ht", bins=100, start=0, stop=500, label=r"Jet HT [GeV]")] ),
        "met_pt": HistConf( [Axis(coll="MET", field="pt", bins=50, start=0, stop=200, label=r"MET $p_T$ [GeV]")] ),
        "met_phi": HistConf( [Axis(coll="MET", field="phi", bins=64, start=-math.pi, stop=math.pi, label=r"MET $phi$")] ),

    }
)
