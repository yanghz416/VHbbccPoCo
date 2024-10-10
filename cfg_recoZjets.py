from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
from pocket_coffea.lib.weights.common.common import common_weights
import vjet_weights
from vjet_weights import *
import workflow_ZjetsReco
from workflow_ZjetsReco import ZjetsBaseProcessor

import CommonSelectors
from CommonSelectors import *

import cloudpickle
cloudpickle.register_pickle_by_value(workflow_ZjetsReco)
cloudpickle.register_pickle_by_value(CommonSelectors)
cloudpickle.register_pickle_by_value(vjet_weights)

import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/zjets_objects.yaml",
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
files_Run3 = [
    f"{localdir}/datasets/Run3_MC_VJets.json",
    f"{localdir}/datasets/Run3_MC_OtherBkg.json",
    f"{localdir}/datasets/Run3_DATA.json",
    f"{localdir}/datasets/Run3_MC_Sig.json",
]

Zcut = Cut(
    name="lalaland",
    function=diLepton,
    params={
        "pt_leading_lep": 50,
        "mll": {'low': 60, 'high': 120},
        "pt_dilep": 60
    },
)

cfg = Configurator(
    parameters = parameters,
    weights_classes = common_weights + [custom_weight_vjet],
    datasets = {
        "jsons": files_2017 + files_2018,
        #"jsons": files_Run3,

        "filter" : {
            "samples": [
                "DATA_DoubleMuon",
                "DATA_DoubleEG",
                "DATA_EGamma",
                #"DATA_SingleMuon",
                #"DATA_SingleElectron",
                "DYJetsToLL_MLM",
                "DYJetsToLL_FxFx",
                #"DYJetsToLL_MiNNLO_MuMu",
                #"DYJetsToLL_MiNNLO_EE",
                #"TTTo2L2Nu"
            ],
            "samples_exclude" : [],
            "year": ['2017','2018']
            #"year": ['2022_postEE']
        }
    },

    workflow = ZjetsBaseProcessor,

    #skim = [get_HLTsel(primaryDatasets=["SingleMuon","SingleEle"])],
    skim = [get_HLTsel(primaryDatasets=["DoubleMuon","DoubleEle"]),
            get_nObj_min(4, 18., "Jet")],
    preselections = [dilepton],
    categories = {
        "baseline_ll": [passthrough],
        "baseline_mumu": [mumu_channel],
        "baseline_ee": [ee_channel],
        "ll_1j_no_ctag_MZ": [dilepton, one_jet],
        "ll_2j": [Zcut, two_jets],
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
            "DYJetsToLL_FxFx": { "inclusive": ["weight_vjet"] }
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
        **lepton_hists(coll="LeptonGood", pos=0),
        **lepton_hists(coll="LeptonGood", pos=1),
        **count_hist(name="nElectronGood", coll="ElectronGood",bins=5, start=0, stop=5),
        **count_hist(name="nMuonGood", coll="MuonGood",bins=5, start=0, stop=5),
        **count_hist(name="nJets", coll="JetGood",bins=8, start=0, stop=8),
        #**count_hist(name="nBJets", coll="BJetGood",bins=8, start=0, stop=8),
        **jet_hists(coll="JetGood", pos=0),
        **jet_hists(coll="JetGood", pos=1),
        "dilep_m" : HistConf( [Axis(coll="ll", field="mass", bins=40, start=70, stop=110, label=r"$M_{\ell\ell}$ [GeV]")] ),
        "dilep_dr" : HistConf( [Axis(coll="ll", field="deltaR", bins=50, start=0, stop=2.5, label=r"$\Delta R_{\ell\ell}$")] ),
        "dilep_pt" : HistConf( [Axis(coll="ll", field="pt", bins=100, start=0, stop=400, label=r"$p_T{\ell\ell}$ [GeV]")] ),

        "dilep_dijet_ratio": HistConf( [Axis(field="ZH_pt_ratio", bins=100, start=0, stop=2, label=r"$\frac{p_T(jj)}{p_T(\ell\ell)}$")] ),
        "dilep_dijet_dphi": HistConf( [Axis(field="ZH_deltaPhi", bins=50, start=1.5, stop=math.pi, label=r"$\Delta \phi (\ell\ell, jj)$")] ),
        "dilep_dijet_dr": HistConf( [Axis(field="ZH_deltaR", bins=50, start=0, stop=5, label=r"$\Delta R (\ell\ell, jj)$")] ),

        "dijet_m" : HistConf( [Axis(coll="dijet", field="mass", bins=100, start=0, stop=800, label=r"$M_{jj}$ [GeV]")] ),
        "dijet_dr" : HistConf( [Axis(coll="dijet", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),
        "dijet_pt" : HistConf( [Axis(coll="dijet", field="pt", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),

        "HT":  HistConf( [Axis(field="JetGood_Ht", bins=100, start=0, stop=800, label=r"Jet HT [GeV]")] ),
        "met_pt": HistConf( [Axis(coll="MET", field="pt", bins=50, start=0, stop=200, label=r"MET $p_T$ [GeV]")] ),
        "met_phi": HistConf( [Axis(coll="MET", field="phi", bins=64, start=-math.pi, stop=math.pi, label=r"MET $phi$")] ),

        "j1_l1_dr" : HistConf( [Axis(field="j1_l1_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{j_1 l_1}$")] ),
        "j1_l2_dr" : HistConf( [Axis(field="j1_l2_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{j_1 l_2}$")] ),

    }
)
