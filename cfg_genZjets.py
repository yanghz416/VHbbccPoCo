from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
from pocket_coffea.lib.weights.common.common import common_weights
import vjet_weights
from vjet_weights import *
import workflow_ZjetsGen
from workflow_ZjetsGen import ZjetsBaseProcessor

import GenSelectors
from GenSelectors import *

import cloudpickle
cloudpickle.register_pickle_by_value(workflow_ZjetsGen)
cloudpickle.register_pickle_by_value(GenSelectors)
cloudpickle.register_pickle_by_value(vjet_weights)

import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/zjets_objects.yaml",
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

cfg = Configurator(
    parameters = parameters,
    weights_classes = common_weights + [custom_weight_vjet],
    datasets = {
        #"jsons": files_2017 + files_2018,
        "jsons": files_Run3,

        "filter" : {
            "samples": [
                "DYJetsToLL_MLM",
                "DYJetsToLL_FxFx",
                #"DYJetsToLL_MiNNLO_MuMu",
                #"DYJetsToLL_MiNNLO_EE",
            ],
            "samples_exclude" : [],
            "year": ['2022_postEE']
        }
    },

    workflow = ZjetsBaseProcessor,

    skim = [get_nObj_min(4, 18., "GenJet")],
    preselections = [get_nObj_min(2, 18., "MyGenLeptons")],
    categories = {
        "baseline_2l": [passthrough],
        "ll_2j": [get_nObj_min(2, 25., "MyGenJets")],
        "ZplusDijet_": [GenZplus2j],
        "ZplusDijet_Nj1": [GenZplus2j, LHE_Nj(1)],
        "ZplusDijet_Nj2": [GenZplus2j, LHE_Nj(2)],
        "ZplusDijet_Nj3": [GenZplus2j, LHE_Nj(3)],
        "ZplusDijet_m0": [GenZplus2j, dijet_m0to60],
        "ZplusDijet_m1": [GenZplus2j, dijet_m60to120],
        "ZplusDijet_m2": [GenZplus2j, dijet_m120to2000],
        #"baseline_mumu": [dijet_0to60],
        #"baseline_ee": [ee_channel],
    },

    weights = {
        "common": {
            "inclusive": ["genWeight","lumi","XS",
                          ],
            "bycategory" : {
            }
        },
        "bysample": {
            "DYJetsToLL_FxFx": {
                "inclusive": ["weight_vjet"]
            }
        }
    },

    variations = {"weights": { "common": { "inclusive": [],},  },},

    variables = {
        **lepton_hists(coll="MyGenLeptons", pos=0),
        **genjet_hists(coll="MyGenJets", pos=0),
        **count_hist(name="nMyGenJets", coll="MyGenJets",bins=8, start=0, stop=8),

        "dilep_m" : HistConf( [Axis(coll="ll", field="mass", bins=40, start=70, stop=110, label=r"$M_{\ell\ell}$ [GeV]")] ),
        "dilep_dr" : HistConf( [Axis(coll="ll", field="deltaR", bins=50, start=0, stop=4.0, label=r"$\Delta R_{\ell\ell}$")] ),
        "dilep_pt" : HistConf( [Axis(coll="ll", field="pt", bins=100, start=0, stop=400, label=r"$p_T{\ell\ell}$ [GeV]")] ),

        "dijet_m" : HistConf( [Axis(coll="dijet", field="mass", bins=100, start=0, stop=800, label=r"$M_{jj}$ [GeV]")],
                              exclude_categories = ['baseline_2l']),
        "dijet_dr" : HistConf( [Axis(coll="dijet", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")],
                               exclude_categories = ['baseline_2l']),
        "dijet_pt" : HistConf( [Axis(coll="dijet", field="pt", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")],
                               exclude_categories = ['baseline_2l']),

        #"dilep_dijet_ratio": HistConf( [Axis(field="ZH_pt_ratio", bins=100, start=0, stop=2, label=r"$\frac{p_T(jj)}{p_T(\ell\ell)}$")] ),
        #"dilep_dijet_dphi": HistConf( [Axis(field="ZH_deltaPhi", bins=50, start=1.5, stop=math.pi, label=r"$\Delta \phi (\ell\ell, jj)$")] ),
        #"dilep_dijet_dr": HistConf( [Axis(field="ZH_deltaR", bins=50, start=0, stop=5, label=r"$\Delta R (\ell\ell, jj)$")] ),

        "LHE_HT":  HistConf( [Axis(coll="LHE", field="HT", bins=100, start=0, stop=1000, label=r"LHE HT [GeV]")] ),
        "LHE_VPT":  HistConf( [Axis(coll="LHE", field="Vpt", bins=100, start=0, stop=400, label=r"LHE Vpt [GeV]")] ),
        "LHE_Njets":  HistConf( [Axis(coll="LHE", field="Njets", bins=6, start=0, stop=6, label=r"LHE Njets")] ),
        "LHE_NpNLO":  HistConf( [Axis(coll="LHE", field="NpNLO", bins=6, start=0, stop=6, label=r"LHE NpNLO")] ),

        #"j1_l1_dr" : HistConf( [Axis(field="j1_l1_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{j_1 l_1}$")] ),
        #"j1_l2_dr" : HistConf( [Axis(field="j1_l2_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{j_1 l_2}$")] ),

    }
)
