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
files_Run3 = [
    f"{localdir}/datasets/Run3_MC_VJets.json",
    f"{localdir}/datasets/Run3_MC_OtherBkg.json",
    f"{localdir}/datasets/Run3_DATA.json",
]

parameters["proc_type"] = "ZLL"
#parameters["run_period"] = "Run2" # Run2 Or Run3

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": files_2016 + files_2017 + files_2018,
        #"jsons": files_Run3,
        
        "filter" : {
            "samples": [
                "DATA_DoubleMuon",
                "DATA_DoubleEG", # in 2016/2017
                #"DATA_EGamma",   # in 2018/2022/2023
	        "WW", "WZ", "ZZ",
                "DYJetsToLL_FxFx",
                "DYJetsToLL_MLM",
                #"TTToSemiLeptonic",
                "DYJetsToLL_MiNNLO",
                #"DYJetsToLL_MiNNLO_ZptWei",
                "TTTo2L2Nu",
            ],
            "samples_exclude" : [],
            "year": ['2017']
            #"year": ['2016_PreVFP', '2016_PostVFP','2017','2018']
            #"year": ['2022_preEE','2022_postEE','2023_preBPix','2023_postBPix']
        }
    },

    workflow = VHccBaseProcessor,

    #skim = [get_HLTsel(primaryDatasets=["SingleMuon","SingleEle"])],
    skim = [get_HLTsel(primaryDatasets=["DoubleMuon","DoubleEle"]),
            get_nObj_min(4, 18., "Jet")],

    preselections = [ll_2j],
    categories = {
        #"baseline_2L2J": [passthrough],
        "baseline_2L2J_no_ctag": [passthrough],
        #"baseline_2L2J_ctag": [passthrough],
        #"baseline_2L2J_ctag_calib": [passthrough],
        "presel_mumu_2J": [mumu_2j],
        "presel_ee_2J": [ee_2j],
        "SR_ll_2J_cJ": [ll_2j, ctag_j1],
        "SR_mumu_2J_cJ": [mumu_2j, ctag_j1],
        "SR_ee_2J_cJ": [ee_2j, ctag_j1],
    },

    weights = {
        "common": {
            "inclusive": ["signOf_genWeight","lumi","XS",
                          "pileup", #Not in 2022/2023
                          #"sf_mu_id","sf_mu_iso",
                          "sf_ele_reco","sf_ele_id",
                          #"sf_ctag", "sf_ctag_calib"
                          ],
            "bycategory" : {
                #"baseline_2L2J_ctag" : ["sf_ctag"],
                #"baseline_2L2J_ctag_calib": ["sf_ctag","sf_ctag_calib"]
            }
        },
        #"bysample": { "DYJetsToLL_MiNNLO_ZptWei": {"inclusive": ["genWeight"] } }
    },
    
    variations = {
        "weights": {
            "common": {
                "inclusive": [
                    "pileup",
                    #"sf_mu_id", "sf_mu_iso",
                    "sf_ele_reco", "sf_ele_id",
                    #"sf_ctag"
                ]
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

        **jet_hists(coll="JetsCvsL", pos=0),
	**jet_hists(coll="JetsCvsL", pos=1),

        "nJet": HistConf( [Axis(field="nJet", bins=15, start=0, stop=15, label=r"nJet direct from NanoAOD")] ),
        "dilepton_m_low": HistConf( [Axis(field="dilep_mass_low", bins=100, start=0, stop=200, label=r"$M_{\ell\ell}$ [GeV], 50 < $p_{T_{\ell\ell}}$ < 150")] ),
        "dilepton_pt_low": HistConf( [Axis(field="dilep_pt_low", bins=100, start=0, stop=200, label=r"$p_{T_{\ell\ell}}$ [GeV], 50 < $p_{T_{\ell\ell}}$ < 150")] ),
        "dilepton_dphi_low": HistConf( [Axis(field="dilep_phi_low", bins=64, start=-1, stop=3.5, label=r"$\Delta \phi_{\ell\ell}$, 50 < $p_{T_{\ell\ell}}$ < 150")] ),
        "dilepton_deta_low": HistConf( [Axis(field="dilep_eta_low", bins=100, start=-1, stop=4, label=r"$\Delta \eta_{\ell\ell}$, 50 < $p_{T_{\ell\ell}}$ < 150")] ),
        "dilepton_dijet_ratio_low": HistConf( [Axis(field="pt_ratio_low", bins=100, start=0, stop=2, label=r"$\frac{p_T(\ell\ell)}{p_T(jj)}$, 50 < $p_{T_{\ell\ell}}$ < 150")] ),
        "dilepton_dijet_dphi_low": HistConf( [Axis(field="ZH_delphi_low", bins=100, start=-4, stop=4, label=r"$\Delta \phi (\ell\ell, jj)$, 50 < $p_{T_{\ell\ell}}$ < 150")] ),
        "dilepton_l2j1_low": HistConf( [Axis(field="deltaPhi_l2_j1_low", bins=100, start=-1, stop=3.5, label=r"$\Delta \phi (\ell_2, j_1)$, 50 < $p_{T_{\ell\ell}}$ < 150")] ),
        "dilepton_l2j2_low": HistConf( [Axis(field="deltaPhi_l2_j2_low", bins=100, start=-1, stop=3.5, label=r"$\Delta \phi (\ell_2, j_2)$, 50 < $p_{T_{\ell\ell}}$ < 150")] ),
        
        "dilepton_m_high": HistConf( [Axis(field="dilep_mass_high", bins=100, start=0, stop=200, label=r"$M_{\ell\ell}$ [GeV], 150 < $p_{T_{\ell\ell}}$")] ),
        "dilepton_pt_high": HistConf( [Axis(field="dilep_pt_high", bins=100, start=0, stop=200, label=r"$p_{T_{\ell\ell}}$ [GeV], 150 < $p_{T_{\ell\ell}}$")] ),
        "dilepton_dphi_high": HistConf( [Axis(field="dilep_phi_high", bins=64, start=-1, stop=3.5, label=r"$\Delta \phi_{\ell\ell}$, 150 < $p_{T_{\ell\ell}}$")] ),
        "dilepton_deta_high": HistConf( [Axis(field="dilep_eta_high", bins=100, start=-1, stop=4, label=r"$\Delta \eta_{\ell\ell}$, 150 < $p_{T_{\ell\ell}}$")] ),
        "dilepton_dijet_ratio_high": HistConf( [Axis(field="pt_ratio_high", bins=100, start=0, stop=2, label=r"$\frac{p_T(\ell\ell)}{p_T(jj)}$, 150 < $p_{T_{\ell\ell}}$")] ),
        "dilepton_dijet_dphi_high": HistConf( [Axis(field="ZH_delphi_high", bins=100, start=-4, stop=4, label=r"$\Delta \phi (\ell\ell, jj)$, 150 < $p_{T_{\ell\ell}}$")] ),
        "dilepton_l2j1_high": HistConf( [Axis(field="deltaPhi_l2_j1_high", bins=100, start=-1, stop=3.5, label=r"$\Delta \phi (\ell_2, j_1)$, 150 < $p_{T_{\ell\ell}}$")] ),
        "dilepton_l2j2_high": HistConf( [Axis(field="deltaPhi_l2_j2_high", bins=100, start=-1, stop=3.5, label=r"$\Delta \phi (\ell_2, j_2)$, 150 < $p_{T_{\ell\ell}}$")] ),
        
        "dilep_m" : HistConf( [Axis(coll="ll", field="mass", bins=100, start=0, stop=200, label=r"$M_{\ell\ell}$ [GeV]")] ),
        "dilep_m_zoom" : HistConf( [Axis(coll="ll", field="mass", bins=40, start=70, stop=110, label=r"$M_{\ell\ell}$ [GeV]")] ),
        "dilep_dr" : HistConf( [Axis(coll="ll", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{\ell\ell}$")] ),
        "dilep_pt" : HistConf( [Axis(coll="ll", field="pt", bins=100, start=0, stop=400, label=r"$p_T{\ell\ell}$ [GeV]")] ),

        "dijet_m_low" : HistConf( [Axis(field="dijet_m_low", bins=100, start=0, stop=600, label=r"$M_{jj}$ [GeV], 50 < $p_{T_{\ell\ell}}$ < 150")] ),
        "dijet_dr_low" : HistConf( [Axis(field="dijet_deltaR_low", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$, 50 < $p_{T_{\ell\ell}}$ < 150")] ),
        "dijet_dphi_low": HistConf( [Axis(field="dijet_deltaPhi_low", bins=64, start=-1, stop=3.5, label=r"$\Delta \phi_{jj}$, 50 < $p_{T_{\ell\ell}}$ < 150")] ),
        "dijet_deta_low": HistConf( [Axis(field="dijet_deltaEta_low", bins=100, start=-1, stop=4, label=r"$\Delta \eta_{jj}$, 50 < $p_{T_{\ell\ell}}$ < 150")] ),
        "dijet_pt_low" : HistConf( [Axis(field="dijet_pt_low", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV], 50 < $p_{T_{\ell\ell}}$ < 150")] ),
        
        "dijet_m_high" : HistConf( [Axis(field="dijet_m_high", bins=100, start=0, stop=600, label=r"$M_{jj}$ [GeV], 150 < $p_{T_{\ell\ell}}$")] ),
        "dijet_dr_high" : HistConf( [Axis(field="dijet_deltaR_high", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$, 150 < $p_{T_{\ell\ell}}$")] ),
        "dijet_dphi_high": HistConf( [Axis(field="dijet_deltaPhi_high", bins=64, start=-1, stop=3.5, label=r"$\Delta \phi_{jj}$, 150 < $p_{T_{\ell\ell}}$")] ),
        "dijet_deta_high": HistConf( [Axis(field="dijet_deltaEta_high", bins=100, start=-1, stop=4, label=r"$\Delta \eta_{jj}$, 150 < $p_{T_{\ell\ell}}$")] ),
        "dijet_pt_high" : HistConf( [Axis(field="dijet_pt_high", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV], 150 < $p_{T_{\ell\ell}}$")] ),

        "dijet_csort_m" : HistConf( [Axis(coll="dijet_csort", field="mass", bins=100, start=0, stop=600, label=r"$M_{jj}$ [GeV]")] ),
        "dijet_csort_dr" : HistConf( [Axis(coll="dijet_csort", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),
        "dijet_csort_pt" : HistConf( [Axis(coll="dijet_csort", field="pt", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),


        "HT":  HistConf( [Axis(field="JetGood_Ht", bins=100, start=0, stop=700, label=r"Jet HT [GeV]")] ),
        "met_pt": HistConf( [Axis(coll="MET", field="pt", bins=50, start=0, stop=200, label=r"MET $p_T$ [GeV]")] ),
        "met_phi": HistConf( [Axis(coll="MET", field="phi", bins=64, start=-math.pi, stop=math.pi, label=r"MET $phi$")] ),

        # 2D plots
        "Njet_Ht": HistConf([ Axis(coll="events", field="nJetGood",bins=[0,2,3,4,8],
                                   type="variable",   label="N. Jets (good)"),
                              Axis(coll="events", field="JetGood_Ht",
                                   bins=[0,80,150,200,300,450,700],
                                   type="variable",
                                   label="Jets $H_T$ [GeV]")]),
        "dphi_jj_dr_jj": HistConf([ Axis(field="dijet_deltaR_low", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$, 50 < $p_{T_{\ell\ell}}$ < 150"),
                              Axis(field="dijet_deltaPhi_low", bins=64, start=-1, stop=3.5, label=r"$\Delta \phi_{jj}$, 50 < $p_{T_{\ell\ell}}$ < 150")]),
    }
)


run_options = {
    "executor"       : "parsl/condor",
    "env"            : "conda",
    "workers"        : 1,
    "scaleout"       : 10,
    "walltime"       : "00:60:00",
#    "mem_per_worker_parsl" : 2, # For Parsl
    "mem_per_worker" : "2GB", # For Dask
    "exclusive"      : False,
    "skipbadfiles"   : False,
    "chunk"          : 500000,
    "retries"        : 10,
    "treereduction"  : 20,
    "adapt"          : False,
    "requirements": (
        '( TotalCpus >= 8) &&'
        '( Machine != "lx3a44.physik.rwth-aachen.de" ) && ' 
        '( Machine != "lx3b80.physik.rwth-aachen.de" )'
    ),
    
}
