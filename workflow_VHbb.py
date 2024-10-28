import awkward as ak
import numpy as np
import uproot
import pandas as pd
import math
import warnings
import os
import lightgbm as lgb
import tensorflow as tf
import gc
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.models import load_model
import CommonSelectors
from CommonSelectors import *
from pocket_coffea.utils.utils import dump_ak_array

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.hist_manager import Axis
from pocket_coffea.lib.deltaR_matching import delta_phi
from pocket_coffea.lib.objects import (
    jet_correction,
    lepton_selection,
    jet_selection,
    btagging,
    CvsLsorted,
    get_dilepton,
    get_dijet
)
import awkward as ak
import numpy as np
  
# def phi_mpi_pi(angle):
#     """Returns the phi angle in the interval (-PI, PI]."""
    
#     if -math.pi < angle <= math.pi:
#         return angle
    
#     if angle > 0:
#         n = int((angle + math.pi) / (2 * math.pi))
#         angle -= 2 * math.pi * n
#     else:
#         n = int(-(angle - math.pi) / (2 * math.pi))
#         angle += 2 * math.pi * n

#     return angle
  
# def delta_R(float eta_1, float eta_2, float phi_1, float phi_2){
#     deta = eta_1 - eta_2
#     dphi = phi_mpi_pi(phi_1 - phi_2)
#     dRsq = pow(deta,2) + pow(dphi,2)
#     return math.sqrt(dRsq)
# }

def get_nu_4momentum(Lepton, MET):
    mW = 80.38

    # Convert pt, eta, phi, m to px, py, pz, E
    px = Lepton.pt * np.cos(Lepton.phi)
    py = Lepton.pt * np.sin(Lepton.phi)
    pz = Lepton.pt * np.sinh(Lepton.eta)
    E = np.sqrt(Lepton.mass**2 + Lepton.pt**2 * np.cosh(Lepton.eta)**2)
    
    MET_px = MET.pt * np.cos(MET.phi)
    MET_py = MET.pt * np.sin(MET.phi)
    
    MisET2 = (MET_px**2 + MET_py**2)
    mu = (mW**2) / 2 + MET_px * px + MET_py * py
    a = (mu * pz) / (E**2 - pz**2)
    a2 = a**2
    b = ((E**2) * (MisET2) - mu**2) / (E**2 - pz**2)
    
    condition = a2 - b >= 0

    # Vectorized handling of conditions
    root = np.sqrt(ak.where(condition, a2 - b, ak.zeros_like(a2)))
    pz1 = a + root
    pz2 = a - root
    pznu = ak.where(np.abs(pz1) < np.abs(pz2), pz1, pz2)
    Enu = np.sqrt(MisET2 + pznu**2)

    # Handle cases where condition is False using your fallback logic
    # Adapted to take into account the real parts of the roots if discriminant is negative
    real_part = ak.where(condition, ak.zeros_like(a), a)  # Use 'a' as the real part when condition is False
    pznu = ak.where(condition, pznu, real_part)  # Update pznu to use real_part when condition is False
    Enu = np.sqrt(MisET2 + pznu**2)  # Recalculate Enu with the updated pznu

    p4nu_rec = ak.Array([MET_px, MET_py, pznu, Enu])
    pt = np.sqrt(MET_px**2 + MET_py**2)
    phi = np.arctan2(MET_py, MET_px)
    theta = np.arctan2(pt, pznu)
    eta = -np.log(np.tan(theta / 2))
    m = np.sqrt(np.maximum(Enu**2 - (MET_px**2 + MET_py**2 + pznu**2), 0))

    return ak.zip({"pt": pt, "eta": eta, "phi": phi, "mass": m},with_name="PtEtaPhiMCandidate")

def BvsLsorted(jets, tagger):
    if tagger == "PNet":
        btag = "btagPNetB"
    elif tagger == "DeepFlav":
        btag = "btagDeepFlavB"
    elif tagger == "RobustParT":
        btag = "btagRobustParTAK4B"
    else:
        raise NotImplementedError(f"This tagger is not implemented: {tagger}")
    
    return jets[ak.argsort(jets[btag], axis=1, ascending=False)]

  
def get_dibjet(jets, tagger = 'PNet'):
    
    fields = {
        "pt": 0.,
        "eta": 0.,
        "phi": 0.,
        "mass": 0.,
    }

    jets = ak.pad_none(jets, 2)
    njet = ak.num(jets[~ak.is_none(jets, axis=1)])
    
    dijet = jets[:, 0] + jets[:, 1]

    for var in fields.keys():
        fields[var] = ak.where(
            (njet >= 2),
            getattr(dijet, var),
            fields[var]
        )

    fields["deltaR"] = ak.where( (njet >= 2), jets[:,0].delta_r(jets[:,1]), -1)
    fields["deltaPhi"] = ak.where( (njet >= 2), abs(jets[:,0].delta_phi(jets[:,1])), -1)
    fields["deltaEta"] = ak.where( (njet >= 2), abs(jets[:,0].eta - jets[:,1].eta), -1)
    fields["j1Phi"] = ak.where( (njet >= 2), jets[:,0].phi, -1)
    fields["j2Phi"] = ak.where( (njet >= 2), jets[:,1].phi, -1)
    fields["j1pt"] = ak.where( (njet >= 2), jets[:,0].pt, -1)
    fields["j2pt"] = ak.where( (njet >= 2), jets[:,1].pt, -1)
    fields["j1mass"] = ak.where( (njet >= 2), jets[:,0].mass, -1)
    fields["j2mass"] = ak.where( (njet >= 2), jets[:,1].mass, -1)
    
    if tagger == "PNet":
        BvL = "btagPNetB"
        CvL = "btagPNetCvL"
        CvB = "btagPNetCvB"
    elif tagger == "DeepFlav":
        BvL = "btagDeepFlavB"
        CvL = "btagDeepFlavCvL"
        CvB = "btagDeepFlavCvB"
    elif tagger == "RobustParT":
        BvL = "btagRobustParTAK4B"
        CvL = "btagRobustParTAK4CvL"
        CvB = "btagRobustParTAK4CvB"
    else:
        raise NotImplementedError(f"This tagger is not implemented: {tagger}")

    if tagger:
        fields["j1BvsL"] = ak.where( (njet >= 2), jets[:,0][BvL], -1)
        fields["j2BvsL"] = ak.where( (njet >= 2), jets[:,1][BvL], -1)
        fields["j1CvsL"] = ak.where( (njet >= 2), jets[:,0][CvL], -1)
        fields["j2CvsL"] = ak.where( (njet >= 2), jets[:,1][CvL], -1)
        fields["j1CvsB"] = ak.where( (njet >= 2), jets[:,0][CvB], -1)
        fields["j2CvsB"] = ak.where( (njet >= 2), jets[:,1][CvB], -1)
    
    dibjet = ak.zip(fields, with_name="PtEtaPhiMCandidate")
    return dibjet
  
def get_additionalleptons(electrons, muons, transverse=False):

    if muons is None and electrons is None:
        raise("Must specify either muon or electron collection in get_dilepton() function")
    elif muons is None and electrons is not None:
        leptons = ak.pad_none(ak.with_name(electrons, "PtEtaPhiMCandidate"), 2)
    elif electrons is None and muons is not None:
        leptons = ak.pad_none(ak.with_name(muons, "PtEtaPhiMCandidate"), 2)
    else:
        leptons = ak.pad_none(ak.with_name(ak.concatenate([ muons[:, 0:2], electrons[:, 0:2]], axis=1), "PtEtaPhiMCandidate"), 2)
        
    nlep = ak.num(leptons[~ak.is_none(leptons, axis=1)])
    NAL = nlep - 2
    return NAL
  
class VHbbBaseProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

        self.proc_type   = self.params["proc_type"]
        self.save_arrays = self.params["save_arrays"]
        self.run_dnn     = self.params["run_dnn"]
        #self.bdt_model = lgb.Booster(model_file=self.params.LightGBM_model)
        #self.bdt_low_model = lgb.Booster(model_file=self.params.LigtGBM_low)
        #self.bdt_high_model = lgb.Booster(model_file=self.params.LigtGBM_high)
        #self.dnn_model = load_model(self.params.DNN_model)
        #self.dnn_low_model = load_model(self.params.DNN_low)
        #self.dnn_high_model = load_model(self.params.DNN_high)
    
        # Define the prediction functions with @tf.function
        #self.predict_dnn = tf.function(self.model.predict, reduce_retracing=True)
        #self.predict_dnn_low = tf.function(self.model_low.predict, reduce_retracing=True)
        #self.predict_dnn_high = tf.function(self.model_high.predict, reduce_retracing=True)

        print("Processor initialized")
        
    def apply_object_preselection(self, variation):
        '''
        
        '''
        # Include the supercluster pseudorapidity variable
        electron_etaSC = self.events.Electron.eta + self.events.Electron.deltaEtaSC
        self.events["Electron"] = ak.with_field(
            self.events.Electron, electron_etaSC, "etaSC"
        )
        # Build masks for selection of muons, electrons, jets, fatjets
        self.events["MuonGood"] = lepton_selection(
            self.events, "Muon", self.params
        )
        self.events["ElectronGood"] = lepton_selection(
            self.events, "Electron", self.params
        )
        leptons = ak.with_name(
            ak.concatenate((self.events.MuonGood, self.events.ElectronGood), axis=1),
            name='PtEtaPhiMCandidate',
        )
        self.events["LeptonGood"] = leptons[ak.argsort(leptons.pt, ascending=False)]

        self.events["ll"] = get_dilepton(
            self.events.ElectronGood, self.events.MuonGood
        )
        
        self.events["JetGood"], self.jetGoodMask = jet_selection(
            self.events, "Jet", self.params, self._year, "LeptonGood"
        )
        
        self.events['EventNr'] = self.events.event
        
        self.events["BJetGood"] = btagging(
            self.events["JetGood"], self.params.btagging.working_point[self._year], wp=self.params.object_preselection.bJetWP)
            
    def count_objects(self, variation):
        self.events["nMuonGood"] = ak.num(self.events.MuonGood)
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood)
        self.events["nLeptonGood"] = ak.num(self.events.LeptonGood)

        self.events["nJet"] = ak.num(self.events.Jet)
        self.events["nJetGood"] = ak.num(self.events.JetGood)
        self.events["NaJ"] = self.events.nJetGood - 2  # additional jets
        self.events["nBJetGood"] = ak.num(self.events.BJetGood)
        # self.events["nfatjet"]   = ak.num(self.events.FatJetGood)


    def evaluateBDT(self, data):
        print(self.params.Models.BDT[self.channel][self.events.metadata["year"]].model_file)
        print()
        # Read the model file
        model = lgb.Booster(model_file=self.params.Models.BDT[self.channel][self.events.metadata["year"]].model_file)
        #bdt_score = self.bdt_model.predict(data)
        bdt_score = model.predict(data)
        # Release memory
        del model
        gc.collect()

        return bdt_score
    
    def evaluateseparateBDTs(self, data):
        data_df_low = data[data['dilep_pt'] < 150]
        data_df_high = data[data['dilep_pt'] >= 150]
        
        # Initialize empty arrays for scores
        bdt_score_low = np.array([])
        bdt_score_high = np.array([])
        
        # Read the model files
        model_low = lgb.Booster(model_file=self.params.Models.BDT[f'{self.channel}_low'][self.events.metadata["year"]].model_file)
        model_high = lgb.Booster(model_file=self.params.Models.BDT[f'{self.channel}_high'][self.events.metadata["year"]].model_file)
        
        # Predict only if data_df_low is non-empty
        if not data_df_low.empty:
            #bdt_score_low = self.bdt_low_model.predict(data_df_low)
            bdt_score_low = model_low.predict(data_df_low)
        
        # Predict only if data_df_high is non-empty
        if not data_df_high.empty:
            #bdt_score_high = self.bdt_high_model.predict(data_df_high)
            bdt_score_high = model_high.predict(data_df_high)
        
        # Concatenate the scores from low and high dataframes
        bdt_score = np.concatenate((bdt_score_low, bdt_score_high), axis=0)
        
        # Release memory
        del model_low, model_high
        gc.collect()

        return bdt_score
    
    def evaluateDNN(self, data):
        
        print("Evaluating DNN...")
        print(self.params.Models.DNN[self.channel][self.events.metadata["year"]].model_file)
        print()
        # Load the model on demand
        with tf.device('/CPU:0'):  # Use CPU to avoid GPU memory issues
            model = load_model(self.params.Models.DNN[self.channel][self.events.metadata["year"]].model_file)
            dnn_score = model.predict(data, batch_size=32).ravel()
        # Release memory
        tf.keras.backend.clear_session()
        del model
        gc.collect()
        print("DNN evaluation completed.")
        return dnn_score
    
    def evaluateseparateDNNs(self, data):
        
        data_df_low = data[data['dilep_pt'] < 150]
        data_df_high = data[data['dilep_pt'] >= 150]
        
        # Initialize empty arrays for scores
        dnn_score_low = np.array([])
        dnn_score_high = np.array([])
        
        # Read the model file
        model_low = load_model(self.params.Models.DNN[f'{self.channel}_low'][self.events.metadata["year"]].model_file)
        model_high = load_model(self.params.Models.DNN[f'{self.channel}_high'][self.events.metadata["year"]].model_file)
        
        # Predict only if data_df_low is non-empty
        if not data_df_low.empty:
            print("Predicting for low dilep_pt...")
            with tf.device('/CPU:0'):  # Use CPU to avoid GPU memory issues
                model_low = load_model(self.params.DNN_low)
                dnn_score_low = model_low.predict(data_df_low, batch_size=32).ravel()
            tf.keras.backend.clear_session()
            del model_low
            gc.collect()
            print("Prediction for low dilep_pt completed.")
        
        # Predict only if data_df_high is non-empty
        if not data_df_high.empty:
            print("Predicting for high dilep_pt...")
            with tf.device('/CPU:0'):  # Use CPU to avoid GPU memory issues
                model_high = load_model(self.params.DNN_high)
                dnn_score_high = model_high.predict(data_df_high, batch_size=32).ravel()
            tf.keras.backend.clear_session()
            del model_high
            gc.collect()
            print("Prediction for high dilep_pt completed.")
        
        dnn_score = np.concatenate((dnn_score_low, dnn_score_high), axis=0)
        
        print("Separate DNN evaluation completed.")
        
        return dnn_score
    
    # Function that defines common variables employed in analyses and save them as attributes of `events`
    def define_common_variables_before_presel(self, variation):
        self.events["JetGood_Ht"] = ak.sum(abs(self.events.JetGood.pt), axis=1)
        
        self.myJetTagger = self.params.ctagging[self._year]["tagger"]
        print("After presel len(events_nr)", len(self.events.EventNr), self.events.EventNr)
        print()
        
        self.events["dijet"] = get_dijet(self.events.JetGood)
        # self.events["JetsCvsL"] = CvsLsorted(self.events["JetGood"], tagger = self.myJetTagger)
        self.events["JetsCvsL"] = CvsLsorted(self.events["JetGood"],
                                             tagger = self.params.object_preselection.bJet_algorithm)
        # self.events["dijet_csort"] = get_dijet(self.events.JetsCvsL, tagger = self.myJetTagger)
        self.events["dijet_csort"] = get_dibjet(self.events.JetsCvsL, 
                                                tagger = self.params.object_preselection.bJet_algorithm)
        
        self.events["JetsBvsL"] = BvsLsorted(self.events["JetGood"], self.params.object_preselection.bJet_algorithm)
        self.events["dijet_bsort"] = get_dibjet(self.events.JetsBvsL, self.params.object_preselection.bJet_algorithm)
        
        self.events["MET_used"] = ak.zip({
                                        "pt": self.events.MET.pt,
                                        "eta": ak.zeros_like(self.events.MET.pt),
                                        "phi": self.events.MET.phi,
                                        "mass": ak.zeros_like(self.events.MET.pt),
                                        "charge": ak.zeros_like(self.events.MET.pt),
                                        },with_name="PtEtaPhiMCandidate")
        
        self.events["pt_miss"] = self.events.MET_used.pt
        
        self.events["NaL"] = get_additionalleptons(
              self.events.ElectronGood, self.events.MuonGood
            ) # number of additional leptons


    def define_common_variables_after_presel(self, variation):
        
        odd_event_mask = (self.events.EventNr % 2 == 1)
        
        self.events["nGenPart"] = ak.num(self.events.GenPart)
        self.events["GenPart_eta"] = self.events.GenPart.eta
        self.events["GenPart_genPartIdxMother"] = self.events.GenPart.genPartIdxMother
        self.events["GenPart_mass"] = self.events.GenPart.mass
        self.events["GenPart_pdgId"] = self.events.GenPart.pdgId
        self.events["GenPart_phi"] = self.events.GenPart.phi
        self.events["GenPart_pt"] = self.events.GenPart.pt
        self.events["GenPart_status"] = self.events.GenPart.status
        self.events["GenPart_statusFlags"] = self.events.GenPart.statusFlags
        
        self.events["LHE_AlphaS"] = self.events.LHE.AlphaS
        self.events["LHE_HT"] = self.events.LHE.HT
        self.events["LHE_HTIncoming"] = self.events.LHE.HTIncoming
        self.events["LHE_Nb"] = self.events.LHE.Nb
        self.events["LHE_Nc"] = self.events.LHE.Nc
        self.events["LHE_Nglu"] = self.events.LHE.Nglu
        self.events["LHE_Njets"] = self.events.LHE.Njets
        self.events["LHE_NpLO"] = self.events.LHE.NpLO
        self.events["LHE_NpNLO"] = self.events.LHE.NpNLO
        self.events["LHE_Nuds"] = self.events.LHE.Nuds
        self.events["LHE_Vpt"] = self.events.LHE.Vpt

        self.events["LHEPart_eta"] = self.events.LHEPart.eta
        self.events["LHEPart_incomingpz"] = self.events.LHEPart.incomingpz
        self.events["LHEPart_mass"] = self.events.LHEPart.mass
        self.events["LHEPart_pdgId"] = self.events.LHEPart.pdgId
        self.events["LHEPart_phi"] = self.events.LHEPart.phi
        self.events["LHEPart_pt"] = self.events.LHEPart.pt
        self.events["LHEPart_spin"] = self.events.LHEPart.spin
        self.events["LHEPart_status"] = self.events.LHEPart.status
        self.events["nLHEPart"] = ak.num(self.events.LHEPart)
                
        if self.proc_type=="ZLL":

            ### General
            self.events["dijet_m"] = self.events.dijet_csort.mass
            self.events["dijet_pt"] = self.events.dijet_csort.pt
            self.events["dijet_dr"] = self.events.dijet_csort.deltaR
            self.events["dijet_deltaPhi"] = self.events.dijet_csort.deltaPhi
            self.events["dijet_deltaEta"] = self.events.dijet_csort.deltaEta
            self.events["dijet_BvsL_max"] = self.events.dijet_csort.j1BvsL
            self.events["dijet_BvsL_min"] = self.events.dijet_csort.j2BvsL
            self.events["dijet_CvsL_max"] = self.events.dijet_csort.j1CvsL
            self.events["dijet_CvsL_min"] = self.events.dijet_csort.j2CvsL
            self.events["dijet_CvsB_max"] = self.events.dijet_csort.j1CvsB
            self.events["dijet_CvsB_min"] = self.events.dijet_csort.j2CvsB
            self.events["dijet_pt_max"] = self.events.dijet_csort.j1pt
            self.events["dijet_pt_min"] = self.events.dijet_csort.j2pt
            
            self.events["dibjet_m"] = self.events.dijet_bsort.mass
            self.events["dibjet_pt"] = self.events.dijet_bsort.pt
            self.events["dibjet_eta"] = self.events.dijet_bsort.eta
            self.events["dibjet_phi"] = self.events.dijet_bsort.phi
            self.events["dibjet_dr"] = self.events.dijet_bsort.deltaR
            self.events["dibjet_deltaPhi"] = self.events.dijet_bsort.deltaPhi
            self.events["dibjet_deltaEta"] = self.events.dijet_bsort.deltaEta
            self.events["dibjet_BvsL_max"] = self.events.dijet_bsort.j1BvsL
            self.events["dibjet_BvsL_min"] = self.events.dijet_bsort.j2BvsL
            self.events["dibjet_CvsL_max"] = self.events.dijet_bsort.j1CvsL
            self.events["dibjet_CvsL_min"] = self.events.dijet_bsort.j2CvsL
            self.events["dibjet_CvsB_max"] = self.events.dijet_bsort.j1CvsB
            self.events["dibjet_CvsB_min"] = self.events.dijet_bsort.j2CvsB
            self.events["dibjet_pt_max"] = self.events.dijet_bsort.j1pt
            self.events["dibjet_pt_min"] = self.events.dijet_bsort.j2pt
            self.events["dibjet_mass_max"] = self.events.dijet_bsort.j1mass
            self.events["dibjet_mass_min"] = self.events.dijet_bsort.j2mass

            self.events["dilep_m"] = self.events.ll.mass
            self.events["dilep_pt"] = self.events.ll.pt
            self.events["dilep_eta"] = self.events.ll.eta
            self.events["dilep_phi"] = self.events.ll.phi
            self.events["dilep_dr"] = self.events.ll.deltaR
            self.events["dilep_deltaPhi"] = self.events.ll.deltaPhi
            self.events["dilep_deltaEta"] = self.events.ll.deltaEta
            
            self.events["ZH_pt_ratio"] = self.events.dijet_pt/self.events.dilep_pt
            self.events["ZH_deltaPhi"] = np.abs(self.events.ll.delta_phi(self.events.dijet_csort))
            
            self.events["ZHbb_pt_ratio"] = self.events.dibjet_pt/self.events.dilep_pt
            self.events["VHbb_pt_ratio"] = self.events.ZHbb_pt_ratio
            
            self.events["ZHbb_deltaPhi"] = np.abs(self.events.ll.delta_phi(self.events.dijet_bsort))
            self.events["VHbb_deltaPhi"] = self.events.ZHbb_deltaPhi
            
            self.events["ZHbb_deltaR"] = np.abs(self.events.ll.delta_r(self.events.dijet_bsort))
            self.events["VHbb_deltaR"] = self.events.ZHbb_deltaR
            
            # Create a record of variables to be dumped as root/parquete file:
            variables_to_process = ak.zip({
                "dilep_m": self.events["dilep_m"],
                "dilep_pt": self.events["dilep_pt"],
                "dilep_dr": self.events["dilep_dr"],
                "dilep_deltaPhi": self.events["dilep_deltaPhi"],
                "dilep_deltaEta": self.events["dilep_deltaEta"],
                
                "dijet_m": self.events["dijet_m"],
                "dijet_pt": self.events["dijet_pt"],
                "dijet_dr": self.events["dijet_dr"],
                "dijet_deltaPhi": self.events["dijet_deltaPhi"],
                "dijet_deltaEta": self.events["dijet_deltaEta"],
                "dijet_CvsL_max": self.events["dijet_CvsL_max"],
                "dijet_CvsL_min": self.events["dijet_CvsL_min"],
                "dijet_CvsB_max": self.events["dijet_CvsB_max"],
                "dijet_CvsB_min": self.events["dijet_CvsB_min"],
                "dijet_pt_max": self.events["dijet_pt_max"],
                "dijet_pt_min": self.events["dijet_pt_min"],
                
                "ZH_pt_ratio": self.events["ZH_pt_ratio"],
                "ZH_deltaPhi": self.events["ZH_deltaPhi"],
                # "deltaPhi_l2_j1": self.events["deltaPhi_l2_j1"]
            })
            
            df = ak.to_pandas(variables_to_process)
            # columns_to_exclude = ['dilep_m']
            # df = df.drop(columns=columns_to_exclude, errors='ignore')
#             print("variables selected", df.columns)
#             if not self.params.separate_models: 
#                 self.events["BDT"] = self.evaluateBDT(df)
#                 self.events["DNN"] = self.evaluateDNN(df)
#             else:
#                 self.events["BDT"] = self.evaluateseparateBDTs(df)
#                 self.events["DNN"] = self.evaluateseparateDNNs(df)
            
            mask = (
              (self.events.nJetGood >= 2) 
              & (self.events.nLeptonGood>=2) 
              & (self.events.dilep_m > 75) 
              & (self.events.dilep_m < 105) 
              & (self.events.dibjet_pt > 50) 
              & (self.events.VHbb_deltaPhi > 2.5) 
              & (self.events.VHbb_deltaR < 3.6)
              & (self.events.pt_miss < 60)
              & (self.events.dibjet_deltaEta < 1.0)
              & (self.events.NaL == 0)
              & (self.events.dibjet_pt_max >= 30)
              & (self.events.dibjet_pt_min >= 30)
              & ( (self.events.dibjet_mass_max >= 5) & (self.events.dibjet_mass_max <= 30) )
              & ( (self.events.dibjet_mass_min >= 5) & (self.events.dibjet_mass_min <= 30) )
              & ( (self.events.VHbb_pt_ratio >= 0.5) & (self.events.VHbb_pt_ratio <= 2) )
            )
    
            selection_ZLL = ak.where(ak.is_none(mask), False, mask)

                
        if self.proc_type=="WLNu":
            self.events["lead_lep"] = ak.firsts(self.events.LeptonGood)
            self.events["W_candidate"] = self.events.lead_lep + self.events.MET_used
            #print("W_candidate", self.events.W_candidate, self.events.W_candidate.mass, self.events.W_candidate.pt)
            self.events["W_m"] = self.events.W_candidate.mass
            self.events["W_pt"] = self.events.W_candidate.pt
            self.events["W_mt"] = np.sqrt(2*self.events.lead_lep.pt*self.events.MET_used.pt*(1-np.cos(self.events.lead_lep.delta_phi(self.events.MET_used))))
            self.events["pt_miss"] = self.events.MET_used.pt
            # Step 1: Calculate delta_r for each b_jet with respect to lead_lep
            delta_rs = self.events.BJetGood.delta_r(self.events.lead_lep)

            # Step 2: Find the index of the b_jet with the minimum delta_r
            min_delta_r_index = ak.argmin(delta_rs, axis=1, keepdims=True)

            # Step 3: Select the b_jet with the minimum delta_r
            self.events["b_jet"] = self.events.BJetGood[min_delta_r_index]
            
            self.events["dijet_m"] = self.events.dijet_csort.mass
            self.events["dijet_pt"] = self.events.dijet_csort.pt
            self.events["dijet_dr"] = self.events.dijet_csort.deltaR
            self.events["dijet_deltaPhi"] = self.events.dijet_csort.deltaPhi
            self.events["dijet_deltaEta"] = self.events.dijet_csort.deltaEta
            self.events["dijet_CvsL_max"] = self.events.dijet_csort.j1CvsL
            self.events["dijet_CvsL_min"] = self.events.dijet_csort.j2CvsL
            self.events["dijet_CvsB_max"] = self.events.dijet_csort.j1CvsB
            self.events["dijet_CvsB_min"] = self.events.dijet_csort.j2CvsB
            self.events["dijet_pt_max"] = self.events.dijet_csort.j1pt
            self.events["dijet_pt_min"] = self.events.dijet_csort.j2pt
            
            self.events["deltaPhi_jet1_MET"] = np.abs(self.events.MET.delta_phi(self.events.JetGood[:,0]))
            self.events["deltaPhi_jet2_MET"] = np.abs(self.events.MET.delta_phi(self.events.JetGood[:,1]))
        
            self.events["WH_deltaPhi"] = np.abs(self.events.W_candidate.delta_phi(self.events.dijet_csort))
            self.events["deltaPhi_l1_j1"] = np.abs(delta_phi(self.events.lead_lep.phi, self.events.dijet_csort.j1Phi))
            self.events["deltaPhi_l1_MET"] = np.abs(delta_phi(self.events.lead_lep.phi, self.events.MET_used.phi))
            self.events["deltaPhi_l1_b"] = np.abs(delta_phi(self.events.lead_lep.phi, self.events.b_jet.phi))
            self.events["deltaEta_l1_b"] = np.abs(self.events.lead_lep.eta - self.events.b_jet.eta)
            self.events["deltaR_l1_b"] = np.sqrt((self.events.lead_lep.eta - self.events.b_jet.eta)**2 + (self.events.lead_lep.phi - self.events.b_jet.phi)**2)
            self.events["b_CvsL"] = self.events.b_jet.btagDeepFlavCvL
            self.events["b_CvsB"] = self.events.b_jet.btagDeepFlavCvB
            self.events["b_Btag"] = self.events.b_jet.btagDeepFlavB
            self.events["neutrino_from_W"] = get_nu_4momentum(self.events.lead_lep, self.events.MET_used)
            self.events["top_candidate"] = self.events.lead_lep + self.events.b_jet + self.events.neutrino_from_W
            #print("top_candidate", self.events.top_candidate, self.events.top_candidate.mass, self.events.top_candidate.pt)

            self.events["top_mass"] = (self.events.lead_lep + self.events.b_jet + self.events.neutrino_from_W).mass
            variables_to_process = ak.zip({
                "dijet_m": self.events["dijet_m"],
                "dijet_pt": self.events["dijet_pt"],
                "dijet_dr": self.events["dijet_dr"],
                "dijet_deltaPhi": self.events["dijet_deltaPhi"],
                "dijet_deltaEta": self.events["dijet_deltaEta"],
                "dijet_CvsL_max": self.events["dijet_CvsL_max"],
                "dijet_CvsL_min": self.events["dijet_CvsL_min"],
                "dijet_CvsB_max": self.events["dijet_CvsB_max"],
                "dijet_CvsB_min": self.events["dijet_CvsB_min"],
                "dijet_pt_max": self.events["dijet_pt_max"],
                "dijet_pt_min": self.events["dijet_pt_min"],
                "W_mt": self.events["W_mt"],
                "W_pt": self.events["W_pt"],
                "pt_miss": self.events["pt_miss"],
                "WH_deltaPhi": self.events["WH_deltaPhi"],
                "deltaPhi_l1_j1": self.events["deltaPhi_l1_j1"],
                "deltaPhi_l1_MET": self.events["deltaPhi_l1_MET"],
                "deltaPhi_l1_b": self.events["deltaPhi_l1_b"],
                "deltaEta_l1_b": self.events["deltaEta_l1_b"],
                "deltaR_l1_b": self.events["deltaR_l1_b"],
                "b_CvsL": self.events["b_CvsL"],
                "b_CvsB": self.events["b_CvsB"],
                "b_Btag": self.events["b_Btag"],
                "top_mass": self.events["top_mass"]})
            df = ak.to_pandas(variables_to_process)
            #columns_to_exclude = []
            #df = df.drop(columns=columns_to_exclude, errors='ignore')
            self.events["BDT"] = self.evaluateBDT(df)
            self.events["DNN"] = self.evaluateDNN(df)
            mask = (self.events.nJetGood >= 2) & (self.events.W_pt > 100) & ((self.events.nJetGood >= 2) & (self.events.dijet_csort.mass > 75) & (self.events.dijet_csort.mass < 200)) & ((self.events.JetsCvsL.btagDeepFlavCvL[:,0]>0.2) & (self.events.JetsCvsL.btagDeepFlavCvB[:,0]>0.4))
            selection_WLNu = ak.where(ak.is_none(mask), False, mask)

        #print("Pt sort pt:", self.events["JetGood"][self.events["nJetGood"]>=3].pt)
        #print("CvsL sort pt:", self.events["JetsCvsL"][self.events["nJetGood"]>=3].pt)

        #print("Pt sort CvsL:", self.events["JetGood"][self.events["nJetGood"]>=3].btagDeepFlavCvL)
        #print("CvsL sort CvsL:", self.events["JetsCvsL"][self.events["nJetGood"]>=3].btagDeepFlavCvL)

        if self.proc_type=="ZNuNu":
            ### General
            self.events["Z_candidate"] = self.events.MET_used
            self.events["Z_pt"] = self.events.Z_candidate.pt
            
            self.events["dijet_m"] = self.events.dijet_csort.mass
            self.events["dijet_pt"] = self.events.dijet_csort.pt
            self.events["dijet_dr"] = self.events.dijet_csort.deltaR
            self.events["dijet_deltaPhi"] = self.events.dijet_csort.deltaPhi
            self.events["dijet_deltaEta"] = self.events.dijet_csort.deltaEta
            self.events["dijet_CvsL_max"] = self.events.dijet_csort.j1CvsL
            self.events["dijet_CvsL_min"] = self.events.dijet_csort.j2CvsL
            self.events["dijet_CvsB_max"] = self.events.dijet_csort.j1CvsB
            self.events["dijet_CvsB_min"] = self.events.dijet_csort.j2CvsB
            self.events["dijet_pt_max"] = self.events.dijet_csort.j1pt
            self.events["dijet_pt_min"] = self.events.dijet_csort.j2pt
            
            self.events["ZH_pt_ratio"] = self.events.dijet_csort.pt/self.events.Z_candidate.pt
            self.events["ZH_deltaPhi"] = np.abs(self.events.Z_candidate.delta_phi(self.events.dijet_csort))
            self.events["deltaPhi_jet1_MET"] = np.abs(self.events.MET.delta_phi(self.events.JetGood[:,0]))
            self.events["deltaPhi_jet2_MET"] = np.abs(self.events.MET.delta_phi(self.events.JetGood[:,1]))
            
            # Create a record of variables to be dumped as root/parquete file:
            variables_to_process = ak.zip({
                "dijet_m": self.events["dijet_m"],
                "dijet_pt": self.events["dijet_pt"],
                "dijet_dr": self.events["dijet_dr"],
                "dijet_deltaPhi": self.events["dijet_deltaPhi"],
                "dijet_deltaEta": self.events["dijet_deltaEta"],
                "dijet_CvsL_max": self.events["dijet_CvsL_max"],
                "dijet_CvsL_min": self.events["dijet_CvsL_min"],
                "dijet_CvsB_max": self.events["dijet_CvsB_max"],
                "dijet_CvsB_min": self.events["dijet_CvsB_min"],
                "dijet_pt_max": self.events["dijet_pt_max"],
                "dijet_pt_min": self.events["dijet_pt_min"],
                "ZH_pt_ratio": self.events["ZH_pt_ratio"],
                "ZH_deltaPhi": self.events["ZH_deltaPhi"],
                "Z_pt": self.events["Z_pt"]
            })
            
            df = ak.to_pandas(variables_to_process)
            #columns_to_exclude = []
            #df = df.drop(columns=columns_to_exclude, errors='ignore')
            self.events["BDT"] = self.evaluateBDT(df)
            self.events["DNN"] = self.evaluateDNN(df)
            mask = ((self.events.nJetGood >= 2) & (self.events.dijet_csort.pt > 120)) &  ( (self.events.deltaPhi_jet1_MET > 0.6) & (self.events.deltaPhi_jet2_MET > 0.6)) & ((self.events.JetsCvsL.btagDeepFlavCvL[:,0]>0.2) & (self.events.JetsCvsL.btagDeepFlavCvB[:,0]>0.4)) & ((self.events.nJetGood >= 2) & (self.events.dijet_csort.mass > 75) & (self.events.dijet_csort.mass < 200))
            selection_ZNuNu = ak.where(ak.is_none(mask), False, mask)


        if self.save_arrays:
            if self.proc_type=="WLNu":
                SR_Data = self.events[selection_WLNu]
                variables_to_save = ak.zip({
                    "dijet_m": SR_Data["dijet_m"],
                    "dijet_pt": SR_Data["dijet_pt"],
                    "dijet_dr": SR_Data["dijet_dr"],
                    "dijet_deltaPhi": SR_Data["dijet_deltaPhi"],
                    "dijet_deltaEta": SR_Data["dijet_deltaEta"],
                    "dijet_CvsL_max": SR_Data["dijet_CvsL_max"],
                    "dijet_CvsL_min": SR_Data["dijet_CvsL_min"],
                    "dijet_CvsB_max": SR_Data["dijet_CvsB_max"],
                    "dijet_CvsB_min": SR_Data["dijet_CvsB_min"],
                    "dijet_pt_max": SR_Data["dijet_pt_max"],
                    "dijet_pt_min": SR_Data["dijet_pt_min"],
                    "W_mt": SR_Data["W_mt"],
                    "W_pt": SR_Data["W_pt"],
                    "pt_miss": SR_Data["pt_miss"],
                    "WH_deltaPhi": SR_Data["WH_deltaPhi"],
                    "deltaPhi_l1_j1": SR_Data["deltaPhi_l1_j1"],
                    "deltaPhi_l1_MET": SR_Data["deltaPhi_l1_MET"],
                    "deltaPhi_l1_b": SR_Data["deltaPhi_l1_b"],
                    "deltaEta_l1_b": SR_Data["deltaEta_l1_b"],
                    "deltaR_l1_b": SR_Data["deltaR_l1_b"],
                    "b_CvsL": SR_Data["b_CvsL"],
                    "b_CvsB": SR_Data["b_CvsB"],
                    "b_Btag": SR_Data["b_Btag"],
                    "top_mass": SR_Data["top_mass"]})
                
            if self.proc_type=="ZLL":
                SR_Data = self.events[selection_ZLL]
                variables_to_save = ak.zip({
                    "nGenPart": SR_Data["nGenPart"],
#                     "GenPart_eta": SR_Data["GenPart_eta"],
#                     "GenPart_genPartIdxMother": SR_Data["GenPart_genPartIdxMother"],
#                     "GenPart_mass": SR_Data["GenPart_mass"],
#                     "GenPart_pdgId": SR_Data["GenPart_pdgId"],
#                     "GenPart_phi": SR_Data["GenPart_phi"],
#                     "GenPart_pt": SR_Data["GenPart_pt"],
#                     "GenPart_status": SR_Data["GenPart_status"],
#                     "GenPart_statusFlags": SR_Data["GenPart_statusFlags"],

#                     "LHE_AlphaS": SR_Data["LHE_AlphaS"],
#                     "LHE_HT": SR_Data["LHE_HT"],
#                     "LHE_HTIncoming": SR_Data["LHE_HTIncoming"],
#                     "LHE_Nb": SR_Data["LHE_Nb"],
#                     "LHE_Nc": SR_Data["LHE_Nc"],
#                     "LHE_Nglu": SR_Data["LHE_Nglu"],
#                     "LHE_Njets": SR_Data["LHE_Njets"],
#                     "LHE_NpLO": SR_Data["LHE_NpLO"],
#                     "LHE_NpNLO": SR_Data["LHE_NpNLO"],
#                     "LHE_Nuds": SR_Data["LHE_Nuds"],
#                     "LHE_Vpt": SR_Data["LHE_Vpt"],

#                     "LHEPart_eta": SR_Data["LHEPart_eta"],
#                     "LHEPart_incomingpz": SR_Data["LHEPart_incomingpz"],
#                     "LHEPart_mass": SR_Data["LHEPart_mass"],
#                     "LHEPart_pdgId": SR_Data["LHEPart_pdgId"],
#                     "LHEPart_phi": SR_Data["LHEPart_phi"],
#                     "LHEPart_pt": SR_Data["LHEPart_pt"],
#                     "LHEPart_spin": SR_Data["LHEPart_spin"],
#                     "LHEPart_status": SR_Data["LHEPart_status"],
#                     "nLHEPart": SR_Data["nLHEPart"],

                    "dijet_m": SR_Data["dijet_m"],
                    "dijet_pt": SR_Data["dijet_pt"],
                    "dijet_dr": SR_Data["dijet_dr"],
                    "dijet_deltaPhi": SR_Data["dijet_deltaPhi"],
                    "dijet_deltaEta": SR_Data["dijet_deltaEta"],
                    "dijet_BvsL_max": SR_Data["dijet_BvsL_max"],
                    "dijet_BvsL_min": SR_Data["dijet_BvsL_min"],
                    "dijet_CvsL_max": SR_Data["dijet_CvsL_max"],
                    "dijet_CvsL_min": SR_Data["dijet_CvsL_min"],
                    "dijet_CvsB_max": SR_Data["dijet_CvsB_max"],
                    "dijet_CvsB_min": SR_Data["dijet_CvsB_min"],
                    "dijet_pt_max": SR_Data["dijet_pt_max"],
                    "dijet_pt_min": SR_Data["dijet_pt_min"],

                    "dibjet_m": SR_Data["dibjet_m"],
                    "dibjet_pt": SR_Data["dibjet_pt"],
                    "dibjet_eta": SR_Data["dibjet_eta"],
                    "dibjet_phi": SR_Data["dibjet_phi"],
                    "dibjet_dr": SR_Data["dibjet_dr"],
                    "dibjet_deltaPhi": SR_Data["dibjet_deltaPhi"],
                    "dibjet_deltaEta": SR_Data["dibjet_deltaEta"],
                    "dibjet_BvsL_max": SR_Data["dibjet_BvsL_max"],
                    "dibjet_BvsL_min": SR_Data["dibjet_BvsL_min"],
                    "dibjet_CvsL_max": SR_Data["dibjet_CvsL_max"],
                    "dibjet_CvsL_min": SR_Data["dibjet_CvsL_min"],
                    "dibjet_CvsB_max": SR_Data["dibjet_CvsB_max"],
                    "dibjet_CvsB_min": SR_Data["dibjet_CvsB_min"],
                    "dibjet_pt_max": SR_Data["dibjet_pt_max"],
                    "dibjet_pt_min": SR_Data["dibjet_pt_min"],
                    "dibjet_mass_max": SR_Data["dibjet_mass_max"],
                    "dibjet_mass_min": SR_Data["dibjet_mass_min"],

                    "dilep_m": SR_Data["dilep_m"],
                    "dilep_pt": SR_Data["dilep_pt"],
                    "dilep_eta": SR_Data["dilep_eta"],
                    "dilep_phi": SR_Data["dilep_phi"],
                    "dilep_dr": SR_Data["dilep_dr"],
                    "dilep_deltaPhi": SR_Data["dilep_deltaPhi"],
                    "dilep_deltaEta": SR_Data["dilep_deltaEta"],

                    "ZH_pt_ratio": SR_Data["ZH_pt_ratio"],
                    "ZH_deltaPhi": SR_Data["ZH_deltaPhi"],

                    "ZHbb_pt_ratio": SR_Data["ZHbb_pt_ratio"],
                    "VHbb_pt_ratio": SR_Data["VHbb_pt_ratio"],
                    "ZHbb_deltaPhi": SR_Data["ZHbb_deltaPhi"],
                    "VHbb_deltaPhi": SR_Data["VHbb_deltaPhi"],

                    "ZHbb_deltaR": SR_Data["ZHbb_deltaR"],
                    "VHbb_deltaR": SR_Data["VHbb_deltaR"],

                    "pt_miss": SR_Data["pt_miss"],
                    
                })
                
            if self.proc_type=="ZNuNu":
                SR_Data = self.events[selection_ZNuNu]
                variables_to_save = ak.zip({
                    "dijet_m": SR_Data["dijet_m"],
                    "dijet_pt": SR_Data["dijet_pt"],
                    "dijet_dr": SR_Data["dijet_dr"],
                    "dijet_deltaPhi": SR_Data["dijet_deltaPhi"],
                    "dijet_deltaEta": SR_Data["dijet_deltaEta"],
                    "dijet_CvsL_max": SR_Data["dijet_CvsL_max"],
                    "dijet_CvsL_min": SR_Data["dijet_CvsL_min"],
                    "dijet_CvsB_max": SR_Data["dijet_CvsB_max"],
                    "dijet_CvsB_min": SR_Data["dijet_CvsB_min"],
                    "dijet_pt_max": SR_Data["dijet_pt_max"],
                    "dijet_pt_min": SR_Data["dijet_pt_min"],
                    "ZH_pt_ratio": SR_Data["ZH_pt_ratio"],
                    "ZH_deltaPhi": SR_Data["ZH_deltaPhi"],
                    "Z_pt": SR_Data["Z_pt"]
                })
            # Here we write to root  and parquete files

            with warnings.catch_warnings():
                # Suppress FutureWarning
                warnings.filterwarnings("ignore", category=FutureWarning)
                
                # Check if the directory exists
                if not os.path.exists(f"Saved_root_files_{self.proc_type}_Hbb/{self.events.metadata['dataset']}"):
                    # If not, create it
                    os.system(f"mkdir -p Saved_root_files_{self.proc_type}_Hbb/{self.events.metadata['dataset']}")
                    
                # Write the events to a ROOT file
                with uproot.recreate(f"Saved_root_files_{self.proc_type}_Hbb/{self.events.metadata['dataset']}/{self.events.metadata['filename'].split('/')[-1].replace('.root','')}_{int(self.events.metadata['entrystart'])}_{int(self.events.metadata['entrystop'])}.root") as f:
                  df = ak.to_pandas(variables_to_save)
                  f["tree"] = df                   

                # Write the events to a Parquet file
                # ak.to_pandas(variables_to_save).to_parquet(f"Saved_root_files_{self.proc_type}_Hbb/{self.events.metadata['dataset']}/{self.events.metadata['filename'].split('/')[-1].replace('.root','')}_{int(self.events.metadata['entrystart'])}_{int(self.events.metadata['entrystop'])}_vars.parquet")