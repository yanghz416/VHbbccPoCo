import awkward as ak
import numpy as np
import uproot
import pandas as pd
import warnings
import os


from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.hist_manager import Axis
from pocket_coffea.lib.objects import (
    jet_correction,
    lepton_selection,
    jet_selection,
    btagging,
    CvsLsorted,
    get_dilepton,
    get_dijet
)


class VHccBaseProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

        self.proc_type = self.params["proc_type"]
        
        #self.isRun3 = True if self.params["run_period"]=='Run3' else False
        
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
            self.events, "Jet", self.params, "LeptonGood"
        )
        self.events["BJetGood"] = btagging(
            self.events["JetGood"], self.params.btagging.working_point[self._year], wp=self.params.object_preselection.bJetWP)
            
    def count_objects(self, variation):
        self.events["nMuonGood"] = ak.num(self.events.MuonGood)
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood)
        self.events["nLeptonGood"] = ak.num(self.events.LeptonGood)

        self.events["nJet"] = ak.num(self.events.Jet)
        self.events["nJetGood"] = ak.num(self.events.JetGood)
        self.events["nBJetGood"] = ak.num(self.events.BJetGood)
        # self.events["nfatjet"]   = ak.num(self.events.FatJetGood)

    # Function that defines common variables employed in analyses and save them as attributes of `events`
    def define_common_variables_before_presel(self, variation):
        self.events["JetGood_Ht"] = ak.sum(abs(self.events.JetGood.pt), axis=1)

    def define_common_variables_after_presel(self, variation):
        self.events["dijet"] = get_dijet(self.events.JetGood)
        
        
        
        #self.events["dijet_pt"] = self.events.dijet.pt
        
        if self.proc_type=="ZLL":
            high_pt_mask = (self.events.ll.pt > 150)
            low_pt_mask = (self.events.ll.pt >= 50) & (self.events.ll.pt < 150)
            
            # Determine the maximum length
            max_length = len(self.events.ll)
            ll_low = self.events.ll[low_pt_mask]
            ll_high = self.events.ll[high_pt_mask]
            dijet_low = self.events.dijet[low_pt_mask]
            dijet_high = self.events.dijet[high_pt_mask]
            # Pad the arrays to match the maximum length
            ll_low_padded = ak.pad_none(ll_low, max_length, axis=0)
            ll_high_padded = ak.pad_none(ll_high, max_length, axis=0)
            dijet_low_padded = ak.pad_none(dijet_low, max_length, axis=0)
            dijet_high_padded = ak.pad_none(dijet_high, max_length, axis=0)

            # Concatenate the jagged arrays along the first axis
            self.events["ll_low"] = ll_low_padded
            self.events["ll_high"] = ll_high_padded
            self.events["dijet_low"] = dijet_low_padded
            self.events["dijet_high"] = dijet_high_padded
            
            ### General
            self.events["dijet_deltaR"] = self.events.dijet.deltaR
            self.events["dijet_deltaPhi"] = self.events.dijet.deltaPhi
            self.events["dijet_deltaEta"] = self.events.dijet.deltaEta
            self.events["dijet_m"] = self.events.dijet.mass
            self.events["dijet_pt"] = self.events.dijet.pt
            self.events["dilep_deltaR"] = self.events.ll.deltaR
            self.events["dilep_pt"] = self.events.ll.pt
            self.events["dilep_mass"] = self.events.ll.mass
            self.events["dilep_phi"] = self.events.ll.deltaPhi
            self.events["dilep_eta"] = self.events.ll.deltaEta
            self.events["pt_ratio"] = self.events.ll.pt/self.events.dijet.pt
            self.events["ZH_delphi"] = np.abs(self.events.ll.delta_phi(self.events.dijet))
            self.angle21_gen = (abs(self.events.ll.l2phi - self.events.dijet.j1Phi) < np.pi)
            self.angle22_gen = (abs(self.events.ll.l2phi - self.events.dijet.j2Phi) < np.pi)
            self.events["deltaPhi_l2_j1"] = ak.where(self.angle21_gen, abs(self.events.ll.l2phi - self.events.dijet.j1Phi), 2*np.pi - abs(self.events.ll.l2phi - self.events.dijet.j1Phi))              
            self.events["deltaPhi_l2_j2"] = ak.where(self.angle22_gen, abs(self.events.ll.l2phi - self.events.dijet.j2Phi), 2*np.pi - abs(self.events.ll.l2phi - self.events.dijet.j2Phi))
            
            
            ### Low_pt dilepton
            self.events["dijet_deltaR_low"] = self.events.dijet_low.deltaR
            self.events["dijet_deltaPhi_low"] = self.events.dijet_low.deltaPhi
            self.events["dijet_deltaEta_low"] = self.events.dijet_low.deltaEta
            self.events["dijet_m_low"] = self.events.dijet_low.mass
            self.events["dijet_pt_low"] = self.events.dijet_low.pt
            self.events["dilep_deltaR_low"] = self.events.ll_low.deltaR
            self.events["dilep_pt_low"] = self.events.ll_low.pt
            self.events["dilep_mass_low"] = self.events.ll_low.mass
            self.events["dilep_phi_low"] = self.events.ll_low.deltaPhi
            self.events["dilep_eta_low"] = self.events.ll_low.deltaEta
            self.events["pt_ratio_low"] = self.events.ll_low.pt/self.events.dijet_low.pt
            self.events["ZH_delphi_low"] = np.abs(self.events.ll_low.delta_phi(self.events.dijet_low))
            self.angle21 = (abs(self.events.ll_low.l2phi - self.events.dijet_low.j1Phi) < np.pi)
            self.angle22 = (abs(self.events.ll_low.l2phi - self.events.dijet_low.j2Phi) < np.pi)
            self.events["deltaPhi_l2_j1_low"] = ak.where(self.angle21, abs(self.events.ll_low.l2phi - self.events.dijet_low.j1Phi), 2*np.pi - abs(self.events.ll_low.l2phi - self.events.dijet_low.j1Phi))              
            self.events["deltaPhi_l2_j2_low"] = ak.where(self.angle22, abs(self.events.ll_low.l2phi - self.events.dijet_low.j2Phi), 2*np.pi - abs(self.events.ll_low.l2phi - self.events.dijet_low.j2Phi))
            
            ### High_pt dilepton
            
            self.events["dijet_deltaR_high"] = self.events.dijet_high.deltaR
            self.events["dijet_deltaPhi_high"] = self.events.dijet_high.deltaPhi
            self.events["dijet_deltaEta_high"] = self.events.dijet_high.deltaEta
            self.events["dijet_m_high"] = self.events.dijet_high.mass
            self.events["dijet_pt_high"] = self.events.dijet_high.pt
            self.events["dilep_deltaR_high"] = self.events.ll_high.deltaR
            self.events["dilep_pt_high"] = self.events.ll_high.pt
            self.events["dilep_mass_high"] = self.events.ll_high.mass
            self.events["dilep_phi_high"] = self.events.ll_high.deltaPhi
            self.events["dilep_eta_high"] = self.events.ll_high.deltaEta
            self.events["pt_ratio_high"] = self.events.ll_high.pt/self.events.dijet_high.pt
            self.events["ZH_delphi_high"] = np.abs(self.events.ll_high.delta_phi(self.events.dijet_high))
            self.angle21_h = (abs(self.events.ll_high.l2phi - self.events.dijet_high.j1Phi) < np.pi)
            self.angle22_h = (abs(self.events.ll_high.l2phi - self.events.dijet_high.j2Phi) < np.pi)
            self.events["deltaPhi_l2_j1_high"] = ak.where(self.angle21_h, abs(self.events.ll_high.l2phi - self.events.dijet_high.j1Phi), 2*np.pi - abs(self.events.ll_high.l2phi - self.events.dijet_high.j1Phi))
            self.events["deltaPhi_l2_j2_high"] = ak.where(self.angle22_h, abs(self.events.ll_high.l2phi - self.events.dijet_high.j2Phi), 2*np.pi - abs(self.events.ll_high.l2phi - self.events.dijet_high.j2Phi))
            
            # Create a record for the low_ variables
            # Create a record for the low_ variables
            low_variables = ak.zip({
                "dilep_deltaR": self.events["dilep_deltaR_low"],
                "dilep_pt": self.events["dilep_pt_low"],
                "dilep_mass": self.events["dilep_mass_low"],
                "dilep_phi": self.events["dilep_phi_low"],
                "dilep_eta": self.events["dilep_eta_low"],
                "pt_ratio": self.events["pt_ratio_low"],
                "ZH_delphi": self.events["ZH_delphi_low"],
                "deltaPhi_l2_j1": self.events["deltaPhi_l2_j1_low"],
                "deltaPhi_l2_j2": self.events["deltaPhi_l2_j2_low"],
                "dijet_deltaR": self.events["dijet_deltaR_low"],
                "dijet_deltaPhi": self.events["dijet_deltaPhi_low"],
                "dijet_deltaEta": self.events["dijet_deltaEta_low"],
                "dijet_m": self.events["dijet_m_low"],
                "dijet_pt": self.events["dijet_pt_low"]
            })

            # Create a record for the high_ variables
            high_variables = ak.zip({
                "dilep_deltaR": self.events["dilep_deltaR_high"],
                "dilep_pt": self.events["dilep_pt_high"],
                "dilep_mass": self.events["dilep_mass_high"],
                "dilep_phi": self.events["dilep_phi_high"],
                "dilep_eta": self.events["dilep_eta_high"],
                "pt_ratio": self.events["pt_ratio_high"],
                "ZH_delphi": self.events["ZH_delphi_high"],
                "deltaPhi_l2_j1": self.events["deltaPhi_l2_j1_high"],
                "deltaPhi_l2_j2": self.events["deltaPhi_l2_j2_high"],
                "dijet_deltaR": self.events["dijet_deltaR_high"],
                "dijet_deltaPhi": self.events["dijet_deltaPhi_high"],
                "dijet_deltaEta": self.events["dijet_deltaEta_high"],
                "dijet_m": self.events["dijet_m_high"],
                "dijet_pt": self.events["dijet_pt_high"]
            })


            # Add the records to the events
            self.events["low_variables"] = low_variables
            self.events["high_variables"] = high_variables

        
        

        if self.proc_type=="ZNuNu":
            self.events["deltaPhi_jet1_MET"] = np.abs(self.events.MET.delta_phi(self.events.JetGood[:,0]))
            self.events["deltaPhi_jet2_MET"] = np.abs(self.events.MET.delta_phi(self.events.JetGood[:,1]))
        
        self.events["JetsCvsL"] = CvsLsorted(self.events["JetGood"], self.params.ctagging.working_point[self._year])

        #print("Pt sort pt:", self.events["JetGood"][self.events["nJetGood"]>=3].pt)
        #print("CvsL sort pt:", self.events["JetsCvsL"][self.events["nJetGood"]>=3].pt)

        #print("Pt sort CvsL:", self.events["JetGood"][self.events["nJetGood"]>=3].btagDeepFlavCvL)
        #print("CvsL sort CvsL:", self.events["JetsCvsL"][self.events["nJetGood"]>=3].btagDeepFlavCvL)

        self.events["dijet_csort"] = get_dijet(self.events.JetsCvsL)
        
        # Suppress FutureWarning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            
             # write to root files
            # Check if the directory exists
            if not os.path.exists(f"Saved_root_files/{self.events.metadata['dataset']}"):
                # If not, create it
                os.system(f"mkdir -p Saved_root_files/{self.events.metadata['dataset']}")
            # Write the events to a ROOT file
            # Write the events to a ROOT file
            with uproot.recreate(f"Saved_root_files/{self.events.metadata['dataset']}/{self.events.metadata['filename'].split('/')[-1].replace('.root','')}_{int(self.events.metadata['entrystart'])}_{int(self.events.metadata['entrystop'])}.root") as f: 
                f["low_variables"] = ak.to_pandas(low_variables)
                f["high_variables"] = ak.to_pandas(high_variables) 
            ak.to_pandas(self.events["low_variables"]).to_parquet(f"Saved_root_files/{self.events.metadata['dataset']}/{self.events.metadata['filename'].split('/')[-1].replace('.root','')}_{int(self.events.metadata['entrystart'])}_{int(self.events.metadata['entrystop'])}_low_vars.parquet")
            ak.to_pandas(self.events["high_variables"]).to_parquet(f"Saved_root_files/{self.events.metadata['dataset']}/{self.events.metadata['filename'].split('/')[-1].replace('.root','')}_{int(self.events.metadata['entrystart'])}_{int(self.events.metadata['entrystop'])}_high_vars.parquet")  