import awkward as ak
import numpy as np
#import uproot
#import pandas as pd
#import warnings
#import os
from CommonSelectors import *


from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.hist_manager import Axis
from pocket_coffea.lib.objects import (
    jet_correction,
    lepton_selection,
    jet_selection,
#    btagging,
#    CvsLsorted,
    get_dilepton,
    get_dijet
)

def delta_phi(a, b):
    """Compute difference in angle two phi values
    Returns a value within [-pi, pi)
    """
    return (a - b + np.pi) % (2 * np.pi) - np.pi

class ZjetsBaseProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    def apply_object_preselection(self, variation):
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
        #self.events["BJetGood"] = btagging(
        #    self.events["JetGood"], self.params.btagging.working_point[self._year], wp=self.params.object_preselection.bJetWP)

    def count_objects(self, variation):
        self.events["nMuonGood"] = ak.num(self.events.MuonGood)
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood)
        self.events["nLeptonGood"] = ak.num(self.events.LeptonGood)

        self.events["nJet"] = ak.num(self.events.Jet)
        self.events["nJetGood"] = ak.num(self.events.JetGood)
        #self.events["nBJetGood"] = ak.num(self.events.BJetGood)
    

    # Function that defines common variables employed in analyses and save them as attributes of `events`
    def define_common_variables_before_presel(self, variation):
        self.events["JetGood_Ht"] = ak.sum(abs(self.events.JetGood.pt), axis=1)

    def define_common_variables_after_presel(self, variation):
        self.events["dijet"] = get_dijet(self.events.JetGood)

        jets = ak.pad_none(self.events.JetGood, 2)
        njet = ak.num(jets[~ak.is_none(jets, axis=1)])
        j1 = jets[:,0]
        j2 = jets[:,1]

        l1 = self.events.LeptonGood[:,0]
        l2 = self.events.LeptonGood[:,1]
        self.events["j1_l1_dr"] = ak.where( (njet >= 1), j1.delta_r(l1), -1)
        self.events["j1_l2_dr"] = ak.where( (njet >= 1), j1.delta_r(l2), -1)
        self.events["j2_l1_dr"] = ak.where( (njet >= 2), j2.delta_r(l1), -1)
        self.events["j2_l2_dr"] = ak.where( (njet >= 2), j2.delta_r(l2), -1)
            
        #self.events["JetsCvsL"] = CvsLsorted(self.events["JetGood"], self.params.ctagging.working_point[self._year])

        self.events["ZH_pt_ratio"] = self.events.dijet.pt/self.events.ll.pt
        self.events["ZH_deltaPhi"] = np.abs(self.events.ll.delta_phi(self.events.dijet))
        self.events["ZH_deltaR"] = np.abs(self.events.ll.delta_r(self.events.dijet))


    def compute_weights_extra(self, variation):
        pass
