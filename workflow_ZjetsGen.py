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
    Gen_leptons,
    Gen_jets,
    get_dilepton,
    get_dijet
)

class ZjetsBaseProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    def apply_object_preselection(self, variation):

        self.events["MyGenLeptons"] = Gen_leptons(
            self.events, "Both", self.params
        )
        self.events["MyGenLeptons","charge"] = np.sign(self.events["MyGenLeptons"]["pdgId"])

        self.events["ll"] = get_dilepton(self.events.MyGenLeptons, None)

        self.events["MyGenJets"] =  Gen_jets(self.events, "MyGenLeptons", self.params)
        self.events["MyGenJets", "pdgId"] = self.events.MyGenJets.partonFlavour
        
        self.events["dijet"] = get_dijet(self.events.MyGenJets)


    def count_objects(self, variation):
        self.events["nMyGenJets"] = ak.num(self.events.MyGenJets)
        self.events["nMyGenLeptons"] = ak.num(self.events.MyGenLeptons)

    def define_common_variables_before_presel(self, variation):
        pass

    def define_common_variables_after_presel(self, variation):

        pass
