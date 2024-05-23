import awkward as ak
from pocket_coffea.lib.cut_definition import Cut

def CvsLsorted(jets, ctag):
    # This returns Jets sorted by CvL score (or other tagger defined in params/ctagging.yaml )
    return jets[ak.argsort(jets[ctag["tagger"]], axis=1, ascending=False)]


def diLepton(events, params, year, sample, **kwargs):

    # Masks for same-flavor (SF) and opposite-sign (OS)
    EE = ((events.nMuonGood == 0) & (events.nElectronGood == 2))
    MuMu = ((events.nMuonGood == 2) & (events.nElectronGood == 0))

    OS = events.ll.charge == 0

    mask = (
        (MuMu | EE) & OS
        & (ak.firsts(events.LeptonGood.pt) > params["pt_leading_lep"])
        & (events.ll.mass > params["mll"]["low"])
        & (events.ll.mass < params["mll"]["high"])
        & (events.ll.pt > params["pt_dilep"])
    )
    
    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)

def TwoMuons(events, **kwargs):
    mask = (events.nMuonGood >= 2)
    return ak.where(ak.is_none(mask), False, mask)

def TwoElectrons(events, **kwargs):
    mask = (events.nElectronGood >= 2)
    return ak.where(ak.is_none(mask), False, mask)

def TwoJets(events, **kwargs):
    mask = (events.nJetGood >= 2)
    return ak.where(ak.is_none(mask), False, mask)

def OneJet(events, **kwargs):
    mask = (events.nJetGood >= 1)
    return ak.where(ak.is_none(mask), False, mask)

def TwoLepTwoJets(events, params, **kwargs):
    mask = ( (events.nJetGood >= 2)
             & ( ( (params["lep_flav"]=="mu") & (events.nMuonGood>=2) ) |
                 ( (params["lep_flav"]=="el") & (events.nElectronGood>=2) )  |
                 ( (params["lep_flav"]=="both") & (events.nLeptonGood>=2) )
                )
             & (events.ll.pt > params["pt_dilep"])
             & (events.ll.mass > params["mll"]["low"])
             & (events.ll.mass < params["mll"]["high"])
            )
    return ak.where(ak.is_none(mask), False, mask)

def AntiZFourJets(events, params, **kwargs):
    # This mask is used for TTbar CR
    mask = ( (events.nJetGood >= 4)
             & ( ( (params["lep_flav"]=="mu") & (events.nMuonGood>=2) ) |
                 ( (params["lep_flav"]=="el") & (events.nElectronGood>=2) )  |
                 ( (params["lep_flav"]=="both") & (events.nLeptonGood>=2) )
                )
             & (events.ll.pt > params["pt_dilep"])
             & ( (events.ll.mass < params["mll"]["low"]) | (events.ll.mass > params["mll"]["high"]) )
             & (events.ll.mass > 50) 
            )
    return ak.where(ak.is_none(mask), False, mask)

def OneLeptonPlusMet(events, params, **kwargs):
    #mask = (events.nLeptonGood == 1 )
    mask = ( (events.nLeptonGood == 1 )
             & (ak.firsts(events.LeptonGood.pt) > params["pt_lep"])
             & (events.MET.pt > params["pt_met"])
            )
    return ak.where(ak.is_none(mask), False, mask)

def LepMetTwoJets(events, params, **kwargs):
    mask = ( (events.nLeptonGood == 1 )
             & (ak.firsts(events.LeptonGood.pt) > params["pt_lep"])
             & (events.MET.pt > params["pt_met"])
             & (events.nJetGood >= 2)
            )
    return ak.where(ak.is_none(mask), False, mask)

def MetTwoJetsNoLep(events, params, **kwargs):    
    mask = ( (events.nLeptonGood == 0 )
             & (events.MET.pt > params["pt_met"])
             & (events.nJetGood >= 2)
             & (ak.firsts(events.JetGood.pt) > params["pt_jet1"])
             & (ak.pad_none(events.JetGood.pt, 2, axis=1)[:,1] > params["pt_jet2"])
            )
    return ak.where(ak.is_none(mask), False, mask)

def WLNuTwoJets(events, params, **kwargs):

    fields = {
        "pt": events.MET.pt,
	"eta": ak.zeros_like(events.MET.pt),
        "phi": events.MET.phi,
        "mass": ak.zeros_like(events.MET.pt),
        "charge": ak.zeros_like(events.MET.pt),
    }

    METs = ak.zip(fields, with_name="PtEtaPhiMCandidate")
    LepPlusMet = METs + ak.firsts(events.LeptonGood)
    mask = ( (events.nJetGood >= 2)
             & ( ( (params["lep_flav"]=="mu") & (events.nMuonGood==1) ) |
                 ( (params["lep_flav"]=="el") & (events.nElectronGood==1) )  |
                 (params["lep_flav"]=="both") 
                )
             & (LepPlusMet.pt > params["pt_w"])
        )
    return ak.where(ak.is_none(mask), False, mask)

def jettag(events, params, **kwargs):
    #print(events.JetsCvsL.btagDeepFlavCvL[:, 0]>0.2)
    if params["ctag"]:
        CvL = (events.JetsCvsL.btagDeepFlavCvL[:,0]>params["cut_CvL"])
    else:
        CvL = (events.JetsCvsL.btagDeepFlavCvL[:,0]<params["cut_CvL"])

    if params["btag"]:
        CvB = (events.JetsCvsL.btagDeepFlavCvB[:,0]<params["cut_CvB"])
    else:
        CvB = (events.JetsCvsL.btagDeepFlavCvB[:,0]>params["cut_CvB"])

    mask = CvL & CvB
    
    return ak.where(ak.is_none(mask), False, mask)


def DiJetPtCut(events, params, **kwargs):
    mask = (  (events.nJetGood >= 2)
              & (events.dijet.pt > params["pt_dijet"])
              #& (events.dijet_csort.pt > params["pt_dijet"])
            )
    return ak.where(ak.is_none(mask), False, mask)

def DiJetMassCut(events, params, **kwargs):

    if params["invert"]:
        mask = (  (events.nJetGood >= 2)          
                  & ( (events.dijet_csort.mass < params["mjj"]["low"])
                      | (events.dijet_csort.mass > params["mjj"]["high"]) )
                )        
    else:
        mask = (  (events.nJetGood >= 2)          
                  & (events.dijet_csort.mass > params["mjj"]["low"])
                  & (events.dijet_csort.mass < params["mjj"]["high"])
                )
    return ak.where(ak.is_none(mask), False, mask)


def DeltaPhiJetMetCut(events, params, **kwargs):
    mask = ( (events['deltaPhi_jet1_MET'] > params["jet_met_dphi_cut"])
             & (events['deltaPhi_jet2_MET'] > params["jet_met_dphi_cut"])
            )
    return ak.where(ak.is_none(mask), False, mask)



# General cuts

one_jet = Cut(
    name="one_jet",
    function=OneJet,
    params={}
    
)

ctag_j1 = Cut(
    name="ctag_j1",
    function=jettag,
    params={
        "ctag": True,
        "btag": False,
        "cut_CvL": 0.2,
        "cut_CvB": 0.4
    }
)
antictag_j1 = Cut(
    name="antictag_j1",
    function=jettag,
    params={
        "ctag": False,
        "btag": False,
        "cut_CvL": 0.2,
        "cut_CvB": 0.4
    }
)
btag_j1 = Cut(
    name="btag_j1",
    function=jettag,
    params={
        "ctag": True,
        "btag": True,
        "cut_CvL": 0.2,
        "cut_CvB": 0.4
    }
)

dijet_pt_cut = Cut(
    name="dijet_pt_cut",
    function=DiJetPtCut,
    params={
	"pt_dijet": 120,
    },
)

dijet_mass_cut = Cut(
    name="dijet_mass_cut",
    function=DiJetMassCut,
    params={
        "invert": False,
	"mjj": {'low': 75, 'high': 200}
    },
)
dijet_invmass_cut = Cut(
    name="dijet_invmass_cut",
    function=DiJetMassCut,
    params={
        "invert": True,
	"mjj": {'low': 75, 'high': 200}
    },
)

jet_met_dphi_cut = Cut(
    name='jet_met_dphi_cut',
    function=DeltaPhiJetMetCut,
    params={
	"jet_met_dphi_cut": 0.6,
    },
)
# Cuts for 0-Lep channel

met_2jets_0lep = Cut(
    name="met_2jets_0lep",
    function=MetTwoJetsNoLep,
    params={
        "pt_met": 170,
        "pt_jet1": 60,
        "pt_jet2": 35,
    },
)

# Cuts for 1-Lep channel

onelep_plus_met = Cut(
    name="onelep_plus_met",
    function=OneLeptonPlusMet,
    params={
        "pt_lep": 33,
        "pt_met": 10,
    },
)

lep_met_2jets = Cut(
    name="lep_met_2jets",
    function=LepMetTwoJets,
    params={
        "pt_lep": 33,
        "pt_met": 10,
    },
)

wlnu_plus_2j = Cut(
    name="w_plus_2j",
    function=WLNuTwoJets,
    params={
        "lep_flav": "both",
        "pt_w": 100
    }
)

wmunu_plus_2j = Cut(
    name="w_plus_2j",
    function=WLNuTwoJets,
    params={
        "lep_flav": "mu",
        "pt_w": 100
    }
)

welnu_plus_2j = Cut(
    name="w_plus_2j",
    function=WLNuTwoJets,
    params={
        "lep_flav": "el",
        "pt_w": 100
    }
)

# Cuts for 2-Lep channel

dilepton = Cut(
    name="dilepton",
    function=diLepton,
    params={
        "pt_leading_lep": 27,
        "mll": {'low': 81, 'high': 101},
        "pt_dilep": 15
    },
)

mumu_channel = Cut(
    name = 'mumu',
    function=TwoMuons,
    params=None
)
ee_channel = Cut(
    name = 'ee',
    function=TwoElectrons,
    params=None
)


ll_2j = Cut(
    name = 'll_2j',
    function=TwoLepTwoJets,
    params={"lep_flav": "both",
            "pt_dilep": 60,
            "mll": {'low': 50, 'high': 400}
            }
)
mumu_2j = Cut(
    name = 'mumu_2j',
    function=TwoLepTwoJets,
    params={"lep_flav": "mu",
            "pt_dilep": 60,
            "mll": {'low': 50, 'high': 400}
        }
)
ee_2j = Cut(
    name = 'ee_2j',
    function=TwoLepTwoJets,
    params={"lep_flav": "el",
            "pt_dilep": 60,
            "mll": {'low': 50, 'high': 400}
        }
)

Zll_2j = Cut(
    name = 'Zll_2j',
    function=TwoLepTwoJets,
    params={"lep_flav": "both",
            "pt_dilep": 60,
            "mll": {'low': 75, 'high': 115}
            }
)
Zmumu_2j = Cut(
    name = 'Zmumu_2j',
    function=TwoLepTwoJets,
    params={"lep_flav": "mu",
            "pt_dilep": 60,
            "mll": {'low': 75, 'high': 115}
        }
)
Zee_2j = Cut(
    name = 'Zee_2j',
    function=TwoLepTwoJets,
    params={"lep_flav": "el",
            "pt_dilep": 60,
            "mll": {'low': 75, 'high': 115}
        }
)


# Cut for ttbar control region
ll_antiZ_4j = Cut(
    name = 'll_antiZ_4j',
    function=AntiZFourJets,
    params={"lep_flav": "both",
            "pt_dilep": 60,
            "mll": {'low': 75, 'high': 120}
            }
)


