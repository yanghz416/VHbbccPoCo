import awkward as ak
from pocket_coffea.lib.cut_definition import Cut

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

def ctag(events, params, **kwargs):
    #print(events.JetsCvsL.btagDeepFlavCvL[:, 0]>0.2)
    mask = (events.JetsCvsL.btagDeepFlavCvL[:,0]>0.2)
    return ak.where(ak.is_none(mask), False, mask)

def CvsLsorted(jets, ctag):
    return jets[ak.argsort(jets[ctag["tagger"]], axis=1, ascending=False)]


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
ctag_j1 = Cut(
    name="ctag_j1",
    function=ctag,
    params={}
    
)

dilepton = Cut(
    name="dilepton",
    function=diLepton,
    params={
        "pt_leading_lep": 33,
        "mll": {'low': 60, 'high': 120},
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
            "mll": {'low': 70, 'high': 120}
            }
)
mumu_2j = Cut(
    name = 'mumu_2j',
    function=TwoLepTwoJets,
    params={"lep_flav": "mu",
            "pt_dilep": 60,
            "mll": {'low': 70, 'high': 120}
        }
)
ee_2j = Cut(
    name = 'ee_2j',
    function=TwoLepTwoJets,
    params={"lep_flav": "el",
            "pt_dilep": 60,
            "mll": {'low': 70, 'high': 120}
        }
)
