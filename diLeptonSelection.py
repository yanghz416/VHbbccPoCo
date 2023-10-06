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

def TwoMuTwoJets(events, params, **kwargs):
    mask = ( (events.nJetGood >= 2)
             & (events.nMuonGood >= 2)
             & (events.ll.pt > params["pt_dilep"]) 
             & (events.ll.mass > params["mll"]["low"])
             & (events.ll.mass < params["mll"]["high"])
         )
    return ak.where(ak.is_none(mask), False, mask)

def TwoEleTwoJets(events, params, **kwargs):
    mask = ( (events.nJetGood >= 2)
             & (events.nElectronGood >= 2)
             & (events.ll.pt > params["pt_dilep"]) 
             & (events.ll.mass > params["mll"]["low"])
             & (events.ll.mass < params["mll"]["high"])
         )
    return ak.where(ak.is_none(mask), False, mask)


dilepton_presel = Cut(
    name="dilepton",
    function=diLepton,
    params={
        "pt_leading_lep": 33,
        "mll": {'low': 0, 'high': 2000},
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

mumu_2j_channel = Cut(
    name = 'mumu_2j',
    function=TwoMuTwoJets,
    params={"pt_dilep": 60,
            "mll": {'low': 70, 'high': 120}
        }
)
ee_2j_channel = Cut(
    name = 'ee_2j',
    function=TwoEleTwoJets,
    params={"pt_dilep": 60,
            "mll": {'low': 70, 'high': 120}
        }
)
