import awkward as ak
from pocket_coffea.lib.cut_definition import Cut

def GenTwoLepTwoJets(events, params, **kwargs):
    mask = ( (events.nMyGenJets >= 2)
             & (events.nMyGenLeptons >= 2)
             & (ak.firsts(events.MyGenLeptons.pt) > params["pt_leading_lep"])             
             & (events.ll.pt > params["pt_dilep"])
             & (events.ll.mass > params["mll"]["low"])
             & (events.ll.mass < params["mll"]["high"])
             & (events.dijet.pt > params["pt_dijet"])
             & (events.dijet.mass > params["mjj"]["low"])
             & (events.dijet.mass < params["mjj"]["high"])
            )
    return ak.where(ak.is_none(mask), False, mask)


def GenDiJetMassCut(events, params, **kwargs):
    if params["invert"]:
        mask = (  (events.nMyGenJets >= 2)
                  & ( (events.dijet.mass < params["mjj"]["low"])
                      | (events.dijet.mass > params["mjj"]["high"]) )
                )
    else:
        mask = (  (events.nMyGenJets >= 2)
                  & (events.dijet.mass > params["mjj"]["low"])
                  & (events.dijet.mass < params["mjj"]["high"])
                )
    return ak.where(ak.is_none(mask), False, mask)


def LHE_NjetCut(events, params, **kwargs):
    mask = (events.LHE.Njets == params["Njets"])
    
    return ak.where(ak.is_none(mask), False, mask)

def LHE_Nj(Nj):
    return Cut(name="LHE_Nj",function=LHE_NjetCut,  params={ "Njets": Nj})

GenZplus2j = Cut(
    name="gen_Zplus2j",
    function=GenTwoLepTwoJets,
    params={
        "pt_leading_lep": 25,
        "mll": {'low': 60, 'high': 120},
        "pt_dilep": 90,
        "mjj": {'low': 0, 'high': 2000},
        "pt_dijet": 0
    },
)


dijet_m0to60 = Cut(
    name="dijet_m0to60",
    function=GenDiJetMassCut,
    params={
        "invert": False,
        "mjj": {'low': 0, 'high': 60}
    },
)
dijet_m60to120 = Cut(
    name="dijet_m50to120",
    function=GenDiJetMassCut,
    params={
        "invert": False,
        "mjj": {'low': 60, 'high': 120}
    },
)
dijet_m120to2000 = Cut(
    name="dijet_m120to2000",
    function=GenDiJetMassCut,
    params={
        "invert": False,
        "mjj": {'low': 120, 'high': 2000}
    },
)
