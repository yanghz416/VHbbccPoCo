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

def NJets(events, params, **kwargs):
    mask = (events.nJetGood >= params['nj'])
    return ak.where(ak.is_none(mask), False, mask)

def TwoLepTwoJets(events, params, **kwargs):
    if params["lep_flav"] not in ['mu','el','both']:
        print("This lepton flavor is not supported:", params["lep_flav"])
        raise Exception("The lepton flavor is not supported")
    
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
    if params["lep_flav"] not in ['mu','el','both']:
        print("This lepton flavor is not supported:", params["lep_flav"])
        raise Exception("The lepton flavor is not supported")
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
    if params['tagger'] == "PNet":
        CvL = "btagPNetCvL"
        CvB = "btagPNetCvB"
    elif params['tagger'] == "DeepFlav":
        CvL = "btagDeepFlavCvL"
        CvB = "btagDeepFlavCvB"
    elif params['tagger'] == "RobustParT":
        CvL = "btagRobustParTAK4CvL"
        CvB = "btagRobustParTAK4CvB"
    else:
        raise NotImplementedError(f"This tagger is not implemented: {params['tagger']}")

    #print(events.JetsCvsL.[ctag][:, 0]>0.2)

    if params["ctag"]:
        mask_CvL = (events.JetsCvsL[CvL][:,0]>params["cut_CvL"])
    else:
        mask_CvL = (events.JetsCvsL[CvL][:,0]<params["cut_CvL"])

    if params["btag"]:
        mask_CvB = (events.JetsCvsL[CvB][:,0]<params["cut_CvB"])
    else:
        mask_CvB = (events.JetsCvsL[CvB][:,0]>params["cut_CvB"])

    mask = mask_CvL & mask_CvB
    
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


def DiLeptonPtCut(events, params, **kwargs):
    mask = (  (events.ll.pt > params["ptll"]["low"]) & (events.ll.pt < params["ptll"]["high"]) )
    return ak.where(ak.is_none(mask), False, mask)


def DeltaPhiJetMetCut(events, params, **kwargs):
    mask = ( (events['deltaPhi_jet1_MET'] > params["jet_met_dphi_cut"])
             & (events['deltaPhi_jet2_MET'] > params["jet_met_dphi_cut"])
            )
    return ak.where(ak.is_none(mask), False, mask)


def TrueJetFlavors(events, params, **kwargs):
    gen_jets = events.GenJet
    
    cGenJetTot = ak.sum((gen_jets.hadronFlavour == 4) & (gen_jets.pt > 20) & (abs(gen_jets.eta) < 2.4), axis=1)
    bGenJetTot = ak.sum((gen_jets.hadronFlavour == 5) & (gen_jets.pt > 20) & (abs(gen_jets.eta) < 2.4), axis=1)

    tag_cc = (cGenJetTot >= 2)
    tag_bb = (bGenJetTot >= 2)
    tag_bc = (bGenJetTot == 1) & (cGenJetTot == 1)
    tag_cl = (cGenJetTot == 1) & (bGenJetTot == 0)
    tag_bl = (bGenJetTot == 1) & (cGenJetTot == 0)
    tag_ll = (cGenJetTot == 0) & (bGenJetTot == 0)

    mask = (  ((params['jj_flav']=='cc') & tag_cc) |
              ((params['jj_flav']=='bb') & tag_bb) |
              ((params['jj_flav']=='bc') & tag_bc) |
              ((params['jj_flav']=='cl') & tag_cl) |
              ((params['jj_flav']=='bl') & tag_bl) |
              ((params['jj_flav']=='bx') & (tag_bb | tag_bc | tag_bl)) |
              ((params['jj_flav']=='cx') & (tag_cc | tag_cl)) |
              ((params['jj_flav']=='ll') & tag_ll)
            )
    
    #mask = ( (params['jj_flav']=='cc') & tag_cc)
    """
    sampleFlavSplit = 1 * tag_cc  +  2 * tag_bb  +  3 * tag_bc  +  4 * tag_cl  +  5 * tag_bl  +  6 * tag_ll 

    mask = (  (params['jj_flav']=='cc' & (sampleFlavSplit==1)) |
              (params['jj_flav']=='bb' & (sampleFlavSplit==2)) |
              (params['jj_flav']=='bc' & (sampleFlavSplit==3)) |
              (params['jj_flav']=='cl' & (sampleFlavSplit==4)) |
              (params['jj_flav']=='bl' & (sampleFlavSplit==5)) |
              (params['jj_flav']=='udsg' & (sampleFlavSplit==6)) 
            )
    """
    
    return ak.where(ak.is_none(mask), False, mask)

# General cuts


DiJet_bx = Cut(
    name="DiJet_bx",
    function=TrueJetFlavors,
    params={"jj_flav": "bx"}
)
DiJet_cx = Cut(
    name="DiJet_bcx",
    function=TrueJetFlavors,
    params={"jj_flav": "cx"}
)
DiJet_ll = Cut(
    name="DiJet_ll",
    function=TrueJetFlavors,
    params={"jj_flav": "ll"}
)

one_jet = Cut(
    name="one_jet",
    function=NJets,
    params={'nj': 1}
)
two_jets = Cut(
    name="two_jets",
    function=NJets,
    params={'nj': 2}
)
four_jets = Cut(
    name="four_jets",
    function=NJets,
    params={'nj': 4}
)

ctag_j1 = Cut(
    name="ctag_j1",
    function=jettag,
    params={
        "tagger": 'RobustParT',
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
        "tagger": 'RobustParT',
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
        "tagger": 'RobustParT',
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
	"mjj": {'low': 70, 'high': 250}
    },
)
dijet_invmass_cut = Cut(
    name="dijet_invmass_cut",
    function=DiJetMassCut,
    params={
        "invert": True,
	"mjj": {'low': 70, 'high': 250}
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

def wlnu_plus_2j(lep_flav='both'):
    return Cut(
        name="w_plus_2j_lepflav_"+lep_flav,
        function=WLNuTwoJets,
        params={
            "lep_flav": lep_flav,
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


def ll_2j(lep_flav='both'):
    return Cut(
        name = 'll_2j_lepflav_'+lep_flav,
        function=TwoLepTwoJets,
        params={"lep_flav": lep_flav,
                "pt_dilep": 60,
                "mll": {'low': 50, 'high': 400}
                }
    )

def Zll_2j(lep_flav='both'):
    return  Cut(
        name = 'Zll_2j_lepflav_'+lepflav,
        function=TwoLepTwoJets,
        params={"lep_flav": lep_flav,
                "pt_dilep": 60,
                "mll": {'low': 75, 'high': 115}
                }
    )


def dilep_pt(pt_min=60, pt_max=2000):
    return Cut(
        name="dilep_pt_cut_Pt%iTo%i"%(pt_min,pt_max),
        function=DiLeptonPtCut,
        params={
	    "ptll": {'low': pt_min, 'high': pt_max}
        },
    )

# Cuts for ttbar control region
def ll_antiZ_4j(lep_flav='both'):
    return  Cut(
        name = 'll_antiZ_4j_lepflav_'+lep_flav,
        function=AntiZFourJets,
        params={"lep_flav": lep_flav,
                "pt_dilep": 60,
                "mll": {'low': 75, 'high': 120}
                }
    )


