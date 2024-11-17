import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from alive_progress import alive_bar
import numpy as np
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os, math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device, as seen by pytorch:", device)
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class GraphDataset(Dataset):
    def __init__(self, datadict, labels, weights=None, dvc=None):
        if dvc:
            for d in datadict:
                datadict[d] = datadict[d].to(dvc)
        self.data = datadict

        self.labels = labels
        if weights is None:
            self.weights = torch.ones_like(labels)
        else:
            self.weights = weights
        
        if dvc:
            self.labels = self.labels.to(dvc)
            self.weights = self.weights.to(dvc)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return the data and corresponding label at the given index
        return *[self.data[d][idx] for d in self.data], self.labels[idx], self.weights[idx]

class GraphAttentionClassifier(nn.Module):
    def __init__(self, out, jet_dim=2, lep_dim=2, ll_dim=4, num_heads=4, attention_dim=128, num_classes=2, dropout=0.2, pairwisefeats = 7):
        super(GraphAttentionClassifier, self).__init__()
        self.num_heads = num_heads

        attention_dim=int(out.getenv("ATTENTION",256))
        
        hyperembeddim = int(out.getenv("DIM",16))
        self.pairwisefeats = pairwisefeats

        self.embedjets = nn.Linear(jet_dim, hyperembeddim)
        self.multihead_attention_jet = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)

        self.edgejet = nn.Linear(pairwisefeats, hyperembeddim)
        self.conv1d_edgejet = nn.Conv1d(in_channels=pairwisefeats, out_channels=1, kernel_size=1)
        self.multihead_attention_edgejet = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)        
        self.edgejetbn = nn.BatchNorm1d(1)

        self.embedleps = nn.Linear(lep_dim, hyperembeddim)
        self.multihead_attention_lep = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)

        self.edgelep = nn.Linear(pairwisefeats, hyperembeddim)
        self.multihead_attention_edgelep = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)     

        self.embedll = nn.Linear(ll_dim, hyperembeddim)     

        self.embedjl = nn.Linear(pairwisefeats, hyperembeddim)
        self.multihead_attention_jl = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)
        self.embedjjl = nn.Linear(pairwisefeats, hyperembeddim)
        self.multihead_attention_jjl = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)
        self.embedjll = nn.Linear(pairwisefeats, hyperembeddim)
        self.multihead_attention_jll = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)
        self.embedjjll = nn.Linear(pairwisefeats, hyperembeddim)
        self.multihead_attention_jjll = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)

        self.layer_norm = nn.LayerNorm(hyperembeddim)

        self.fc1 = nn.Linear(hyperembeddim*9, attention_dim)
        self.bn1 = nn.BatchNorm1d(attention_dim)
        self.fc2 = nn.Linear(attention_dim, attention_dim)
        self.bn2 = nn.BatchNorm1d(attention_dim)
        self.fc3 = nn.Linear(attention_dim, num_classes)

        self.dropout = nn.Dropout(out.getenv("DROPOUT",dropout))
    
    def selfinteraction(self, x):
        pT = x[:, :, 0]        
        eta = x[:, :, 1]
        phi = x[:, :, 2]
        mass = x[:, :, 3]

        # Compute momentum components
        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)

        # Compute energy for each node
        energy = torch.sqrt(pT**2 + pz**2 + mass**2)

        # Compute pairwise momentum and energy differences
        px1 = px.unsqueeze(2)
        py1 = py.unsqueeze(2)
        pz1 = pz.unsqueeze(2)
        E1 = energy.unsqueeze(2)

        px2 = px.unsqueeze(1)
        py2 = py.unsqueeze(1)
        pz2 = pz.unsqueeze(1)
        E2 = energy.unsqueeze(1)


        # Compute invariant mass squared
        E_sum = E1 + E2 
        px_sum = px1 + px2
        py_sum = py1 + py2
        pz_sum = pz1 + pz2
        p_sum2 = px_sum**2 + py_sum**2 + pz_sum**2

        invariant_mass_squared = E_sum**2 - p_sum2
        invariant_mass_squared = torch.clamp(invariant_mass_squared, min=0.0)

        pairwise_pT = torch.sqrt(px_sum**2 + py_sum**2 + 1e-10)
        

        # Combined eta and phi of the pair
        combined_eta = torch.arctanh(pz_sum/torch.sqrt(p_sum2+1e-10))
        combined_eta = torch.where(torch.isinf(combined_eta),20,combined_eta)
        combined_phi = torch.atan2(py_sum, px_sum)

        # Create mask for invalid edges
        nodemask = torch.all(x == 0, dim=-1).bool()
        edgemask = nodemask.unsqueeze(1) | nodemask.unsqueeze(2) 

        
        deta = eta.unsqueeze(2) - eta.unsqueeze(1)
        dphi = phi.unsqueeze(2) - phi.unsqueeze(1)
        dphi = (dphi + math.pi) % (2 * math.pi) - math.pi  # Wrap to range [-pi, pi]
        distances = torch.sqrt(deta**2 + dphi**2)

        # Compute k_T
        pT1 = pT.unsqueeze(2)  # shape [batch, n, 1]
        pT2 = pT.unsqueeze(1)  # shape [batch, 1, n]
        min_pT = torch.min(pT1, pT2)  # shape [batch, n, n]
        kT = min_pT * distances

        # Compute z
        sum_pT = pT1 + pT2
        z = min_pT / sum_pT

        # Apply clamp to avoid issues with log(0)
        distances = torch.clamp(distances, min=1e-10)
        kT = torch.clamp(kT, min=1e-10)
        z = torch.clamp(z, min=1e-10)
        invariant_mass_squared = torch.clamp(invariant_mass_squared, min=1e-10)
        pairwise_pT = torch.clamp(pairwise_pT, min=1e-10)

        # Compute logarithms of the features
        log_distances = torch.log(distances)
        log_kT = torch.log(kT)
        log_z = torch.log(z)
        log_invariant_mass_squared = torch.log(invariant_mass_squared)
        log_pairwise_pT = torch.log(pairwise_pT)

        # Combine logarithms of edge features
        ftlist = [log_distances, log_kT, log_z, log_invariant_mass_squared, log_pairwise_pT, combined_eta, combined_phi]
        log_edge_features = torch.stack(ftlist[:self.pairwisefeats], dim=-1)
        edgeP4 = torch.stack([pairwise_pT, combined_eta, combined_phi, torch.sqrt(invariant_mass_squared)], dim=-1)

        # Apply mask to edge features
        log_edge_features_masked = log_edge_features.masked_fill(edgemask.unsqueeze(-1), -1e6)

        return log_edge_features_masked, edgeP4


    def pairwise_interaction(self, x, y):
        # Extract features for x and y
        pT_x, eta_x, phi_x, mass_x = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
        pT_y, eta_y, phi_y, mass_y = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]

        # Compute momentum components for x and y
        px_x = pT_x * torch.cos(phi_x)
        py_x = pT_x * torch.sin(phi_x)
        pz_x = pT_x * torch.sinh(eta_x)
        px_y = pT_y * torch.cos(phi_y)
        py_y = pT_y * torch.sin(phi_y)
        pz_y = pT_y * torch.sinh(eta_y)

        # Compute energy for each node in x and y
        energy_x = torch.sqrt(pT_x**2 + pz_x**2 + mass_x**2)
        energy_y = torch.sqrt(pT_y**2 + pz_y**2 + mass_y**2)

        # Compute pairwise momentum and energy sums for invariant mass and kinematic properties
        px1, py1, pz1, E1 = px_x.unsqueeze(2), py_x.unsqueeze(2), pz_x.unsqueeze(2), energy_x.unsqueeze(2)
        px2, py2, pz2, E2 = px_y.unsqueeze(1), py_y.unsqueeze(1), pz_y.unsqueeze(1), energy_y.unsqueeze(1)

        # Calculate invariant mass squared
        E_sum = E1 + E2
        px_sum = px1 + px2
        py_sum = py1 + py2
        pz_sum = pz1 + pz2
        p_sum2 = px_sum**2 + py_sum**2 + pz_sum**2
        invariant_mass_squared = E_sum**2 - p_sum2
        invariant_mass_squared = torch.clamp(invariant_mass_squared, min=0.0)

        # Calculate pairwise transverse momentum (pT)
        pairwise_pT = torch.sqrt(px_sum**2 + py_sum**2 + 1e-10)

        # Combined eta and phi for the pair
        combined_eta = torch.arctanh(pz_sum / torch.sqrt(p_sum2 + 1e-10))
        combined_eta = torch.where(torch.isinf(combined_eta),20,combined_eta)
        combined_phi = torch.atan2(py_sum, px_sum)

        # Create mask for invalid edges
        nodemask_x = torch.all(x == 0, dim=-1).bool()
        nodemask_y = torch.all(y == 0, dim=-1).bool()
        edgemask = nodemask_x.unsqueeze(2) | nodemask_y.unsqueeze(1)

        # Calculate pairwise differences in eta and phi
        deta = eta_x.unsqueeze(2) - eta_y.unsqueeze(1)
        dphi = phi_x.unsqueeze(2) - phi_y.unsqueeze(1)
        dphi = (dphi + math.pi) % (2 * math.pi) - math.pi  # Wrap to range [-pi, pi]
        distances = torch.sqrt(deta**2 + dphi**2)

        # Compute k_T
        pT1 = pT_x.unsqueeze(2)
        pT2 = pT_y.unsqueeze(1)
        min_pT = torch.min(pT1, pT2)
        kT = min_pT * distances

        # Compute z
        sum_pT = pT1 + pT2
        z = min_pT / sum_pT

        # Apply clamp to avoid issues with log(0)
        distances = torch.clamp(distances, min=1e-10)
        kT = torch.clamp(kT, min=1e-10)
        z = torch.clamp(z, min=1e-10)
        invariant_mass_squared = torch.clamp(invariant_mass_squared, min=1e-10)
        pairwise_pT = torch.clamp(pairwise_pT, min=1e-10)

        # Compute logarithms of the features
        log_distances = torch.log(distances)
        log_kT = torch.log(kT)
        log_z = torch.log(z)
        log_invariant_mass_squared = torch.log(invariant_mass_squared)
        log_pairwise_pT = torch.log(pairwise_pT)

        # Combine logarithms of edge features
        ftlist = [log_distances, log_kT, log_z, log_invariant_mass_squared, log_pairwise_pT, combined_eta, combined_phi]
        log_edge_features = torch.stack(ftlist[:self.pairwisefeats], dim=-1)
        edgeP4 = torch.stack([pairwise_pT, combined_eta, combined_phi, torch.sqrt(invariant_mass_squared)], dim=-1)

        # Apply mask to edge features
        log_edge_features_masked = log_edge_features.masked_fill(edgemask.unsqueeze(-1), -1e6)

        return log_edge_features_masked, edgeP4

    def makeUT(self,tensor):
        nnodes = tensor.shape[1]
        indices = torch.triu_indices(nnodes, nnodes, offset=1, dtype=torch.int32)
        return tensor[:, indices[0], indices[1], :]
        

    def pairprocess(self,p4_1,p4_2,embed,mha=None,pool=True):
        ij, _ = self.pairwise_interaction(p4_1,p4_2)
        ij = ij.reshape(ij.shape[0],ij.shape[1]*ij.shape[2],ij.shape[3])
        ij_embed = embed(ij)
        if mha is not None:
            ij_embed, _ = mha(ij_embed,ij_embed,ij_embed)
            ij_embed = self.layer_norm(ij_embed)
        if pool:
            ij_pooled = torch.mean(ij_embed, dim=1)
            return ij_pooled
        return ij_embed

    def forward(self, jet, jetp4, lep, lepp4, ll):
        # Jets
        edge_features, dijetP4 = self.selfinteraction(jetp4)            # BxNxNxF
        nnodes = edge_features.shape[1]
        edge_features_conc = edge_features.view(edge_features.shape[0], nnodes*nnodes, edge_features.shape[3])   # BxN^2xF

        e_attn = self.conv1d_edgejet(edge_features_conc.permute(0,2,1))      #BxFxN^2
        e_attn = F.relu(self.edgejetbn(e_attn))
        e_attn = e_attn.view(e_attn.shape[0],e_attn.shape[1],nnodes,nnodes) #BxFxNxN
        e_attn = e_attn.repeat(1, self.num_heads, 1, 1)
        e_attn = e_attn.view(e_attn.shape[0]*e_attn.shape[1],e_attn.shape[2],e_attn.shape[3])  #B*FxNxN

        jetmask = torch.all(jet == 0, dim=-1)
        edgemask = jetmask.unsqueeze(1) | jetmask.unsqueeze(2)
        edgemask = edgemask.view(edgemask.shape[0],edgemask.shape[1]*edgemask.shape[2])     #BxN^2

        jet = self.embedjets(jet)
        attn_output_jet, _ = self.multihead_attention_jet(jet, jet, jet, key_padding_mask=jetmask.float(), attn_mask=e_attn)
        attn_output_jet = self.layer_norm(attn_output_jet)

        e = self.edgejet(edge_features_conc)
        attn_output_edgejet, _ = self.multihead_attention_edgejet(e, e, e, key_padding_mask=edgemask)
        attn_output_edgejet = self.layer_norm(attn_output_edgejet)

        pooled_output_jets = torch.sum(attn_output_jet, dim=1)
        pooled_output_edgejet = torch.mean(attn_output_edgejet, dim=1)


        # Leps
        edge_features, _ = self.selfinteraction(lepp4)
        nnodes = edge_features.shape[1]
        edge_features_avg = edge_features.view(edge_features.shape[0], nnodes*nnodes, edge_features.shape[3])


        lep_emb = self.embedleps(lep)
        attn_output_lep, _ = self.multihead_attention_lep(lep_emb, lep_emb, lep_emb)
        attn_output_lep = self.layer_norm(attn_output_lep)

        e = self.edgelep(edge_features_avg)
        attn_output_edgelep, _ = self.multihead_attention_edgelep(e, e, e)
        attn_output_edgelep = self.layer_norm(attn_output_edgelep)

        pooled_output_leps = torch.sum(attn_output_lep, dim=1)
        pooled_output_edgelep = torch.mean(attn_output_edgelep, dim=1)

        # ll
        ll_emb = self.embedll(ll)

        #jet with lep interactions
        jl_out = self.pairprocess(jetp4,lepp4,self.embedjl,self.multihead_attention_jl)

        # jj with lep interactions
        jjp4 = self.makeUT(dijetP4)
        jjl_out = self.pairprocess(jjp4,lepp4,self.embedjjl,self.multihead_attention_jjl)


        # j with ll interactions
        ll = ll.unsqueeze(1)
        jll_out = self.pairprocess(jetp4,ll,self.embedjll,self.multihead_attention_jll)

        # jj with ll interactions
        jjll_out = self.pairprocess(jjp4,ll,self.embedjjll,self.multihead_attention_jjll)

        allout = torch.cat([pooled_output_jets,pooled_output_edgejet,pooled_output_leps,pooled_output_edgelep,ll_emb,jl_out,jjl_out,jll_out,jjll_out],dim=1)

        # Feedforward neural network
        x = self.fc1(allout)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return torch.sigmoid(x)

def runGNNtraining(tensordict, y, outdir, test=False):
    # if test:
    torch.autograd.set_detect_anomaly(True)

    y = torch.tensor(y.to_numpy(),dtype=torch.float32)
    graph = GraphDataset(tensordict,y)
    
    outhandle = outhandler(outdir)

    if device=="cpu":
        nloaders = os.cpu_count()
    else:
        nloaders = 4
    nepochs = 1000
    lr = outhandle.getenv("LR",1e-3)

    print(f"Will use {nloaders} cpus for dataloaders.")

    model = GraphAttentionClassifier(out=outhandle).to(device)

    modelsize = count_parameters(model)
    print("Using model of size:",modelsize)
    
    # if torch.cuda.device_count() > 1:
    #     print(f"Will use {torch.cuda.device_count()} GPUs.")
    #     model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience = 15)

    criterion = nn.BCELoss()

    trainfrac = 0.8
    dataset_size = len(graph)
    batch_size = 1024*128
    val_batch_size = 1024*32
    if test:
        batch_size = 16
        val_batch_size = 16
    train_size = int(trainfrac * dataset_size)
    val_size = dataset_size - train_size
   
    train_dataset, val_dataset = random_split(graph, [train_size, val_size])
     
    train_weights = graph.weights[train_dataset.indices]
    train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=train_size, replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=nloaders, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=nloaders, pin_memory=True)

    print(f"Train data size to model size ratio: {train_size/modelsize}")
    if train_size/modelsize < 10:
        print(f"WARNING: Training data size={train_size} is less than 10 times the modelsize={modelsize}!!!")

    earlystop = 80
    reportevery = 20
    bestloss = 1e9
    bestepoch = 0
    bestmodel = None
    trainlosses = []
    vallosses = []
    bestlosses = []
    lrs = []

    print("Using batch size",batch_size)
    outdir = outhandle.outdir
    print("Working dir:", outdir)
    os.system(f"mkdir -p {outdir}/checkpoints")

    with alive_bar(nepochs,title="Epoch") as epochbar:
        for iepoch in range(nepochs):
            model.train()
            total_loss = 0.
            for data in train_loader:
                jet,jetp4,lep,lepp4,llp4,label,weight = data
                jet,jetp4,lep,lepp4,llp4,label,weight = jet.to(device),jetp4.to(device),lep.to(device),lepp4.to(device),llp4.to(device),label.to(device),weight.to(device)

                optimizer.zero_grad()
                outputs = model(jet,jetp4,lep,lepp4,llp4)
                loss = criterion(outputs[:,0], label)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.) 
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            val_loss = 0.
            for data in val_loader:
                jet,jetp4,lep,lepp4,llp4,label,weight = data
                jet,jetp4,lep,lepp4,llp4,label,weight = jet.to(device),jetp4.to(device),lep.to(device),lepp4.to(device),llp4.to(device),label.to(device),weight.to(device)
                with torch.no_grad():
                    outputs = model(jet,jetp4,lep,lepp4,llp4)
                    loss = criterion(outputs[:,0], label)

                    val_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            if avg_val_loss < bestloss:
                bestloss = avg_val_loss
                bestepoch = iepoch
                bestmodel = model.state_dict()

            scheduler.step(avg_val_loss)
            nowlr = optimizer.param_groups[-1]['lr']
            status = f"LR: {nowlr:.6f}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Best Val Loss: {bestloss:.6f}, Best epoch: {bestepoch}"

            trainlosses.append(avg_train_loss)
            vallosses.append(avg_val_loss)
            bestlosses.append(bestloss)
            lrs.append(nowlr)

            if iepoch > 0 and iepoch % reportevery == 0:
                print(status)
                dumploss(trainlosses,bestlosses,f"{outdir}/losses_inprogress.png",lrcurve=lrs,valloss=vallosses)
                torch.save(bestmodel,f"{outdir}/checkpoints/gnn_{iepoch}.pt")


            if iepoch - bestepoch > earlystop:
                print(f"Loss did not improve in {earlystop} epochs. Exiting.")
                break


            epochbar.text(status)
            epochbar()

    print(status)

    
    
    torch.save(bestmodel,f"{outdir}/gnn.pt")
    dumploss(trainlosses,bestlosses,f"{outdir}/losses.png",lrcurve=lrs,valloss=vallosses)

    # Eval
    model.load_state_dict(bestmodel)
    model.eval()
    allouts = []
    truth = []
    for data in val_loader:
        jet,jetp4,lep,lepp4,llp4,label,weight = data
        jet,jetp4,lep,lepp4,llp4,label,weight = jet.to(device),jetp4.to(device),lep.to(device),lepp4.to(device),llp4.to(device),label.to(device),weight.to(device)
        with torch.no_grad():
            outputs = model(jet,jetp4,lep,lepp4,llp4)
            allouts.append(outputs[:,0])
            truth.append(label)
    out = torch.cat(allouts,dim=0).cpu()
    yvals = torch.cat(truth,dim=0).cpu()
    return out,yvals

class outhandler():
    def __init__(self,outdir):
        self.outdir = outdir

    def getenv(self,envvarname,default):
        #Useful for hyperparameter optimization
        if os.getenv(envvarname):
            out = os.getenv(envvarname)
            print(f"Setting {envvarname} to {out}.")            
            newval = float(out)
        else:
            newval = default
        self.outdir += f"_{envvarname}{newval}"
        return newval

def dumploss(loss,minloss,outfl,lrcurve=None,valloss=None):
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(range(1,len(loss)+1), loss, label="Loss")
    valtxt = ""
    if valloss:
        ax1.plot(range(1,len(valloss)+1), valloss, label=f"Val loss")
        valtxt = "val "
    ax1.plot(range(1,len(minloss)+1), minloss, label=f"Best {valtxt}loss\n({minloss[-1]:.6f})")
    
    plt.title('Loss Curve')
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("# epoch")
    # ax1.set_yscale("log")
    ax1.legend()
    if lrcurve:        
        ax2 = ax1.twinx()
        ax2.plot(range(1,len(lrcurve)+1), lrcurve, label="LR",color = "gray", alpha=0.5)
        ax2.set_ylabel('Learning Rate', color='gray')
        ax2.set_yscale("log")
        ax2.legend(loc=9)    
    plt.tight_layout()
    plt.savefig(outfl)
    plt.close()