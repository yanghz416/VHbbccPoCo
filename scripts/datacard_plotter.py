import uproot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import copy
from functools import partial

def extendarr(arr):
    return np.append(arr,arr[-1])

def plot_histograms(root_file_path, config, eras, categ_to_var,plotdir="plot_datacards", samplelist=None,brstr="Shape_"):
    """
    Generate CMS-style stack plots for histograms directly from a ROOT file.

    Parameters:
    - root_file_path: Path to the ROOT file containing histograms.
    - config: Configuration dictionary from the YAML file.
    - eras: List of eras (e.g., ["2022_preEE"]).
    - categ_to_var: Mapping of categories to variables and names.
    """
    signal_scaling = config["plotting"].get("signal_scaling", 1)
    signal_scaling_cc = config["plotting"].get("signal_scaling_cc", 100)
    blinding_config = config["plotting"].get("blinding", {})
    flatx = config["plotting"].get("flatx", [])

    with uproot.open(root_file_path) as root_file:
        for era in eras:
            for cat, var_and_name in categ_to_var.items():
                variable, newCatName = var_and_name
                blinded_region = blinding_config.get(cat, {})

                # Collect histograms
                if samplelist is None:
                    stack_components = {"TT": None, "VJet": None, "VV": None, "ZH_hbb": None, "ZH_hcc": None}
                else:
                    stack_components = {i:None for i in samplelist}
                signal_hist = None
                signal_cc_hist = None
                data_hist = None

                # Load histograms for stack components
                for proc in stack_components.keys():
                    hist_name = f"{era}_{newCatName}/{proc}_{brstr}nominal"
                    if hist_name in root_file:
                        hist = root_file[hist_name]
                        values, edges = hist.to_numpy(flow=False)

                        if variable in flatx: edges = range(len(edges))
                        stack_components[proc] = copy.deepcopy((values, edges))

                # Load signal histogram
                bbsignals = ["ZH_hbb","ggZH_hbb","WH_hbb"]
                signal_hist = None
                for signame in bbsignals:
                    signal_name = f"{era}_{newCatName}/{signame}_{brstr}nominal"
                    if signal_name in root_file:
                        hist = root_file[signal_name]
                        values, edges = hist.to_numpy(flow=False)

                        if variable in flatx: edges = range(len(edges))
                        if signal_hist is None:
                            signal_hist = [np.array(values), copy.deepcopy(edges)]

                        else:
                            signal_hist[0] += np.array(values)
                    
                # Load signal histogram
                ccsignals = ["ZH_hcc","ggZH_hcc","WH_hcc"]
                signal_cc_hist = None
                for signame in ccsignals:
                    signal_cc_name = f"{era}_{newCatName}/{signame}_{brstr}nominal"
                    if signal_cc_name in root_file:
                        hist = root_file[signal_cc_name]
                        values, edges = hist.to_numpy(flow=False)
                        
                        if variable in flatx: edges = range(len(edges))
                        if signal_cc_hist is None:
                            signal_cc_hist = [np.array(values), copy.deepcopy(edges)]
                        else:
                            signal_cc_hist[0] += np.array(values)

                # Load data histogram
                data_name = f"{era}_{newCatName}/data_obs_{brstr}nominal"
                if data_name in root_file:
                    hist = root_file[data_name]
                    values, edges = hist.to_numpy(flow=False)      
                    # Blinding logic
                    if blinded_region and variable in blinded_region:
                        lower, upper = blinded_region[variable]
                        lower = float(lower) if isinstance(lower, (int, str)) else lower
                        upper = float(upper) if isinstance(upper, (int, str)) else upper
                        mask = (edges[:-1] > (lower if lower is not None else -float('inf'))) & \
                                (edges[:-1] < (upper if upper is not None else float('inf')))
                        values[mask] = 0  # Blind the region
                        
                    if variable in flatx: edges = range(len(edges))
                    data_hist = copy.deepcopy((values, edges))

                # Prepare plot
                fig = plt.figure(figsize=(6, 6))
                gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
                ax_main = fig.add_subplot(gs[0])
                ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

                bottoms = None

                # Plot stack components
                colors = {"TT": "#00BFFF", "VJet": "#f5f768", "VV": "#91bfdb", "ZH_hbb": "#ff0000", "ZH_hcc": "#b700ff","ggZH_hbb": "#ff0000", "ggZH_hcc": "#b700ff","WH_hbb": "#ff0000", "WH_hcc": "#b700ff",
                    "WW": "#FFA07A",
                    "WZ": "#FFA07A",
                    "ZZ": "#FFA07A",
                    "QCD": "#D3D3D3",
                    "ST": "#2B193D",

                    "Zj_bj":  "#196f3d",
                    "Zj_cj":  "#52be80",
                    "Zj_ll":  "#ADFF2F",
                    
                    "WJetsToLNu_FxFx": "#F6D608",
                    "ZJetsToNuNu_FxFx": "#ADFF2F"}

                bottoms = None
                stack_uncertainty = None  # Initialize stack uncertainty
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] + [cm.viridis(i) for i in range(0, 256, 26)]
                colid = 0
                for proc, hist_data in stack_components.items():
                    if hist_data is not None:
                        values, edges = hist_data
                        bin_centers = edges[:-1] + np.diff(edges) / 2  # Compute bin centers
                        bin_widths = np.diff(edges)  # Compute bin widths

                        # Use bin centers for alignment with signal and data
                        if any([c in proc for c in colors]):
                            matchedproc = [c for c in colors if c in proc][0]
                            col = colors[matchedproc]
                        else:
                            col = color_cycle[colid]
                            colid += 1
                        lab = proc

                        ax_main.bar(bin_centers, values, width=bin_widths, bottom=bottoms, color=col, label=lab)
                        if bottoms is None:
                            bottoms = values
                        else:
                            bottoms += values
                            
                # Calculate stack uncertainties (upper and lower bounds)
                stack_uncertainty = np.sqrt(bottoms)            #TODO: This is not right; fix it with per sample Poisson
                maxMC = np.max(bottoms)

                # Plot statistical uncertainty band on the upper plot
                ax_main.fill_between(edges, extendarr(bottoms - stack_uncertainty), extendarr(bottoms + stack_uncertainty), 
                                     color="gray", alpha=0.3, step="post", label="Stat. Unc.")

                # Plot signal
                if signal_hist:
                    values, edges = signal_hist
                    values *= signal_scaling
                    ax_main.step(edges, extendarr(values), where="post", color="#ff0000", linestyle="--", label=f"Signal x{signal_scaling}")
                    
                if signal_cc_hist:
                    values, edges = signal_cc_hist
                    values *= signal_scaling_cc

                    ax_main.step(edges, extendarr(values), where="post", color="#b700ff", linestyle="--", label=f"Signal x{signal_scaling_cc}")

                # Plot data
                if data_hist:
                    values, edges = data_hist
                    bin_centers = edges[:-1] + np.diff(edges) / 2
                    ax_main.errorbar(bin_centers, values, yerr=np.sqrt(values), fmt="o", color="black", label="Data")
                # Ratio plot: Data/MC
                if data_hist and bottoms is not None:
                    data_values, edges = data_hist
                    bin_centers = edges[:-1] + np.diff(edges) / 2
                    
                    # Ensure no division by zero
                    safe_bottoms = np.where(bottoms <= 0, 1e-10, bottoms)
                    ratio = data_values / safe_bottoms
                    ratio_unc = np.sqrt(data_values) / safe_bottoms  # Uncertainty in the ratio
                    
                    # Statistical uncertainty band
                    stack_uncertainty = np.sqrt(bottoms)
                    ratio_band = stack_uncertainty / safe_bottoms

                    # Plot the ratio
                    ax_ratio.errorbar(bin_centers, ratio, yerr=ratio_unc, fmt="o", color="black", label="Data/MC")
                    # Plot uncertainty band
                    ax_ratio.fill_between(edges, extendarr(1 - ratio_band), extendarr(1 + ratio_band), 
                                          color="gray", alpha=0.3, step="post", label="Stat. Unc.")
                    
                    ax_ratio.axhline(1, color="gray", linestyle="--")  # Reference line at ratio=1
                    ax_ratio.set_ylabel("Data/MC")
                    ax_ratio.set_ylim(0.5, 1.5)  # Adjust y-axis limits for better visualization
                    ax_ratio.grid(axis="y", linestyle="--", alpha=0.5)

                # Formatting for the main plot
                ax_main.set_ylabel("Events")
                ax_main.legend(ncol=3,fontsize=8,loc="best")

                ax_main.grid(axis="y", linestyle="--", alpha=0.5)
                ax_main.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels on the upper plot
                ax_main.set_ylim(bottom=0,top=maxMC*1.3)

                # Formatting for the ratio plot
                ax_ratio.set_xlabel(variable + " (bin #)" if variable in flatx else "")
                ax_ratio.set_ylabel("Data/MC")
                ax_ratio.grid(axis="y", linestyle="--", alpha=0.5)

                # CMS text
                ax_main.text(0, 1.02, "CMS", fontsize=16, fontweight="bold", transform=ax_main.transAxes)
                ax_main.text(0.2, 1.02, "Private Work", fontsize=12, style="italic", transform=ax_main.transAxes)
                ax_main.text(1, 1.02, f"{era}: {newCatName}", fontsize=12, ha="right", transform=ax_main.transAxes)

                # Save plot
                plot_file = f"{plotdir}/stack_plot_{newCatName}_{era}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                print(f"Saved plot: {plot_file}")

                # Set y-axis to log scale
                ax_main.set_yscale("log")
                ax_main.set_ylim(bottom=0.1,top=maxMC*30)  # Ensure the lower limit is positive for log scale
                ax_main.grid(axis="y", linestyle="--", alpha=0.5)  # Update grid for log scale

                # Save log scale plot
                plot_file_log = f"{plotdir}/stack_plot_{newCatName}_{era}_log.png"
                plt.savefig(plot_file_log, dpi=300, bbox_inches="tight")
                print(f"Saved log-scale plot: {plot_file_log}")

                plt.close()
