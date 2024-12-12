import numpy as np
import matplotlib.pyplot as plt
import boost_histogram as bh
import copy
import uproot

def asimovSigSquare(s, b, b_var):
    """
    Calculate the asimov significance
    """
    term1 = (s+b)*np.log(((s+b)*(b+b_var))/(b*b+(s+b)*b_var))
    term2 = (b*b/b_var)*np.log(1+(b_var*s)/(b*(b+b_var)))
    sig = 2*(term1-term2)
    sig = np.where(b!=0, sig, 0.0)
    return(sig)

def lowerBins(bkgHist, sigHist, sigLoss=0.01, minimumSignal=0.0, epsilon=0.05, doPlot=False):
    """
    Lower the number of bins by merging while maintaining the asimov significance
    and a roughly decreasing number of signal events in each bin
    
    Arguments:
        bkgHist: boost_histogram object with the background processes
        sigHist: boost_histogram object with the signal processes
        sigLoss: amount the significance can drop with each bin merging, default
                 is 1%
        minimumSignal: minimum number of signal events in the last bin, 
                       default is 0
        epsilon: how much the monotonically decreasing nature can be violated
                 from the left, default 5%
        doPlot: whether to show the final plots, default false
    Returns:
        bkgHist: boost_histogram object with the background processes with merged bins
        sigHist: boost_histogram object with the signal processes with merged bins
    
    """
    
    # extract data from the histograms
    sigValues = sigHist.values()
    bkgValues = bkgHist.values()
    sigVariances = sigHist.variances()
    bkgVariances = bkgHist.variances()
    bins = sigHist.axes.edges[0]
    
    # get the asimov significance
    sig = asimovSigSquare(sigValues, bkgValues, bkgVariances)
    sigSum = np.cumsum(sig)
    
    # start lists with the merged bins, values, and variances
    finalBins = [bins[-1]]
    finalSigValues = []
    finalBkgValues = []
    finalSigVariances = []
    finalBkgVariances = []
    
    currentBin = len(sigValues) - 1 # starting bin
    targetSig = np.sqrt(sigSum[-1]) # initial significance
    currentSig = targetSig
    requiredSig = minimumSignal # amount of signal required in the next bin
    
    for currentBin in range(len(sigValues) - 1, 0, -1):
        
        # get the significance of the current bin and the one to merge into
        binSigSquare = asimovSigSquare(sigValues[currentBin], bkgValues[currentBin], bkgVariances[currentBin])    
        nextBinSigSquare = asimovSigSquare(sigValues[currentBin - 1], bkgValues[currentBin - 1], bkgVariances[currentBin - 1])
        
        # Get the counts, variance, and significance for the proposed new bin
        mergedS = sigValues[currentBin] + sigValues[currentBin - 1]
        mergedB = bkgValues[currentBin] + bkgValues[currentBin - 1]
        mergedBVar = bkgVariances[currentBin] + bkgVariances[currentBin - 1] 
        mergedSigSquare = asimovSigSquare(mergedS, mergedB, mergedBVar)
        
        # find the new significance
        totalMergedSignificance = np.sqrt(currentSig*currentSig - binSigSquare - nextBinSigSquare + mergedSigSquare)
        
        # Don't merge these bins if there is too much significance loss but there is enough signal if we don't merge
        if(totalMergedSignificance < targetSig*(1-sigLoss) and sigValues[currentBin]>requiredSig*(1-epsilon)):
            targetSig = currentSig # Update the target significance
            
            # Store the current histogram values
            finalBins.append(bins[currentBin])
            finalSigValues.append(sigValues[currentBin])
            finalBkgValues.append(bkgValues[currentBin])
            finalSigVariances.append(sigVariances[currentBin])
            finalBkgVariances.append(bkgVariances[currentBin])
            
            # Update the required significance
            requiredSig = max(sigValues[currentBin], requiredSig)
        else: # Merge if not enough signal or there is minimal significance loss
            # update the target significance is this increases the significance    
            if(totalMergedSignificance>targetSig):
                targetSig = totalMergedSignificance
                
            # Update current significance and merge the bins
            currentSig = totalMergedSignificance
            
            bkgValues[currentBin - 1] += bkgValues[currentBin]
            sigValues[currentBin - 1] += sigValues[currentBin]
            
            bkgVariances[currentBin - 1] += bkgVariances[currentBin]
            sigVariances[currentBin - 1] += sigVariances[currentBin]
    
    # Store the last bin and reverse the lists into the correct order 
    finalBins.append(bins[0])
    finalSigValues.append(sigValues[0])
    finalBkgValues.append(bkgValues[0])
    finalSigVariances.append(sigVariances[0])
    finalBkgVariances.append(bkgVariances[0])
    
    
    finalBins = np.flip(finalBins)
    finalSigValues= np.flip(finalSigValues)
    finalBkgValues = np.flip(finalBkgValues)
    finalSigVariances = np.flip(finalSigVariances)
    finalBkgVariances = np.flip(finalBkgVariances)
    
    # Store in histogram objects
    bkgHist = bh.Histogram(bh.axis.Variable(finalBins), storage=bh.storage.Weight())
    sigHist = bh.Histogram(bh.axis.Variable(finalBins), storage=bh.storage.Weight())
    bkgHist[...] = np.stack([finalBkgValues, finalBkgVariances], axis=-1)
    sigHist[...] = np.stack([finalSigValues, finalSigVariances], axis=-1)
    
    # Plot the results if requested
    if(doPlot):
        plt.figure()
        plt.stairs(sigHist.values(),edges=range(len(sigHist.values())+1), label="Signal",linewidth=1.5, color="r")
        plt.stairs(bkgHist.values(),edges=range(len(bkgHist.values())+1), label="Background",linewidth=1.5, color="b")
        plt.errorbar(np.linspace(0.5, len(sigHist.values())-0.5,len(sigHist.values())), sigHist.values(), yerr=np.sqrt(sigHist.variances()), ls="none")
        plt.errorbar(np.linspace(0.5, len(bkgHist.values())-0.5,len(bkgHist.values())), bkgHist.values(), yerr=np.sqrt(bkgHist.variances()), ls="none")
        plt.yscale("log")
        plt.legend()
        plt.savefig("rebinplot1.png")
    
    return(bkgHist, sigHist)
    

def lowerVariance(bkgHist, sigHist, targetUncert, doPlot=False):
    """
    Merge neighboring bins to makes sure the uncertainty is properly low 
    everywhere
    
    Arguments:
        bkgHist: boost_histogram object with the background processes
        sigHist: boost_histogram object with the signal processes
        targetUncert: maximum value for bkg bin uncertainty over bkg bin counts
        doPlot: whether to show the final plots, default false
    Returns:
        bkgHist: boost_histogram object with the background processes with merged bins
        sigHist: boost_histogram object with the signal processes with merged bins
    
    """
    
    # extract data from the histograms
    sigValues = sigHist.values()
    bkgValues = bkgHist.values()
    sigVariances = sigHist.variances()
    bkgVariances = bkgHist.variances()
    bins = sigHist.axes.edges[0]
    
    # Plot the input histograms if requested
    if(doPlot):
        plt.figure()
        plt.stairs(sigHist.values(),edges=range(len(sigHist.values())+1), label="Signal",linewidth=1.5, color="r")
        plt.stairs(bkgHist.values(),edges=range(len(bkgHist.values())+1), label="Background",linewidth=1.5, color="b")
        plt.errorbar(np.linspace(0.5, len(sigHist.values())-0.5,len(sigHist.values())), sigHist.values(), yerr=np.sqrt(sigHist.variances()), ls="none")
        plt.errorbar(np.linspace(0.5, len(bkgHist.values())-0.5,len(bkgHist.values())), bkgHist.values(), yerr=np.sqrt(bkgHist.variances()), ls="none")
        plt.yscale("log")
        plt.legend()
        plt.savefig("rebinplot2.png")
        
    # start lists with the merged bins, values, and variances
    finalBins = [bins[-1]]
    finalSigValues = []
    finalBkgValues = []
    finalSigVariances = []
    finalBkgVariances = []
    
    currentBin = len(sigValues) - 1 # starting bin
    
    for currentBin in range(len(sigValues)-1, -1, -1):
        # calculate the uncertainty of this bin
        binUncert = np.sqrt(bkgVariances[currentBin])/bkgValues[currentBin]
        
        # if the bin has too large an uncertainty, or there are <=0 signal or 
        # background events, merge the bins
        if(currentBin>0 and (binUncert > targetUncert or bkgValues[currentBin]<=0 or sigValues[currentBin]<=0)):
            
            # merge bins
            bkgValues[currentBin - 1] += bkgValues[currentBin]
            sigValues[currentBin - 1] += sigValues[currentBin]
            
            bkgVariances[currentBin - 1] += bkgVariances[currentBin]
            sigVariances[currentBin - 1] += sigVariances[currentBin]
            
        # This bin passes, store it
        else:
            finalBins.append(bins[currentBin])
            finalSigValues.append(sigValues[currentBin])
            finalBkgValues.append(bkgValues[currentBin])
            finalSigVariances.append(sigVariances[currentBin])
            finalBkgVariances.append(bkgVariances[currentBin])
    
    # Reverse the lists into the correct order
    finalBins = np.flip(finalBins)
    finalSigValues= np.flip(finalSigValues)
    finalBkgValues = np.flip(finalBkgValues)
    finalSigVariances = np.flip(finalSigVariances)
    finalBkgVariances = np.flip(finalBkgVariances)
    
    # Store in histogram objects
    bkgHist = bh.Histogram(bh.axis.Variable(finalBins), storage=bh.storage.Weight())
    sigHist = bh.Histogram(bh.axis.Variable(finalBins), storage=bh.storage.Weight())
    bkgHist[...] = np.stack([finalBkgValues, finalBkgVariances], axis=-1)
    sigHist[...] = np.stack([finalSigValues, finalSigVariances], axis=-1)
    
    # Plot the results if requested
    if(doPlot):
        plt.figure()
        plt.stairs(sigHist.values(),edges=range(len(sigHist.values())+1), label="Signal",linewidth=1.5, color="r")
        plt.stairs(bkgHist.values(),edges=range(len(bkgHist.values())+1), label="Background",linewidth=1.5, color="b")
        plt.errorbar(np.linspace(0.5, len(sigHist.values())-0.5,len(sigHist.values())), sigHist.values(), yerr=np.sqrt(sigHist.variances()), ls="none")
        plt.errorbar(np.linspace(0.5, len(bkgHist.values())-0.5,len(bkgHist.values())), bkgHist.values(), yerr=np.sqrt(bkgHist.variances()), ls="none")
        plt.yscale("log")
        plt.legend()
        plt.savefig("rebinplot3.png")
    
    return(bkgHist, sigHist)


    

def doRebin(fileName, regionDirectories, signalProcesses, targetUncert=0.3,
            sigLoss=0.01, minimumSignal=0, epsilon=0.05, doPlot=False):
    """
    Opens a root file, find the signal and background histograms in a
    particular directory, and runs the rebinning algorithm on them
    
    Arguments:
        fileName: root file to read from
        regionDirectories: list of directories to read from
        signalProcesses: list of strings corresponding to the signal process
        targetUncert: maximum value for bkg bin uncertainty over bkg bin counts,
                      default is 0.3
        sigLoss: amount the significance can drop with each bin merging, 
                 default is 1%
        minimumSignal: minimum number of signal events in the last bin, 
                       default is 0
        epsilon: how much the monotonically decreasing nature can be violated
                 from the left, default 5%
        doPlot: whether to show the final plots, default false
    Returns:
        mergeDict: dictionary mapping from directories to rebinnings
    
    """
    # dictionary with the rebinning
    mergeDict = dict()
    with uproot.open(fileName) as rf:
        bkg = None
        sig = None
        # check each key in the root file
        for key in rf.keys():
            
            # Don't want data or the top directories
            if("data" in key or "/" not in key):
                continue
            regionFound = False
            
            # Check if this key is in one of the directories we want
            for region in regionDirectories:
                if(region in key):
                    regionFound = True
            if(not regionFound):
                continue
                
            # Check if it's a signal process or background process
            # Get the boost histogram
            sigProcess = False
            for proc in signalProcesses:
                if(proc in key):
                    sigProcess = True
            if(sigProcess):
                if(sig is None):
                    sig = rf[key].to_boost()
                else:
                    sig += rf[key].to_boost()
            elif(bkg is None):
                bkg = rf[key].to_boost()
            else:
                bkg += rf[key].to_boost()
                
        # get the original bins, then do the rebinning
        originalBins = bkg.axes.edges[0]
        bkg, sig = lowerVariance(bkg, sig, targetUncert, doPlot=doPlot)
        bkg, sig = lowerBins(bkg, sig, sigLoss, epsilon=epsilon, doPlot=doPlot)
        newBins = bkg.axes.edges[0]
        
        # Find the bin merging locations
        binMerging = np.searchsorted(originalBins, newBins)
        
        # Store these mreged bins for each relavent region 
        for val in regionDirectories:
            mergeDict[val] = binMerging
    return(mergeDict)



def doRebinDict(histDictOrig, regionDirectories, signalProcesses, targetUncert=0.3,
                sigLoss=0.01, minimumSignal=0, epsilon=0.05, doPlot=False):
    """
    Load a dictionary mimicing a root file, find the signal and background 
    histograms in a particular directory, and runs the rebinning algorithm on 
    them
    
    Arguments:
        histDict: root file to read from
        regionDirectories: list of directories to read from
        signalProcesses: list of strings corresponding to the signal process
        targetUncert: maximum value for bkg bin uncertainty over bkg bin counts,
                      default is 0.3
        sigLoss: amount the significance can drop with each bin merging, 
                 default is 1%
        minimumSignal: minimum number of signal events in the last bin, 
                       default is 0
        epsilon: how much the monotonically decreasing nature can be violated
                 from the left, default 5%
        doPlot: whether to show the final plots, default false
    Returns:
        mergeDict: dictionary mapping from directories to rebinnings
    
    """

    histDict = copy.deepcopy(histDictOrig)

    print("Will rebin:",regionDirectories)

    # Make sure it is formated correctly
    for x in range(len(regionDirectories)):
        reg = regionDirectories[x].strip()
        if(reg[-1]!="/"):
            reg+="/"
        regionDirectories[x] = reg
    # dictionary with the rebinning
    mergeDict = dict()
    bkg = None
    sig = None
    # check each key in the root file
    for key in histDict.keys():

        # Don't want data or the top directories
        if("data" in key or "/" not in key or "nominal" not in key):
            continue
        regionFound = False
        
        # Check if this key is in one of the directories we want
        for region in regionDirectories:
            if(region in key):
                regionFound = True
        if(not regionFound):
            continue
            
        # Check if it's a signal process or background process
        # Get the boost histogram
        sigProcess = False
        for proc in signalProcesses:
            if(proc in key):
                sigProcess = True
        if(sigProcess):

            print("\tSig:", key)
            if(sig is None):
                sig = histDict[key]
            else:
                sig += histDict[key]
        elif(bkg is None):

            print("\tBkg:", key)
            bkg = histDict[key]
        else:
            print("\tBkg:", key)
            bkg += histDict[key]
            
    # get the original bins, then do the rebinning
    originalBins = bkg.axes.edges[0]
    bkg, sig = lowerVariance(bkg, sig, targetUncert, doPlot=doPlot)
    bkg, sig = lowerBins(bkg, sig, sigLoss, epsilon=epsilon, doPlot=doPlot)
    newBins = bkg.axes.edges[0]
    
    # Find the bin merging locations
    binMerging = np.searchsorted(originalBins, newBins)
    
    # Store these mreged bins for each relavent region 
    for val in regionDirectories:
        mergeDict[val] = binMerging

    return mergeDict

def rebinHist(histogram, bins):
    """
    Merge neighboring bins to makes sure the uncertainty is properly low 
    everywhere
    
    Arguments:
        histogram: boost_histogram object to rebin
        bins: New bin edges
    Returns:
        histogram: rebinned boost_histogram object
    """
    
    # Get histogram values and variances
    values = histogram.values()
    variances = histogram.variances()
    
    # Arrays for new values and variances
    newValues = []
    newVariances = []
    for x in range(1,len(bins)):
        # Slice and sum the values and variances to merge bins
        newValues.append(np.sum(values[bins[x-1]:bins[x]]))
        newVariances.append(np.sum(variances[bins[x-1]:bins[x]]))
        
    # Make new histogram object and store the new values and variances
    histogram = bh.Histogram(bh.axis.Variable(bins), storage=bh.storage.Weight())
    histogram[...] = np.stack([newValues, newVariances], axis=-1)
    return(histogram)

def writeMergedBins(inputFileName, outputFileName, binDict):
    """
    Copy histograms from inputFile to outputFile. If a directory shows up in 
    binDict rebin the histograms first.

    Arguments:    
        inputFileName: string with name of root file to read from 
        outputFileName : string with name of root file to create
        binDict: dictionary mapping from directories to rebinnings
    """
    with uproot.open(inputFileName) as rf:
        with uproot.recreate(outputFileName) as wf:
            for key in rf:
                if("/") in key:
                    directory = key.split("/")[0]+"/"
                    if(directory in binDict):
                        wf[key] = rebinHist(rf[key].to_boost(), binDict[directory])
                    else:
                        wf[key] = rf[key]
         
def main():
        
    fileName = "vhcc_shapes_2022_preEE_2L.root"
    mergedFileName = "vhcc_shapes_2022_preEE_2L_rebin.root"
    regionDirectories = [["2022_preEE_Zmm_SR_hiZPT/","2022_preEE_Zee_SR_hiZPT/"],["2022_preEE_Zmm_SR_loZPT/","2022_preEE_Zee_SR_loZPT/"],["2022_preEE_Zll_SR/"]]
    signalProcesses = [["ZH_hcc"],["ZH_hcc"],["ZH_hcc"]]
    # do the bin merging
    mergeDict = dict()
    for x in range(len(regionDirectories)):
        # each call is one root file, as list of regions, a list of signal processes, and then the optional keyword arguments
        mergeDict.update(doRebin(fileName, regionDirectories[x], signalProcesses[x], doPlot=False))
    # Display the results
    print(mergeDict)
    # Save the bin merging
    writeMergedBins(fileName, mergedFileName, mergeDict)
    
if __name__ == "__main__":
    main()