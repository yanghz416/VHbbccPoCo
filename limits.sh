set -e

cd ..
indir=$1
era=2022_postEE
channel=$2
python VHccPoCo/scripts/convertToRoot.py $indir/output_all.coffea -c VHccPoCo/params/shapemaker_vhcc_run3/$channel.yaml -e $era
cd CMSSW_14_1_0_pre4/src/
. /cvmfs/cms.cern.ch/cmsset_default.sh
cmsenv
cd CombineHarvester/VHccCoHa
mv -f ../../../../vhcc*$era*.root .
python3 scripts/vhcc_cards.py -c $channel -y $era
ulimit -s unlimited

if [ "$channel" == "Zll" ]; then
  dirname='{Zmm,Zee}'
elif [ "$channel" == "Wln" ]; then
  dirname='W'
elif [ "$channel" == "Znn" ]; then
  dirname='Znn'
else
  echo "Error: Unknown channel value '$channel'."
  exit 1
fi

expanded_files=$(eval echo "output/vhcc_Run3_$era/vhcc_$dirname*.txt")
combineTool.py -M T2W --cc combined.txt -o output/vhcc_Run3_$era/ws$channel$era.root -i $expanded_files
combineTool.py -M AsymptoticLimits -d output/vhcc_Run3_$era/ws$channel$era.root --there --run blind

#Full
combineTool.py -M T2W --cc combined.txt -o output/vhcc_Run3_$era/wsfull_$era.root -i output/vhcc_Run3_$era/vhcc_*.txt &> /dev/null &
combineTool.py -M AsymptoticLimits -d output/vhcc_Run3_$era/wsfull_$era.root --there --run blind
