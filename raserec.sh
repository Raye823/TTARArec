#DUORec
python run_basic.py --dataset Amazon_Beauty
python run_basic.py --dataset Amazon_Sports_and_Outdoors
#SASRec
python run_basic.py --dataset Amazon_Beauty --config_file sasrec.yaml --model SASRec
python run_basic.py --dataset Amazon_Sports_and_Outdoors --config_file sasrec.yaml --model SASRec
#TTAREc
python run_ttararec.py -d Amazon_Beauty -pt duorec -pp pretrainmodel/DuoRec/Amazon_Beauty/model.pth
python run_ttararec.py -d Amazon_Sports_and_Outdoors -pt duorec -pp pretrainmodel/DuoRec/Amazon_Sports_and_Outdoors/model.pth
#EVALUATE
python run_ttararec.py -mp pretrainmodel/TTARARec/Amazon_Beauty/model.pth -e