#SASRec
python run_basic.py --dataset Amazon_Beauty --config_file sasrec.yaml --model SASRec
#DuoRec
python run_basic.py --dataset Amazon_Beauty --config_file duorec.yaml --model DuoRec
#TTAREc(+DuoRec)
python run_read.py -d Amazon_Office -pt duorec -pp pretrainmodel/DuoRec/Amazon_Office/model.pth
python run_read.py -d Amazon_Beauty -pt duorec -pp pretrainmodel/DuoRec/Amazon_Beauty/model.pth
python run_read.py -d Amazon_Sports_and_Outdoors -pt duorec -pp pretrainmodel/DuoRec/Amazon_Sports_and_Outdoors/model.pth
python run_read.py -d Amazon_Home -pt duorec -pp pretrainmodel/DuoRec/Amazon_Home/model.pth
python run_read.py -d ml-1m -pt duorec -pp pretrainmodel/DuoRec/ml-1m/model.pth
#EVALUATE
python run_read.py -mp pretrainmodel/TTARARec/Amazon_Office/model.pth -e
python run_read.py -mp pretrainmodel/TTARARec/Amazon_Beauty/model.pth -e
python run_read.py -mp pretrainmodel/TTARARec/Amazon_Sports_and_Outdoors/model.pth -e
python run_read.py -mp pretrainmodel/TTARARec/Amazon_Home/model.pth -e
python run_read.py -mp pretrainmodel/TTARARec/ml-1m/model.pth -e