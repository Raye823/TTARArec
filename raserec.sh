# Collaborative-based Pre-training Stage
python run_seq.py --dataset='Amazon_Sports_and_Outdoors' --gpu_id=0  --metrics="['Recall', 'NDCG', 'MRR']" --valid_metric="MRR@10" --train_batch_size=1024 --lmd=0.1 --lmd_sem=0.1 --model='DuoRec' --contrast='us_x' --sim='dot' --tau=1 --nproc=2 --epochs=100 --data_path="./recbole/dataset"
# Retrieval-Augmented Fine-tuning Stage
python run_seq.py --dataset='Amazon_Sports_and_Outdoors'  --nprobe=1 --attn_tau=1.0 --dropout_rate=0.5 --alpha=0.5 --beta=1.0 --top_k=10 --metrics="['Recall', 'NDCG']" --valid_metric="Recall@10"  --train_batch_size=1024 --model='RaSeRec' --sim='dot' --tau=1 --nproc=2 --epochs=100 --data_path="./recbole/dataset" --pre_training_ckt="./log/DuoRec/Amazon_Sports_and_Outdoors/bs1024-lmd0.1-sem0.1-us_x-Sep-07-2025_18-38-29-lr0.001-l20-tau1-dot-DPh0.5-DPa0.5/model.pth"
#gru4
python run_seq.py --dataset='Amazon_Sports_and_Outdoors' --gpu_id=0 --metrics="['Recall', 'NDCG', 'MRR']" --valid_metric="MRR@10" --train_batch_size=1024 --model='GRU4Rec' --epochs=100 --data_path="./recbole/dataset"
#TTAREc
python run_ttararec.py --dataset Amazon_Beauty --pretrained_model_path "./log/DuoRec/Amazon_Beauty/bs1024-lmd0.1-sem0.1-us_x-Mar-19-2025_21-16-57-lr0.001-l20-tau1-dot-DPh0.5-DPa0.5/model.pth"
python run_ttararec.py --dataset Amazon_Sports_and_Outdoors --pretrained_model_path "./log/DuoRec/Amazon_Sports_and_Outdoors/bs1024-lmd0.1-sem0.1-us_x-Sep-07-2025_18-38-29-lr0.001-l20-tau1-dot-DPh0.5-DPa0.5/model.pth"
