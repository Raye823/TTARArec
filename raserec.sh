# Collaborative-based Pre-training Stage
python run_seq.py --dataset='beauty' --gpu_id=0  --metrics="['Recall', 'NDCG', 'MRR']" --valid_metric="MRR@10" --train_batch_size=1024 --lmd=0.1 --lmd_sem=0.1 --model='DuoRec' --contrast='us_x' --sim='dot' --tau=1 --nproc=2 --epochs=100 --data_path="./recbole/dataset"
# Retrieval-Augmented Fine-tuning Stage
python run_seq.py --dataset='Amazon_Beauty' --data_path="./recbole/dataset" --pre_training_ckt="./log/DuoRec/Amazon_Beauty/bs1024-lmd0.1-sem0.1-us_x-Mar-19-2025_21-16-57-lr0.001-l20-tau1-dot-DPh0.5-DPa0.5/model.pth"
#gru4_cmd
python run_seq.py --dataset='Amazon_Beauty' --gpu_id=0 --metrics="['Recall', 'NDCG', 'MRR']" --valid_metric="MRR@10" --train_batch_size=1024 --model='GRU4Rec' --epochs=100 --data_path="./recbole/dataset"

python run_seq.py --dataset='Amazon_Beauty'  --pretrained_path="./log/DuoRec/Amazon_Beauty/bs1024-lmd0.1-sem0.1-us_x-Mar-19-2025_21-16-57-lr0.001-l20-tau1-dot-DPh0.5-DPa0.5/model.pth"

python run_ttararec.py --dataset Amazon_Beauty --pretrained_model_path "./log/DuoRec/Amazon_Beauty/bs1024-lmd0.1-sem0.1-us_x-Mar-19-2025_21-16-57-lr0.001-l20-tau1-dot-DPh0.5-DPa0.5/model.pth"

1.attention层结构,没有训练value矩阵
2.检索评分是否有必要加上mlp;适应检索任务(待验证)
3.attention结构以及应用位置编码;没什么提升
4.推荐损失权重设置为1;单设推荐损失没用
5.kl散度太大;受索引数量k值影响累加/受batchsize影响/受dropout影响
6.融合改为序列
7.知识库构建方式切分序列
8.学习率、参数数量调整
