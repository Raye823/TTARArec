# @Time   : 2020/9/16
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

"""
recbole.data.sequential_dataset
###############################
"""

import copy

import numpy as np

from recbole.data.dataset import Dataset


class SequentialDataset(Dataset):
    """:class:`SequentialDataset` is based on :class:`~recbole.data.dataset.dataset.Dataset`,
    and provides augmentation interface to adapt to Sequential Recommendation,
    which can accelerate the data loader.

    Attributes:
        uid_list (numpy.ndarray): List of user id after augmentation.

        item_list_index (numpy.ndarray): List of indexes of item sequence after augmentation.

        target_index (numpy.ndarray): List of indexes of target item id after augmentation.

        item_list_length (numpy.ndarray): List of item sequences' length after augmentation.

    """

    def __init__(self, config):
        super().__init__(config)

    def prepare_data_augmentation(self):
        """Augmentation processing for sequential dataset.

        For SASRec model: Align with TTA4SR - each user generates exactly 3 samples.
        For other models: Use original data augmentation strategy.
        
        TTA4SR style (for user with sequence [i1, i2, i3, i4, i5, i6, i7]):
        - Train sample:  input=[i1, i2, i3, i4]       target=i5  (items[:-3] -> target=items[-3])
        - Valid sample:  input=[i1, i2, i3, i4, i5]    target=i6  (items[:-2] -> target=items[-2])
        - Test sample:   input=[i1, i2, i3, i4, i5, i6] target=i7  (items[:-1] -> target=items[-1])

        Each user sequence generates EXACTLY 3 samples, no sliding window.
        """
        # 检查模型类型
        model_name = self.config['model'].lower()
        
        if model_name == 'sasrec':
            self.logger.debug('prepare_data_augmentation - ttararec)')
            self._prepare_ttararec_augmentation()
        elif model_name == 'ttararec':
            self.logger.debug('prepare_data_augmentation - ttararec)')
            self._prepare_ttararec_augmentation()
        else:
            self.logger.debug('prepare_data_augmentation - 原始模式')
            self._prepare_original_augmentation()

    def _prepare_ttararec_augmentation(self):
        """每个用户固定生成最后生成3个样本，不使用滑动窗口
        
        核心差异：
        1. 不使用滑动窗口
        2. 每个用户固定生成3个样本用于train/valid/test
        3. 这样训练样本数 = 用户数（而非滑动窗口的N倍）
        """
        self._check_field('uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        
        # 按用户分组，收集每个用户的所有交互索引
        user_interactions = {}
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if uid not in user_interactions:
                user_interactions[uid] = []
            user_interactions[uid].append(i)
        
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        
        for uid, indices in user_interactions.items():
            seq_len = len(indices)
            
            # 至少需要5个交互才能生成3个样本 (train需要至少4个历史+1个target)
            if seq_len < 5:
                self.logger.debug(f'用户 {uid} 交互数不足5，跳过')
                continue
            
            # 🔧 对齐TTA4SR: 生成恰好3个样本
            # Sample 1 (for train): input=indices[:-3], target=indices[-3]
            # Sample 2 (for valid): input=indices[:-2], target=indices[-2]  
            # Sample 3 (for test):  input=indices[:-1], target=indices[-1]
            
            for offset in [3, 2, 1]:
                # target是倒数第offset个
                target_idx = indices[-offset]
                
                # input是从开始到target之前的所有
                input_len = len(indices) - offset  # indices[:-offset]的长度
                
                # 如果input序列太长，截取最后max_item_list_len个
                if input_len > max_item_list_len:
                    # 取最后max_item_list_len个作为输入
                    seq_start = indices[input_len - max_item_list_len]
                    seq_end = target_idx
                else:
                    # 序列不长，全部作为输入
                    seq_start = indices[0]
                    seq_end = target_idx
                
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, seq_end))
                target_index.append(target_idx)
                item_list_length.append(seq_end - seq_start)

        self.uid_list = np.array(uid_list)
        self.item_list_index = np.array(item_list_index)
        self.target_index = np.array(target_index)
        self.item_list_length = np.array(item_list_length, dtype=np.int64)
        self.mask = np.ones(len(self.inter_feat), dtype=np.bool_)
        
        num_users = len(user_interactions)
        self.logger.info(f'TTA4SR对齐数据增强完成: {num_users}个用户 -> {len(self.uid_list)}个样本 (每用户3个)')
        self.logger.info(f'平均每用户样本数: {len(self.uid_list) / max(num_users, 1):.2f}')

    def _prepare_original_augmentation(self):
        """原始的数据增强策略：保持兼容性"""
        self._check_field('uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        self.uid_list = np.array(uid_list)
        self.item_list_index = np.array(item_list_index)
        self.target_index = np.array(target_index)
        self.item_list_length = np.array(item_list_length, dtype=np.int64)
        self.mask = np.ones(len(self.inter_feat), dtype=np.bool_)

    def semantic_augmentation(self, target_index):
        aug_path = self.config['data_path'] + '/semantic_augmentation.npy'
        import os
        if os.path.exists(aug_path):
            same_target_index = np.load(aug_path, allow_pickle=True)
        else:
            print(f"[DEBUG] 开始语义增强计算，target_index长度: {len(target_index)}")
            target_item = self.inter_feat['item_id'][target_index].numpy()
            
            # 优化版本：使用字典预分组，避免O(n²)复杂度
            item_to_indices = {}
            for index, item_id in enumerate(target_item):
                if item_id not in item_to_indices:
                    item_to_indices[item_id] = []
                item_to_indices[item_id].append(index)
            
            same_target_index = []
            for index, item_id in enumerate(target_item):
                all_indices = item_to_indices[item_id]
                # 移除当前索引
                same_indices = [idx for idx in all_indices if idx != index]
                same_target_index.append(np.array(same_indices))
            
            same_target_index = np.array(same_target_index, dtype=object)
            print(f"[DEBUG] 语义增强计算完成，保存到: {aug_path}")
            np.save(aug_path, same_target_index)
        
        return same_target_index

    def leave_one_out(self, group_by, leave_one_num=1):
        self.logger.debug(f'Leave one out, group_by=[{group_by}], leave_one_num=[{leave_one_num}].')
        if group_by is None:
            raise ValueError('Leave one out strategy require a group field.')
        if group_by != self.uid_field:
            raise ValueError('Sequential models require group by user.')

        self.prepare_data_augmentation()
        grouped_index = self._grouped_index(self.uid_list)
        next_index = self._split_index_by_leave_one_out(grouped_index, leave_one_num)

        self._drop_unused_col()
        next_ds = []
        for index in next_index:
            ds = copy.copy(self)
            for field in ['uid_list', 'item_list_index', 'target_index', 'item_list_length']:
                setattr(ds, field, np.array(getattr(ds, field)[index]))
            setattr(ds, 'mask', np.ones(len(self.inter_feat), dtype=np.bool_))
            next_ds.append(ds)
        
        # 🔧 对齐TTA4SR的mask逻辑：
        # TTA4SR中:
        #   Valid时: train_matrix=items[:-2]，过滤items[:-2]，评估items[-2]
        #   Test时:  train_matrix=items[:-1]，过滤items[:-1]，评估items[-1]
        
        # Train dataset: mask掉valid和test的targets
        next_ds[0].mask[self.target_index[next_index[1] + next_index[2]]] = False
        
        # Valid dataset: 也mask掉valid和test的targets（关键！）
        # 这样inter_matrix=items[:-2]，评估items[-2]时不会被过滤
        next_ds[1].mask[self.target_index[next_index[1] + next_index[2]]] = False
        
        # Test dataset: 只mask掉test的target
        # 这样inter_matrix=items[:-1]，评估items[-1]时不会被过滤
        next_ds[2].mask[self.target_index[next_index[2]]] = False
        
        # semantic augmentation for training and only for the train dataset
        if self.config['SSL_AUG'] == 'DuoRec':
            self.same_target_index = self.semantic_augmentation(next_ds[0].target_index)
            setattr(next_ds[0], 'same_target_index', self.same_target_index)
        
        return next_ds

    def inter_matrix(self, form='coo', value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.
        Sparse matrix has shape (user_num, item_num).
        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset does not exist uid/iid, thus can not converted to sparse matrix.')

        self.logger.warning(
            'Load interaction matrix may lead to label leakage from testing phase, this implementation '
            'only provides the interactions corresponding to specific phase'
        )
        local_inter_feat = self.inter_feat[self.mask]  # TODO: self.mask will applied to _history_matrix() in future
        return self._create_sparse_matrix(local_inter_feat, self.uid_field, self.iid_field, form, value_field)

    def build(self, eval_setting):
        self._change_feat_format()

        ordering_args = eval_setting.ordering_args
        if ordering_args['strategy'] == 'shuffle':
            raise ValueError('Ordering strategy `shuffle` is not supported in sequential models.')
        elif ordering_args['strategy'] == 'by':
            if ordering_args['field'] != self.time_field:
                raise ValueError('Sequential models require `TO` (time ordering) strategy.')
            if ordering_args['ascending'] is not True:
                raise ValueError('Sequential models require `time_field` to sort in ascending order.')

        group_field = eval_setting.group_field

        split_args = eval_setting.split_args
        if split_args['strategy'] == 'loo':
            return self.leave_one_out(group_by=group_field, leave_one_num=split_args['leave_one_num'])
        else:
            raise ValueError('Sequential models require `loo` (leave one out) split strategy.')
