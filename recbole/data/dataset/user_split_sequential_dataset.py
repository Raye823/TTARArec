# @Time   : 2025/1/21
# @Author : Assistant
# @Email  : 

"""
recbole.data.dataset.user_split_sequential_dataset
##################################################
"""

import copy
import numpy as np
import torch
from recbole.data.dataset import Dataset
from recbole.utils import FeatureType, FeatureSource


class UserSplitSequentialDataset(Dataset):
    """用户分组的序列数据集类
    
    直接按用户比例分割数据集，每个用户的完整序列作为一个样本。
    不进行数据增强，每个用户只产生一个预测任务：使用前n-1个物品预测第n个物品。
    
    Features:
    - 每个用户 = 一个样本
    - 不做复杂的数据增强
    - 直接按用户比例分组
    - 避免用户间信息泄露
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # 用户序列相关属性
        self.uid_list = None           # 用户ID列表
        self.item_list_index = None    # 历史序列索引
        self.target_index = None       # 目标物品索引  
        self.item_list_length = None   # 序列长度
        self.mask = None               # 数据隔离mask
        
    def prepare_user_sequences(self):
        """准备用户序列，每个用户生成一个样本
        
        与 prepare_data_augmentation 不同，这里每个用户只产生一个样本：
        用户序列 [A, B, C, D] → 一个样本: [A, B, C] → D
        """
        self.logger.debug('准备用户序列，每用户一个样本')
        
        self._check_field('uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        
        # 按用户和时间排序
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        
        # 获取每个用户的交互序列
        user_interactions = {}
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if uid not in user_interactions:
                user_interactions[uid] = []
            user_interactions[uid].append(i)
        
        # 为每个用户创建一个样本
        for uid, interactions in user_interactions.items():
            if len(interactions) >= 2:  # 至少需要2个交互（1个历史+1个目标）
                # 限制序列长度
                if len(interactions) > max_item_list_len + 1:
                    # 保留最后 max_item_list_len+1 个交互
                    interactions = interactions[-(max_item_list_len + 1):]
                
                # 创建样本：前n-1个作为历史，最后1个作为目标
                history_interactions = interactions[:-1]  # 历史序列
                target_interaction = interactions[-1]     # 目标物品
                
                uid_list.append(uid)
                item_list_index.append(slice(history_interactions[0], history_interactions[-1] + 1))
                target_index.append(target_interaction)
                item_list_length.append(len(history_interactions))
        
        self.uid_list = np.array(uid_list)
        self.item_list_index = np.array(item_list_index)
        self.target_index = np.array(target_index)
        self.item_list_length = np.array(item_list_length, dtype=np.int64)
        self.mask = np.ones(len(self.inter_feat), dtype=np.bool_)
        
        self.logger.info(f'用户序列准备完成，总样本数: {len(self.uid_list)}')
    
    def split_by_users(self, ratios, group_by=None):
        """按用户比例分割数据集
        
        Args:
            ratios (list): 分割比例列表，如 [8, 1, 1] 表示 8:1:1
            group_by (str): 分组字段，必须是用户ID字段
            
        Returns:
            list: 分割后的数据集列表 [train_dataset, valid_dataset, test_dataset]
        """
        self.logger.debug(f'用户分组分割，比例: {ratios}, 分组字段: {group_by}')
        
        if group_by is None:
            raise ValueError('用户分组策略需要指定分组字段')
        if group_by != self.uid_field:
            raise ValueError('序列模型要求按用户分组')
        
        # 准备用户序列（不做数据增强）
        self.prepare_user_sequences()
        
        # 获取所有唯一用户
        unique_users = np.unique(self.uid_list)
        total_users = len(unique_users)
        
        self.logger.info(f'总用户数: {total_users}，总样本数: {len(self.uid_list)}')
        
        # 计算每个分割的用户数量
        total_ratio = sum(ratios)
        ratios = [r / total_ratio for r in ratios]  # 归一化比例
        
        split_sizes = []
        remaining_users = total_users
        for i, ratio in enumerate(ratios[:-1]):
            split_size = int(total_users * ratio)
            split_sizes.append(split_size)
            remaining_users -= split_size
        split_sizes.append(remaining_users)  # 最后一个分割取剩余所有用户
        
        self.logger.info(f'用户分割数量: {split_sizes}')
        
        # 随机打乱用户（确保随机性）
        np.random.seed(self.config['seed'])
        np.random.shuffle(unique_users)
        
        # 按用户分组
        user_groups = []
        start_idx = 0
        for split_size in split_sizes:
            user_groups.append(unique_users[start_idx:start_idx + split_size])
            start_idx += split_size
        
        # 为每个用户组创建样本索引
        group_indices = []
        for user_group in user_groups:
            # 找到属于当前用户组的所有样本索引
            user_mask = np.isin(self.uid_list, user_group)
            indices = np.where(user_mask)[0]
            group_indices.append(indices)
        
        # 创建分割后的数据集
        self._drop_unused_col()
        next_ds = []
        for i, indices in enumerate(group_indices):
            ds = copy.copy(self)
            # 设置各个字段
            for field in ['uid_list', 'item_list_index', 'target_index', 'item_list_length']:
                setattr(ds, field, np.array(getattr(ds, field)[indices]))
            
            # 设置mask实现用户组隔离
            ds.mask = self._create_user_group_mask(user_groups, i)
            next_ds.append(ds)
        
        # 为DataLoader兼容性设置必要的字段
        for ds in next_ds:
            ds._setup_dataloader_fields()
        
        self.logger.info(f'用户分组完成，各组用户数: {[len(group) for group in user_groups]}')
        self.logger.info(f'各组样本数: {[len(ds.uid_list) for ds in next_ds]}')
        
        return next_ds
    
    def _setup_dataloader_fields(self):
        """设置DataLoader所需的字段，确保兼容性"""
        list_suffix = self.config['LIST_SUFFIX']
        
        # 为每个交互字段创建对应的list字段
        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = field + list_suffix
                setattr(self, f'{field}_list_field', list_field)
                ftype = self.field2type[field]
                
                if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ
                
                if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                    list_len = (self.config['MAX_ITEM_LIST_LENGTH'], self.field2seqlen[field])
                else:
                    list_len = self.config['MAX_ITEM_LIST_LENGTH']
                
                self.set_field_property(list_field, list_ftype, FeatureSource.INTERACTION, list_len)
        
        # 设置item_list_length_field
        self.item_list_length_field = self.config['ITEM_LIST_LENGTH_FIELD']
        self.set_field_property(self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)
    
    def _create_user_group_mask(self, user_groups, current_group_idx):
        """为指定用户组创建mask，实现组间隔离
        
        Args:
            user_groups (list): 所有用户组列表
            current_group_idx (int): 当前组的索引
            
        Returns:
            np.ndarray: mask数组
        """
        # 创建基础mask（所有交互都可见）
        mask = np.ones(len(self.inter_feat), dtype=np.bool_)
        
        # 训练集：只能看到自己组的用户交互
        if current_group_idx == 0:  # 训练集
            train_users = user_groups[0]
            # 只保留训练用户的交互
            train_user_mask = np.isin(self.inter_feat[self.uid_field].numpy(), train_users)
            mask = train_user_mask
            
        # 验证集：可以看到训练集+自己组的用户交互  
        elif current_group_idx == 1:  # 验证集
            visible_users = np.concatenate([user_groups[0], user_groups[1]])
            visible_user_mask = np.isin(self.inter_feat[self.uid_field].numpy(), visible_users)
            mask = visible_user_mask
            
        # 测试集：可以看到所有前面组的用户交互
        else:  # 测试集
            visible_users = np.concatenate(user_groups[:current_group_idx + 1])
            visible_user_mask = np.isin(self.inter_feat[self.uid_field].numpy(), visible_users)
            mask = visible_user_mask
        
        return mask
    
    def build(self, eval_setting):
        """根据评估设置构建数据集
        
        Args:
            eval_setting: 评估设置对象
            
        Returns:
            list: 构建后的数据集列表
        """
        self._change_feat_format()
        
        # 检查排序策略
        ordering_args = eval_setting.ordering_args
        if ordering_args['strategy'] == 'shuffle':
            raise ValueError('序列模型不支持随机排序策略')
        elif ordering_args['strategy'] == 'by':
            if ordering_args['field'] != self.time_field:
                raise ValueError('序列模型需要按时间排序')
            if ordering_args['ascending'] is not True:
                raise ValueError('序列模型需要时间字段按升序排序')
        
        group_field = eval_setting.group_field
        split_args = eval_setting.split_args
        
        # 只支持用户分组策略
        if split_args['strategy'] == 'user_split':
            return self.split_by_users(split_args['ratios'], group_by=group_field)
        elif split_args['strategy'] == 'by_ratio':
            # 兼容原有的比例分割配置，但实际执行用户分组
            return self.split_by_users(split_args['ratios'], group_by=group_field)
        else:
            raise ValueError(f'UserSplitSequentialDataset 只支持用户分组策略，不支持: {split_args["strategy"]}')
