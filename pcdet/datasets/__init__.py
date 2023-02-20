import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .waymo.waymo_dataset import WaymoDataset
from .pandaset.pandaset_dataset import PandasetDataset
from .lyft.lyft_dataset import LyftDataset
from .once.once_dataset import ONCEDataset
from .custom.custom_dataset import CustomDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'WaymoDataset': WaymoDataset,
    'PandasetDataset': PandasetDataset,
    'LyftDataset': LyftDataset,
    'ONCEDataset': ONCEDataset,
    'CustomDataset': CustomDataset
}
# 데이터 셋 마다 데이터 로더 클래스가 모두 정의되어 있다.
# 구체적인 파라미터만 결정하여 생성하면 되는데
# 그 파라미터는 파일에 저장하거나 파라미터를 통해 받는것 같다.

class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    """

    Args:
        dataset_cfg: model.yaml에 있는 DATA_CONFIG 속성 정보
        class_names: 분류 가능한 class들 이름
        batch_size:
        dist: 분산 학습 여부
        root_path:
        workers:
        seed: 셔플 등에 랜덤이 들어갈텐데 이때를 위한 random seed
        logger:
        training:
        merge_all_iters_to_one_epoch:
        total_epochs:

    Returns:

    """

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )
    # 데이터 셋에 맞는 데이터 로더 클래스를 생성한다.
    # 파일과 파라미터를 통해 구체화된 값들을 인자로 넘긴다.

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
    # torch.utils.data.Dataset을 확장할 때 추가한 기능인 것 같은데
    # 해당 코드를 더 봐야 알 수 있겠지만 일단 패스

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
            # multi-GPU 사용 방법중 하나로 distributed data parallel이 있다.
            # 이걸 쓸 때는 DataLoader에 Sampler로 DistributedSampler가 들어가야 한다고...
            # 이것의 역할은 전체 인덱스를 GPU별로 나누고 셔플하여 데이터를 분산할 수 있도록 도와준다.
    else:
        sampler = None

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )
    # multi-GPU 사용 여부 에따라 적절한 DataLoader 생성

    return dataset, dataloader, sampler
