from copy import deepcopy
import deepmist.datasets.NUDTMIRSDTDataset as NUDTMIRSDTDataset
from deepmist.datasets.MISTDataset import MISTDataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


def build_dataset(dataset_cfg, mode='train'):
    dataset_cfg = deepcopy(dataset_cfg)
    dataset_name = dataset_cfg.pop('name')
    if dataset_name == 'MIST':
        train_dataset = MISTDataset(**dataset_cfg, split='train')
        val_dataset = MISTDataset(**dataset_cfg, split='val_all')
        val_hard_dataset = MISTDataset(**dataset_cfg, split='val_hard')
    elif dataset_name == 'NUDT-MIRSDT':
        train_dataset = NUDTMIRSDTDataset.TrainDataset(dataset_cfg)
        val_dataset = NUDTMIRSDTDataset.ValDataset(dataset_cfg, split='all')
        val_hard_dataset = NUDTMIRSDTDataset.ValDataset(dataset_cfg, split='lSCR')
    else:
        raise NotImplementedError(
            f"Invalid dataset name '{dataset_name}'. Only MIST and NUDT-MIRSDT are supported.")

    if mode == 'train':
        return train_dataset, val_dataset, val_hard_dataset
    elif mode == 'val':
        return val_dataset, val_hard_dataset
    else:
        raise ValueError(f"Invalid mode '{mode}'. It must be 'train' or 'val'.")


class DataLoaderX(DataLoader):
    def _iter_(self):
        return BackgroundGenerator(super()._iter_())
