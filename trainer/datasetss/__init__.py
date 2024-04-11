from hydra.core.config_store import ConfigStore

from trainer.datasetss.clip_hf_dataset import CLIPHFDatasetConfig

from trainer.datasetss.wikihow_dataset import WikiHowDatasetConfig

cs = ConfigStore.instance()
cs.store(group="dataset", name="wikihow-im", node=WikiHowDatasetConfig)
