from hydra.core.config_store import ConfigStore

from trainer.models.sd_model import SDModelConfig#, ShowNotTellPipelineConfig

cs = ConfigStore.instance()
cs.store(group="model", name="sd", node=SDModelConfig)
# cs.store(group="model", name="pipeline", node=ShowNotTellPipelineConfig)

