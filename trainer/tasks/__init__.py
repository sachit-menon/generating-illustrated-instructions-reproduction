
from hydra.core.config_store import ConfigStore

from trainer.tasks.sd_task import SDTaskConfig

cs = ConfigStore.instance()
cs.store(group="task", name="sd", node=SDTaskConfig)
