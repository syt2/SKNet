import logging

from torch.optim.lr_scheduler import MultiStepLR

from schedulers.schedulers_fn import ConstantLR

logger = logging.getLogger("SKNet")

key2scheduler = {
    "constant_lr": ConstantLR,
    "multi_step": MultiStepLR,
}


def get_scheduler(optimizer, cfg):
    if cfg["training"]["lr_schedule"] is None:
        logger.info("Using No LR Scheduling")
        return ConstantLR(optimizer)

    scheduler_dict = cfg["training"]["lr_schedule"]

    s_type = scheduler_dict["name"]
    scheduler_dict.pop("name")

    logging.info("Using {} scheduler with {} params".format(s_type, scheduler_dict))

    return key2scheduler[s_type](optimizer, **scheduler_dict)
