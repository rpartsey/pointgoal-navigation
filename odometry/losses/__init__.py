from . import losses as losses_module


def make_loss(loss_config):
    loss_type = getattr(losses_module, loss_config.type)
    loss = loss_type(
        **loss_config.params
    )

    return loss
