import pytest
import torch

from odometry.config.default import get_config
from odometry.models import make_model


@pytest.mark.parametrize(
    "vo_config_path", [
        "test/config_files/odometry/vonet.yaml",
        "test/config_files/odometry/vonet_v2.yaml",
        "test/config_files/odometry/vonet_v3.yaml",
        "test/config_files/odometry/vonet_v4.yaml"
    ]
)
def test_vo_model_create_and_forward(vo_config_path: str, batch_size=16, gpu_device_id=0):
    vo_model_config = get_config(vo_config_path, new_keys_allowed=True)
    encoder_params = vo_model_config.model.params.encoder.params
    device = torch.device(f"cuda:{gpu_device_id}")
    try:
        vo_model = make_model(vo_model_config.model).to(device)

        input_size = (batch_size, encoder_params.in_channels, encoder_params.in_height, encoder_params.in_width)
        encoder_input = torch.randn(*input_size).to(device)
        with torch.no_grad():
            output = vo_model(encoder_input, action=torch.ones(batch_size, dtype=torch.int32).to(device))
    except Exception as e:
        assert False, f"Exception raised: {e}"
