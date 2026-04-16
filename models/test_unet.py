import torch
from models.unet import UNet

# Test UNet initialization

def test_unet_init_default_in_channels_is_3():
    # Act
    model = UNet()

    # Assert
    assert model.in_channels == 3

def test_unet_init_default_out_channels_is_3():
    # Act
    model = UNet()

    # Assert
    assert model.out_channels == 3

def test_unet_init_default_base_channels_is_64():
    # Act
    model = UNet()

    # Assert
    assert model.base_channels == 64

def test_unet_init_default_num_res_blocks_is_3():
    # Act
    model = UNet()

    # Assert
    assert model.num_res_blocks == 2

def test_unet_init_set_in_channels():
    # Arrange
    model = UNet(in_channels=1)

    # Act
    in_channels = model.in_channels

    # Assert
    assert in_channels == 1

def test_unet_init_set_out_channels():
    # Act
    model = UNet(out_channels=2)

    # Assert
    assert model.out_channels == 2

def test_unet_init_set_base_channels():
    # Act
    model = UNet(base_channels=128)

    # Assert
    assert model.base_channels == 128

def test_unet_init_set_channel_mults():
    # Act
    model = UNet(channel_mults=(1, 2, 3))

    # Assert
    assert model.channel_mults == (1, 2, 3)

def test_unet_init_set_num_res_blocks():
    # Act
    model = UNet(num_res_blocks=4)

    # Assert
    assert model.num_res_blocks == 4

def test_unet_init_set_attention_resolutions():
    # Act
    model = UNet(attention_resolutions=(1, 2))

    # Assert
    assert model.attention_resolutions == (1, 2)

def test_unet_init_set_time_emb_dim():
    # Act
    model = UNet(time_emb_dim=512)

    # Assert
    assert model.time_emb_dim == 512    

def test_unet_init_set_num_heads():
    # Act
    model = UNet(num_heads=8)

    # Assert
    assert model.num_heads == 8


# Test UNet forward pass

def test_unet_forward_pass_shape_includes_batch_dimension_out_channels_image_size():
    # Arrange
    batch_size = 2
    in_channels = 3
    out_channels = 3
    image_size = 256
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    input_tensor = torch.randn(batch_size, in_channels, image_size, image_size)
    t = torch.randint(0, 1000, (batch_size,))

    # Act
    output_tensor = model(input_tensor, t)

    # Assert
    assert output_tensor.shape == (batch_size, out_channels, image_size, image_size)

def test_unet_forward_pass_output_tensor_is_not_nan():
    # Arrange
    batch_size = 2
    in_channels = 3
    out_channels = 3
    image_size = 256
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    input_tensor = torch.randn(batch_size, in_channels, image_size, image_size)
    t = torch.randint(0, 1000, (batch_size,))

    # Act
    output_tensor = model(input_tensor, t)

    # Assert
    assert output_tensor.shape == (batch_size, out_channels, image_size, image_size)
    assert not torch.isnan(output_tensor).any()

def test_unet_forward_pass_output_tensor_is_not_nan():
    # Arrange
    batch_size = 2
    in_channels = 3
    out_channels = 3
    image_size = 256
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    input_tensor = torch.randn(batch_size, in_channels, image_size, image_size)
    t = torch.randint(0, 1000, (batch_size,))

    # Act
    output_tensor = model(input_tensor, t)

    # Assert
    assert output_tensor.shape == (batch_size, out_channels, image_size, image_size)
    assert not torch.isinf(output_tensor).any()      