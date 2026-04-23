import torch
from blocks import SinusoidalPositionEmbedding, ResidualBlock, AttentionBlock, Downsample, Upsample

# Test SinusoidalPositionEmbedding

def test_sinusoidal_position_embedding_output_shape():
    # Arrange
    batch_size = 2
    time_emb_dim = 64
    t = torch.randint(0, 1000, (batch_size,))
    embedding = SinusoidalPositionEmbedding(time_emb_dim)

    # Act
    output = embedding(t)

    # Assert
    assert output.shape == (batch_size, time_emb_dim)

# Test ResidualBlock

def test_residual_block_output_shape():
    # Arrange
    batch_size = 2
    # default groups=8 in ResidualBlock, so in_channels should be divisible by 8
    in_channels = 8 
    out_channels = 64
    time_emb_dim = 256
    x = torch.randn(batch_size, in_channels, 16, 16)
    t = torch.randn(batch_size, time_emb_dim)
    block = ResidualBlock(in_channels, out_channels, time_emb_dim)

    # Act
    output = block(x, t)

    # Assert
    assert output.shape == (batch_size, out_channels, 16, 16)

def test_residual_block_identity_shortcut_when_channels_equal():
    # Arrange
    channels = 64
    time_emb_dim = 256
    x = torch.randn(1, channels, 16, 16)
    t = torch.randn(1, time_emb_dim)

    # Act
    block = ResidualBlock(channels, channels, time_emb_dim)
    output = block(x, t)

    # Assert
    assert output.shape == (1, channels, 16, 16)

def test_residual_block_gradients_flow_through_time_embedding():
    # Arrange
    in_channels, out_channels, time_emb_dim = 8, 64, 256
    x = torch.randn(1, in_channels, 16, 16, requires_grad=True)
    t = torch.randn(1, time_emb_dim, requires_grad=True)
    block = ResidualBlock(in_channels, out_channels, time_emb_dim)

    # Act
    block(x, t).sum().backward()

    # Assert
    assert x.grad is not None
    assert t.grad is not None

def test_residual_block_preserves_spatial_dims():
    # Arrange
    in_channels, out_channels, time_emb_dim = 8, 64, 256
    x = torch.randn(1, in_channels, 8, 8)
    t = torch.randn(1, time_emb_dim)
    block = ResidualBlock(in_channels, out_channels, time_emb_dim)

    # Act
    output = block(x, t)

    # Assert
    assert output.shape == (1, out_channels, 8, 8)

# Test ResidualBlock adagn=True

def test_residual_block_adagn_output_shape():
    # Assert
    in_channels, out_channels, time_emb_dim = 8, 64, 256
    x = torch.randn(2, in_channels, 16, 16)
    t = torch.randn(2, time_emb_dim)
    block = ResidualBlock(in_channels, out_channels, time_emb_dim, adagn=True)

    # Act
    output = block(x, t)

    # Assert
    assert output.shape == (2, out_channels, 16, 16)

def test_residual_block_adagn_norm2_has_no_affine_params():
    # Act
    block = ResidualBlock(8, 64, 256, adagn=True)

    # Assert
    assert block.norm2.weight is None
    assert block.norm2.bias is None

def test_residual_block_adagn_time_mlp_outputs_scale_and_shift():
    # Arrange
    in_channels, out_channels, time_emb_dim = 8, 64, 256
    block = ResidualBlock(in_channels, out_channels, time_emb_dim, adagn=True)
    t = torch.randn(1, time_emb_dim)

    # Act
    time_emb = block.time_mlp(t)
    scale, shift = time_emb.chunk(2, dim=1)

    # Assert
    assert scale.shape == (1, out_channels)
    assert shift.shape == (1, out_channels)

def test_residual_block_adagn_different_timesteps_produce_different_outputs():
    # Arrange
    in_channels, out_channels, time_emb_dim = 8, 64, 256
    x = torch.randn(1, in_channels, 16, 16)
    block = ResidualBlock(in_channels, out_channels, time_emb_dim, adagn=True)
    block.eval()

    # Act
    t1 = torch.zeros(1, time_emb_dim)
    t2 = torch.ones(1, time_emb_dim) * 100.0

    # Asseert
    assert not torch.allclose(block(x, t1), block(x, t2))

# Test ResidualBlock adagn=False (additive)

def test_residual_block_additive_output_shape():
    # Arrange
    in_channels, out_channels, time_emb_dim = 8, 64, 256
    x = torch.randn(2, in_channels, 16, 16)
    t = torch.randn(2, time_emb_dim)
    block = ResidualBlock(in_channels, out_channels, time_emb_dim, adagn=False)

    # Act
    output = block(x, t)

    # Assert
    assert output.shape == (2, out_channels, 16, 16)

def test_residual_block_additive_norm2_has_affine_params():
    # Act
    block = ResidualBlock(8, 64, 256, adagn=False)

    # Assert
    assert block.norm2.weight is not None
    assert block.norm2.bias is not None

def test_residual_block_additive_time_mlp_outputs_single_embedding():
    # Arrange
    in_channels, out_channels, time_emb_dim = 8, 64, 256
    block = ResidualBlock(in_channels, out_channels, time_emb_dim, adagn=False)
    t = torch.randn(1, time_emb_dim)

    # Act
    time_emb = block.time_mlp(t)

    # Assert
    assert time_emb.shape == (1, out_channels)

# Test ResidualBlock zero_init_conv=True

def test_residual_block_zero_init_conv_weights_are_zero():
    # Act
    block = ResidualBlock(8, 64, 256, zero_init_conv=True)

    # Assert
    assert torch.all(block.conv2.weight == 0)
    assert torch.all(block.conv2.bias == 0)

# New tests for Adaptive Group Normalization

def test_residual_block_time_embedding_modulates_output():
    # Arrange
    in_channels, out_channels, time_emb_dim = 8, 64, 256
    x = torch.randn(1, in_channels, 16, 16)
    block = ResidualBlock(in_channels, out_channels, time_emb_dim)
    block.eval()

    # Act
    t1 = torch.zeros(1, time_emb_dim)
    t2 = torch.ones(1, time_emb_dim) * 100.0

    # Assert
    assert not torch.allclose(block(x, t1), block(x, t2))

def test_residual_block_time_mlp_outputs_scale_and_shift():
    # Arrange
    in_channels, out_channels, time_emb_dim = 8, 64, 256
    block = ResidualBlock(in_channels, out_channels, time_emb_dim, adagn=True)
    t = torch.randn(1, time_emb_dim)
    time_emb = block.time_mlp(t)

    # Act
    scale, shift = time_emb.chunk(2, dim=1)

    # Assert
    assert scale.shape == (1, out_channels)
    assert shift.shape == (1, out_channels)
    assert not torch.allclose(scale, shift)

def test_residual_block_norm2_has_no_affine_params():
    # Act
    block = ResidualBlock(8, 64, 256, adagn=True)

    # Assert
    assert block.norm2.weight is None
    assert block.norm2.bias is None

def test_residual_block_output_not_nan_or_inf_with_extreme_time():

    # Arrange
    in_channels, out_channels, time_emb_dim = 8, 64, 256
    x = torch.randn(1, in_channels, 16, 16)
    t = torch.full((1, time_emb_dim), 1e6)
    block = ResidualBlock(in_channels, out_channels, time_emb_dim)

    # Act
    output = block(x, t)

    # Assert
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_residual_block_shortcut_carries_gradient():
    # Arrange
    in_channels, out_channels, time_emb_dim = 8, 64, 256
    x = torch.randn(1, in_channels, 16, 16, requires_grad=True)
    t = torch.randn(1, time_emb_dim)
    block = ResidualBlock(in_channels, out_channels, time_emb_dim)

    # Act
    block(x, t).sum().backward()

    # Assert
    assert not torch.all(x.grad == 0)

# Test AttentionBlock

def test_attention_block_batch_independence():
    # Arrange
    batch_size = 2
    channels = 64
    # The smaller 16x16 size speeds up the test
    x = torch.randn(batch_size, channels, 16, 16)
    
    # Act
    block = AttentionBlock(channels)
    block.eval()

    # Each sample in a batch should be processed independently
    full_batch_output = block(x)
    single_output = block(x[0:1])

    assert torch.allclose(full_batch_output[0:1], single_output, atol=1e-5)

def test_attention_block_deterministic_in_eval_mode():
    # Arrange
    channels = 64
    x = torch.randn(1, channels, 16, 16)

    # Act
    block = AttentionBlock(channels)
    block.eval()

    # Assert
    # Each input should produce the same output when not training
    assert torch.equal(block(x), block(x))

def test_attention_block_gradients_backprop():
    channels = 64
    x = torch.randn(1, channels, 16, 16, requires_grad=True)

    # Act
    block = AttentionBlock(channels)
    # Gradients should backpropagate through the block
    block(x).sum().backward()

    assert x.grad is not None
    assert not torch.all(x.grad == 0)

def test_attention_block_output_differs_from_input():
    channels = 64
    x = torch.randn(1, channels, 16, 16)

    # Act
    block = AttentionBlock(channels)

    # Assert
    assert not torch.equal(block(x), x)

def test_attention_block_output_shape():
    # Arrange
    batch_size = 2
    channels = 64

    x = torch.randn(batch_size, channels, 16, 16)
    block = AttentionBlock(channels)

    # Act
    output = block(x)

    # Assert
    assert output.shape == (batch_size, channels, 16, 16)

def test_attention_block_output_not_nan():
    # Arrange
    batch_size = 2
    channels = 64
    x = torch.randn(batch_size, channels, 16, 16)
    block = AttentionBlock(channels)

    # Act
    output = block(x)

    # Assert
    assert not torch.isnan(output).any() 

def test_attention_block_output_not_inf():
    # Arrange
    batch_size = 2
    channels = 64
    x = torch.randn(batch_size, channels, 16, 16)
    block = AttentionBlock(channels)

    # Act
    output = block(x)

    # Assert
    assert not torch.isinf(output).any()


def test_attention_block_output_not_all_zeros():
    # Arrange
    batch_size = 2
    channels = 64
    x = torch.randn(batch_size, channels, 16, 16)
    block = AttentionBlock(channels)

    # Act
    output = block(x)

    # Assert
    assert not torch.all(output == 0)   

def test_attention_block_output_not_all_same_value():
    # Arrange
    batch_size = 2
    channels = 64
    x = torch.randn(batch_size, channels, 16, 16)
    block = AttentionBlock(channels)

    # Act
    output = block(x)

    # Assert
    assert not torch.all(output == output[0, 0, 0, 0])   

def test_attention_block_custom_num_heads():
    channels = 64
    x = torch.randn(1, channels, 16, 16)
    block = AttentionBlock(channels, num_heads=8)

    output = block(x)

    assert output.shape == (1, channels, 16, 16)

# Test Upsample

def test_upsample_block_spatial_dims_double():
    # Arrange
    batch_size = 2
    in_channels = 3
    out_channels = 3
    x = torch.randn(batch_size, in_channels, 8, 8)
    doubled_dim = 16
    block = Upsample(in_channels)

    # Act
    output = block(x)

    # Assert
    assert output.shape == (batch_size, out_channels, doubled_dim, doubled_dim)   

# Test Downsample

def test_downsample_block_spatial_dims_halve():
    # Arrange
    batch_size = 2
    in_channels = 3
    out_channels = 3

    dim = 16
    halved_dim = dim // 2
    x = torch.randn(batch_size, in_channels, dim, dim)
    block = Downsample(in_channels)

    # Act
    output = block(x)

    # Assert
    assert output.shape == (batch_size, out_channels, halved_dim, halved_dim)    
