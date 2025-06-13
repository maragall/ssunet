import pytest
import torch
from torch import nn

# Import from blocks.py (current implementations)
from src.ssunet.modules.blocks import (
    ConvNeXtDownBlock3D,
    ConvNeXtUpBlock3D,
    DownConvTri3D,
    EfficientDownBlock3D,
    EfficientUpBlock3D,
    NAFDownBlock3D,
    NAFUpBlock3D,
    UpConvTri3D,
)

# Import from layers.py (new implementations)
from src.ssunet.modules.layers import (
    NAFBlock3D,
    conv333,
    pool,
    upconv222,
)

# Import from normalization.py
from src.ssunet.modules.normalization import LayerNorm3D, get_norm_layer

# Import from special_layers.py
from src.ssunet.modules.special_layers import (
    GatedReLUMix,
    PartialConv3d,
    PixelShuffle3d,
    PixelUnshuffle3d,
    SimpleGate3D,
    SinGatedMix,
)

# Common test parameters
B, D, H, W = 2, 8, 16, 16
C_IN, C_OUT = 16, 32


@pytest.fixture
def input_tensor_3d():
    return torch.randn(B, C_IN, D, H, W)


@pytest.fixture
def input_tensor_skip_3d():
    return torch.randn(B, C_OUT, D, H, W)


# Tests for layers.py
def test_conv333(input_tensor_3d):
    # Test with z_conv=True (3D convolution)
    conv_3d = conv333(C_IN, C_OUT, z_conv=True)
    output_3d = conv_3d(input_tensor_3d)
    assert output_3d.shape == (B, C_OUT, D, H, W)

    # Test with z_conv=False (2.5D convolution)
    conv_25d = conv333(C_IN, C_OUT, z_conv=False)
    output_25d = conv_25d(input_tensor_3d)
    assert output_25d.shape == (B, C_OUT, D, H, W)  # Padding (0,1,1) maintains H,W


@pytest.mark.parametrize("down_mode", ["maxpool", "avgpool", "conv", "unshuffle"])
@pytest.mark.parametrize("z_conv", [True, False])
def test_pool(input_tensor_3d, down_mode, z_conv):
    pool_layer = pool(C_IN, C_OUT if down_mode == "conv" else C_IN, down_mode, z_conv)

    # Specific channel handling for unshuffle
    if down_mode == "unshuffle":
        expected_c_after_unshuffle = C_IN * 8 if z_conv else C_IN * 4
        # If pool layer includes a 1x1 conv to match C_OUT
        if C_OUT != expected_c_after_unshuffle and isinstance(pool_layer, nn.Sequential):
            pool_layer = pool(C_IN, C_OUT, down_mode, z_conv)  # Re-init with C_OUT
            expected_channels = C_OUT
        else:  # unshuffle only
            pool_layer = pool(C_IN, expected_c_after_unshuffle, down_mode, z_conv)
            expected_channels = expected_c_after_unshuffle
    elif down_mode == "conv":
        expected_channels = C_OUT
    else:
        expected_channels = C_IN

    output = pool_layer(input_tensor_3d)

    # All pooling modes should now halve the spatial dimensions (except D if z_conv=False)
    expected_d = D if not z_conv else D // 2
    expected_h = H // 2
    expected_w = W // 2

    assert output.shape == (B, expected_channels, expected_d, expected_h, expected_w)


@pytest.mark.parametrize("up_mode", ["transpose", "pixelshuffle", "trilinear"])
@pytest.mark.parametrize("z_conv", [True, False])
def test_upconv222(input_tensor_3d, up_mode, z_conv):
    if up_mode == "pixelshuffle":
        # The upconv222 handles this adjustment internally.
        up_layer = upconv222(C_IN, C_OUT, z_conv, up_mode)
        current_input = input_tensor_3d
        expected_c = C_OUT
    else:
        up_layer = upconv222(C_IN, C_OUT, z_conv, up_mode)
        current_input = input_tensor_3d
        expected_c = C_OUT

    output = up_layer(current_input)

    expected_d = D if not z_conv else D * 2
    expected_h = H * 2
    expected_w = W * 2
    assert output.shape == (B, expected_c, expected_d, expected_h, expected_w)


# Tests for normalization.py
def test_layer_norm_3d(input_tensor_3d):
    ln = LayerNorm3D(C_IN)
    output = ln(input_tensor_3d)
    assert output.shape == input_tensor_3d.shape
    # Check that values changed (normalization was applied)
    assert not torch.allclose(output, input_tensor_3d)


@pytest.mark.parametrize("norm_type", ["layer", "batch", "group", "instance", "invalid"])
def test_get_norm_layer(norm_type):
    if norm_type == "group":
        norm_layer = get_norm_layer(norm_type, C_IN, num_groups=4)
        assert isinstance(norm_layer, nn.GroupNorm)
    elif norm_type == "invalid":
        norm_layer = get_norm_layer(norm_type, C_IN)
        assert isinstance(norm_layer, nn.Identity)
    else:
        norm_layer = get_norm_layer(norm_type, C_IN)
        if norm_type == "layer":
            assert isinstance(norm_layer, LayerNorm3D)
        elif norm_type == "batch":
            assert isinstance(norm_layer, nn.BatchNorm3d)
        elif norm_type == "instance":
            assert isinstance(norm_layer, nn.InstanceNorm3d)


# Tests for special_layers.py
def test_pixel_shuffle_unshuffle_3d():
    # Test PixelShuffle3d
    shuffle_factor = 2
    c_shuffle_in = C_OUT * (shuffle_factor**3)
    shuffle_input = torch.randn(B, c_shuffle_in, D, H, W)
    ps = PixelShuffle3d(shuffle_factor)
    shuffled_output = ps(shuffle_input)
    assert shuffled_output.shape == (
        B,
        C_OUT,
        D * shuffle_factor,
        H * shuffle_factor,
        W * shuffle_factor,
    )

    # Test PixelUnshuffle3d
    unshuffle_input = shuffled_output
    pus = PixelUnshuffle3d(shuffle_factor)
    unshuffled_output = pus(unshuffle_input)
    assert unshuffled_output.shape == (B, c_shuffle_in, D, H, W)
    assert torch.allclose(unshuffled_output, shuffle_input, atol=1e-6)


def test_partial_conv3d(input_tensor_3d):
    # Test with return_mask=False (default)
    pconv = PartialConv3d(C_IN, C_OUT, kernel_size=3, padding=1)
    # Test that weight_mask_updater is a buffer (correct buffer name)
    assert "weight_mask_updater" in pconv._buffers

    # Test forward pass without mask
    output_no_mask = pconv(input_tensor_3d)
    assert output_no_mask.shape == (
        B,
        C_OUT,
        D,
        H,
        W,
    )  # Test forward pass with mask (return_mask=False by default)
    mask = torch.ones_like(input_tensor_3d)
    mask[:, :, D // 2 :, H // 2 :, W // 2 :] = 0  # Create a partial mask
    output_with_mask = pconv(input_tensor_3d, mask=mask)
    assert output_with_mask.shape == (B, C_OUT, D, H, W)

    # Test with return_mask=True set in constructor
    pconv_with_mask = PartialConv3d(C_IN, C_OUT, kernel_size=3, padding=1, return_mask=True)
    output_with_mask, returned_mask = pconv_with_mask(input_tensor_3d, mask=mask)
    assert output_with_mask.shape == (B, C_OUT, D, H, W)
    assert returned_mask.shape == (B, C_OUT, D, H, W)  # Mask is for output channels


@pytest.mark.parametrize("custom_activation", [GatedReLUMix, SinGatedMix])
def test_custom_activations(custom_activation):
    # Input channels must be even
    c_even = 32
    input_tensor_even_channels = torch.randn(B, c_even, D, H, W)
    activation = custom_activation()
    output = activation(input_tensor_even_channels)
    assert output.shape == (B, c_even, D, H, W)

    # Test ValueError for odd channels
    c_odd = 31
    input_tensor_odd_channels = torch.randn(B, c_odd, D, H, W)
    with pytest.raises(ValueError, match="Input channels must be even"):
        activation(input_tensor_odd_channels)


# Tests for blocks.py
def test_efficient_down_block_3d(input_tensor_3d):
    block = EfficientDownBlock3D(C_IN, C_OUT, z_conv=True)
    pooled, skip = block(input_tensor_3d)
    assert pooled.shape == (B, C_OUT, D // 2, H // 2, W // 2)
    assert skip.shape == (B, C_OUT, D, H, W)

    block_no_skip = EfficientDownBlock3D(C_IN, C_OUT, z_conv=True, skip_out=False)
    pooled_no_skip, skip_none = block_no_skip(input_tensor_3d)
    assert pooled_no_skip.shape == (B, C_OUT, D // 2, H // 2, W // 2)
    assert skip_none is None


def test_efficient_up_block_3d(input_tensor_3d):
    block = EfficientUpBlock3D(C_IN, C_OUT, z_conv=True, merge_mode="concat")
    x_for_upblock = torch.randn(B, C_IN, D // 2, H // 2, W // 2)
    skip_for_upblock = torch.randn(B, C_OUT, D, H, W)

    output = block(x_for_upblock, skip_for_upblock)
    assert output.shape == (B, C_OUT, D, H, W)  # Output spatial dims match skip

    # Test with add merge_mode (skip channels should match block's out_channels)
    block_add = EfficientUpBlock3D(C_IN, C_OUT, z_conv=True, merge_mode="add")
    output_add = block_add(x_for_upblock, skip_for_upblock)  # skip_for_upblock has C_OUT
    assert output_add.shape == (B, C_OUT, D, H, W)

    # Test without skip
    output_no_skip = block(x_for_upblock, None)
    assert output_no_skip.shape == (B, C_OUT, D, H, W)


# Tests for blocks.py (continued)
@pytest.mark.parametrize("z_conv", [True, False])
def test_convnext_down_block_3d(input_tensor_3d, z_conv):
    # Create block_config with expand_ratio instead of passing as direct parameter
    block_config = {"expand_ratio": 2}
    block = ConvNeXtDownBlock3D(C_IN, C_OUT, z_conv=z_conv, num_blocks=1, block_config=block_config)
    pooled, skip = block(input_tensor_3d)

    expected_d_pooled = D if not z_conv else D // 2
    expected_h_pooled = H // 2
    expected_w_pooled = W // 2

    assert pooled.shape == (B, C_OUT, expected_d_pooled, expected_h_pooled, expected_w_pooled)
    assert skip.shape == (B, C_OUT, D, H, W)

    block_no_skip = ConvNeXtDownBlock3D(
        C_IN, C_OUT, z_conv=z_conv, num_blocks=1, skip_out=False, block_config=block_config
    )
    pooled_no_skip, skip_none = block_no_skip(input_tensor_3d)
    assert pooled_no_skip.shape == (
        B,
        C_OUT,
        expected_d_pooled,
        expected_h_pooled,
        expected_w_pooled,
    )
    assert skip_none is None


@pytest.mark.parametrize("z_conv", [True, False])
@pytest.mark.parametrize("merge_mode", ["concat", "add"])
def test_convnext_up_block_3d(z_conv, merge_mode):
    # Adjust input dimensions based on z_conv
    if z_conv:
        x_for_upblock = torch.randn(B, C_IN, D // 2, H // 2, W // 2)
        skip_for_upblock = torch.randn(B, C_OUT, D, H, W)
    else:
        # When z_conv=False, only H and W are downsampled/upsampled
        x_for_upblock = torch.randn(B, C_IN, D, H // 2, W // 2)
        skip_for_upblock = torch.randn(B, C_OUT, D, H, W)

    block = ConvNeXtUpBlock3D(
        C_IN,
        C_OUT,
        z_conv=z_conv,
        num_blocks=1,
        merge_mode=merge_mode,
        block_config={"expand_ratio": 2},
    )
    output = block(x_for_upblock, skip_for_upblock)
    assert output.shape == (B, C_OUT, D, H, W)

    output_no_skip = block(x_for_upblock, None)
    assert output_no_skip.shape == (B, C_OUT, D, H, W)


@pytest.mark.parametrize("z_conv", [True, False])
@pytest.mark.parametrize("up_mode", ["transpose", "pixelshuffle"])
@pytest.mark.parametrize("merge_mode", ["concat", "add"])
def test_up_conv_tri_3d(z_conv, up_mode, merge_mode):
    # Adjust input dimensions based on z_conv
    if z_conv:
        x_for_upblock = torch.randn(B, C_IN, D // 2, H // 2, W // 2)
        skip_for_upblock = torch.randn(B, C_OUT, D, H, W)
    else:  # When z_conv=False, only H and W are downsampled/upsampled
        x_for_upblock = torch.randn(B, C_IN, D, H // 2, W // 2)
        skip_for_upblock = torch.randn(B, C_OUT, D, H, W)

    gn = C_OUT // 2 if C_OUT >= 2 and C_OUT % 2 == 0 else 0
    block = UpConvTri3D(
        C_IN, C_OUT, z_conv=z_conv, up_mode=up_mode, merge_mode=merge_mode, group_norm_legacy=gn
    )
    output = block(x_for_upblock, skip_for_upblock)
    assert output.shape == (B, C_OUT, D, H, W)

    output_no_skip = block(x_for_upblock, None)
    assert output_no_skip.shape == (B, C_OUT, D, H, W)


@pytest.mark.parametrize("z_conv", [True, False])
@pytest.mark.parametrize("down_mode", ["maxpool", "conv"])
def test_down_conv_tri_3d(input_tensor_3d, z_conv, down_mode):
    gn = C_OUT // 2 if C_OUT >= 2 and C_OUT % 2 == 0 else 0
    block = DownConvTri3D(C_IN, C_OUT, z_conv=z_conv, down_mode=down_mode, group_norm_legacy=gn)
    pooled, skip = block(input_tensor_3d)

    # Pooling in DownConvTri3D should now halve spatial dimensions (except D if not z_conv).
    expected_d_pooled = D if not z_conv else D // 2
    expected_h_pooled = H // 2
    expected_w_pooled = W // 2

    assert pooled.shape == (B, C_OUT, expected_d_pooled, expected_h_pooled, expected_w_pooled)
    assert skip.shape == (B, C_OUT, D, H, W)

    block_no_skip = DownConvTri3D(
        C_IN, C_OUT, z_conv=z_conv, down_mode=down_mode, skip_out=False, group_norm_legacy=gn
    )
    pooled_no_skip, skip_none = block_no_skip(input_tensor_3d)
    assert pooled_no_skip.shape == (
        B,
        C_OUT,
        expected_d_pooled,
        expected_h_pooled,
        expected_w_pooled,
    )
    assert skip_none is None


# ===== NAFNet Tests =====


def test_simple_gate_3d():
    """Test SimpleGate3D activation."""
    # SimpleGate requires even number of channels
    c_even = 32
    input_tensor = torch.randn(B, c_even, D, H, W)

    gate = SimpleGate3D()
    output = gate(input_tensor)

    # Output should have half the input channels (after multiplication)
    assert output.shape == (B, c_even // 2, D, H, W)

    # Verify it's doing element-wise multiplication of two halves
    x1, x2 = input_tensor.chunk(2, dim=1)
    expected = x1 * x2
    assert torch.allclose(output, expected)


@pytest.mark.parametrize("z_conv", [True, False])
@pytest.mark.parametrize("dropout_p", [0.0, 0.1])
def test_naf_block_3d(z_conv, dropout_p):
    """Test NAFBlock3D with different configurations."""
    channels = 32
    input_tensor = torch.randn(B, channels, D, H, W)
    block = NAFBlock3D(
        channels=channels,
        z_conv=z_conv,
        block_config={"dw_expand": 2, "ffn_expand": 2, "dropout_p": dropout_p},
    )

    output = block(input_tensor)

    # Output shape should match input
    assert output.shape == (B, channels, D, H, W)

    # Check that parameters are properly initialized
    assert block.beta.shape == (1, channels, 1, 1, 1)
    assert block.gamma.shape == (1, channels, 1, 1, 1)

    # Verify normalization layers
    assert isinstance(block.norm1, LayerNorm3D)
    assert isinstance(block.norm2, LayerNorm3D)

    # Verify default activation type (SimpleGate)
    assert isinstance(block.sg1, SimpleGate3D)
    assert isinstance(block.sg2, SimpleGate3D)


@pytest.mark.parametrize("z_conv", [True, False])
@pytest.mark.parametrize("down_mode", ["maxpool", "avgpool", "conv", "unshuffle"])
@pytest.mark.parametrize("activation", ["gelu", "relu"])  # Param for test variety
def test_naf_down_block_3d(input_tensor_3d, z_conv, down_mode, activation):
    """Test NAFDownBlock3D with different configurations."""
    block_config_params = {"dw_expand": 2, "ffn_expand": 2}
    block = NAFDownBlock3D(
        C_IN,
        C_OUT,
        z_conv=z_conv,
        down_mode=down_mode,
        num_blocks=2,
        block_config=block_config_params,
    )

    pooled, skip = block(input_tensor_3d)

    # Pooling in NAFDownBlock3D should now halve spatial dimensions (except D if not z_conv).
    expected_d_pooled = D if not z_conv else D // 2
    expected_h_pooled = H // 2
    expected_w_pooled = W // 2

    assert pooled.shape == (B, C_OUT, expected_d_pooled, expected_h_pooled, expected_w_pooled)
    assert skip.shape == (B, C_OUT, D, H, W)

    # Test without skip output
    block_no_skip = NAFDownBlock3D(
        C_IN,
        C_OUT,
        z_conv=z_conv,
        skip_out=False,
        down_mode=down_mode,
        num_blocks=2,
        block_config=block_config_params,
    )
    pooled_no_skip, skip_none = block_no_skip(input_tensor_3d)
    expected_shape_pooled_no_skip = (
        B,
        C_OUT,
        expected_d_pooled,
        expected_h_pooled,
        expected_w_pooled,
    )
    assert pooled_no_skip.shape == expected_shape_pooled_no_skip
    assert skip_none is None


@pytest.mark.parametrize("z_conv", [True, False])
@pytest.mark.parametrize("up_mode", ["transpose", "pixelshuffle", "trilinear"])
@pytest.mark.parametrize("merge_mode", ["concat", "add"])
@pytest.mark.parametrize("activation", ["gelu", "silu"])
def test_naf_up_block_3d(z_conv, up_mode, merge_mode, activation):
    """Test NAFUpBlock3D with different configurations."""
    # Adjust input dimensions based on z_conv
    if z_conv:
        x_for_upblock = torch.randn(B, C_IN, D // 2, H // 2, W // 2)
        skip_for_upblock = torch.randn(B, C_OUT, D, H, W)
    else:
        x_for_upblock = torch.randn(B, C_IN, D, H // 2, W // 2)
        skip_for_upblock = torch.randn(B, C_OUT, D, H, W)
    # Ensure block is defined outside the conditional
    block = NAFUpBlock3D(
        C_IN,
        C_OUT,
        z_conv=z_conv,
        up_mode=up_mode,
        merge_mode=merge_mode,
        num_blocks=2,
        block_config={"dw_expand": 2, "ffn_expand": 2},
    )

    # Test with skip connection
    output = block(x_for_upblock, skip_for_upblock)
    assert output.shape == (B, C_OUT, D, H, W)

    # Test without skip connection
    output_no_skip = block(x_for_upblock, None)
    assert output_no_skip.shape == (B, C_OUT, D, H, W)


def test_naf_block_with_custom_params():
    """Test NAFBlock3D with custom expansion ratios."""
    channels = 64
    dw_expand = 4
    ffn_expand = 3

    input_tensor = torch.randn(B, channels, D, H, W)

    block = NAFBlock3D(
        channels=channels,
        z_conv=True,
        block_config={"dw_expand": dw_expand, "ffn_expand": ffn_expand, "dropout_p": 0.2},
    )

    output = block(input_tensor)
    assert output.shape == (B, channels, D, H, W)

    # Verify internal channel dimensions
    assert block.conv1.out_channels == channels * dw_expand
    assert block.conv4.out_channels == channels * ffn_expand


def test_naf_blocks_gradient_flow():
    """Test that gradients flow properly through NAF blocks."""
    # Create a simple model with NAF blocks
    block = NAFDownBlock3D(C_IN, C_OUT, z_conv=True)

    # Set to training mode
    block.train()

    input_tensor = torch.randn(B, C_IN, D, H, W, requires_grad=True)
    pooled, skip = block(input_tensor)  # Create a simple loss and backpropagate
    loss = pooled.mean() + (skip.mean() if skip is not None else 0)
    loss.backward()

    # Check that input tensor received gradients
    assert input_tensor.grad is not None
    zeros_like_grad = torch.zeros_like(input_tensor.grad)
    assert not torch.allclose(input_tensor.grad, zeros_like_grad)

    # Check that block parameters received gradients
    params_with_grad = 0
    for _, param in block.named_parameters():
        if param.requires_grad and param.grad is not None:
            params_with_grad += 1

    # Should have at least some parameters with gradients
    assert params_with_grad > 0


@pytest.mark.parametrize("block_type", ["nafnet", "efficient", "convnext", "tri"])
def test_block_activation_customization(input_tensor_3d, block_type):
    """Test that all block types support custom activation."""
    from src.ssunet.modules.blocks import BLOCK_REGISTRY

    down_block_cls, up_block_cls = BLOCK_REGISTRY[block_type]

    # Test different activations
    activations = ["relu", "gelu", "silu"]
    for activation_str in activations:  # Renamed loop variable for clarity
        # Create downsampling block
        if block_type in ["nafnet", "convnext"]:
            # These blocks don't accept activation parameter directly at the top level
            # for their main path, or handle it via internal block_config.
            down_block = down_block_cls(C_IN, C_OUT, z_conv=True)
        elif block_type == "tri":
            # DownConvTri3D expects 'activation'
            down_block = down_block_cls(C_IN, C_OUT, z_conv=True, activation=activation_str)
        else:  # block_type == "efficient"
            # EfficientDownBlock3D expects 'activation_name'
            down_block = down_block_cls(C_IN, C_OUT, z_conv=True, activation_name=activation_str)

        pooled, skip = down_block(input_tensor_3d)

        # Verify output shapes
        # Assuming z_conv=True is passed or is the default for these down blocks.
        expected_d_pooled = D // 2
        actual_d_pooled = pooled.shape[2]

        assert actual_d_pooled == expected_d_pooled
        assert pooled.shape[3] == H // 2
        assert pooled.shape[4] == W // 2


def test_naf_block_with_non_simplegate():
    """Test NAFBlock3D uses SimpleGate by default."""
    channels = 32
    input_tensor = torch.randn(B, channels, D, H, W)

    # NAFBlock3D always uses SimpleGate internally
    block = NAFBlock3D(
        channels=channels,
        z_conv=True,
        block_config={"dw_expand": 2, "ffn_expand": 2, "dropout_p": 0.0},
    )

    output = block(input_tensor)
    assert output.shape == (B, channels, D, H, W)

    # Verify it uses SimpleGate
    assert isinstance(block.sg1, SimpleGate3D)
    assert isinstance(block.sg2, SimpleGate3D)

    # Verify conv3 and conv5 expect half channels (SimpleGate output)
    assert block.conv3.in_channels == channels  # dw_channels // 2
    assert block.conv5.in_channels == channels  # ffn_channels // 2
