## Lesson 1: Understanding the Vision Transformer (ViT) Backbone

In this lesson, you will learn about the Vision Transformer backbone used in DepthPro and how it is loaded and configured.

**Note:** ### Loading the ViT Backbone

We start by importing the `create_vit` function from `vit_factory.py`. This function helps us create and load a ViT backbone module configured according to a preset.

The Vision Transformer (ViT) is a key component in the DepthPro model, providing powerful feature extraction capabilities by leveraging self-attention mechanisms. See **Section 3.1** of the paper for more details on the ViT backbone.

We create a ViT model using the `'dinov2l16_384'` preset, without loading pretrained weights.

```python
from depth_pro.network.vit_factory import create_vit

vit_model = create_vit(preset='dinov2l16_384', use_pretrained=False)
```

**Note:** ### Adjusting the Forward Method

We set the `forward` method of the ViT model to be `forward_features`. This ensures that when we call `vit_model(input)`, it returns the features extracted by the backbone.

```python
vit_model.forward = vit_model.forward_features
```

**Note:** ### Resizing the ViT Model

We resize the ViT model to the desired image size of 384×384 pixels. This involves adjusting the positional embeddings to match the new image size, as described in the `resize_vit` function in `vit.py`.

```python
img_size = (384, 384)

vit_model = resize_vit(vit_model, img_size=img_size)
```

**Note:** ### Adjusting the Patch Embedding

We adjust the patch embedding size to 16×16 pixels using the `resize_patch_embed` function. This is important for matching the input resolution and ensuring that the model works correctly with the given patch size.

```python
patch_size = (16, 16)

vit_model = resize_patch_embed(vit_model, new_patch_size=patch_size)
```

## Lesson 2: Building the DepthProEncoder

In this lesson, you will learn how the DepthProEncoder combines patch and image encoders to create multi-resolution encodings.

**Note:** ### Initializing DepthProEncoder

We import the `DepthProEncoder` class from `encoder.py` and create an instance of it. The encoder uses both a patch encoder and an image encoder (both using the ViT model), and combines their features to produce multi-resolution encodings.

- `dims_encoder` specifies the expected dimensions at each level from the encoder.
- `patch_encoder` and `image_encoder` are the ViT models we created earlier.
- `hook_block_ids` are the indices of the ViT blocks where we will hook to get intermediate features.
- `decoder_features` sets the number of features in the decoder.

Refer to **Section 3.1** of the paper for details on how the encoder handles multi-scale feature extraction.

```python
from depth_pro.network.encoder import DepthProEncoder

encoder = DepthProEncoder(
    dims_encoder=[256, 512, 1024, 1024],
    patch_encoder=vit_model,
    image_encoder=vit_model,
    hook_block_ids=[5, 11],
    decoder_features=256
)
```

**Note:** ### Registering Forward Hooks

We register forward hooks on the ViT model's blocks to extract intermediate features during the forward pass. These features are captured in the encoder's `_hook0` and `_hook1` methods.

This allows the encoder to obtain features from specific layers of the ViT model, which are then used for multi-resolution encoding.

```python
# Adding hooks to extract intermediate features

vit_model.blocks[5].register_forward_hook(encoder._hook0)
vit_model.blocks[11].register_forward_hook(encoder._hook1)
```

**Note:** ### Understanding the Forward Method

The `forward` method of `DepthProEncoder` processes the input image to produce multi-resolution encodings.

It performs the following steps:

1. **Image Pyramid Creation**: Creates a 3-level image pyramid with different resolutions.
2. **Patch Extraction**: Splits images into overlapping patches at each scale.
3. **Feature Extraction**: Processes the patches through the ViT encoder to obtain features.
4. **Feature Merging**: Merges the features from patches back into feature maps.

See **Algorithm 1** in the paper for a high-level overview of the encoder's forward pass.

```python
def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
    # Implementation of the forward method
    encodings = self.encoder(x)
    # Return list of encoded features
    return encodings
```

## Lesson 3: Understanding the Multiresolution Decoder

In this lesson, you will learn how the MultiresConvDecoder combines features from the encoder to produce high-resolution depth predictions.

**Note:** ### Initializing MultiresConvDecoder

We import the `MultiresConvDecoder` from `decoder.py` and create an instance. The decoder is designed to fuse features from different resolutions and project them into a common feature space.

- `dims_encoder` includes the decoder features followed by the encoder feature dimensions.
- `dim_decoder` is the dimensionality of the decoder features.

Refer to **Section 3.1** of the paper, particularly the section on the decoder architecture.

```python
from depth_pro.network.decoder import MultiresConvDecoder

decoder = MultiresConvDecoder(
    dims_encoder=[256, 256, 512, 1024, 1024],
    dim_decoder=256
)
```

**Note:** ### Decoding the Multi-Resolution Encodings

The `forward` method of the decoder takes the multi-resolution encodings from the encoder and processes them to produce the final features for depth prediction.

It uses convolutional layers and feature fusion blocks to combine the features from different resolutions. This is crucial for capturing both global context and fine details.

```python
# Forward method

def forward(self, encodings: List[torch.Tensor]) -> torch.Tensor:
    # Implementation of the forward method
    features, lowres_features = self.decoder(encodings)
    return features, lowres_features
```

**Note:** ### Feature Fusion Blocks

The `FeatureFusionBlock2d` class is used within the decoder to fuse features from different levels.

- It includes residual blocks with convolutional layers.
- It optionally performs upsampling (deconvolution) to align feature maps spatially.

This fusion process is described by the equation:

$$
F_i = \text{Conv}(F_{i-1}) + \text{Upsample}(F_i)
$$

See **Equation (2)** in the paper, which describes the fusion process using residual connections and convolutional layers.

```python
# FeatureFusionBlock2d class definition

class FeatureFusionBlock2d(nn.Module):
    def __init__(self, num_features: int, deconv: bool = False, batch_norm: bool = False):
        # Implementation...
    
    def forward(self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Implementation...
```

## Lesson 4: Field of View (FOV) Estimation

In this lesson, you will learn how the DepthPro model estimates the field of view using the FOVNetwork.

**Note:** ### Initializing FOVNetwork

We import the `FOVNetwork` class from `fov.py` and create an instance. The FOVNetwork is responsible for estimating the field of view (FOV) from the input image and the features extracted by the encoder.

This is crucial for producing metric depth predictions without relying on provided camera intrinsics. Refer to **Section 3.3** in the paper for details on FOV estimation.

```python
from depth_pro.network.fov import FOVNetwork

fov_network = FOVNetwork(
    num_features=256,
    fov_encoder=vit_model
)
```

**Note:** ### Understanding the FOV Estimation Process

The `forward` method of the `FOVNetwork` processes the input image and low-resolution features to predict the FOV.

- It down-samples the input image and extracts features using an optional encoder.
- It combines these features with the low-resolution features from the decoder.
- The combined features are passed through convolutional layers to estimate the FOV in degrees.

This predicted FOV is then used to adjust the depth predictions accordingly.

```python
def forward(self, x: torch.Tensor, lowres_feature: torch.Tensor) -> torch.Tensor:
    # Implementation of the forward method
    fov_deg = self.head(features)
    return fov_deg
```

**Note:** ### Computing the Focal Length

Using the estimated FOV, we compute the focal length in pixels:

$$
 f_{\text{px}} = \frac{W}{2 \tan\left(\frac{\text{FOV}_{\text{deg}}}{2}\right)}
$$

Where:

- ( W ) is the width of the image in pixels.
- ( \text{FOV}_{\text{deg}} ) is the estimated field of view in degrees.

See **Equation (3)** in the paper for this formula.

```python
# Focal length estimation formula

f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg))
```

## Lesson 5: Putting It All Together: The DepthPro Model

In this final lesson, you will see how the encoder, decoder, and FOV estimation are integrated into the DepthPro model for zero-shot metric monocular depth estimation.

**Note:** ### Initializing the DepthPro Model

We import the `DepthPro` class from `depth_pro.py` and create an instance, passing in the encoder, decoder, and FOV encoder.

- `last_dims` specifies the dimensions for the final convolution layers.
- `use_fov_head` is set to `True` to include FOV estimation.

Refer to **Section 3** of the paper for an overview of the DepthPro model architecture.

```python
from depth_pro.depth_pro import DepthPro

model = DepthPro(
    encoder=encoder,
    decoder=decoder,
    last_dims=(32, 1),
    use_fov_head=True,
    fov_encoder=vit_model
)
```

**Note:** ### Understanding the Forward Pass

The `forward` method of `DepthPro` processes the input image through the encoder, decoder, and FOV network to produce the canonical inverse depth map and optionally the estimated FOV.

The depth predictions are adjusted using the estimated FOV to produce metric depth maps, as discussed in **Section 3.2** and **Section 3.3** of the paper.

```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Implementation of the forward method
    canonical_inverse_depth, fov_deg = self.forward(x)
    return canonical_inverse_depth, fov_deg
```

**Note:** ### Inference Method

The `infer` method provides an easy interface to perform depth estimation on an input image, handling resizing and adjusting for the FOV.

- If the focal length (`f_px`) is not provided, it uses the estimated FOV to compute it.
- This allows the model to produce metric depth predictions without requiring camera intrinsics, fulfilling the goal stated in the paper.

```python
def infer(self, x: torch.Tensor, f_px: Optional[float] = None) -> Mapping[str, torch.Tensor]:
    # Implementation of the inference method
    prediction = self.infer(x)
    return prediction
```

**Note:** ### Using the DepthPro Model

We can now use the `infer` method to perform depth estimation on an input image. The output is a dictionary containing the depth map and optionally the focal length in pixels.

This demonstrates the end-to-end process of zero-shot monocular depth estimation with DepthPro.

```python
# Example usage

image = ...  # Load an image tensor

prediction = model.infer(image)
depth = prediction['depth']
```


