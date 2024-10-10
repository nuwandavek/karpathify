**5-Lesson Plan to Understand the DepthPro Project**

Welcome to this 5-lesson journey to understand the DepthPro project and its accompanying paper on sharp monocular metric depth estimation. Each lesson is centered around a Python program from the repository and introduces specific concepts to build your understanding step by step.

---

### **Lesson 1: Introduction to Monocular Depth Estimation and Running the Model**

**Program:** `cli/run.py`

#### **Concepts Introduced:**

- **Running the Pre-trained Model with Command-line Interface**
- **Understanding the Inference Pipeline**

#### **Lesson Overview:**

In this lesson, we'll start by exploring the `run.py` script in the `cli` directory, which serves as the entry point for running the DepthPro model on input images. This script demonstrates how to load the pre-trained model, preprocess input images, perform inference to estimate depth, and handle the output.

#### **Key Points:**

1. **Command-line Interface (CLI) Parsing:**
   - The script uses the `argparse` module to handle command-line arguments, allowing users to specify input images, output paths, and other options.
   - Example snippet:
     ```python
     parser = argparse.ArgumentParser(description="Inference scripts of DepthPro with PyTorch models.")
     parser.add_argument("-i", "--image-path", type=Path, default="./data/example.jpg", help="Path to input image.")
     # ... other arguments
     ```

2. **Model Loading and Device Configuration:**
   - Detection of the available hardware (CPU, CUDA, MPS) to run the model efficiently.
   - Loading the pre-trained model and associated transforms using the `create_model_and_transforms` function.
   - Example snippet:
     ```python
     model, transform = create_model_and_transforms(device=get_torch_device(), precision=torch.half)
     model.eval()
     ```

3. **Image Processing and Inference:**
   - Reading and preprocessing input images using functions from `utils.py`, like `load_rgb`.
   - Performing inference using the `model.infer` method to obtain depth predictions.
   - Handling of focal length (`f_px`) if available or estimating it if not provided.

4. **Output Handling:**
   - Saving the depth maps and color-mapped visualizations.
   - Displaying results using `matplotlib` if not skipped.

#### **Additional Reading:**

- **Monocular Depth Estimation Basics:**
  - *Understanding how depth can be estimated from a single image using cues like texture, shading, and familiarity with object sizes.*
  - [Monocular Depth Estimation Survey](https://arxiv.org/abs/1901.09402)

- **PyTorch Model Evaluation:**
  - *Learn about setting models to evaluation mode and avoiding gradient computations during inference.*
  - [PyTorch Model Evaluation](https://pytorch.org/docs/stable/generated/torch.nn.Module.eval.html)

- **Command-line Interfaces in Python:**
  - *Using `argparse` to handle command-line arguments in scripts.*
  - [Python argparse Tutorial](https://docs.python.org/3/howto/argparse.html)

#### **Exercises:**

- Modify the `run.py` script to process a directory of images and save the outputs in a structured way.
- Add an argument to change the output resolution and observe how it affects the depth estimation.

---

### **Lesson 2: Understanding the Model Architecture - The DepthPro Class**

**Program:** `depth_pro.py`

#### **Concepts Introduced:**

- **DepthPro Class Structure**
- **Forward Pass and Inference Method**

#### **Lesson Overview:**

In this lesson, we'll dive into the core of the model by exploring the `DepthPro` class defined in `depth_pro.py`. We'll understand how the encoder and decoder are combined to perform depth estimation and how the model handles focal length estimation.

#### **Key Points:**

1. **DepthPro Class Components:**
   - The `DepthPro` class inherits from `nn.Module` and encapsulates the entire depth estimation model.
   - It composes an encoder, a decoder, and additional heads for depth and optionally for field-of-view (FOV) estimation.

2. **Encoder and Decoder:**
   - The encoder (`DepthProEncoder`) extracts multi-resolution features from the input image.
   - The decoder (`MultiresConvDecoder`) processes these features to produce a depth map.

3. **Forward Method:**
   - The `forward` method defines the computation performed at every call.
   - It takes input images, passes them through the encoder and decoder, and computes the canonical inverse depth.
   - Example snippet:
     ```python
     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
         # Encoding
         encodings = self.encoder(x)
         # Decoding
         features, features_0 = self.decoder(encodings)
         # Depth prediction
         canonical_inverse_depth = self.head(features)
         # FOV estimation (optional)
         fov_deg = None
         if hasattr(self, "fov"):
             fov_deg = self.fov.forward(x, features_0.detach())
         return canonical_inverse_depth, fov_deg
     ```

4. **Inference Method:**
   - The `infer` method handles inference, including resizing the input if necessary and adjusting the depth predictions based on the focal length.
   - It ensures that the output depth has the correct scale, especially if the focal length is estimated rather than provided.
   - Example snippet:
     ```python
     @torch.no_grad()
     def infer(self, x: torch.Tensor, f_px: Optional[Union[float, torch.Tensor]] = None):
         # Handle resizing and focal length estimation
         # Perform forward pass
         # Adjust depth based on focal length
         return {"depth": depth.squeeze(), "focallength_px": f_px}
     ```

5. **Understanding Focal Length Handling:**
   - The model can estimate the focal length if it is not provided, which is crucial for metric depth estimation.
   - The focal length estimation is handled by the FOV head if available.

#### **Additional Reading:**

- **PyTorch nn.Module and Custom Models:**
  - *Understanding how to build custom neural network modules in PyTorch.*
  - [Creating PyTorch Modules](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#nn-module)

- **Forward Pass in Neural Networks:**
  - *Learn about defining the forward computations in a neural network module.*
  - [Defining the Forward Method](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#define-the-network)

- **Metric vs. Relative Depth Estimation:**
  - *Difference between predicting absolute depth values and depth up to a scale (relative depth).*
  - [Metric Depth Estimation in Computer Vision](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yin_Enforcing_Geometric_Constraints_of_Virtual_Normal_for_Depth_Prediction_CVPR_2020_paper.pdf)

#### **Exercises:**

- Modify the `DepthPro` class to include an additional output, such as surface normals, and implement the necessary changes in the forward method.
- Implement unit tests for the `infer` method to check its behavior with and without provided focal lengths.

---

### **Lesson 3: The Encoder Module - Multi-scale Vision Transformers**

**Program:** `network/encoder.py` and `network/vit_factory.py`

#### **Concepts Introduced:**

- **DepthProEncoder Class and Multi-resolution Encoding**
- **Vision Transformers (ViT) for Feature Extraction**

#### **Lesson Overview:**

This lesson focuses on how the encoder uses Vision Transformers to extract features at multiple resolutions. We'll explore how the `DepthProEncoder` creates an image pyramid and processes patches through the ViT backbones.

#### **Key Points:**

1. **DepthProEncoder Overview:**
   - The encoder generates multi-resolution encodings from the input image.
   - It creates an image pyramid and processes overlapping patches at each level using a ViT encoder.
   - Example snippet (image pyramid creation):
     ```python
     def _create_pyramid(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
         x0 = x
         x1 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
         x2 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
         return x0, x1, x2
     ```

2. **Vision Transformer Backbone:**
   - The ViT model is used to process image patches and extract features.
   - ViTs process images by dividing them into patches, embedding them, and passing through transformer layers.
   - The `vit_factory.py` provides functions to create and load ViT models suitable for the encoder.

3. **Patch Processing and Merging:**
   - The encoder splits the images into overlapping patches to capture local details.
   - After processing, the patches are merged back into feature maps that correspond to the full image.
   - Example snippet (splitting and merging):
     ```python
     x_pyramid_patches = torch.cat((x0_patches, x1_patches, x2_patches), dim=0)
     x_pyramid_encodings = self.patch_encoder(x_pyramid_patches)
     # ... later merging ...
     x0_features = self.merge(x0_encodings, batch_size=batch_size, padding=3)
     ```

4. **Multi-resolution Feature Fusion:**
   - The features from different scales are upsampled and fused to create rich representations.
   - This multi-scale approach helps in capturing both global context and fine details.

5. **Use of Hooks for Intermediate Features:**
   - The encoder uses forward hooks to extract intermediate features from the transformer blocks.
   - These features are used to capture fine-grained details for depth prediction.
   - Example snippet:
     ```python
     self.patch_encoder.blocks[self.hook_block_ids[0]].register_forward_hook(self._hook0)
     ```

#### **Additional Reading:**

- **Vision Transformers (ViT):**
  - *Understand the architecture and functioning of Vision Transformers.*
  - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
  - [Visual Guide to Transformers](https://jalammar.github.io/illustrated-transformer/)

- **Multi-scale Feature Extraction:**
  - *Learn about processing images at multiple scales to capture different levels of detail.*
  - [Multi-scale Representation Learning](https://www.sciencedirect.com/science/article/abs/pii/S1077314215001180)

- **Forward Hooks in PyTorch:**
  - *Using hooks to access intermediate activations in a network.*
  - [PyTorch Hooks Tutorial](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks)

#### **Exercises:**

- Experiment with different patch sizes in the encoder and observe how it affects the model's ability to capture details.
- Implement a new function that visualizes the feature maps at different scales.

---

### **Lesson 4: The Decoder Module and Multi-resolution Fusion**

**Program:** `network/decoder.py`

#### **Concepts Introduced:**

- **MultiresConvDecoder Class**
- **Residual Blocks and Feature Fusion Blocks**

#### **Lesson Overview:**

In this lesson, we'll explore how the decoder takes the multi-resolution features from the encoder and fuses them to predict the depth map. We'll study the implementation of residual connections and feature fusion techniques used for depth prediction.

#### **Key Points:**

1. **MultiresConvDecoder Overview:**
   - The decoder aligns and fuses features from different resolutions to produce a final depth prediction.
   - It uses convolutional layers to project encoder features to a common dimensionality.

2. **FeatureFusionBlock2d:**
   - A key component in the decoder that fuses features using residual connections.
   - Combines upsampled features with skip connections from the encoder.
   - Example snippet:
     ```python
     class FeatureFusionBlock2d(nn.Module):
         def __init__(self, num_features: int, deconv: bool = False, batch_norm: bool = False):
             # Initialization code
         def forward(self, x0: torch.Tensor, x1: torch.Tensor | None = None) -> torch.Tensor:
             # Fusion logic
     ```

3. **Residual Blocks:**
   - Implemented via the `ResidualBlock` class to facilitate training deeper networks.
   - Helps in propagating gradients and avoiding vanishing gradient problems.
   - Example snippet:
     ```python
     class ResidualBlock(nn.Module):
         def __init__(self, residual: nn.Module, shortcut: nn.Module | None = None) -> None:
             # Initialization code
         def forward(self, x: torch.Tensor) -> torch.Tensor:
             delta_x = self.residual(x)
             if self.shortcut is not None:
                 x = self.shortcut(x)
             return x + delta_x
     ```

4. **Upsampling and Projection Layers:**
   - The decoder uses transposed convolutions (deconvolutions) to upsample feature maps.
   - Convolutional layers are used to project features to the desired number of channels.

5. **Building the Decoder Pipeline:**
   - The decoder processes the features in a hierarchical manner, starting from the lowest resolution.
   - At each level, features are upsampled and fused with higher-resolution features.

#### **Additional Reading:**

- **Convolutional Neural Network (CNN) Decoders:**
  - *Understanding how CNNs can be used to decode features into desired outputs.*
  - [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)

- **Residual Networks:**
  - *Learning about residual connections and how they help in training deep networks.*
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

- **Feature Fusion Techniques:**
  - *Study methods for combining features from different layers or modalities.*
  - [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

#### **Exercises:**

- Modify the `MultiresConvDecoder` to experiment with different fusion strategies, such as attention mechanisms.
- Add batch normalization layers to the decoder and observe the effect on training stability and performance.

---

### **Lesson 5: Focal Length Estimation and Boundary Metrics**

**Program:** `network/fov.py` and `eval/boundary_metrics.py`

#### **Concepts Introduced:**

- **Field of View Estimation Module**
- **Evaluation Metrics for Sharp Boundaries**

#### **Lesson Overview:**

In the final lesson, we'll examine how the model estimates the field of view (FOV) from input images and how sharpness in depth boundaries is evaluated using custom metrics. Understanding these components is crucial for improving depth prediction accuracy and assessing model performance.

#### **Key Points:**

1. **FOVNetwork Class:**
   - The `FOVNetwork` predicts the field of view, which is then used to estimate the focal length.
   - It can optionally use features from a separate encoder to enhance estimation.
   - Example snippet:
     ```python
     class FOVNetwork(nn.Module):
         def __init__(self, num_features: int, fov_encoder: Optional[nn.Module] = None):
             # Initialization code
         def forward(self, x: torch.Tensor, lowres_feature: torch.Tensor) -> torch.Tensor:
             # FOV estimation logic
     ```

2. **Estimating Focal Length:**
   - The estimated FOV in degrees is converted to focal length in pixels, which is important for metric depth estimation.
   - The model can work with or without provided focal length, making it flexible for various inputs.

3. **Boundary Metrics in `boundary_metrics.py`:**
   - Custom metrics are used to evaluate the sharpness of depth boundaries.
   - The module defines functions to compute precision, recall, and F1 scores for depth edges.
   - Understanding these metrics helps in assessing and improving the model's ability to capture fine details.

4. **Edge Detection and NMS:**
   - The module uses Non-Maximum Suppression (NMS) to thin out edges and focus on the most significant boundaries.
   - It identifies foreground-background relationships between neighboring pixels based on depth differences.

5. **Application in Model Evaluation:**
   - These metrics are especially useful for datasets where ground truth might not be accurate at object boundaries.
   - They provide a quantitative way to measure improvements in boundary sharpness.

#### **Additional Reading:**

- **Focal Length and Field of View Estimation:**
  - *Methods for estimating camera parameters from images.*
  - [Single Image Camera Calibration Techniques](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470050118.ecse256)

- **Edge Detection and Image Gradients:**
  - *Understanding techniques for detecting edges in images.*
  - [Canny Edge Detector](https://en.wikipedia.org/wiki/Canny_edge_detector)

- **Evaluation Metrics in Image Segmentation:**
  - *Learn about precision, recall, and F1 scores in the context of evaluating segmentation and edge detection.*
  - [Metrics for Evaluating Image Segmentation](https://www.sciencedirect.com/science/article/pii/S1361841516300704)

#### **Exercises:**

- Implement an alternative method for FOV estimation and compare its performance with the existing `FOVNetwork`.
- Use the boundary metrics to evaluate depth maps produced by different models and analyze the results.

---

**Final Notes:**

By completing these lessons, you should have a solid understanding of the DepthPro model's architecture and how it performs sharp monocular metric depth estimation. To deepen your knowledge, consider reading the DepthPro paper in detail, focusing on the sections that discuss the design choices, experiments, and results.

For further exploration:

- **Deep Learning for Depth Estimation:**
  - Understanding the advancements in deep learning approaches for depth estimation.
  - [Deep Learning for Monocular Depth Estimation: A Review](https://arxiv.org/abs/2006.01316)

- **Advanced Topics in Vision Transformers:**
  - Investigate recent developments and applications of Vision Transformers in computer vision tasks.
  - [A Survey on Visual Transformer Models](https://arxiv.org/abs/2012.12556)

- **Implementing and Training Custom Models:**
  - Experiment with modifying the DepthPro model or implementing your depth estimation models.
  - [PyTorch Tutorials](https://pytorch.org/tutorials/)

Good luck on your learning journey!