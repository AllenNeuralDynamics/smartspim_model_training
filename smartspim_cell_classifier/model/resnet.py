from typing import Dict, List, Literal, Optional, Tuple, Union

try:
    from keras import KerasTensor as Tensor
except ImportError:
    from keras.src.backend.common.keras_tensor import KerasTensor as Tensor

from keras import Model
import keras.ops as ops
from keras.initializers import Initializer
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,  # Keep import for backwards compatibility
    Concatenate,
    Conv3D,
    Dense,
    Dropout,
    GlobalAveragePooling3D,
    Input,
    Layer,
    MaxPooling3D,
    Multiply,
    ZeroPadding3D,
)
from keras.optimizers import Adam, Optimizer
from keras.regularizers import l2 as l2_regularizer
import keras

# from functools import partial  # Removed - causes serialization issues

#####################################################################
# Define the types of ResNet
#####################################################################

layer_type = Literal[
    "18-layer", "34-layer", "50-layer", "101-layer", "152-layer"
]

resnet_unit_blocks: Dict[layer_type, List[int]] = {
    "18-layer": [2, 2, 2, 2],
    "34-layer": [3, 4, 6, 3],
    "50-layer": [3, 4, 6, 3],
    "101-layer": [3, 4, 23, 3],
    "152-layer": [3, 6, 36, 3],
}

network_residual_bottleneck: Dict[layer_type, bool] = {
    "18-layer": False,
    "34-layer": False,
    "50-layer": True,
    "101-layer": True,
    "152-layer": True,
}

#####################################################################
# CUSTOM LAYERS - GROUP NORMALIZATION
#####################################################################

@keras.saving.register_keras_serializable(package="Custom")
class GroupNormalization3D(Layer):
    """
    Group Normalization for 3D data.
    
    Normalizes features within groups, making the model invariant to:
    - Different imaging power levels
    - Batch composition
    - Per-sample intensity variations
    
    This is CRITICAL for generalization across datasets with different
    acquisition settings.
    
    Args:
        groups: Number of groups to split channels into
        epsilon: Small constant for numerical stability
        center: If True, add learned offset (beta)
        scale: If True, add learned scale (gamma)
    """
    
    def __init__(self, groups=8, epsilon=1e-5, center=True, scale=True, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        
    def build(self, input_shape):
        # Input shape: (batch, depth, height, width, channels)
        self.channels = input_shape[-1]
        
        if self.channels % self.groups != 0:
            raise ValueError(
                f'Number of channels ({self.channels}) must be divisible by '
                f'number of groups ({self.groups})'
            )
        
        shape = (self.channels,)
        
        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=shape,
                initializer='ones',
                trainable=True
            )
        else:
            self.gamma = None
            
        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=shape,
                initializer='zeros',
                trainable=True
            )
        else:
            self.beta = None
            
        super().build(input_shape)
        
    def call(self, inputs):
        # Input shape: (N, D, H, W, C)
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        
        # Reshape to (N, D, H, W, groups, C // groups)
        x = ops.reshape(
            inputs,
            [batch_size, input_shape[1], input_shape[2], input_shape[3], 
             self.groups, self.channels // self.groups]
        )
        
        # Compute mean and variance over spatial dims and channels within each group
        # Axis: (1, 2, 3, 5) = (D, H, W, channels_per_group)
        mean = ops.mean(x, axis=[1, 2, 3, 5], keepdims=True)
        variance = ops.var(x, axis=[1, 2, 3, 5], keepdims=True)
        
        # Normalize
        x = (x - mean) / ops.sqrt(variance + self.epsilon)
        
        # Reshape back to (N, D, H, W, C)
        x = ops.reshape(x, input_shape)
        
        # Apply scale and shift
        if self.scale:
            x = x * self.gamma
        if self.center:
            x = x + self.beta
            
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'groups': self.groups,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
        })
        return config


@keras.saving.register_keras_serializable(package="Custom")
class ReduceMean3D(Layer):
    """Reduce mean along channel axis - replaces Lambda layer."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs, mask=None):
        return ops.mean(inputs, axis=-1, keepdims=True)
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)
    
    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="Custom")
class ReduceMax3D(Layer):
    """Reduce max along channel axis - replaces Lambda layer."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs, mask=None):
        return ops.max(inputs, axis=-1, keepdims=True)
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)
    
    def get_config(self):
        return super().get_config()



# ============================================================================
# SERIALIZABLE LOSS CLASSES (for model saving)
# ============================================================================

@keras.saving.register_keras_serializable(package="Custom")
class BinaryFocalLoss(keras.losses.Loss):
    """
    Binary Focal Loss as a proper Keras Loss class.
    Fully serializable - fixes the functools.partial error.
    """
    
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        """Compute binary focal loss."""
        y_pred = ops.cast(y_pred, "float32")
        y_true = ops.cast(y_true, "float32")
        
        epsilon = keras.backend.epsilon()
        y_pred = ops.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Binary cross entropy
        bce = -(y_true * ops.log(y_pred) + (1 - y_true) * ops.log(1 - y_pred))
        
        # Focal weight: (1 - pt)^gamma
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = ops.power(1.0 - pt, self.gamma)
        
        # Alpha balancing
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        return focal_weight * alpha_weight * bce
    
    def get_config(self):
        """Return config for serialization."""
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
        })
        return config


@keras.saving.register_keras_serializable(package="Custom")
class CategoricalFocalLoss(keras.losses.Loss):
    """
    Categorical Focal Loss as a proper Keras Loss class.
    Fully serializable - fixes the functools.partial error.
    """
    
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        """Compute categorical focal loss."""
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        
        epsilon = keras.backend.epsilon()
        y_pred = ops.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Cross entropy
        ce = -y_true * ops.log(y_pred)
        
        # pt
        pt = ops.sum(y_true * y_pred, axis=-1, keepdims=True)
        
        # Focal weight
        focal_weight = ops.power(1.0 - pt, self.gamma)
        loss = focal_weight * ce
        
        # Alpha weighting
        if isinstance(self.alpha, (list, tuple)):
            alpha_tensor = ops.convert_to_tensor(self.alpha, dtype="float32")
            alpha_weight = y_true * alpha_tensor
            loss = alpha_weight * loss
        else:
            loss = self.alpha * loss
        
        return ops.sum(loss, axis=-1)
    
    def get_config(self):
        """Return config for serialization."""
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
        })
        return config


#####################################################################
# ATTENTION MECHANISMS
#####################################################################

def spatial_attention_3d(x: Tensor, kernel_size: int = 5, name_prefix: str = 'spatial_attention') -> Tensor:
    """
    Spatial Attention for 3D data.
    
    Learns to emphasize coherent blob regions and suppress scattered noise.
    The 5x5x5 kernel sees enough context to distinguish "blob-like" patterns
    from isolated pixels.
    
    Args:
        x: Input features (batch, h, w, d, channels)
        kernel_size: Size of attention convolution. Larger = more context.
                    Recommended: 5 for 14x14x26 inputs
        name_prefix: Prefix for layer names
    
    Returns:
        Attention-weighted features
    """
    # FIXED: Replace Lambda layers with custom serializable layers
    avg_pool = ReduceMean3D(name=f'{name_prefix}_avg_pool')(x)
    max_pool = ReduceMax3D(name=f'{name_prefix}_max_pool')(x)
    
    concat = Concatenate(axis=-1, name=f'{name_prefix}_concat')([avg_pool, max_pool])
    
    # Learn spatial attention map
    attention = Conv3D(
        filters=1,
        kernel_size=(kernel_size, kernel_size, kernel_size),
        padding='same',
        activation='sigmoid',
        name=f'{name_prefix}_conv'
    )(concat)
    
    # Apply attention weights
    return Multiply(name=f'{name_prefix}_multiply')([x, attention])

#####################################################################
# MODEL BUILDERS
#####################################################################


def build_improved_model(
    shape: Tuple[int, int, int, int] = (14, 14, 26, 2),
    network_depth: layer_type = "18-layer",
    learning_rate: float = 0.0003,
    starting_features: int = 64,
    dropout_rate: float = 0.3,
    axis = 3,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    use_attention: bool = True,
    attention_kernel_size: int = 5,
    l2_reg: float = 1e-4,
    optimizer: Optional[Optimizer] = None,
    number_classes: int = 1,
    classification_activation: str = None,
    multiscale_reduction: int = 1,
    use_group_norm: bool = True,  # NEW: Toggle for GroupNorm vs BatchNorm
    num_groups: int = 8,  # NEW: Number of groups for GroupNorm
) -> Model:
    """
    Build improved ResNet model with optional Group Normalization.
    
    NEW PARAMETERS:
        use_group_norm: If True, uses GroupNormalization instead of BatchNormalization.
                       HIGHLY RECOMMENDED for datasets with varying imaging conditions.
        num_groups: Number of groups for GroupNormalization. 
                   Rule of thumb: 4-16 channels per group works well.
                   For 64 features: use 8 groups (8 channels per group)
                   For 32 features: use 8 groups (4 channels per group)
    """

    blocks, bottleneck = get_resnet_blocks_and_bottleneck(network_depth)
    
    inputs = Input(shape, name='input')
    x = non_residual_block(
        inputs, 
        starting_features, 
        axis=axis,
        use_group_norm=use_group_norm,
        num_groups=num_groups
    )
    
    # Multi-scale feature collection
    feature_maps = []
    
    # Residual units with progressive downsampling
    for i, num_blocks in enumerate(blocks):
        features = starting_features * (2 ** i)
        
        for block in range(num_blocks):
            residual_func = residual_block(
                features,
                resnet_unit_id=i,
                block_id=block,
                bottleneck=bottleneck,
                axis=axis,
                use_group_norm=use_group_norm,
                num_groups=num_groups
            )
            x = residual_func(x)
        
        # Collect features from each scale for multi-scale processing
        if multiscale_reduction > 1 and i < len(blocks) - 1:
            feature_maps.append(x)
    
    # Multi-scale feature fusion
    if multiscale_reduction > 1 and len(feature_maps) > 0:
        # Upsample lower resolution features to match final resolution
        target_shape = ops.shape(x)[1:4]  # (D, H, W)
        upsampled_features = []
        
        for feat_map in feature_maps:
            # Simple nearest neighbor upsampling for 3D
            current_shape = ops.shape(feat_map)[1:4]
            
            # Calculate upsampling factors
            factors = [
                target_shape[i] / current_shape[i] 
                for i in range(3)
            ]
            
            # Note: Keras doesn't have native 3D upsampling
            # Using Conv3DTranspose as alternative
            num_filters = ops.shape(feat_map)[-1]
            upsampled = Conv3D(
                filters=num_filters,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                name=f'multiscale_up_{len(upsampled_features)}'
            )(feat_map)
            upsampled_features.append(upsampled)
        
        # Concatenate all scales
        if len(upsampled_features) > 0:
            x = Concatenate(axis=-1)([x] + upsampled_features)
            
            # Reduce channels with 1x1x1 conv
            final_features = starting_features * (2 ** (len(blocks) - 1))
            x = Conv3D(
                filters=final_features,
                kernel_size=(1, 1, 1),
                padding='same',
                name='multiscale_fusion'
            )(x)
            
            if use_group_norm:
                x = GroupNormalization3D(
                    groups=num_groups,
                    name='multiscale_fusion_gn'
                )(x)
            else:
                x = BatchNormalization(
                    axis=axis,
                    name='multiscale_fusion_bn'
                )(x)
    
    # Apply spatial attention if requested
    if use_attention:
        x = spatial_attention_3d(
            x, 
            kernel_size=attention_kernel_size,
            name_prefix='final_attention'
        )
        
        # Add dropout after attention to prevent over-focusing
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name='attention_dropout')(x)
    
    # Global pooling
    x = GlobalAveragePooling3D(name='global_avg_pool')(x)
    
    # Optional dropout before classification
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name='pre_classification_dropout')(x)
    
    # Classification head
    if classification_activation is None:
        classification_activation = 'sigmoid' if number_classes == 1 else 'softmax'
    
    x = Dense(
        number_classes,
        activation=classification_activation,
        name='classification',
        kernel_regularizer=l2_regularizer(l2_reg) if l2_reg > 0 else None
    )(x)
    
    # Compile model
    if use_focal_loss:
        if number_classes == 1:
            loss = BinaryFocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        else:
            loss = CategoricalFocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    else:
        loss = 'binary_crossentropy' if number_classes == 1 else 'categorical_crossentropy'
    
    model = Model(inputs=inputs, outputs=x)

    if optimizer is None:
        optimizer = Adam(learning_rate=learning_rate)

    recall_at_90p = keras.metrics.RecallAtPrecision(
        precision=0.90,
        num_thresholds=200,
        class_id=1,
        name='recall_at_p90'
    )

    recall_at_95p = keras.metrics.RecallAtPrecision(
        precision=0.95,
        num_thresholds=200,
        class_id=1,
        name='recall_at_p95'
    )

    model.compile(
        optimizer, 
        loss=loss, 
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision', class_id=1),
            keras.metrics.Recall(name='recall', class_id=1),
            recall_at_90p,
            recall_at_95p
        ]
    )
    return model

def get_resnet_blocks_and_bottleneck(
    network_depth: layer_type,
):
    """
    Parses dicts, and returns how many resnet blocks are in each unit, along
    with whether they are bottleneck blocks or not

    :param network_depth:
    :return:
    """
    blocks = resnet_unit_blocks[network_depth]
    bottleneck = network_residual_bottleneck[network_depth]
    return blocks, bottleneck


def non_residual_block(
    inputs: Tensor,
    starting_features: int,
    conv_kernel: Tuple[int, int, int] = (7, 7, 3),
    strides: Tuple[int, int, int] = (2, 2, 2),
    padding: int = 3,
    max_pool_size: Tuple[int, int, int] = (3, 3, 2),
    activation: str = "relu",
    use_bias: bool = False,
    bn_epsilon: float = 1e-5,
    pooling_padding: str = "valid",
    axis: int = 3,
    use_group_norm: bool = True,
    num_groups: int = 8,
):
    """
    Non-residual unit from He et al. (2015). Corresponds to "conv1" and the
    max pool.
    
    Modified to support Group Normalization.
    """

    x = ZeroPadding3D(padding=padding, name="conv1_padding")(inputs)
    x = Conv3D(
        starting_features,
        conv_kernel,
        strides=strides,
        use_bias=use_bias,
        name="conv1",
    )(x)
    
    # CRITICAL CHANGE: Use GroupNorm instead of BatchNorm
    if use_group_norm:
        x = GroupNormalization3D(
            groups=num_groups,
            epsilon=bn_epsilon,
            name="conv1_gn"
        )(x)
    else:
        x = BatchNormalization(axis=axis, epsilon=bn_epsilon, name="conv1_bn")(x)
    
    x = Activation(activation, name="conv1_activation")(x)

    x = MaxPooling3D(
        max_pool_size,
        strides=strides,
        padding=pooling_padding,
        name="max_pool",
    )(x)

    return x


def residual_block(
    output_features: Tensor,
    resnet_unit_id: int,
    block_id: int,
    conv_kernel: Tuple[int, int, int] = (3, 3, 3),
    bottleneck_conv_kernel: Tuple[int, int, int] = (1, 1, 1),
    bottleneck: int = False,
    activation: str = "relu",
    use_bias: bool = False,
    kernel_initializer: str = "he_normal",
    bn_epsilon: float = 1e-5,
    axis: int = 3,
    use_group_norm: bool = True,
    num_groups: int = 8,
):
    """
    Residual unit from He et al. (2015)
    
    Modified to support Group Normalization.
    """

    stride = get_stride(resnet_unit_id, block_id)
    resnet_unit_label = resnet_unit_id + 2

    def f(x: Tensor) -> Tensor:
        if bottleneck:
            y = Conv3D(
                output_features,
                bottleneck_conv_kernel,
                strides=stride,
                use_bias=use_bias,
                name=f"resunit{resnet_unit_label}_block{block_id}_conv_a",
                kernel_initializer=kernel_initializer,
            )(x)
        else:
            y = ZeroPadding3D(
                padding=1,
                name=f"resunit{resnet_unit_label}_block{block_id}_pad_a",
            )(x)

            y = Conv3D(
                output_features,
                conv_kernel,
                strides=stride,
                use_bias=use_bias,
                name=f"resunit{resnet_unit_label}_block{block_id}_conv_a",
                kernel_initializer=kernel_initializer,
            )(y)

        # CRITICAL CHANGE: Use GroupNorm instead of BatchNorm
        if use_group_norm:
            y = GroupNormalization3D(
                groups=num_groups,
                epsilon=bn_epsilon,
                name=f"resunit{resnet_unit_label}_block{block_id}_gn_a",
            )(y)
        else:
            y = BatchNormalization(
                axis=axis,
                epsilon=bn_epsilon,
                name=f"resunit{resnet_unit_label}_block{block_id}_bn_a",
            )(y)

        y = Activation(
            activation,
            name=f"resunit{resnet_unit_label}_block{block_id}_activation_a",
        )(y)

        y = ZeroPadding3D(
            padding=1, name=f"resunit{resnet_unit_label}_block{block_id}_pad_b"
        )(y)

        y = Conv3D(
            output_features,
            conv_kernel,
            use_bias=use_bias,
            name=f"resunit{resnet_unit_label}_block{block_id}_conv_b",
            kernel_initializer=kernel_initializer,
        )(y)

        # CRITICAL CHANGE: Use GroupNorm instead of BatchNorm
        if use_group_norm:
            y = GroupNormalization3D(
                groups=num_groups,
                epsilon=bn_epsilon,
                name=f"resunit{resnet_unit_label}_block{block_id}_gn_b",
            )(y)
        else:
            y = BatchNormalization(
                axis=axis,
                epsilon=bn_epsilon,
                name=f"resunit{resnet_unit_label}_block{block_id}_bn_b",
            )(y)

        if bottleneck:
            y = Activation(
                activation,
                name=f"resunit{resnet_unit_label}_block"
                f"{block_id}_activation_b",
            )(y)

            y = Conv3D(
                output_features * 4,
                bottleneck_conv_kernel,
                use_bias=use_bias,
                name=f"resunit{resnet_unit_label}_block{block_id}_conv_c",
                kernel_initializer=kernel_initializer,
            )(y)

            # CRITICAL CHANGE: Use GroupNorm instead of BatchNorm
            if use_group_norm:
                y = GroupNormalization3D(
                    groups=num_groups,
                    epsilon=bn_epsilon,
                    name=f"resunit{resnet_unit_label}_block{block_id}_gn_c",
                )(y)
            else:
                y = BatchNormalization(
                    axis=axis,
                    epsilon=bn_epsilon,
                    name=f"resunit{resnet_unit_label}_block{block_id}_bn_c",
                )(y)

            identity_shortcut = get_shortcut(
                x,
                resnet_unit_label,
                block_id,
                output_features * 4,
                stride,
                axis=axis,
                use_group_norm=use_group_norm,
                num_groups=num_groups,
            )
        else:
            identity_shortcut = get_shortcut(
                x,
                resnet_unit_label,
                block_id,
                output_features,
                stride,
                axis=axis,
                use_group_norm=use_group_norm,
                num_groups=num_groups,
            )

        y = Add(name=f"resunit{resnet_unit_label}_block{block_id}_add")(
            [y, identity_shortcut]
        )

        y = Activation(
            activation,
            name=f"resunit{resnet_unit_label}_block{block_id}_activation_c",
        )(y)

        return y

    return f


def get_shortcut(
    inputs: Tensor,
    resnet_unit_label: int,
    block_id: int,
    features: int,
    stride: int,
    use_bias: bool = False,
    kernel_initializer: Union[str, Initializer] = "he_normal",
    bn_epsilon: float = 1e-5,
    axis: int = 3,
    use_group_norm: bool = True,
    num_groups: int = 8,
):
    """
    Create shortcut. For none-bottleneck residual units, this is just the
    identity. Otherwise, the input is reshaped to match the output of the
    bottleneck unit
    
    Modified to support Group Normalization.
    """

    if block_id == 0:
        shortcut = Conv3D(
            features,
            (1, 1, 1),
            strides=stride,
            use_bias=use_bias,
            name=f"resunit{resnet_unit_label}_block{block_id}_shortcut_conv",
            kernel_initializer=kernel_initializer,
        )(inputs)

        # CRITICAL CHANGE: Use GroupNorm instead of BatchNorm
        if use_group_norm:
            shortcut = GroupNormalization3D(
                groups=num_groups,
                epsilon=bn_epsilon,
                name=f"resunit{resnet_unit_label}_block{block_id}_shortcut_gn",
            )(shortcut)
        else:
            shortcut = BatchNormalization(
                axis=axis,
                epsilon=bn_epsilon,
                name=f"resunit{resnet_unit_label}_block{block_id}_shortcut_bn",
            )(shortcut)
        return shortcut
    else:
        return inputs


def get_stride(resnet_unit_id: int, block_id: int):
    """
    Determines the convolution stride.

    :param int resnet_unit_id: Which resnet unit
    (e.g. 0 is conv2_x, 1 is conv3_x)
    :param int block_id: Which block in the resnet unit (i.e. in the third
    unit of a 152-layer resnet, block_id=36)
    :return: Stride
    """
    if resnet_unit_id == 0 or block_id != 0:
        return 1
    else:
        return 2