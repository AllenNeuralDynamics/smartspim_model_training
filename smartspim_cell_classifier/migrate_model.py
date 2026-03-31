"""
Simplified Model Migration Script
Loads model with Lambda layers and re-saves with custom serializable layers.

Usage:
    python migrate_model_simple.py --input old_model.keras --output new_model.keras
"""

import argparse
import keras
import tensorflow as tf
from pathlib import Path
import sys

# Try to import from different possible locations
try:
    from improved_resnet_v2 import ReduceMean3D, ReduceMax3D, binary_focal_loss, categorical_focal_loss, f1_loss
except ImportError:
    try:
        from utils.improved_resnet_v2 import ReduceMean3D, ReduceMax3D, binary_focal_loss, categorical_focal_loss, f1_loss
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from improved_resnet_v2 import ReduceMean3D, ReduceMax3D, binary_focal_loss, categorical_focal_loss, f1_loss


def migrate_model(input_path: str, output_path: str):
    """
    Load model with Lambda layers using safe_mode=False, then re-save.
    The custom layers are already registered via decorator.
    """
    print(f"Loading model from: {input_path}")
    
    # Define custom objects for loading
    custom_objects = {
        'ReduceMean3D': ReduceMean3D,
        'ReduceMax3D': ReduceMax3D,
        'binary_focal_loss': binary_focal_loss,
        'categorical_focal_loss': categorical_focal_loss,
        'f1_loss': f1_loss,
    }
    
    # Load the old model with safe_mode=False
    # This allows loading Lambda layers
    old_model = keras.models.load_model(
        input_path,
        custom_objects=custom_objects,
        safe_mode=False,
        compile=True  # Load with compilation
    )
    
    print("✓ Model loaded successfully!")
    print(f"\nModel summary:")
    old_model.summary()
    
    # Get model config and weights
    config = old_model.get_config()
    weights = old_model.get_weights()
    
    # Check for Lambda layers
    lambda_count = sum(1 for layer_cfg in config.get('layers', []) 
                       if layer_cfg.get('class_name') == 'Lambda')
    print(f"\nFound {lambda_count} Lambda layers in model")
    
    # Manually replace Lambda layers in config
    print("\nReplacing Lambda layers...")
    replaced = replace_lambda_layers_in_config(config)
    print(f"Replaced {replaced} Lambda layers")
    
    # Rebuild model from modified config
    print("\nRebuilding model from modified config...")
    
    # Use Keras 3 custom_object_scope
    with keras.saving.custom_object_scope(custom_objects):
        new_model = keras.Model.from_config(config)
    
    print("✓ Model rebuilt successfully!")
    
    # Set weights
    print("\nCopying weights...")
    try:
        new_model.set_weights(weights)
        print("✓ Weights copied successfully!")
    except Exception as e:
        print(f"Warning: Could not copy all weights: {e}")
        print("Copying weights layer by layer...")
        copy_weights_by_layer(old_model, new_model)
    
    # Compile if original was compiled
    if hasattr(old_model, 'optimizer') and old_model.optimizer is not None:
        print("\nRecompiling model...")
        new_model.compile(
            optimizer=old_model.optimizer.__class__.from_config(old_model.optimizer.get_config()),
            loss=old_model.loss,
            metrics=[m.name if hasattr(m, 'name') else m for m in old_model.metrics]
        )
        print("✓ Model recompiled!")
    
    # Save the new model
    print(f"\nSaving migrated model to: {output_path}")
    new_model.save(output_path)
    
    print("\n✓ Migration complete!")
    print(f"✓ New model saved to: {output_path}")
    
    # Verify it can be loaded normally
    print("\nVerifying model can be loaded...")
    try:
        # The model still has a lambda in the loss function, so we need safe_mode=False
        # But the Lambda LAYERS have been replaced with ReduceMean3D/ReduceMax3D
        test_model = keras.models.load_model(
            output_path,
            custom_objects=custom_objects,
            safe_mode=False  # Still needed for the loss function lambda
        )
        print("✓ Verification successful!")
        print("\nNote: The model can now be loaded, but still requires safe_mode=False")
        print("because the loss function uses a lambda. The Lambda LAYERS have been")
        print("successfully replaced with ReduceMean3D and ReduceMax3D.")
    except Exception as e:
        print(f"! Verification warning: {e}")
        print("The model was saved successfully, but may require custom_objects when loading.")
    
    return new_model


def replace_lambda_layers_in_config(config):
    """Replace Lambda layers with custom layers in the config dict."""
    replaced = 0
    
    if 'layers' not in config:
        return replaced
    
    for layer_cfg in config['layers']:
        if layer_cfg.get('class_name') != 'Lambda':
            continue
        
        layer_name = layer_cfg.get('config', {}).get('name', '')
        
        if 'avg_pool' in layer_name or 'mean' in layer_name.lower():
            # Replace with ReduceMean3D
            old_name = layer_cfg['config'].get('name', 'unknown')
            layer_cfg['class_name'] = 'ReduceMean3D'
            layer_cfg['module'] = 'Custom'  # IMPORTANT: Set the registered package
            layer_cfg['registered_name'] = 'Custom>ReduceMean3D'
            layer_cfg['config'] = {
                'name': old_name,
                'trainable': layer_cfg['config'].get('trainable', True),
                'dtype': layer_cfg['config'].get('dtype', 'float32'),
            }
            replaced += 1
            print(f"  ✓ {old_name}: Lambda → ReduceMean3D")
            
        elif 'max_pool' in layer_name or 'max' in layer_name.lower():
            # Replace with ReduceMax3D
            old_name = layer_cfg['config'].get('name', 'unknown')
            layer_cfg['class_name'] = 'ReduceMax3D'
            layer_cfg['module'] = 'Custom'  # IMPORTANT: Set the registered package
            layer_cfg['registered_name'] = 'Custom>ReduceMax3D'
            layer_cfg['config'] = {
                'name': old_name,
                'trainable': layer_cfg['config'].get('trainable', True),
                'dtype': layer_cfg['config'].get('dtype', 'float32'),
            }
            replaced += 1
            print(f"  ✓ {old_name}: Lambda → ReduceMax3D")
        else:
            print(f"  ! Warning: Unknown Lambda layer: {layer_name}")
    
    return replaced


def copy_weights_by_layer(old_model, new_model):
    """Copy weights layer by layer, skipping layers without weights."""
    old_layers = {layer.name: layer for layer in old_model.layers}
    
    copied = 0
    skipped = 0
    
    for new_layer in new_model.layers:
        if new_layer.name not in old_layers:
            # New layer (probably replacing a Lambda)
            skipped += 1
            continue
        
        old_layer = old_layers[new_layer.name]
        old_weights = old_layer.get_weights()
        
        if len(old_weights) == 0:
            # No weights to copy
            continue
        
        try:
            new_layer.set_weights(old_weights)
            copied += 1
        except Exception as e:
            print(f"    ! Could not copy weights for {new_layer.name}: {e}")
            skipped += 1
    
    print(f"  ✓ Copied weights for {copied} layers, skipped {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate Keras model from Lambda layers to serializable custom layers'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input model file (.keras, .h5)'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to output model file (.keras recommended)'
    )
    
    args = parser.parse_args()
    
    # Verify input exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Run migration
    try:
        migrate_model(args.input, args.output)
        return 0
    except Exception as e:
        print(f"\n✗ Migration failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())