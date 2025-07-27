# PWM_validation_example.py
# Example usage of the PWM validation framework (Final Version)
# Updated to match exact training data format and model interface
# -----------------------------------------------------------------------------
# Data format from training:
# - observations: (batch_size, sequence_length, 84, 84) - normalized [0,1]
# - actions: (batch_size, sequence_length) - long tensor
# -----------------------------------------------------------------------------

import torch
import torch.utils.data as data
import os
import sys
from model.PWM import PWM  # Import your PWM model
from PWM_validation import PWMValidator, run_validation_example, create_validation_dataloader

def get_config(config_name: str = "normal"):
    """
    Configuration function matching the training script.
    """
    if config_name not in ["light", "normal", "large"]:
        config_name = "normal"
    d_models = {"light": 256, "normal": 512, "large": 768}
    d_model = d_models.get(config_name, 512)
    configs = {
        "light": {
            "d_model": d_model, "dropout": 0.1, "seq_length": 32, "pos_encoding_type": "learnable", "loss_type": "combined",
            "img_encoder_blocks": 2, "img_encoder_groups": 4, "action_encoder_hidden_dim": d_model,
            "transformer_layers": 3, "transformer_heads": 4, "transformer_d_ff": d_model * 4,
            "variance_weight": 0.0, "num_actions": 18
        },
        "normal": {
            "d_model": d_model, "dropout": 0.1, "seq_length": 32, "pos_encoding_type": "learnable", "loss_type": "combined",
            "img_encoder_blocks": 4, "img_encoder_groups": 8, "action_encoder_hidden_dim": d_model,
            "transformer_layers": 6, "transformer_heads": 8, "transformer_d_ff": d_model * 4,
            "variance_weight": 0.0, "num_actions": 18
        },
        "large": {
            "d_model": d_model, "dropout": 0.1, "seq_length": 32, "pos_encoding_type": "learnable", "loss_type": "combined",
            "img_encoder_blocks": 5, "img_encoder_groups": 16, "action_encoder_hidden_dim": d_model,
            "transformer_layers": 8, "transformer_heads": 12, "transformer_d_ff": d_model * 4,
            "variance_weight": 0.0, "num_actions": 18
        },
    }
    return configs.get(config_name)

def load_trained_model(model_path, config_name='normal', device='cuda'):
    """
    Load a trained PWM model from checkpoint, matching training script format.
    
    Args:
        model_path: Path to the model checkpoint
        config_name: Configuration name used during training
        device: Device to load the model on
    
    Returns:
        Loaded PWM model
    """
    # Get the same configuration used during training
    config = get_config(config_name)
    
    # Initialize model
    model = PWM(config)
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
        
        print(f"✅ Model loaded successfully from {model_path}")
    else:
        print(f"⚠️  Warning: Model file {model_path} not found. Using randomly initialized model for testing.")
    
    model.to(device)
    model.eval()
    
    return model

def main():
    """
    Main validation pipeline matching training script format.
    """
    # Configuration - Update these paths to match your setup
    MODEL_PATH = 'Output/episode_0/ckp/EvalBest.pth'  # Update this path
    DATA_DIR = 'testData'  # Directory containing .h5 files (same as training)
    CONFIG_NAME = 'normal'  # Should match the config used during training
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 8
    NUM_VALIDATION_BATCHES = 50  # Set to None to validate on entire dataset
    
    print("="*70)
    print("PWM MODEL VALIDATION PIPELINE (FINAL)")
    print("="*70)
    print("📊 Data format:")
    print("   - Observations: (batch_size, sequence_length, 84, 84) [normalized 0-1]")
    print("   - Actions: (batch_size, sequence_length) [long tensor]")
    print("🎯 Validation outputs:")
    print("   - Similarity heatmaps (correlation-like visualization)")
    print("   - Hinton diagram (output activation patterns)")
    print("   - Performance curves (MSE & cosine distance vs sequence)")
    print("="*70)
    
    # 1. Load trained model
    print("\n1. Loading trained PWM model...")
    model = load_trained_model(MODEL_PATH, CONFIG_NAME, DEVICE)
    print(f"   Device: {DEVICE}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model config: {CONFIG_NAME}")
    
    # 2. Create validation dataloader
    print("\n2. Setting up validation data...")
    
    if os.path.exists(DATA_DIR) and len([f for f in os.listdir(DATA_DIR) if f.endswith('.h5')]) > 0:
        print(f"   📁 Loading H5 data from: {DATA_DIR}")
        dataloader = create_validation_dataloader(
            DATA_DIR, 
            batch_size=BATCH_SIZE,
            num_workers=0,  # Set to 0 for debugging, increase for faster loading
            shuffle=False,
            normalize_obs=True  # Match training normalization
        )
        print(f"   ✅ Validation dataset loaded")
        print(f"   📊 Expected batch format:")
        print(f"      - observations: ({BATCH_SIZE}, seq_length, 84, 84)")
        print(f"      - actions: ({BATCH_SIZE}, seq_length)")
    else:
        print("   ⚠️  H5 data directory not found or empty. Using mock data for demonstration...")
        from PWM_validation import create_mock_dataloader
        dataloader = create_mock_dataloader(
            batch_size=BATCH_SIZE, 
            seq_length=16,  # Use shorter sequence for testing
            num_batches=10
        )
        print("   🧪 Mock dataloader created for testing")
    
    # 3. Test data format compatibility
    print("\n3. Testing data format compatibility...")
    try:
        sample_batch = next(iter(dataloader))
        obs_shape = sample_batch['observations'].shape
        actions_shape = sample_batch['actions'].shape
        print(f"   ✅ Observations shape: {obs_shape}")
        print(f"   ✅ Actions shape: {actions_shape}")
        print(f"   ✅ Observations range: [{sample_batch['observations'].min():.3f}, {sample_batch['observations'].max():.3f}]")
        print(f"   ✅ Actions range: [{sample_batch['actions'].min()}, {sample_batch['actions'].max()}]")
        
        # Test model forward pass
        sample_batch = {k: v.to(DEVICE) for k, v in sample_batch.items()}
        with torch.no_grad():
            # Prepare for model input (match training format)
            obs = sample_batch['observations']
            actions = sample_batch['actions']
            batch_size, seq_length, height, width = obs.shape
            
            # Convert to model expected format
            first_obs = obs[:, 0:1, :, :].unsqueeze(2).repeat(1, 1, 4, 1, 1)
            rest_obs = obs.unsqueeze(2).repeat(1, 1, 4, 1, 1)
            model_observations = torch.cat([first_obs, rest_obs], dim=1)
            
            model_output = model(observations=model_observations, actions=actions)
            print(f"   ✅ Model forward pass successful")
            print(f"   ✅ Model output shapes: predicted {model_output['predicted'].shape}, target {model_output['target'].shape}")
        
    except Exception as e:
        print(f"   ❌ Error in data format test: {e}")
        print("   Please check your model and data compatibility")
        return
    
    # 4. Initialize validator
    print("\n4. Initializing PWM validator...")
    validator = PWMValidator(
        model, 
        device=DEVICE, 
        save_dir='validation_results'
    )
    print("   ✅ Validator initialized")
    
    # 5. Run validation
    print("\n5. Running validation...")
    print("   📊 Computing image-to-image similarities (normalized observations)")
    print("   📈 Computing output-to-output similarities (MSE & cosine)")
    print("   🎯 Computing prediction accuracy (per-step analysis)")
    print("   🖼️  Generating visualizations...")
    
    results = validator.run_validation(
        dataloader, 
        num_batches=NUM_VALIDATION_BATCHES
    )
    
    # 6. Print detailed results
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    print("\n📊 IMAGE SIMILARITY METRICS:")
    for key, value in results.items():
        if 'image_' in key and isinstance(value, (int, float)):
            print(f"   {key}: {value:.6f}")
    
    print("\n📈 OUTPUT SIMILARITY METRICS:")
    for key, value in results.items():
        if 'output_' in key and isinstance(value, (int, float)):
            print(f"   {key}: {value:.6f}")
    
    print("\n🎯 PREDICTION ACCURACY METRICS:")
    for key, value in results.items():
        if 'pred_gt_' in key and isinstance(value, (int, float)):
            print(f"   {key}: {value:.6f}")
    
    print("\n📋 PER-STEP PERFORMANCE:")
    if 'per_step_mse_mean' in results:
        mse_values = results['per_step_mse_mean']
        display_mse = [f"{v:.4f}" for v in mse_values[:8]]
        if len(mse_values) > 8:
            display_mse.append("...")
        print(f"   MSE per step: {display_mse}")
    
    if 'per_step_cosine_distance_mean' in results:
        cos_values = results['per_step_cosine_distance_mean']
        display_cos = [f"{v:.4f}" for v in cos_values[:8]]
        if len(cos_values) > 8:
            display_cos.append("...")
        print(f"   Cosine distance per step: {display_cos}")
    
    # 7. Analysis and recommendations
    print("\n" + "="*70)
    print("ANALYSIS & RECOMMENDATIONS")
    print("="*70)
    
    analyze_results(results)
    
    print(f"\n✅ Validation completed! Generated files:")
    print(f"   📄 validation_results/validation_metrics_*.json - Complete metrics")
    print(f"   📊 validation_results/similarity_heatmaps_*.png - Correlation-like heatmaps")
    print(f"   🎯 validation_results/hinton_diagram_*.png - Output activation patterns")
    print(f"   📈 validation_results/performance_curves_*.png - Error vs sequence length")
    print(f"   📋 validation_results/similarity_details_*.csv - Detailed data for analysis")
    print(f"   💾 validation_results/step_outputs_*.pt - Raw outputs for custom plotting")
    
    return results

def analyze_results(results):
    """
    Analyze validation results and provide specific recommendations.
    """
    print("\n🔍 PERFORMANCE ANALYSIS:")
    
    # Check prediction accuracy
    if 'pred_gt_mse_overall_batch_mean' in results:
        mse_score = results['pred_gt_mse_overall_batch_mean']
        if mse_score < 0.1:
            print("   ✅ Excellent MSE performance (< 0.1)")
        elif mse_score < 0.5:
            print("   ⚠️  Moderate MSE performance (0.1-0.5)")
        else:
            print("   ❌ Poor MSE performance (> 0.5)")
            print("      Recommendation: Check training convergence and data quality")
    
    if 'pred_gt_cosine_distance_overall_batch_mean' in results:
        cos_dist_score = results['pred_gt_cosine_distance_overall_batch_mean']
        if cos_dist_score < 0.2:
            print("   ✅ Excellent cosine similarity (distance < 0.2)")
        elif cos_dist_score < 0.4:
            print("   ⚠️  Moderate cosine similarity (distance 0.2-0.4)")
        else:
            print("   ❌ Poor cosine similarity (distance > 0.4)")
            print("      Recommendation: Consider adjusting loss function or model architecture")
    
    # Check sequence consistency (error accumulation)
    if 'per_step_mse_mean' in results:
        step_mse = results['per_step_mse_mean']
        if len(step_mse) > 2:
            initial_error = step_mse[0]
            final_error = step_mse[-1]
            error_growth = (final_error - initial_error) / initial_error if initial_error > 0 else 0
            
            if error_growth > 0.5:  # 50% increase
                print("   ⚠️  Significant error accumulation over sequence")
                print("      Recommendation: Consider shorter sequences or architectural improvements")
            elif error_growth > 0.2:  # 20% increase
                print("   ⚠️  Moderate error accumulation detected")
                print("      Recommendation: Monitor longer sequences carefully")
            else:
                print("   ✅ Stable prediction quality across sequence")
    
    # Check output diversity
    if 'output_mse_similarity_mean_batch_mean' in results:
        output_diversity = results['output_mse_similarity_mean_batch_mean']
        if output_diversity < 0.01:
            print("   ⚠️  Very low output diversity - possible mode collapse")
            print("      Recommendation: Check training data diversity and loss function")
        elif output_diversity < 0.1:
            print("   ⚠️  Low output diversity detected")
            print("      Recommendation: Consider adding diversity regularization")
        else:
            print("   ✅ Good output diversity")
    
    # Check image processing consistency
    if 'image_mse_similarity_mean_batch_mean' in results:
        img_similarity = results['image_mse_similarity_mean_batch_mean']
        print(f"   📊 Average image similarity: {img_similarity:.4f}")
        if img_similarity > 2.0:
            print("      Note: High image dissimilarity suggests good visual diversity")
        elif img_similarity < 0.5:
            print("      Note: Low image dissimilarity - check for repetitive frames")
    
    print("\n💡 NEXT STEPS:")
    print("   1. 📊 Examine similarity_heatmaps_*.png for visual correlation patterns")
    print("   2. 🎯 Check hinton_diagram_*.png for output activation distribution")
    print("   3. 📈 Analyze performance_curves_*.png for error accumulation trends")
    print("   4. 📋 Import similarity_details_*.csv into your analysis tools")
    print("   5. 💾 Load step_outputs_*.pt for custom visualizations:")
    print("      ```python")
    print("      data = torch.load('validation_results/step_outputs_*.pt')")
    print("      predictions = data['all_predictions']  # For custom analysis")
    print("      ```")

def quick_test():
    """
    Quick test function using a randomly initialized model.
    Perfect for testing the validation framework without a trained model.
    """
    print("🧪 Running quick test with randomly initialized model...")
    print("   This will test all validation features:")
    print("   📊 Similarity heatmaps")
    print("   🎯 Hinton diagram")
    print("   📈 Performance curves")
    
    # Create a random model with light config for faster testing
    config = get_config('light')
    model = PWM(config)
    
    print(f"   ✅ Random model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Run validation with mock data
    results = run_validation_example(model, dataloader=None)
    
    print("\n🎉 Quick test completed successfully!")
    print("   Check 'validation_results' directory for generated visualizations.")
    print("   This confirms the validation framework is working correctly.")
    
    return results

def validate_trained_model(model_path, data_dir, config_name='normal'):
    """
    Convenience function to validate a specific trained model.
    
    Args:
        model_path: Path to the trained model checkpoint
        data_dir: Directory containing H5 validation data
        config_name: Model configuration ('light', 'normal', 'large')
    """
    print(f"🎯 Validating trained model: {model_path}")
    
    # Load model
    model = load_trained_model(model_path, config_name)
    
    # Create dataloader
    if os.path.exists(data_dir):
        dataloader = create_validation_dataloader(data_dir, batch_size=4, shuffle=False)
        print(f"   📁 Using validation data from: {data_dir}")
    else:
        print(f"   ⚠️  Data directory not found, using mock data")
        from PWM_validation import create_mock_dataloader
        dataloader = create_mock_dataloader(batch_size=4, seq_length=16, num_batches=5)
    
    # Run validation
    validator = PWMValidator(model, save_dir=f'validation_results_{config_name}')
    results = validator.run_validation(dataloader, num_batches=10)
    
    print(f"✅ Validation completed for {config_name} model!")
    return results

if __name__ == "__main__":
    # Uncomment the function you want to run:
    
    # For full validation with your trained model:
    main()
    
    # For quick testing without trained model:
    # quick_test()
    
    # For validating a specific model:
    # validate_trained_model('Output/episode_0/ckp/EvalBest.pth', 'testData', 'normal')