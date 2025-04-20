import mrcfile
import numpy as np
import os

def calculate_error_metrics(original, predicted, epsilon=1e-8):
    """
    Calculate error metrics focused on percentage/ratio between predicted and actual values.
    
    Args:
        original: Ground truth values
        predicted: Predicted values
        epsilon: Small value to avoid division by zero
    
    Returns:
        Dictionary of error metrics
    """
    ratio = predicted / (original + epsilon)
    rel_error = np.abs(predicted - original) / (np.abs(original) + epsilon) * 100
    
    return {
        'mean_ratio': np.mean(ratio),
        'median_ratio': np.median(ratio),
        'mean_rel_error_percent': np.mean(rel_error),
        'median_rel_error_percent': np.median(rel_error),
        'within_10_percent': np.mean(np.abs(ratio - 1.0) <= 0.10) * 100,
        'within_20_percent': np.mean(np.abs(ratio - 1.0) <= 0.20) * 100,
        'within_30_percent': np.mean(np.abs(ratio - 1.0) <= 0.30) * 100,
        'overestimation_percent': np.mean(ratio > 1.0) * 100,
        'underestimation_percent': np.mean(ratio < 1.0) * 100,
        'perfect_prediction_percent': np.mean(ratio == 1.0) * 100,
        'rel_error_array': rel_error
    }

def evaluate_reconstruction(original_dir, reconstructed_dir, output_dir, threshold=0.05, file_names=None):
    """
    Evaluate reconstruction quality across all density values above threshold.
    
    Args:
        original_dir: Directory with original .mrc files
        reconstructed_dir: Directory with reconstructed .mrc files
        output_dir: Directory to save results
        threshold: Minimum density threshold for evaluation (adjustable)
        file_names: List of file names to evaluate (without .mrc extension)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if file_names is None:
        file_names = [f.replace('.mrc', '') for f in os.listdir(original_dir) if f.endswith('.mrc')]
    print(f"Evaluating {len(file_names)} files: {file_names}")

    summary_path = os.path.join(output_dir, "error_summary.csv")
    with open(summary_path, 'w') as f:
        f.write("Filename,Num_Points,Mean_Ratio,Median_Ratio,Mean_Error(%),Median_Error(%),Within_5%,Within_10%,Within_20%,Overestimation(%),Underestimation(%)\n")
    
    for fname in file_names:
        print(f"\nEvaluating File: {fname}")
        
        # Load original .mrc file
        original_mrc_path = os.path.join(original_dir, f"{fname}.mrc")
        if not os.path.exists(original_mrc_path):
            print(f"Original file not found: {original_mrc_path}")
            continue
        original_data = mrcfile.read(original_mrc_path)
        
        # Load reconstructed .mrc file
        reconstructed_mrc_path = os.path.join(reconstructed_dir, f"{fname}_reconstructed.mrc")
        if not os.path.exists(reconstructed_mrc_path):
            print(f"Reconstructed file not found: {reconstructed_mrc_path}")
            continue
        with mrcfile.open(reconstructed_mrc_path, mode='r') as mrc:
            reconstructed_data = mrc.data
        
        # Verify shapes match
        if original_data.shape != reconstructed_data.shape:
            print(f"Shape mismatch: Original {original_data.shape} vs Reconstructed {reconstructed_data.shape}")
            continue
        
        # Mask to points where original density > threshold
        positive_mask = original_data > threshold
        num_positive_points = np.sum(positive_mask)
        if num_positive_points == 0:
            print(f"No points with density > {threshold} found in original data for {fname}")
            continue
        
        original_positive = original_data[positive_mask]
        reconstructed_positive = reconstructed_data[positive_mask]
        
        print(f"Evaluating {num_positive_points:,} points with density > {threshold} "
              f"({100 * num_positive_points / original_data.size:.2f}% of total)")

        # Calculate metrics for all points above threshold
        metrics = calculate_error_metrics(original_positive, reconstructed_positive)
        
        print(f"\nResults for density > {threshold}:")
        print(f"Mean Relative Error: {metrics['mean_rel_error_percent']:.2f}%")
        print(f"Median Relative Error: {metrics['median_rel_error_percent']:.2f}%")
        print(f"Within 10% error: {metrics['within_10_percent']:.2f}%")
        print(f"Within 20% error: {metrics['within_20_percent']:.2f}%")
        print(f"Within 30% error: {metrics['within_30_percent']:.2f}%")
        # print(f"Overestimation: {metrics['overestimation_percent']:.2f}%")
        # print(f"Underestimation: {metrics['underestimation_percent']:.2f}%")
        
        # Write to summary
        with open(summary_path, 'a') as f:
            f.write(f"{fname},{num_positive_points},{metrics['mean_ratio']:.4f},{metrics['median_ratio']:.4f},"
                    f"{metrics['mean_rel_error_percent']:.2f},{metrics['median_rel_error_percent']:.2f},"
                    f"{metrics['within_10_percent']:.2f},{metrics['within_20_percent']:.2f},"
                    f"{metrics['within_30_percent']:.2f},{metrics['overestimation_percent']:.2f},"
                    f"{metrics['underestimation_percent']:.2f}\n")

def main():
    original_dir = "/home/zcy/NFD/denoised_sample/"
    reconstructed_dir = "/home/zcy/seperate_VAE/output"
    output_dir = "/home/zcy/seperate_VAE/error_percentage_evaluation"
    file_names = None
    threshold = 0.04  # User can adjust this value
    
    evaluate_reconstruction(original_dir, reconstructed_dir, output_dir, threshold, file_names)

if __name__ == "__main__":
    main()