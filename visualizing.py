import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_adversarial_samples(original_images, adv_images, original_labels, adv_preds, clean_preds=None, class_names=None, num_samples=5, filename='adversarial_samples.png'):
    """
    Visualizes and saves adversarial examples.
    
    Args:
        original_images (Tensor): Batch of original images [C, H, W] with values in [0, 1].
        adv_images (Tensor): Batch of adversarial images [C, H, W] with values in [0, 1].
        original_labels (Tensor): True labels of the original images.
        adv_preds (Tensor): Incorrect predictions made by the model for the adversarial images.
        clean_preds (Tensor, optional): Clean predictions made by the model for the original images.
        class_names (list, optional): List of class names (e.g., ['airplane', 'automobile', ...]).
        num_samples (int): Number of samples to visualize. Defaults to 5.
        filename (str): Filename to save the visualization. Defaults to 'adversarial_samples.png'.
    """
    # 1. Create a directory to save the results
    os.makedirs('results', exist_ok=True)
    
    # Limit the number of samples to the actual batch size if it's smaller than num_samples
    num_samples = min(num_samples, len(original_images))
    
    # 2. Setup Matplotlib Figure (nrows=num_samples, ncols=3)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    fig.suptitle(f'Adversarial Attack Visualization ({filename})', fontsize=16)

    for i in range(num_samples):
        # Convert tensors to CPU and NumPy arrays, and change shape from (C, H, W) to (H, W, C)
        orig_img = original_images[i].detach().cpu().permute(1, 2, 0).numpy()
        adv_img = adv_images[i].detach().cpu().permute(1, 2, 0).numpy()
        
        # 3. Calculate perturbation and magnify it for visibility
        # Calculate absolute difference and multiply by a factor (e.g., 50) to make it visible
        perturbation = np.abs(adv_img - orig_img)
        magnified_pert = np.clip(perturbation * 50, 0, 1) # Clip to ensure values stay within [0, 1]
        
        # Prepare label text
        orig_true_text = class_names[original_labels[i]] if class_names else str(original_labels[i].item())
        clean_pred_text = ""
        if clean_preds is not None:
             clean_p = class_names[clean_preds[i]] if class_names else str(clean_preds[i].item())
             clean_pred_text = f"Pred: {clean_p}\n"
        
        adv_label_text = class_names[adv_preds[i]] if class_names else str(adv_preds[i].item())

        # (1) Plot original image
        ax_orig = axes[i, 0] if num_samples > 1 else axes[0]
        # For grayscale images (MNIST), we need to squeeze the last dimension or check shape
        if orig_img.shape[2] == 1:
            ax_orig.imshow(np.clip(orig_img, 0, 1), cmap='gray')
        else:
            ax_orig.imshow(np.clip(orig_img, 0, 1))
        
        ax_orig.set_title(f'Original\n{clean_pred_text}True: {orig_true_text}')
        ax_orig.axis('off')

        # (2) Plot adversarial image (model's incorrect prediction)
        ax_adv = axes[i, 1] if num_samples > 1 else axes[1]
        if adv_img.shape[2] == 1:
            ax_adv.imshow(np.clip(adv_img, 0, 1), cmap='gray')
        else:
            ax_adv.imshow(np.clip(adv_img, 0, 1))
        ax_adv.set_title(f'Adversarial\nPred: {adv_label_text}')
        ax_adv.axis('off')

        # (3) Plot magnified perturbation
        ax_pert = axes[i, 2] if num_samples > 1 else axes[2]
        if magnified_pert.shape[2] == 1:
            ax_pert.imshow(magnified_pert, cmap='gray')
        else:
            ax_pert.imshow(magnified_pert)
        ax_pert.set_title('Perturbation\n(Magnified)')
        ax_pert.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Adjust layout to make room for the main title
    
    # 4. Save the figure as an image file
    save_path = os.path.join('results', filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
    # Close the plot to free memory
    plt.close(fig)