import torch
from tqdm import tqdm
from adversarial_attack import fgsm_targeted, fgsm_untargeted, pgd_targeted, pgd_untargeted
from visualizing import visualize_adversarial_samples


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def process_attack(model, adv_imgs, attack_name, targeted, predicted, labels, inputs, vis_data):
    _, adv_preds = torch.max(model(adv_imgs).data, 1)
    orig_correct = (predicted == labels)
    
    target_labels = (labels + 1) % 10
    
    if targeted:
        success_mask = orig_correct & (adv_preds == target_labels)
    else:
        success_mask = orig_correct & (adv_preds != labels)

    success_count = success_mask.sum().item()
    
    if len(vis_data[attack_name]) < 5:
        success_indices = torch.where(success_mask)[0]
        for idx in success_indices:
            if len(vis_data[attack_name]) < 5:
                vis_data[attack_name].append({
                    'orig_img': inputs[idx].cpu(),
                    'adv_img': adv_imgs[idx].cpu(),
                    'label': labels[idx].cpu(),
                    'adv_pred': adv_preds[idx].cpu()
                })
                
    return success_count

def evaluate(
    model,
    dataloader,
    criterion,
    device,
    use_attack=True,
    attack_eps=0.3,
    class_names=None,
    return_attack_metrics=False,
    save_visuals=True,
):
    model.eval()
    total_loss = 0
    correct = 0
    
    total_attack_samples = 0
    fgsm_u_success = 0
    fgsm_t_success = 0
    pgd_u_success = 0
    pgd_t_success = 0

    # for visualization of adversarial samples (store first batch only)
    vis_data = {'fgsm_u': [], 'fgsm_t': [], 'pgd_u': [], 'pgd_t': []}
    # Iterate over the dataset
    for inputs, labels in tqdm(dataloader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Clean evaluation
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        
        if use_attack:
            # Generate attacks
            fgsm_u_img, fgsm_t_img, pgd_u_img, pgd_t_img = attack(model, inputs, labels, device, eps=attack_eps)
            
            # Evaluate attacks
            with torch.no_grad():
                
                fgsm_u_success += process_attack(model, fgsm_u_img, 'fgsm_u', False, predicted, labels, inputs, vis_data)
                fgsm_t_success += process_attack(model, fgsm_t_img, 'fgsm_t', True, predicted, labels, inputs, vis_data)
                pgd_u_success += process_attack(model, pgd_u_img, 'pgd_u', False, predicted, labels, inputs, vis_data)
                pgd_t_success += process_attack(model, pgd_t_img, 'pgd_t', True, predicted, labels, inputs, vis_data)
                
            total_attack_samples += inputs.size(0)
            
    if use_attack and save_visuals:
        attack_types = [
            ('fgsm_u', 'fgsm_untargeted_vis.png'),
            ('fgsm_t', 'fgsm_targeted_vis.png'),
            ('pgd_u', 'pgd_untargeted_vis.png'),
            ('pgd_t', 'pgd_targeted_vis.png')
        ]
    
        for atk_name, filename in attack_types:
            samples = vis_data[atk_name]
            if len(samples) > 0:
                # Stack the collected tensors into a single batch
                orig_imgs_batch = torch.stack([s['orig_img'] for s in samples])
                adv_imgs_batch = torch.stack([s['adv_img'] for s in samples])
                labels_batch = torch.stack([s['label'] for s in samples])
                preds_batch = torch.stack([s['adv_pred'] for s in samples])
                
                # Call the visualization function
                visualize_adversarial_samples(
                    orig_imgs_batch, adv_imgs_batch, labels_batch, preds_batch, 
                    class_names=class_names, num_samples=len(samples), filename=filename
                )

    accuracy = correct / len(dataloader.dataset)
    
    attack_metrics = None
    if use_attack and total_attack_samples > 0:
        # Change the denominator of ASR to 'total number of correct predictions (correct)' to improve accuracy
        asr_denom = max(correct, 1)
        fgsm_u_asr = fgsm_u_success / asr_denom
        fgsm_t_asr = fgsm_t_success / asr_denom
        pgd_u_asr = pgd_u_success / asr_denom
        pgd_t_asr = pgd_t_success / asr_denom

        attack_metrics = {
            'eps': attack_eps,
            'clean_accuracy': accuracy,
            'clean_correct': correct,
            'total_samples': len(dataloader.dataset),
            'asr': {
                'fgsm_untargeted': fgsm_u_asr,
                'fgsm_targeted': fgsm_t_asr,
                'pgd_untargeted': pgd_u_asr,
                'pgd_targeted': pgd_t_asr,
            },
            'success_count': {
                'fgsm_untargeted': fgsm_u_success,
                'fgsm_targeted': fgsm_t_success,
                'pgd_untargeted': pgd_u_success,
                'pgd_targeted': pgd_t_success,
            },
        }

        print(f"\nModel Clean Accuracy: {accuracy:.4f} ({correct}/{len(dataloader.dataset)})")
        print(f"Attack Success Rate (ASR) over {correct} correctly classified samples:")
        print(f"FGSM Untargeted: {fgsm_u_asr:.4f}")
        print(f"FGSM Targeted:   {fgsm_t_asr:.4f}")
        print(f"PGD Untargeted:  {pgd_u_asr:.4f}")
        print(f"PGD Targeted:    {pgd_t_asr:.4f}")

    if return_attack_metrics:
        return total_loss / len(dataloader), accuracy, attack_metrics
    return total_loss / len(dataloader), accuracy

def attack(model, inputs, labels, device, eps=0.3):
    with torch.enable_grad():
        fgsm_untargeted_image = fgsm_untargeted(model, 
                                    inputs.clone().detach().to(device), 
                                    labels.to(device), eps=eps)
        fgsm_targeted_image = fgsm_targeted(model, 
                        inputs.clone().detach().to(device), 
                        ((labels+1)%10).to(device), eps=eps)
        pgd_untargeted_image = pgd_untargeted(model,
                    inputs.clone().detach().to(device), 
                    labels.to(device), 
                    k=40, eps=eps, eps_step=0.01)
        pgd_targeted_image = pgd_targeted(model,
                    inputs.clone().detach().to(device),
                    ((labels+1)%10).to(device), 
                    k=40, eps=eps, eps_step=0.01)
    return fgsm_untargeted_image, fgsm_targeted_image, pgd_untargeted_image, pgd_targeted_image