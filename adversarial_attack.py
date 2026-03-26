import torch
import torch.nn as nn

def fgsm_targeted(model, x, target, eps):
    '''
    model   : the neural network
    x       : input image tensor (requires_grad should be set)
    target  : the desired (wrong) class label
    eps     : perturbation magnitude (e.g., 0.1, 0.3)
    return  : adversarial image x_adv
    '''
    
    # Set requires_grad attribute of tensor. Important for Attack
    x.requires_grad = True
    
    model.zero_grad()  # Zero all existing gradients
    output = model(x)  # Forward pass
    loss = nn.CrossEntropyLoss()(output, target)  # Calculate loss with respect to
    loss.backward()  # Backward pass to calculate gradients of model parameters and input  
    
    # Collect the sign of the gradients of the input image
    sign_data_grad = x.grad.data.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    x_adv = x - eps * sign_data_grad  # Subtract because it's a targeted
    # Adding clipping to maintain [0,1] range
    x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv

def fgsm_untargeted(model, x, true_label, eps):
    '''
    model       : the neural network
    x           : input image tensor
    true_label  : the correct class label
    eps         : perturbation magnitude 
    return      : adversarial image x_adv
    '''
    x.requires_grad = True
    
    model.zero_grad()  # Zero all existing gradients
    output = model(x)  # Forward pass
    loss = nn.CrossEntropyLoss()(output, true_label)  # Calculate loss with respect to true label
    loss.backward()  # Backward pass to calculate gradients of model parameters and input  
    
    # Collect the sign of the gradients of the input image
    sign_data_grad = x.grad.data.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    x_adv = x + eps * sign_data_grad  # Add because it's an untargeted attack
    # Adding clipping to maintain [0,1] range
    x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv

def pgd_targeted(model, x, target, k, eps, eps_step):
    '''
    model   : the neural network
    x       : input image tensor 
    target  : desired (wrong) class label
    k       : number of iterations (e.g., 10, 40)
    eps     : total perturbation budget
    eps_step: step size per iteration
    return  : adversarial image x_adv
    '''
    
    # Initialize adversarial example as the original input
    x_adv = x.clone().detach()
    
    for i in range(k):
        x_adv = x_adv.detach() # Detach from previous graph to prevent gradient accumulation
        x_adv.requires_grad = True

        model.zero_grad()  # Zero all existing gradients
        output = model(x_adv)  # Forward pass
        loss = nn.CrossEntropyLoss()(output, target)  # Calculate loss with respect to target label
        loss.backward()  # Backward pass to calculate gradients of model parameters and input  
        
        with torch.no_grad():
            
            # Collect the sign of the gradients of the input image
            sign_grad = x_adv.grad.sign()
            # Create the perturbed image by adjusting each pixel of the input image
            x_adv = x_adv - eps_step * sign_grad # Subtract because it's a targeted attack
            
            # Adding clipping to maintain [0,1] range and ensure perturbation is within eps ball
            x_adv = torch.clamp(x_adv, x - eps, x + eps)
            x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv

def pgd_untargeted(model, x, true_label, k, eps, eps_step):
    '''
    model       : the neural network
    x           : input image tensor 
    true_label  : correct class label
    k           : number of iterations (e.g., 10, 40)
    eps         : total perturbation budget
    eps_step    : step size per iteration
    return      : adversarial image x_adv
    '''
    
    # Initialize adversarial example as the original input
    x_adv = x.clone().detach()
    
    for i in range(k):
        
        x_adv.requires_grad = True
        model.zero_grad()  # Zero all existing gradients
        output = model(x_adv)  # Forward pass
        loss = nn.CrossEntropyLoss()(output, true_label)  # Calculate loss with respect to true label
        loss.backward()  # Backward pass to calculate gradients of model parameters and input  
        
        with torch.no_grad():
            # Collect the sign of the gradients of the input image
            sign_grad = x_adv.grad.sign()
            # Create the perturbed image by adjusting each pixel of the input image
            x_adv = x_adv + eps_step * sign_grad  # Add because it's an untargeted attack

            # Adding clipping to maintain [0,1] range and ensure perturbation is within eps ball
            x_adv = torch.clamp(x_adv, x - eps, x + eps)
            x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv

    