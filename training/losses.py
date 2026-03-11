"""
Anti-Gravity Deepfake Detection System
Training Pipeline — Loss Functions

Implements Focal Loss to address class imbalance and focus training on hard-to-classify examples.
Deepfake datasets often have easy real videos and hard fake videos; Focal Loss dynamically scales
the cross entropy loss based on prediction confidence.

Author: Anti-Gravity Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks.
    
    Addresses class imbalance by down-weighting well-classified examples and focusing 
    training on hard, misclassified examples.
    
    Formula: FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha (float): Weighting factor for the rare class (FAKE). Default is 0.25.
            gamma (float): Focusing parameter. Higher gamma reduces the relative loss 
                           for well-classified examples. Default is 2.0.
            reduction (str): Specifies the reduction to apply to the output: 
                             'none' | 'mean' | 'sum'. Default is 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid). Shape (B,).
            targets (torch.Tensor): Ground truth labels (0=REAL, 1=FAKE). Shape (B,).
            
        Returns:
            torch.Tensor: The computed focal loss.
        """
        # Compute binary cross entropy loss with logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Get probabilities from logits
        pt = torch.exp(-bce_loss)  # pt is the probability of the true class
        
        # Compute the focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Optional alpha weighting
        if self.alpha is not None:
            # Assign alpha to the FAKE class (target=1), and (1-alpha) to the REAL class (target=0)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
            
        # Compute the final focal loss
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

if __name__ == "__main__":
    # Smoke test
    y_true = torch.tensor([1.0, 0.0, 1.0])
    y_pred = torch.tensor([0.9, 0.1, -0.5])  # logits
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    loss = criterion(y_pred, y_true)
    
    print(f"Focal Loss: {loss.item():.4f}")
    print("FocalLoss smoke test PASSED ✓")
