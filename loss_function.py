import torch

class SAMLoss(torch.nn.Module):
    
   def __init__(self):
       super(SAMLoss, self).__init__()
   def forward(self, input, label):
       return _sam(input, label)

def _sam(img1, img2):

    inner_product = torch.sum(img1 * img2, 0)
    img1_spectral_norm = torch.sqrt(torch.sum(img1 ** 2, 0))
    img2_spectral_norm = torch.sqrt(torch.sum(img2 ** 2, 0))
    # numerical stability
    cos_theta = torch.clip(inner_product / (img1_spectral_norm * img2_spectral_norm + 1e-10), min=0, max=1)
    return torch.mean(torch.acos(cos_theta))





