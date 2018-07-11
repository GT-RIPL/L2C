import torch.nn as nn

eps = 1e-7  # Avoid calculating log(0). Use the small value of float16. It also works fine using 1e-35 (float32).

class KLDiv(nn.Module):
    # Calculate KL-Divergence
        
    def forward(self, predict, target):
       assert predict.ndimension()==2,'Input dimension must be 2'
       target = target.detach()

       # KL(T||I) = \sum T(logT-logI)
       predict += eps
       target += eps
       logI = predict.log()
       logT = target.log()
       TlogTdI = target * (logT - logI)
       kld = TlogTdI.sum(1)
       return kld
