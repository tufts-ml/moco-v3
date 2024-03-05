import torch
import torch.nn as nn


class MoCoLoss(nn.Module):
    def __init__(self, temp=1.0) -> None:
        """MoCo loss with symmetric batching

        Args:
            temp (float, optional): Softmax temperature. Defaults to 1.0.
        """
        self.temp = temp
        super().__init__()

    def forward(self, q1, q2, k1, k2):
        # separate examples with different SyncNorm
        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)

    def sym_moco(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.temp
        b_size = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(b_size, dtype=torch.long) +
                  b_size * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.temp)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
