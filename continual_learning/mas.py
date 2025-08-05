import torch

class MAS:
    def __init__(self, model, dataloader, device="cpu"):
        self.model = model
        self.device = device
        self.importance = self._compute_importance(dataloader)

        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

    def _compute_importance(self, dataloader):
        importance = {n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for batch in dataloader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
            self.model.zero_grad()
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            s = probs.sum()
            s.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None and p.requires_grad:
                    importance[n] += p.grad.detach().abs()
        for n in importance:
            importance[n] /= len(dataloader)
        return importance

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.importance:
                loss += (self.importance[n] * (p - self.params[n]) ** 2).sum()
        return loss