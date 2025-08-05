import torch

class EWC:
    def __init__(self, model, dataloader, device="cpu"):
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for batch in dataloader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(self.device)
            self.model.zero_grad()
            outputs = self.model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None and p.requires_grad:
                    fisher[n] += p.grad.detach() ** 2
        for n in fisher:
            fisher[n] /= len(dataloader)
        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return 0.5 * loss 