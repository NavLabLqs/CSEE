from torch import nn
import torch


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing  # 置信度，即非平滑部分的权重
        self.smoothing = smoothing  # 平滑系数
        self.cls = classes  # 类别总数
        self.dim = dim  # softmax操作的维度

    def forward(self, pred, target):
        # 对预测结果应用log softmax
        pred = pred.log_softmax(dim=self.dim)
        # 生成平滑的目标分布
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)  # 创建与预测相同形状的张量
            # 对所有类别填充平滑值
            true_dist.fill_(self.smoothing / (self.cls - 1))
            # 在真实类别的位置上设置置信度
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 计算平滑损失
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))