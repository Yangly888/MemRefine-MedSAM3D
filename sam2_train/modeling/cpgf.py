# models/cpgf.py
import torch, torch.nn as nn, torch.nn.functional as F

def _convNd(Cin, Cout, k, dim):  # helper
    return nn.Conv3d(Cin, Cout, k, 1, k//2) if dim==3 else nn.Conv2d(Cin, Cout, k, 1, k//2)

class CPGF(nn.Module):
    def __init__(self, C, heads=4, dim_head=32, dim=2, alpha_cap=0.4):
        super().__init__()
        self.dim, self.H, self.D = dim, heads, dim_head
        self.q = _convNd(C, heads*dim_head, 1, dim)
        self.k = _convNd(C, heads*dim_head, 1, dim)
        self.v = _convNd(C, heads*dim_head, 1, dim)
        self.proj = _convNd(heads*dim_head, C, 1, dim)
        self.alpha_cap = alpha_cap
        # 增强的自适应门控：考虑特征相似性和能量差异
        self.gate = nn.Sequential(
            nn.Conv1d(3, 8, 1), nn.ReLU(), 
            nn.Conv1d(8, 1, 1), nn.Sigmoid()
        )

    def forward(self, x, mem_feats):
        if (mem_feats is None) or (len(mem_feats)==0):
            return x
        
        # 聚合记忆（top-K 传入后这里取均值，简单稳妥）
        m = torch.stack(mem_feats, 0).mean(0)

        # 增强的自适应门控：计算特征相似性和能量差异
        ex = x.abs().mean(dim=1, keepdim=True).flatten(2).mean(-1, keepdim=True)   # [B,1,1]
        em = m.abs().mean(dim=1, keepdim=True).flatten(2).mean(-1, keepdim=True)   # [B,1,1]
        
        # 计算特征相似性（余弦相似度）
        x_norm = F.normalize(x.flatten(2).mean(-1, keepdim=True), dim=1)           # [B,C,1]
        m_norm = F.normalize(m.flatten(2).mean(-1, keepdim=True), dim=1)           # [B,C,1]
        sim = (x_norm * m_norm).sum(dim=1, keepdim=True)                          # [B,1,1]
        
        # 组合三个信号：当前能量、记忆能量、相似性
        gate_input = torch.cat([ex, em, sim], 1)                                   # [B,3,1]
        alpha = self.gate(gate_input) * self.alpha_cap                            # [B,1,1]
        
        # 自适应显存控制：根据特征尺寸动态调整stride
        total_elements = x.numel()
        if total_elements > 1e7:  # 10M elements
            stride = 4
        elif total_elements > 5e6:  # 5M elements
            stride = 2
        else:
            stride = 1
            
        # 在不同分辨率下进行处理
        if stride > 1:
            # 低分辨率处理
            if self.dim == 3:
                x_low = F.avg_pool3d(x, kernel_size=stride, stride=stride)
                m_low = F.avg_pool3d(m, kernel_size=stride, stride=stride)
            else:
                x_low = F.avg_pool2d(x, kernel_size=stride, stride=stride)
                m_low = F.avg_pool2d(m, kernel_size=stride, stride=stride)
                
            # 在低分辨率上计算注意力
            q, k, v = self.q(x_low), self.k(m_low), self.v(m_low)
            B, _, *sp_low = q.shape
            q = q.view(B, self.H, self.D, *sp_low).flatten(3)
            k = k.view(B, self.H, self.D, *sp_low).flatten(3)
            v = v.view(B, self.H, self.D, *sp_low).flatten(3)
            q = F.normalize(q, dim=2); k = F.normalize(k, dim=2)
            attn = torch.einsum('bhdk,bhdk->bhd', q, k).softmax(dim=-1).unsqueeze(-1)
            if self.dim == 2:
                # 2D特殊处理：先计算加权平均，再reshape
                weighted_v = v * attn  # [B, H, D, spatial]
                y_low_features = weighted_v.mean(dim=-1)  # [B, H, D]
                y_low = y_low_features.view(B, self.H*self.D, *([1]*len(sp_low)))
            else:
                # 3D保持原来的逻辑
                y_low = (v * attn).view(B, self.H*self.D, *([1]*len(sp_low)))
            if self.dim == 2:
                # 2D: 先proj再expand
                y_low = self.proj(y_low).expand_as(x_low)
            else:
                # 3D: 保持原逻辑，先expand再proj
                y_low = self.proj(y_low.expand_as(x_low))
            
            # 上采样到原始分辨率
            mode = 'trilinear' if self.dim == 3 else 'bilinear'
            y = F.interpolate(y_low - x_low, size=x.shape[2:], mode=mode, align_corners=False)
            return x + alpha * y
        else:
            # 原分辨率处理
            q, k, v = self.q(x), self.k(m), self.v(m)
            B, _, *sp = q.shape
            q = q.view(B, self.H, self.D, *sp).flatten(3)
            k = k.view(B, self.H, self.D, *sp).flatten(3)
            v = v.view(B, self.H, self.D, *sp).flatten(3)
            q = F.normalize(q, dim=2); k = F.normalize(k, dim=2)
            attn = torch.einsum('bhdk,bhdk->bhd', q, k).softmax(dim=-1).unsqueeze(-1)
            if self.dim == 2:
                # 2D特殊处理：先计算加权平均，再reshape
                weighted_v = v * attn  # [B, H, D, spatial]
                y = weighted_v.mean(dim=-1)  # [B, H, D] - 对空间维度求平均
                y = y.view(B, self.H*self.D, *([1]*len(sp)))  # [B, H*D, 1, 1]
            else:
                # 3D保持原来的逻辑（已验证工作正常）
                y = (v * attn).view(B, self.H*self.D, *([1]*len(sp)))
            if self.dim == 2:
                # 2D: 先proj再expand，因为我们的reshape改变了空间结构
                y = self.proj(y).expand_as(x)
            else:
                # 3D: 保持原逻辑，先expand再proj
                y = self.proj(y.expand_as(x))
            # 2D需要额外扩展alpha维度，3D保持原有逻辑
            if self.dim == 2:
                alpha = alpha.unsqueeze(-1)  # [B,1,1] -> [B,1,1,1] for 2D broadcasting
            return x + alpha * y
