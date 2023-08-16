import torch

def peak_extract(heat, kernel=5, K=25):
    B, C, H, W = heat.size()
    
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)

    keep = (hmax == heat).float()
    
    peak = heat * keep
    
    topk_scores, topk_inds = torch.topk(peak.view(B, C, -1), K)

    topk_inds = topk_inds % (H * W)
    topk_ys   = (topk_inds / W).int().float()
    topk_xs   = (topk_inds % W).int().float()
    
    topk_scores = topk_scores.float().detach().cpu().numpy()
    topk_ys = topk_ys.int().detach().cpu().numpy()
    topk_xs = topk_xs.int().detach().cpu().numpy()
    
    return topk_scores, topk_ys, topk_xs


def smoothing(heat, kernel=3):
    pad = (kernel - 1) // 2
    heat = torch.nn.functional.avg_pool2d(heat, (kernel, kernel), stride=1, padding=pad)

    return heat
