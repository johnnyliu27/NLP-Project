import torch
import torch.nn.functional as F

def cs(x1, x2, dim, eps = 1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return w12 / (w1 * w2).clamp(min=eps)

def loss_function(pid_tensor, other_tensor, margin = 0.001):
    # pid_tensor is of dim batch_size x 1 x output_dim
    # other tensor is of dim batch_size x 21 x output_dim
    expanded = pid_tensor.expand_as(other_tensor)
    similarity = cs(expanded, other_tensor, dim=2).squeeze(2)
    pos_sim = similarity[:,0]
    rest = similarity[:,1:]
    max_rest = torch.max(rest, dim = 1)[0]
    diff = max_rest - pos_sim + margin
    MML = F.relu(diff)
    final_loss = torch.mean(MML)
    return final_loss
