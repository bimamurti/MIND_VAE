from logging import config
import torch
import torch.nn.functional as F
import numpy as np
import math
def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

class Training():

    def __init__(self, proxy_model, beta, device, lr_proxy_model, lr_weights):
        self.proxy_model = proxy_model
        self.lr_p =  lr_proxy_model
        self.lr_w =  lr_weights
        self.optimizer_theta_p_model = torch.optim.Adam(self.proxy_model.parameters(), lr=self.lr_p)
        self.weight_optimizer = None
        self.device = device
        self.eta = 0.5
        self.beta = beta
        self.buffer = []
        self.identity = []

    def init_proxy_model(self):
        for m in self.proxy_model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)

    def train_inner(self, item, task_id, sample_weights, inner_epchos):
        loss = math.inf
        self.optimizer_theta_p_model= torch.optim.Adam(self.proxy_model.parameters(), lr=self.lr_p)
        self.optimizer_theta_p_model.zero_grad()
        
        for i in range(inner_epchos):
            #print("epoch inner:",i)
            #self.optimizer_theta_p_model = torch.optim.SGD(self.proxy_model.parameters(), lr=self.lr_p)
            self.proxy_model.train()
            
            #data = data_S.to(self.device).type(torch.float)
            #target = target_S.to(self.device).type(torch.long)
            sample_weights = sample_weights.to(self.device).type(torch.float).detach()
            out = self.proxy_model(*item)
            loss = self.proxy_model.loss(*out)
            loss = torch.mean(sample_weights * loss["loss"])
            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"Warning: NaN detected in inner loss at epoch {i}, skipping update")
                continue
            loss.backward()
            # Clip gradients to prevent explosion leading to NaN
            torch.nn.utils.clip_grad_norm_(self.proxy_model.parameters(), max_norm=1.0)
            
            # Check and sanitize parameters for NaN after step
            for param in self.proxy_model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0)
            
            self.optimizer_theta_p_model.step()
            
            # Sanitize parameters if they became NaN
            with torch.no_grad():
                for param in self.proxy_model.parameters():
                    if not torch.isfinite(param).all():
                        param.data = param.data.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0)
                        print(f"Warning: Parameter became NaN at inner epoch {i}, sanitized")
            
            self.optimizer_theta_p_model.zero_grad()#
        return loss


    def train_outer(self, items, task_id, data_weights, topk, ref_x=None, ref_y=None):
        #data = data.to(self.device)
        #target = target.to(self.device).type(torch.long)
        sample_weights = data_weights.to(self.device)
        #X_S = data[:].to(self.device)
        #y_S = target[:].to(self.device).type(torch.long)
        return self.update_sample_weights(items, task_id, sample_weights, topk, beta=self.beta, ref_x=ref_x, ref_y=ref_y)


    def update_sample_weights(self, items, task_id,  sample_weights, topk, beta, epsilon=1e-3, ref_x=None, ref_y=None):
        z = torch.normal(0, 1, size=[topk]).cuda()
        # Disable CuDNN for RNNs to support double backward (second-order derivatives)
        # This must wrap ALL operations that involve double backward, not just forward pass
        with torch.backends.cudnn.flags(enabled=False):
            # Sanitize model parameters before forward pass
            # with torch.no_grad():
            #     for param in self.proxy_model.parameters():
            #         if not torch.isfinite(param).all():
            #             param.data = param.data.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0)
            #             print("Warning: Found NaN in parameters before forward pass")
            
            output=self.proxy_model(*items)
            err, kl = output  # Get err and kl before they're averaged
            # Compute per-sample losses: err shape is (horizon, n_samples, 2), kl shape is (horizon, n_samples, z_dim)
            # Average over horizon and feature dimensions to get one loss per sample
            loss_per_sample_err = err.mean(dim=(0, 2))  # Average over horizon and xy dimensions
            loss_per_sample_kl = kl.mean(dim=(0, 2))    # Average over horizon and z dimensions
            loss_outer = loss_per_sample_err + loss_per_sample_kl  # Shape: (n_samples,)
            #ini harus dihitung di luar fungsi loss biar kedeteksi gradiennya, kemarin gradien bisa 0 semua karena learning ratenya 10, kegehdean!
            #print(f"[Debug] err shape: {err.shape}, kl shape: {kl.shape}")
            #print(f"[Debug] loss_outer shape: {loss_outer.shape}, should be (n_samples,)")

            
            # Check for NaN/Inf in loss before proceeding
            # if not torch.isfinite(loss_outer).all():
            #     print(f"Warning: loss_outer contains {torch.isnan(loss_outer).sum()} NaN and {torch.isinf(loss_outer).sum()} Inf values")
            #     loss_outer = torch.nan_to_num(loss_outer, nan=1.0, posinf=10.0, neginf=-10.0)
            
            topk_weights, ind = sample_weights.topk(topk)#get top k weights
            loss_outer_avg = torch.mean(loss_outer) - beta*(topk_weights + epsilon*z).sum()#loss with regularizer
            # if ref_x != None:
            #     loss_buff = []
            #     for i in range(task_id-1):
            #         loss_buff += F.cross_entropy(self.proxy_model(ref_x[i].to(self.device), i+1), ref_y[i].to(self.device), reduction='none')
            #     loss_buff_avg = torch.mean(torch.Tensor(loss_buff))
            #     alpha = 0.1
            #     loss_outer_avg +=  alpha * loss_buff_avg
            d_theta = torch.autograd.grad(loss_outer_avg, self.proxy_model.parameters(), retain_graph=True)
            # Sanitize gradients to prevent NaN propagation
            #d_theta = tuple(g.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0) if g is not None else None for g in d_theta)
            v_0  = d_theta
            # Use the already computed loss_outer, don't recompute it
            loss_inner = torch.mean(F.softmax(sample_weights, dim=-1) * loss_outer)
            
            # Detailed diagnostics
            # print(f"[Bilevel] loss_outer - mean={loss_outer.mean():.6f}, std={loss_outer.std():.6f}, min={loss_outer.min():.6f}, max={loss_outer.max():.6f}")
            # print(f"[Bilevel] loss_outer first 10 values: {loss_outer[:10].detach().cpu().numpy()}")
            # print(f"[Bilevel] loss_outer variance/mean ratio: {(loss_outer.std() / (loss_outer.mean() + 1e-8)):.6f}")
            # print(f"[Bilevel] sample_weights - mean={sample_weights.mean():.6f}, std={sample_weights.std():.6f}, min={sample_weights.min():.6f}, max={sample_weights.max():.6f}")
            # print(f"[Bilevel] loss_inner={loss_inner:.6f}")
            # print(f"[Bilevel] softmax(sample_weights) first 10: {F.softmax(sample_weights, dim=-1)[:10].detach().cpu().numpy()}")
            
            # Check if loss_outer values are too similar (less than 1% variation)
            # if loss_outer.std() < 0.01 * loss_outer.mean():
            #     print(f"[Bilevel] WARNING: loss_outer has very low variance - all samples have similar loss!")
            #     print(f"[Bilevel] This will cause jacobian to be uniform (same values for all samples)")
            #     print(f"[Bilevel] Reason: Proxy model treats all samples equally (not discriminating between easy/hard samples)")
            grads_theta = torch.autograd.grad(loss_inner, self.proxy_model.parameters(), create_graph=True, retain_graph=True)
            G_theta = []
            par=self.proxy_model.parameters()
            #for i in par:
            #    print(i)
            for p, g in zip(self.proxy_model.parameters(), grads_theta):
                if g == None:
                    G_theta.append(None)
                else:
                    G_theta.append(p-self.lr_p*g)
            v_Q = v_0
            #cap=tuple(v_Q)
            for iter_num in range(3):
                # This double backward needs CuDNN disabled
                v_new = torch.autograd.grad(G_theta, self.proxy_model.parameters(), grad_outputs=v_0, retain_graph=True)
                # Sanitize v_new to prevent NaN propagation
                v_0 = [i.detach().nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0) if i is not None else None for i in v_new]
                for i in range(len(v_0)):
                    if v_0[i] is not None:
                        # Check before adding to prevent NaN accumulation
                        if not torch.isfinite(v_0[i]).all():
                            print(f"Warning: v_0[{i}] at iteration {iter_num} contains NaN/Inf")
                        v_Q[i].add_(v_0[i])
            # Sanitize v_Q before computing jacobian
            v_Q = tuple(v.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0) if v is not None else None for v in v_Q)
            jacobian = torch.autograd.grad(grads_theta, sample_weights, grad_outputs=v_Q, retain_graph=False)[0]#seberapa pengaruh sample_weights terhadap loss_outer atau grad_theta
            
            # Diagnostic: Check jacobian values
            # print(f"[Jacobian] mean={jacobian.mean():.6f}, std={jacobian.std():.6f}, min={jacobian.min():.6f}, max={jacobian.max():.6f}")
            # print(f"[Jacobian] Unique values count: {torch.unique(jacobian).numel()} out of {jacobian.numel()}")
            # if torch.unique(jacobian).numel() == 1:
            #     print(f"[Jacobian] WARNING: All jacobian values are identical = {jacobian[0].item():.6f}")
        
        # Sanitize jacobian
        #jacobian = jacobian.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0)
        with torch.no_grad():
            sample_weights -= self.lr_w * jacobian
            # Sanitize sample_weights after update
            sample_weights = sample_weights.nan_to_num(nan=1.0, posinf=1.0, neginf=1.0)
        return  sample_weights.detach(), jacobian.detach(), loss_outer.detach()
    #jacobian bernilai 0 karena saat awal awal nilai loss begitu buruk sehingga tidak ada perubahan pada sample_weights

