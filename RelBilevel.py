import torch
import numpy as np
from social_vae import SocialVAE

class offRELdirect:
    """Off-policy REL helper.

    Maintains a buffer of trajectory samples and provides routines to train
    a hold-out model and a reference model from that buffer.
    """
    def __init__(self, settings, data_train, config, model, lossmain, buffer_size=200, batch_size=16, lr=1e-3,epoch=100):
        self.settings = settings
        self.config = config
        self.device = settings.device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.data_train = data_train
        self.model = model
        self.lossmain = lossmain
        self.epoch=epoch
        #self.train_no_aug = data_nonaug
        # simple list of dict samples
        self.buffer = []

        # hold-out and reference models
        self.holdout_model = self._init_model()
        self.subset_model = self._init_model()
        self.holdout_optim = torch.optim.Adam(self.holdout_model.parameters(), lr=lr)
        self.subset_optim = torch.optim.Adam(self.subset_model.parameters(), lr=lr)

    def _init_model(self):
        model = SocialVAE(
            horizon=self.config.PRED_HORIZON,
            ob_radius=self.config.OB_RADIUS,
            hidden_dim=self.config.RNN_HIDDEN_DIM,
        )
        model.to(self.device)
        return model

    def _flatten_losses(self, losses):
        if torch.is_tensor(losses):
            return losses.detach().cpu().numpy().flatten()
        return np.atleast_1d(np.array(losses)).flatten()
                    
    def _sample_batches(self):
        if not self.buffer:
            return []
        idxs = np.arange(len(self.buffer))
        np.random.shuffle(idxs)
        batches = [idxs[i:i + self.batch_size] for i in range(0, len(idxs), self.batch_size)]
        return batches

    def _train_on_buffer(self, model, optim, epochs=1):
        """Train on the same data the main model was trained on (stored in self.lossmain).

        On the last epoch, collect per-sample err (output[0] mean) aligned with track ids.
        """
        model.train()
        
        # lossmain is [obs, fut, neigh, err, kl, track_ids]
        # if not self.lossmain or len(self.lossmain) == 0:
        #     return None, []
            
        obs, fut, neigh, err, kl, track_ids,cluster, taskno = self.lossmain
        total_samples = obs.shape[1]  # batch dimension
        collected_err = []
        
        for epoch_idx in range(epochs):
            # Shuffle indices
            indices = torch.randperm(total_samples)
            
            # Create mini-batches
            for i in range(0, total_samples, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                
                # Extract batch
                obs_batch = obs[:, batch_idx, :].to(self.device)
                fut_batch = fut[:, batch_idx, :].to(self.device)
                neigh_batch = neigh[:, batch_idx, :, :].to(self.device)
                err_batch = err[batch_idx].to(self.device)
                kl_batch = kl[batch_idx].to(self.device)
                track_ids_batch = track_ids[batch_idx]
                # Forward pass
                output = model(obs_batch, fut_batch, neigh_batch, err_batch, kl_batch)
                loss = model.loss(*output)
                
                # If last epoch, collect err with track ids
                if epoch_idx == epochs - 1:
                    err_per_sample = output[0].mean(dim=(0, 2)).detach().cpu()
                    if torch.is_tensor(track_ids_batch):
                        track_ids_cpu = track_ids_batch.detach().cpu()
                    else:
                        track_ids_cpu = torch.as_tensor(track_ids_batch)
                    for e_val, tid_val in zip(err_per_sample, track_ids_cpu):
                        collected_err.append((int(tid_val), float(e_val)))
                # Backward pass
                loss["loss"].backward()
                optim.step()
                optim.zero_grad()

        return output, collected_err#reformating this output , cek the format
    def _train_subset_on_buffer(self, model, optim, subsetdata):
        """Train on subset data with batching and padding if needed.
        
        If subsetdata has fewer samples than batch_size, replicate/pad data to fill batches.
        """
        model.train()
        subsetdata = subsetdata  # exclude track ids
        
        # Extract components
        obs, fut, neigh, err, kl, tid, ncluster, taskno = subsetdata
        total_samples = obs.shape[1]  # batch dimension
        
        # If subset is smaller than batch_size, pad by repeating
        if total_samples < self.batch_size:
            # Calculate how many times to repeat
            repeat_times = (self.batch_size // total_samples) + 1
            
            # Repeat along batch dimension
            obs = obs.repeat(1, repeat_times, 1)[:, :self.batch_size]
            fut = fut.repeat(1, repeat_times, 1)[:, :self.batch_size]
            neigh = neigh.repeat(1, repeat_times, 1, 1)[:, :self.batch_size]
            err = err.repeat(repeat_times).flatten()[:self.batch_size]
            kl = kl.repeat(repeat_times).flatten()[:self.batch_size]
            tid = tid.repeat(repeat_times).flatten()[:self.batch_size]
            total_samples = self.batch_size
        # Train on batches
        indices = torch.randperm(total_samples)
        if(total_samples>self.batch_size):
            a=0
        for i in range(0, total_samples, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            
            # Extract batch
            obs_batch = obs[:, batch_idx, :].to(self.device)
            fut_batch = fut[:, batch_idx, :].to(self.device)
            neigh_batch = neigh[:, batch_idx, :, :].to(self.device)
            err_batch = err[batch_idx].to(self.device)
            kl_batch = kl[batch_idx].to(self.device)
            tid_batch = tid[batch_idx]
            # Forward pass
            output = model(obs_batch, fut_batch, neigh_batch, err_batch, kl_batch)
            loss = model.loss(*output)
            
            # Backward pass
            loss["loss"].backward()
            optim.step()
            optim.zero_grad()
        
        return output
    def getthelostscorefromtraining(self,model):
        model.train()
        obs, fut, neigh, err, kl, track_ids, cluster, taskno = self.lossmain
        with torch.no_grad():
            output = model(obs, fut, neigh, err, kl)
            loss = model.loss(*output)
        return output,track_ids
    def train_holdout_model(self, epochs=10):
        """Train hold-out model on current buffer."""
        # if not self.buffer:
        #     return
        return self._train_on_buffer(self.holdout_model, self.holdout_optim, epochs)

    def train_subset_model(self,subset):
        """Train reference model on current buffer."""
        if not self.buffer:
            return
        self._train_subset_on_buffer(self.subset_model, self.subset_optim, subset)
    def convertIndextoBuffer(self,items,i):
        #return a subset of items based on index
        i=np.array(i)
        i=i.astype(int)
        if(i.ndim==2):
            idx=i[:,1]
        else:
            idx=i
      
        traj=items[0][:,idx,:]
        future=items[1][:,idx,:]
        neighbour=items[2][:,idx,:,:]
        err=items[3][idx]
        kl=items[4][idx]
        #cluster=np.array(items[5].cpu())
        #cluster=cluster[idx]
        indexes=np.array(items[5].cpu())
        idt=indexes[idx]
        ncluster=np.array(items[6].cpu())
        ncluster=ncluster[idx]
        taskno=np.array(items[7].cpu())
        taskno=taskno[idx]
        return [traj,future,neighbour,err,kl,idt,ncluster,taskno]
    def getdataperbatch(self,i,batch,batchtrained):
        import random
        allowed = [a for a in range(len(self.buffer[3])) if a not in (batchtrained)]
        if(batch<len(allowed)):
            batchtrained =  random.sample(allowed, batch) 
        else:
             batchtrained =  random.sample(allowed, len(allowed))
    def selectbyloss(self,lossholdout,losssubset,subset=None):
        """Select sample IDs based on loss difference: holdout_model_loss - init_model_loss.
        
        Args:
            items: tuple (obs, fut, neigh, err, kl, cluster, track_ids) - batch from training data
            subset: list of indices already selected (to avoid reselection)
        
        Returns:
            selected_ids: list of track IDs with highest loss difference
        """
        if subset is None:
            subset = []
        #items=self.lossmain[:-1]
        # Get loss from init model (subset model)
        loss_init_per_sample=lossholdout
        loss_subset_per_sample=losssubset
        # Get loss from subset model
        losssubset=loss_subset_per_sample[0].mean(dim=(0, 2))
        # Compute difference (holdout - init)
        loss_diff = loss_subset_per_sample[0].mean(dim=(0, 2)) - loss_init_per_sample[0].mean(dim=(0, 2))
        loss_diff_np = loss_diff.detach().cpu().numpy()
        
        # Convert track_ids to numpy
        track_ids = self.lossmain[5]
        if torch.is_tensor(track_ids):
            track_ids_np = track_ids.detach().cpu().numpy()
        else:
            track_ids_np = np.array(track_ids)
        
        # Select indices not in subset, sorted by highest loss difference
        valid_mask = np.array([i not in subset for i in range(len(track_ids_np))])
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return []
        
        valid_diffs = loss_diff_np[valid_indices]
        sorted_idx = np.argsort(valid_diffs)  # ascending
        selected_batch_indices = valid_indices[sorted_idx]
        selected_ids = track_ids_np[selected_batch_indices].tolist()
        loss_sorted = valid_diffs[sorted_idx]
        return selected_ids, selected_batch_indices, loss_sorted
    def adddatatosubset(self,selectedsubset):
                #ini digunakan untuk filter data training yang bagus dan yang enggak berdasarkan mean valuenya atau kalau saat ini msh menggunakan losslist-1
            #print('loss individu',lossarray[i][1])[i for i, val in enumerate(arr) if val < 5]
        temp=selectedsubset
            #t1_cat = torch.cat([t1a, t1b], dim=1) 
        if(temp[0].ndim==2):
            obv=torch.cat((self.buffer[0],temp[0].unsqueeze(1)),dim=1)
            fut=torch.cat((self.buffer[1],temp[1].unsqueeze(1)),dim=1)
            neighb=torch.cat((self.buffer[2],temp[2].unsqueeze(1)),dim=1)
            yt=torch.cat((self.buffer[3],temp[3].unsqueeze(0)),dim=0)
            ft=torch.cat((self.buffer[4],temp[4].unsqueeze(0)),dim=0)
            idt=torch.cat((self.buffer[5],temp[5].unsqueeze(0)),dim=0)
            temp_idt = np.atleast_1d(temp[5])
            idt = np.concatenate((self.buffer[5], temp_idt), axis=0)
            temp_ncluster = np.atleast_1d(temp[6])
            ncluster = np.concatenate((self.buffer[6], temp_ncluster), axis=0)
            temp_idtask = np.atleast_1d(temp[7])
            idtask = np.concatenate((self.buffer[7], temp_idtask), axis=0)
        else:
            obv=torch.cat((self.buffer[0],temp[0]),dim=1)
            fut=torch.cat((self.buffer[1],temp[1]),dim=1)
            neighb=torch.cat((self.buffer[2],temp[2]),dim=1)
            yt=torch.cat((self.buffer[3],temp[3]),dim=0)
            ft=torch.cat((self.buffer[4],temp[4]),dim=0)
            idt = np.concatenate((self.buffer[5], temp[5]), axis=0)
            ncluster = np.concatenate((self.buffer[6], temp[6]), axis=0)
            idtask = np.concatenate((self.buffer[7], temp[7]), axis=0)
        finalloss=[obv,fut,neighb,yt,ft,idt,ncluster,idtask]
        return finalloss
    def outer_loss(self, X, y, task_id, topk, ref_x=None, ref_y=None):
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        n = X.shape[0]
        coreset_weights = 1.0 / n * torch.ones([n], dtype=torch.float, requires_grad=True)

        _, _, outer_loss = self.training_model_op.train_outer(X, y, task_id, coreset_weights, topk, ref_x,
                                                                            ref_y)
        return outer_loss

    def projection_onto_simplex(self, v, b=1):
        v = v.cpu().detach().numpy()
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - b
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        cek=u - cssv / ind
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        w = torch.from_numpy(w).cuda()
        w.requires_grad = True
        return w         
    def update_buffer(self):
        #lossscoreholdout=#self.train_holdout_model(epochs=10)
        # get per-sample losses from hold-out model
        #self.train_subset_model(subset)
        all_selected_index=[]
        while(len(all_selected_index)<self.buffer_size):#cek lagi loopnya
            lossscoresubset,trackids=self.getthelostscorefromtraining(self.subset_model)
            lossscoreholdout,trackids=self.getthelostscorefromtraining(self.model)  
            id_selected, selected_batch_indices, loss_diff_np = self.selectbyloss(lossscoreholdout,lossscoresubset)
            selectedsubset=self.convertIndextoBuffer(self.lossmain,selected_batch_indices[:2])
            if(self.buffer!=[]):
                all_selected_index.extend(selected_batch_indices[:2])
                self.buffer=self.adddatatosubset(selectedsubset)#tambahkan data ke buffer
            else:
                self.buffer=selectedsubset
            self.subset_model = self._init_model()
            self.train_subset_model(self.buffer)
        np.random.seed(self.seed)


    