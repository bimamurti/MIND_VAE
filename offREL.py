import torch
import numpy as np
from social_vae import SocialVAE

class offREL:
    """Off-policy REL helper.

    Maintains a buffer of trajectory samples and provides routines to train
    a hold-out model and a reference model from that buffer.
    """
    def __init__(self, settings,config, data_train=None, model=None, lossmain=None, buffer_size=200, batch_size=16, lr=1e-3,epoch=100,nbest=512):
        self.settings = settings
        self.config = config
        self.device = settings.device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.data_train = data_train
        self.model = model
        self.lossmain = lossmain
        self.epoch=epoch
        self.nbest=nbest
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

    # def update_buffer(self, items, losses, track_ids):
    # 	"""Update buffer with a batch.

    # 	Args:
    # 		items: tuple/list (obs, fut, neigh, err, kl, cluster)
    # 		losses: per-sample loss values
    # 		track_ids: ids aligned to batch dimension
    # 	"""
    # 	obs, fut, neigh, err, kl, cluster = items
    # 	losses_np = self._flatten_losses(losses)
    # 	if torch.is_tensor(track_ids):
    # 		track_ids = track_ids.detach().cpu().numpy()
    # 	track_ids = np.atleast_1d(track_ids).flatten()

    # 	batch_len = obs.shape[1]
    # 	for i in range(batch_len):
    # 		sample = dict(
    # 			obs=obs[:, i].detach().cpu(),
    # 			fut=fut[:, i].detach().cpu(),
    # 			neigh=neigh[:, i].detach().cpu(),
    # 			err=err[i].detach().cpu() if torch.is_tensor(err) else torch.tensor(err[i]),
    # 			kl=kl[i].detach().cpu() if torch.is_tensor(kl) else torch.tensor(kl[i]),
    # 			cluster=cluster[i] if not torch.is_tensor(cluster) else cluster[i].detach().cpu(),
    # 			tid=int(track_ids[i]) if len(track_ids) > i else i,
    # 			score=float(losses_np[i]) if len(losses_np) > i else 0.0,
    # 		)
    # 		# if buffer full, replace lowest-score item when new score higher
    # 		if len(self.buffer) < self.buffer_size:
    # 			self.buffer.append(sample)
    # 		else:
    # 			idx_min = int(np.argmin([s["score"] for s in self.buffer]))
    # 			if sample["score"] > self.buffer[idx_min]["score"]:
    # 				self.buffer[idx_min] = sample
                    
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
        obs, fut, neigh, err, kl, tid, ncluster, taskno,relscore = subsetdata
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
        #get the loss score from training the model, this is used to select the data for buffer
        model.train()
        #self.lossmain.device=model.device
        obs, fut, neigh, err, kl, track_ids, cluster, taskno = self.lossmain
        total_samples = obs.shape[1]
        
        # Process in batches to reduce memory usage
        all_outputs = []
        with torch.no_grad():
            for i in range(0, total_samples, self.batch_size):
                batch_idx = slice(i, min(i + self.batch_size, total_samples))
                
                obs_batch = obs[:, batch_idx, :].to(self.device)
                fut_batch = fut[:, batch_idx, :].to(self.device)
                neigh_batch = neigh[:, batch_idx, :, :].to(self.device)
                err_batch = err[batch_idx].to(self.device)
                kl_batch = kl[batch_idx].to(self.device)
                
                output = model(obs_batch, fut_batch, neigh_batch, err_batch, kl_batch)
                # Move output to CPU immediately to free GPU memory
                output_cpu = tuple(o.cpu() if torch.is_tensor(o) else o for o in output)
                all_outputs.append(output_cpu)
        
        # Concatenate all batch outputs
        concatenated_output = tuple(
            torch.cat([batch[i] for batch in all_outputs], dim=1 if i < 2 else 0)
            for i in range(len(all_outputs[0]))
        )
        #self.lossmain.device="cpu"
        return concatenated_output, track_ids
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
    def convertIndextoBuffer(self,items,i,lossscore):
        #return a subset of items based on index
        i=np.array(i)
        i=i.astype(int)
        if(i.ndim==2):
            idx=i[:,1]
        else:
            idx=i
      
        # Slice tensors and move them to CPU to avoid keeping GPU memory
        traj = items[0][:, idx, :]
        future = items[1][:, idx, :]
        neighbour = items[2][:, idx, :, :]
        err = items[3][idx]
        kl = items[4][idx]

        # Detach and move tensor data to CPU
        traj = traj.detach().cpu().clone() if torch.is_tensor(traj) else traj
        future = future.detach().cpu().clone() if torch.is_tensor(future) else future
        neighbour = neighbour.detach().cpu().clone() if torch.is_tensor(neighbour) else neighbour
        err = err.detach().cpu().clone() if torch.is_tensor(err) else err
        kl = kl.detach().cpu().clone() if torch.is_tensor(kl) else kl

        # IDs, ncluster and taskno: ensure numpy on CPU
        indexes = np.array(items[5].cpu()) if torch.is_tensor(items[5]) else np.array(items[5])
        idt = indexes[idx]
        ncluster = np.array(items[6].cpu()) if torch.is_tensor(items[6]) else np.array(items[6])
        ncluster = ncluster[idx]
        taskno = np.array(items[7].cpu()) if torch.is_tensor(items[7]) else np.array(items[7])
        taskno = taskno[idx]

        return [traj, future, neighbour, err, kl, idt, ncluster, taskno, lossscore]
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
        # valid_diffs=torch.softmax(torch.as_tensor(valid_diffs),dim=0).numpy()
        sorted_idx = np.argsort(valid_diffs)[::-1]  # descending
        selected_batch_indices = valid_indices[sorted_idx]
        selected_ids = track_ids_np[selected_batch_indices].tolist()
        loss_sorted = valid_diffs[sorted_idx]
        return selected_ids, selected_batch_indices, loss_sorted
    def adddatatosubsetnoduplicates(self, selectedsubset):
        """Add data to buffer, but avoid duplicates by checking track_ids.
        If a track_id already exists in buffer, only update its loss score.
        Otherwise, concatenate as new entry.
        
        Returns:
            tuple: (buffer_data, num_duplicates, num_new)
        """
        temp = selectedsubset
        
        # Extract temp IDs and loss scores
        temp_idt = np.atleast_1d(temp[5])
        temp_lossscore = np.atleast_1d(temp[8])
        
        # Extract buffer IDs and loss scores
        buffer_idt = self.buffer[5]
        buffer_lossscore = self.buffer[8]
        
        # Find which temp IDs are duplicates (already in buffer)
        duplicate_mask = np.isin(temp_idt, buffer_idt)
        new_mask = ~duplicate_mask
        
        num_duplicates = np.sum(duplicate_mask)
        num_new = np.sum(new_mask)
        
        # Update loss scores for duplicates
        updated_lossscore = buffer_lossscore.copy()
        for idx, temp_id in enumerate(temp_idt):
            if duplicate_mask[idx]:
                # Find position of this ID in buffer
                buffer_idx = np.where(buffer_idt == temp_id)[0][0]
                # Update loss score (take the higher/worse loss)
                updated_lossscore[buffer_idx] = max(updated_lossscore[buffer_idx], temp_lossscore[idx])
        
        # Concatenate only new IDs
        if np.any(new_mask):
            new_indices = np.where(new_mask)[0]
            
            if temp[0].ndim == 2:
                obv = torch.cat((self.buffer[0], temp[0][:, new_indices,:].unsqueeze(1)), dim=1)
                fut = torch.cat((self.buffer[1], temp[1][:, new_indices,:].unsqueeze(1)), dim=1)
                neighb = torch.cat((self.buffer[2], temp[2][:, new_indices,:,:].unsqueeze(1)), dim=1)
                yt = torch.cat((self.buffer[3], temp[3][new_indices].unsqueeze(0)), dim=0)
                ft = torch.cat((self.buffer[4], temp[4][new_indices].unsqueeze(0)), dim=0)
            else:
                obv = torch.cat((self.buffer[0], temp[0][:, new_indices,:]), dim=1)
                fut = torch.cat((self.buffer[1], temp[1][:, new_indices,:]), dim=1)
                neighb = torch.cat((self.buffer[2], temp[2][:, new_indices,:,:]), dim=1)
                yt = torch.cat((self.buffer[3], temp[3][new_indices]), dim=0)
                ft = torch.cat((self.buffer[4], temp[4][new_indices]), dim=0)
            
            idt = np.concatenate((buffer_idt, temp_idt[new_indices]), axis=0)
            ncluster = np.concatenate((self.buffer[6], temp[6][new_indices]), axis=0)
            idtask = np.concatenate((self.buffer[7], temp[7][new_indices]), axis=0)
            lossscore = np.concatenate((updated_lossscore, temp_lossscore[new_indices]), axis=0)
        else:
            # All are duplicates, just update loss scores
            obv = self.buffer[0]
            fut = self.buffer[1]
            neighb = self.buffer[2]
            yt = self.buffer[3]
            ft = self.buffer[4]
            idt = buffer_idt
            ncluster = self.buffer[6]
            idtask = self.buffer[7]
            lossscore = updated_lossscore
        
        finalloss = [obv, fut, neighb, yt, ft, idt, ncluster, idtask, lossscore]
        return finalloss, num_duplicates, num_new
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
            lossscore=np.atleast_1d(temp[8])
            lossscore = np.concatenate((self.buffer[8], lossscore), axis=0)
        else:
            obv=torch.cat((self.buffer[0],temp[0]),dim=1)
            fut=torch.cat((self.buffer[1],temp[1]),dim=1)
            neighb=torch.cat((self.buffer[2],temp[2]),dim=1)
            yt=torch.cat((self.buffer[3],temp[3]),dim=0)
            ft=torch.cat((self.buffer[4],temp[4]),dim=0)
            idt = np.concatenate((self.buffer[5], temp[5]), axis=0)
            ncluster = np.concatenate((self.buffer[6], temp[6]), axis=0)
            idtask = np.concatenate((self.buffer[7], temp[7]), axis=0)
            lossscore = np.concatenate((self.buffer[8], temp[8]), axis=0)
        finalloss=[obv,fut,neighb,yt,ft,idt,ncluster,idtask,lossscore]
        return finalloss
    #  if(epoch==end_epoch):#ini tentukan kapan training coresetnya
    #                 endtimebcsr=time.time()
    #                 ids,loss=bc.coreset_select(model, item, task_id=0,  topk=int(config.BATCH_SIZE*(int(settings.datacapacity)/100)),out_loss=[],idt=idtracks)
    #                 rng_state = get_rng_state(settings.device)
    #                 candidates_indices.append(ids.cpu().numpy())
    #                 lastbatch.append([item,errindividual,klindividual,torch.as_tensor(idtracks, dtype=torch.int, device=settings.device),torch.as_tensor(ncluster, dtype=torch.double, device=settings.device),torch.as_tensor(idtask, dtype=torch.int, device=settings.device)])
         
    def update_buffer(self):
        lossscoreholdout=self.train_holdout_model(epochs=10)
        # get per-sample losses from hold-out model
        #self.train_subset_model(subset)
        all_selected_index=[]
        lossscoreholdout,trackids=self.getthelostscorefromtraining(self.holdout_model)
        while(len(all_selected_index)<self.buffer_size):#cek lagi loopnya
            print("buffer contains:"+str(len(all_selected_index))+"/"+str(self.buffer_size))
            lossscoresubset,trackids=self.getthelostscorefromtraining(self.subset_model)
            id_selected, selected_batch_indices, loss_diff_np = self.selectbyloss(lossscoreholdout,lossscoresubset)
            selectedsubset=self.convertIndextoBuffer(self.lossmain,selected_batch_indices[:self.nbest],loss_diff_np[:self.nbest])
            if(self.buffer!=[]):
                all_selected_index.extend(selected_batch_indices[:self.nbest])
                self.buffer=self.adddatatosubsetnoduplicates(selectedsubset)#tambahkan data ke buffer (no duplicates)
            else:
                self.buffer=selectedsubset
            self.subset_model = self._init_model()
            self.train_subset_model(self.buffer)
    def update_buffer_new(self):
        # use main model instead of hold-out model for loss comparison
        losscoremain,trackids=self.getthelostscorefromtraining(self.model)
        i=0
        all_selected_index=[]
        while(len(all_selected_index)<self.buffer_size):
            #print(len(self.all_selected_index))
            print("buffer contains:"+str(len(all_selected_index))+"/"+str(self.buffer_size))
            lossscoresubset,trackids=self.getthelostscorefromtraining(self.subset_model)
            id_selected, selected_batch_indices, loss_diff_np = self.selectbyloss(losscoremain,lossscoresubset)
            selectedsubset=self.convertIndextoBuffer(self.lossmain,selected_batch_indices[:self.nbest],loss_diff_np[:self.nbest])
            if(self.buffer!=[]):
                #all_selected_index.extend(selected_batch_indices[:2])
                self.buffer=self.adddatatosubsetnoduplicates(selectedsubset)#tambahkan data ke buffer (no duplicates)
            else:
                self.buffer=selectedsubset
            all_selected_index.extend(selected_batch_indices[:self.nbest])
            self.subset_model = self._init_model()
            self.train_subset_model(self.buffer)

        lossscoresubset,trackids=self.getthelostscorefromtraining(self.subset_model)
        id_selected, selected_batch_indices, loss_diff_np = self.selectbyloss(losscoremain,lossscoresubset)
        return id_selected, selected_batch_indices, loss_diff_np
                      
    def update_buffer_new_noduplicate(self):
        # use main model instead of hold-out model for loss comparison
        losscoremain,trackids=self.getthelostscorefromtraining(self.model)
        i=0
        all_selected_index=[]
        leninside=0
        while(leninside<self.buffer_size):
            #print(len(self.all_selected_index))
            print("buffer contains:"+str(leninside)+"/"+str(self.buffer_size))
            lossscoresubset,trackids=self.getthelostscorefromtraining(self.subset_model)
            id_selected, selected_batch_indices, loss_diff_np = self.selectbyloss(losscoremain,lossscoresubset)
            selectedsubset=self.convertIndextoBuffer(self.lossmain,selected_batch_indices[:self.nbest],loss_diff_np[:self.nbest])
            if(self.buffer!=[]):
                #all_selected_index.extend(selected_batch_indices[:2])
                self.buffer,duplicates,_=self.adddatatosubsetnoduplicates(selectedsubset)#tambahkan data ke buffer
                self.nbest+=duplicates
            else:
                self.buffer=selectedsubset
            
            leninside=len(self.buffer[5])
            #all_selected_index.extend(selected_batch_indices[:self.nbest])
            self.subset_model = self._init_model()
            self.train_subset_model(self.buffer)

        lossscoresubset,trackids=self.getthelostscorefromtraining(self.subset_model)
        id_selected, selected_batch_indices, loss_diff_np = self.selectbyloss(losscoremain,lossscoresubset)
        return id_selected, selected_batch_indices, loss_diff_np    