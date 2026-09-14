import math
from typing import Optional, Sequence, List

import os, sys
import torch
import numpy as np

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
#from scipy.spatial.distance import euclidean
#from fastdtw import fastdtw
import time

import data

class Dataloaderbatch(torch.utils.data.Dataset):

    class FixedNumberBatchSampler(torch.utils.data.sampler.BatchSampler):
        def __init__(self, n_batches, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_batches = n_batches
            self.sampler_iter = None #iter(self.sampler)
            
        def __iter__(self):
            # same with BatchSampler, but StopIteration every n batches
            counter = 0
            batch = []
            while True:
                if counter >= self.n_batches:
                    self.sampler_iter = None
                    break
                if self.sampler_iter is None: 
                    self.sampler_iter = iter(self.sampler)
                try:
                    idx = next(self.sampler_iter)
                except StopIteration:
                    self.sampler_iter = None
                    if self.drop_last: batch = []
                    continue
                batch.append(idx)
                if len(batch) == self.batch_size:
                    counter += 1
                    yield batch
                    batch = []

    def __init__(self, 
        files: List[str], ob_horizon: int, pred_horizon: int,
        batch_size: int, drop_last: bool=False, shuffle: bool=False, batches_per_epoch=None, 
        frameskip: int=1, inclusive_groups: Optional[Sequence]=None,
        batch_first: bool=False, seed: Optional[int]=None,
        device: Optional[torch.device]=None,
        kneighbour:int=10,
        flip: bool=False, rotate: bool=False, scale: bool=False,idtask:int=0
        
    ):
        super().__init__()
        self.ob_horizon = ob_horizon
        self.pred_horizon = pred_horizon
        self.horizon = self.ob_horizon+self.pred_horizon
        self.frameskip = int(frameskip) if frameskip and int(frameskip) > 1 else 1
        self.batch_first = batch_first
        self.kneighbour=kneighbour
        self.flip = flip
        self.rotate = rotate
        self.scale = scale
        self.ncluster=[]
        self.idtrack=[]
        self.idtasks=[]
        self.scores=[]
        self.idtask=idtask
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu") 
        else:
            self.device = device

        if inclusive_groups is None:
            inclusive_groups = [[] for _ in range(len(files))]
        assert(len(inclusive_groups) == len(files))
        self.inclusive_groups = inclusive_groups
        print(" Scanning files...")
        files_ = []
        for path, incl_g in zip(files, inclusive_groups):
            if os.path.isdir(path):
                files_.extend([(os.path.join(root, f), incl_g) \
                    for root, _, fs in os.walk(path) \
                    for f in fs if f.endswith(".csv")])
            elif os.path.exists(path):
                files_.append((path, incl_g))
        data_files = sorted(files_, key=lambda _: _[0])

        data = []
        
        done = 0
        # too large of max_workers will cause the problem of memory usage
        max_workers = min(len(data_files), torch.get_num_threads(), 20)
        max_workers=12
        print('maxworker:',max_workers)
        with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"), max_workers=max_workers) as p:
            #print('before')
            futures = [p.submit(self.__class__.load, self, f, incl_g) for f, incl_g in data_files]
            print('more rise')
            for fut in as_completed(futures):
                #print('masuk a')
                done += 1
                sys.stdout.write("\r\033[K Loading data files...{}/{}".format(
                    done, len(data_files)
                ))
            for fut in futures:
               # print('masuk b')
                dataload = fut.result()
                if dataload is not None:
                    data=dataload
                sys.stdout.write("\r\033[K Loading data files...{}/{} ".format(
                    done, len(data_files)
                ))
        self.data = data
        #del data
        print("\n   {} trajectories loaded.".format(len(self.data)))
        
        self.rng = np.random.RandomState()
        if seed: self.rng.seed(seed)
        shuffle=False
        if shuffle:
            print("shuffle!!")
            sampler = torch.utils.data.sampler.RandomSampler(self)
        else:
            print("not shuffle!!")
            sampler = torch.utils.data.sampler.SequentialSampler(self)
            
        if batches_per_epoch is None:
            self.batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
            self.batches_per_epoch = len(self.batch_sampler) #if use none then use half of the sampler
            self.batch_sampler = self.__class__.FixedNumberBatchSampler(self.batches_per_epoch, sampler, batch_size, drop_last)
        else:
            self.batch_sampler = self.__class__.FixedNumberBatchSampler(batches_per_epoch, sampler, batch_size, drop_last)
            self.batches_per_epoch = batches_per_epoch

    def collate_fn(self, batch):
        alldata=self.createtheneighbour(self, batch, False)
        X, Y, NEIGHBOR = [], [], []
        id_track=[]
        nclusters=[]
        id_tasks=[]
        for item,ncluster,idt,idtask,score in alldata:
            hist, future, neighbor = item[0], item[1], item[2]
            id_track.append(idt)
            nclusters.append(ncluster)
            id_tasks.append(idtask)
            hist_shape = hist.shape
            neighbor_shape = neighbor.shape
            hist = np.reshape(hist, (-1, 2))
            neighbor = np.reshape(neighbor, (-1, 2))
            if self.flip:
               # print('flipped')
                if self.rng.randint(2):
                    hist[..., 1] *= -1
                    future[..., 1] *= -1
                    neighbor[..., 1] *= -1
                if self.rng.randint(2):
                    hist[..., 0] *= -1
                    future[..., 0] *= -1
                    neighbor[..., 0] *= -1
            if self.rotate:
               # print('rotated')
                rot = self.rng.random() * (np.pi+np.pi) 
                s, c = np.sin(rot), np.cos(rot)
                r = np.asarray([
                    [c, -s],
                    [s,  c]
                ])
                hist = (r @ np.expand_dims(hist, -1)).squeeze(-1)
                future = (r @ np.expand_dims(future, -1)).squeeze(-1)
                neighbor = (r @ np.expand_dims(neighbor, -1)).squeeze(-1)
            if self.scale:
               # print('scaled')
                s = self.rng.randn()*0.05 + 1 # N(1, 0.05)
                hist = s * hist
                future = s * future
                neighbor = s * neighbor
            hist = np.reshape(hist, hist_shape)
            neighbor = np.reshape(neighbor, neighbor_shape)

            X.append(hist)
            Y.append(future)
            NEIGHBOR.append(neighbor)
        
        n_neighbors = [n.shape[1] for n in NEIGHBOR]
        max_neighbors = max(n_neighbors) 
        if max_neighbors != min(n_neighbors):
            NEIGHBOR = [
                np.pad(neighbor, ((0, 0), (0, max_neighbors-n), (0, 0)), 
                "constant", constant_values=1e9)
                for neighbor, n in zip(NEIGHBOR, n_neighbors)
            ]
        stack_dim = 0 if self.batch_first else 1
        x = np.stack(X, stack_dim)
        y = np.stack(Y, stack_dim)
        neighbor = np.stack(NEIGHBOR, stack_dim)

        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        neighbor = torch.tensor(neighbor, dtype=torch.float32, device=self.device)
        err=torch.ones(x.shape[1],dtype=torch.float32, device=self.device)
        kl=torch.ones(x.shape[1],dtype=torch.float32, device=self.device)
        return x, y, neighbor,err,kl,nclusters,id_track,id_tasks
    def collate_fnscore(self, batch):
        X, Y, NEIGHBOR = [], [], []
        id_track=[]
        nclusters=[]
        id_tasks=[]
        scores=[]
        for item,ncluster,idt,idtask,score in batch:
            hist, future, neighbor = item[0], item[1], item[2]
            id_track.append(idt)
            nclusters.append(ncluster)
            id_tasks.append(idtask)
            scores.append(score)
            hist_shape = hist.shape
            neighbor_shape = neighbor.shape
            hist = np.reshape(hist, (-1, 2))
            neighbor = np.reshape(neighbor, (-1, 2))
            if self.flip:
               # print('flipped')
                if self.rng.randint(2):
                    hist[..., 1] *= -1
                    future[..., 1] *= -1
                    neighbor[..., 1] *= -1
                if self.rng.randint(2):
                    hist[..., 0] *= -1
                    future[..., 0] *= -1
                    neighbor[..., 0] *= -1
            if self.rotate:
               # print('rotated')
                rot = self.rng.random() * (np.pi+np.pi) 
                s, c = np.sin(rot), np.cos(rot)
                r = np.asarray([
                    [c, -s],
                    [s,  c]
                ])
                hist = (r @ np.expand_dims(hist, -1)).squeeze(-1)
                future = (r @ np.expand_dims(future, -1)).squeeze(-1)
                neighbor = (r @ np.expand_dims(neighbor, -1)).squeeze(-1)
            if self.scale:
               # print('scaled')
                s = self.rng.randn()*0.05 + 1 # N(1, 0.05)
                hist = s * hist
                future = s * future
                neighbor = s * neighbor
            hist = np.reshape(hist, hist_shape)
            neighbor = np.reshape(neighbor, neighbor_shape)

            X.append(hist)
            Y.append(future)
            NEIGHBOR.append(neighbor)
        
        n_neighbors = [n.shape[1] for n in NEIGHBOR]
        max_neighbors = max(n_neighbors) 
        if max_neighbors != min(n_neighbors):
            NEIGHBOR = [
                np.pad(neighbor, ((0, 0), (0, max_neighbors-n), (0, 0)), 
                "constant", constant_values=1e9)
                for neighbor, n in zip(NEIGHBOR, n_neighbors)
            ]
        stack_dim = 0 if self.batch_first else 1
        x = np.stack(X, stack_dim)
        y = np.stack(Y, stack_dim)
        neighbor = np.stack(NEIGHBOR, stack_dim)

        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        neighbor = torch.tensor(neighbor, dtype=torch.float32, device=self.device)
        err=torch.ones(x.shape[1],dtype=torch.float32, device=self.device)
        kl=torch.ones(x.shape[1],dtype=torch.float32, device=self.device)
        return x, y, neighbor,err,kl,nclusters,id_track,id_tasks,scores
    def collate_fntest(self, batch):
        X, Y, NEIGHBOR = [], [], []
        for item in batch:
            hist, future, neighbor = item[0][0], item[0][1], item[0][2]
            
            hist_shape = hist.shape
            neighbor_shape = neighbor.shape
            hist = np.reshape(hist, (-1, 2))
            neighbor = np.reshape(neighbor, (-1, 2))
            if self.flip:
               # print('flipped')
                if self.rng.randint(2):
                    hist[..., 1] *= -1
                    future[..., 1] *= -1
                    neighbor[..., 1] *= -1
                if self.rng.randint(2):
                    hist[..., 0] *= -1
                    future[..., 0] *= -1
                    neighbor[..., 0] *= -1
            if self.rotate:
               # print('rotated')
                rot = self.rng.random() * (np.pi+np.pi) 
                s, c = np.sin(rot), np.cos(rot)
                r = np.asarray([
                    [c, -s],
                    [s,  c]
                ])
                hist = (r @ np.expand_dims(hist, -1)).squeeze(-1)
                future = (r @ np.expand_dims(future, -1)).squeeze(-1)
                neighbor = (r @ np.expand_dims(neighbor, -1)).squeeze(-1)
            if self.scale:
               # print('scaled')
                s = self.rng.randn()*0.05 + 1 # N(1, 0.05)
                hist = s * hist
                future = s * future
                neighbor = s * neighbor
            hist = np.reshape(hist, hist_shape)
            neighbor = np.reshape(neighbor, neighbor_shape)

            X.append(hist)
            Y.append(future)
            NEIGHBOR.append(neighbor)
        
        n_neighbors = [n.shape[1] for n in NEIGHBOR]
        max_neighbors = max(n_neighbors) 
        if max_neighbors != min(n_neighbors):
            NEIGHBOR = [
                np.pad(neighbor, ((0, 0), (0, max_neighbors-n), (0, 0)), 
                "constant", constant_values=1e9)
                for neighbor, n in zip(NEIGHBOR, n_neighbors)
            ]
        stack_dim = 0 if self.batch_first else 1
        x = np.stack(X, stack_dim)
        y = np.stack(Y, stack_dim)
        neighbor = np.stack(NEIGHBOR, stack_dim)

        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        neighbor = torch.tensor(neighbor, dtype=torch.float32, device=self.device)
        ade=torch.ones(x.shape[1],dtype=torch.float32, device=self.device)
        #kl=torch.ones(x.shape[1],dtype=torch.float32, device=self.device)
        return x, y, neighbor,ade

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def load(self, filename, inclusive_groups):
        if os.path.isdir(filename): return None
        print('load:',filename)
        with open(filename, "r") as record:
            data = self.load_traj(record)
            print('load,')
        data = self.extend(data,1)
        #data nomor 5 dan 6 berisi data radius dan jumlah pedestrian
        return data
        
        
    def createtheneighbour(self,data,inclusive_groups):
        horizon = (self.horizon-1)*1
        print('start rising')
        time = np.sort(list(data.keys()))
        if len(time) < horizon+1: return None
        valid_horizon = self.ob_horizon + self.pred_horizon
        traj = []
        trajncluster=[]
        score=[]
        e = len(time)
        tid0 = 0
        while tid0 < e-horizon:
            tid1 = tid0+horizon
            t0 = time[tid0]
            
            idx = [aid for aid, d in data[t0].items() if not inclusive_groups or any(g in inclusive_groups for g in d[-1])]#ambil data di frame
            if idx:
                idx_all = list(data[t0].keys())
                for tid in range(tid0+1, tid1+1, 1):#loop kesemua frame pada horizon
                    t = time[tid]
                    idx_cur = [aid for aid, d in data[t].items() if not inclusive_groups or any(g in inclusive_groups for g in d[-1])]#kumpulan id pada frame setelahnya
                    if not idx_cur: # ignore empty frames
                        tid0 = tid
                        idx = []
                        break
                    idx = np.intersect1d(idx, idx_cur)#mengambil id yang ada pada kedua frame
                    if len(idx) == 0: break
                    idx_all.extend(data[t].keys())
            if len(idx):
                data_dim = 7
                neighbor_idx = np.setdiff1d(idx_all, idx)
                if len(idx) == 1 and len(neighbor_idx) == 0:
                    agents = np.array([
                        [data[time[tid]][idx[0]][:data_dim]] + [[1e9]*data_dim]
                        for tid in range(tid0, tid1+1, 1)#ini pembuatan range sepanjang horizonnya
                    ]) # L x 2 x 6
                else:
                    agents = np.array([
                        [data[time[tid]][i][:data_dim] for i in idx] +
                        [data[time[tid]][j][:data_dim] if j in data[time[tid]] else [1e9]*data_dim for j in neighbor_idx]
                        for tid in range(tid0, tid1+1, 1)#neighbourhood itu setiap frame dan id pedestrian
                    ])  # L X N x 6
                #print('totalidx:',len(idx))
                def neighborunderradius(self,agents,i,radius):
                    traject=[]
                    for valhor in agents:
                        curr=np.array(valhor[i,:2])
                        valhortemp=[]
                        distances=[]
                        for idx,ped in enumerate(valhor):
                            if(idx!=i):
                                neigh=np.array([ped[0],ped[1]])
                                dist=np.linalg.norm(curr-neigh)
                                distances.append(dist)
                                if(dist<radius):
                                    valhortemp.append(ped)
                        traject.append(np.array(valhortemp, dtype=object))
                    return np.array(traject,dtype=object)
                def neighbourtopk(self,agents,i,k):
                    traject=[]
                    for valhor in agents:
                        curr=np.array(valhor[i,:2])
                        valhortemp=[]
                        distances=[]
                        for idx,ped in enumerate(valhor):
                            if(idx!=i):
                                neigh=np.array([ped[0],ped[1]])
                                dist=np.linalg.norm(curr-neigh)
                                #distances.append(dist)
                                valhortemp.append([dist,ped])
                        sorted_data = sorted(valhortemp, key=lambda x: x[0])
                        top10 = [item[1] for item in sorted_data[:10]]
                        traject.append(np.array(top10, dtype=object))
                    return np.array(traject,dtype=object)
                def neighbourtopktensor(self,agents,i,k):#i adalah index agennya
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    curr=torch.tensor(agents[:,i][:, :2],device=device).unsqueeze(1)
                    others=torch.tensor(agents[:, :, :2] ,device=device)
                    #print('curr:',curr.shape)
                    #print('others:',others.shape)
                    diff=(curr-others)**2
                    dist=diff.sum(dim=2)
                    res=torch.sqrt(dist)
                    finalmean=res.mean(dim=0)
                    top_values, top_indices= torch.topk(finalmean, k=k,largest=False) 
                    #a=res<200
                    agents=torch.tensor(agents,device=device)
                    #final = [t[m] for t, m in zip(agents, a)]
                    #tensor_list = [torch.tensor(t) for t in final]
                    tensor_list=agents[:,top_indices,:]
                    final=tensor_list
                    #final=res[a]#[a.cpu().numpy()]
                    # for valhor in agents:
                    #     others=torch.tensor(valhor,device=device)
                    #     diff=(curr-others)**2
                    #     dist=diff.sum(dim=1)
                    #     res=torch.sqrt(dist) 
                    #     a=res<10
                    return final
                
                for i in range(len(idx)):
                    hist = agents[:self.ob_horizon,i]  # L_ob x 6
                    getncluster= hist[:,6]
                    masked_values = np.where(getncluster < 1000, getncluster, np.nan) 
                    histncluster= np.nanmean(masked_values, axis=0) #get the cluster avg data
                    hist=agents[:self.ob_horizon,i,:6]
                    #print(np.max(histncluster))
                    future = agents[self.ob_horizon:valid_horizon,i]  # L_pred x 2
                    getncluster=future[:,6]
                    masked_values = np.where(getncluster < 1000, getncluster, np.nan) 
                    future = agents[self.ob_horizon:valid_horizon,i,:2]
                    futurencluster=np.nanmean(masked_values, axis=0)
                   # neighbor=neighborunderradius(self,agents,i,150)
                 #
                    #neighbor = agents[:valid_horizon, [d for d in range(agents.shape[1]) if d != i]] # and distance(agents,d,i)<10] # L x (N-1) x 6#ini biang keroknya, pembuatan neighbour
                   
                    neighbor=neighbourtopktensor(self,agents,i,self.kneighbour)
                    #print(np.array(neighbors).shape)
                    #print(np.array(neighbor).shape)
                    # hs=np.array(hist)
                    # fu=np.array(future)
                    # nei=np.array(neighbor)
                   
                    #hsn=np.array(histncluster)

                    traj.append((hist, future, neighbor))#cekkk
                    trajncluster.append((histncluster,futurencluster))
                    score.append(histncluster)
                    #print('nei')
                    #print(nei)
                    #print('traj:',len(traj),',',hs.shape,',',fu.shape,',',nei.shape)
                   # print('nei len:',len(nei),', hist len:',len(hs))
                #if(tid0==1):
                   # print(traj[0])
                    #print(nei)
            tid0 += self.frameskip
        #print('check 1')
        items = []
        idtracks=[]
        idtasks=[]
        idtrack=0
        for hist, future, neighbor in traj:
            hist = np.float32(hist)
            future = np.float32(future)
            neighbor = np.float32(np.array(neighbor.cpu().numpy()))
            weight=1
            items.append((hist, future, neighbor,weight))
            idtracks.append(idtrack+(self.idtask*100000))
            idtasks.append(self.idtask)
            idtrack+=1
        scorenorm = (score - min(score)) / (max(score) - min(score))
        #print('finish')
        #testing=np.array(items)
        #print('item shape',testing.shape)
        #self.ncluster=trajncluster
        return items,trajncluster,idtracks,idtasks,scorenorm
    def distance(self,agents,d,i):
        dist=math.sqrt((agents[d][0]-agents[i][0])^2+(agents[d][1]-agents[i][1])^2)
        return dist
    def euclidean_distance(self,x, y):
        return torch.sqrt(torch.sum((x - y) ** 2))

    
    def dtw_normalized_vectors(self,trajectory1, trajectory2):
        distance = self.dtw(trajectory1, trajectory2)
        return distance
    
    def calculatesimilarity(self,agents1,agents2):
        #mengkalkulasi kemiripan dengan dynamic time wrapping dan juga menggunakan metode untuk menghitung kemiripan dari posisi neighbourhood(berarti memperhatikan konteks), kecepatan dan percepatan
        #selanjutnya bikin surat juga buat indo!
        pastagents1=torch.tensor(agents1[0][:,:2],device=self.device)
        pastagents2=torch.tensor(agents2[0][:,:2],device=self.device)
        futureagents1=torch.tensor(agents1[1],device=self.device)
        futureagents2=torch.tensor(agents2[1],device=self.device)
        past=self.dtw_normalized_vectors(pastagents1,pastagents2)
        future=self.dtw_normalized_vectors(futureagents1,futureagents2)
        return past,future
    def filtersimilarity(self,prevdata):
        ktmu=0
        #mencari similarity berdasarkan previous data per scene
        for idx,d in enumerate(self.data):#8000
            agents1=d 
            #agents2=prevdata
            #kalau menggunakan previous data maka data akan diulang n kali, tapi kalau menggunakan allprevious data,
            #konidisi ini akan menghasilkan 3x1000x500, kalau all previous data 1000x500x3 jadi sama ja
            #data previous data masih banyak jadi lebih baik gunakan loop
            #looping similarity bikin masalah cuy! karena perlu crosscheck 2 data!
            #tambahakan strategi untuk early stop!!!!!!!!!!!!!!!!!! sama strategi untuk hitung similaritynya!!!!!!! misalnya pakai 10 jarak pertama kalau ga mirip lgs skip!truss neighbourhoodnya mau dimiripin juga gak!!!
            for batch in prevdata: #8000
                agents=batch[0]
                futureagents=batch[1] 
                start = time.time()
                for pd in range(agents.shape[1]):
                    agents2=[agents[:,pd,:],futureagents[:,pd,:]]
                    past,future=self.calculatesimilarity(agents1,agents2)
                    if(past+future)<20000:
                        self.data=np.delete(self.data,idx)
                        ktmu+=1
                        #self.data=self.data[self.data!=d]
                        #berikan similarity score kepada data yang tersimpan!
                        val=past+future
                end=time.time()
                total=end-start
                #print('ktmu:',ktmu)
                #print('time on',total)
        return val          
    def extend(self, data, frameskip):
        time = np.sort(list(data.keys()))
        dts = np.unique(time[1:] - time[:-1])
        dt = dts.min()
        if np.any(dts % dt != 0):
            raise ValueError("Inconsistent frame interval:", dts)
        i = 0
        while i < len(time)-1:
            if time[i+1] - time[i] != dt:
                time = np.insert(time, i+1, time[i]+dt)
            i += 1
        # ignore those only appearing at one frame
        for tid, t in enumerate(time):
            removed = []
            if t not in data: data[t] = {}
            for idx in data[t].keys():
                t0 = time[tid-frameskip] if tid >= frameskip else None
                t1 = time[tid+frameskip] if tid+frameskip < len(time) else None
                if (t0 is None or t0 not in data or idx not in data[t0]) and \
                (t1 is None or t1 not in data or idx not in data[t1]):
                    removed.append(idx)
            for idx in removed:
                data[t].pop(idx)
        # extend velocity
        for tid in range(len(time)-frameskip):
            t0 = time[tid]
            t1 = time[tid+frameskip]
            if t1 not in data or t0 not in data: continue
            for i, item in data[t1].items():
                if i not in data[t0]: continue
                x0 = data[t0][i][0]
                y0 = data[t0][i][1]
                x1 = data[t1][i][0]
                y1 = data[t1][i][1]
                vx, vy = x1-x0, y1-y0
                data[t1][i].insert(2, vx)
                data[t1][i].insert(3, vy)
                if tid < frameskip or i not in data[time[tid-1]]:
                    data[t0][i].insert(2, vx)
                    data[t0][i].insert(3, vy)
        # extend acceleration
        for tid in range(len(time)-frameskip):
            t_1 = None if tid < frameskip else time[tid-frameskip]
            t0 = time[tid]
            t1 = time[tid+frameskip]
            if t1 not in data or t0 not in data: continue
            for i, item in data[t1].items():
                if i not in data[t0]: continue
                vx0 = data[t0][i][2]
                vy0 = data[t0][i][3]
                vx1 = data[t1][i][2]
                vy1 = data[t1][i][3]
                ax, ay = vx1-vx0, vy1-vy0
                data[t1][i].insert(4, ax)
                data[t1][i].insert(5, ay)
                #data[t1][i].insert(6, ay)
                if t_1 is None or i not in data[t_1]:
                    # first appearing frame, pick value from the next frame
                    data[t0][i].insert(4, ax)
                    data[t0][i].insert(5, ay)
                
        return data

    def load_traj(self, file):
        data = {}
        for row in file.readlines():
            item = row.split()
            if not item: continue
            t = int(float(item[0]))
            idx = int(float(item[1]))
            x = float(item[2])
            y = float(item[3])
            number=float(item[5]) if len(item) > 4 else None 
            #group = item[4].split("/") if len(item) > 4 else None
            if t not in data:
                data[t] = {}
            data[t][idx] = [x, y, number, None]
        #print('finish')
        return data
