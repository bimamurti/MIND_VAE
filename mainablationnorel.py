import os, sys, time
import importlib
import torch
import numpy as np
import random
#import ContinualLearning.PreviousData
from ContinualLearning.previousdata import previousdata
from torch.utils.tensorboard import SummaryWriter

from databatch import Dataloaderbatch
from social_vae import SocialVAE
from data import Dataloader
from utils import *
from ContinualLearning.previousdata import previousdata
from offREL import *
#from memory_profiler import memory_usage
import resource
import argparse
import glob
import time
import gc
from RelBilevel import *
from bcsr_coreset import BCSR_Coreset

parser = argparse.ArgumentParser()#1.21.2
parser.add_argument("--train", nargs='+', default=['data/continualdata_sandbox/task3/train'])
parser.add_argument("--test", nargs='+', default=['data/continualdata_sandbox/task3/test'])#/home/USER/Documents/projects/MIND_VAE/data/continualdata/data90d7/

#parser.add_argument("--train", nargs='+', default=['data/crowdTraj World/dataset cluster world CL/duri_morning_cluster_world/train'])
#parser.add_argument("--test", nargs='+', default=['data/crowdTraj World/dataset cluster world CL/duri_morning_cluster_world/test'])
# parser.add_argument("--train", nargs='+', default=['data/continual_trajcrowd/paisley_cluster/train'])
# parser.add_argument("--test", nargs='+', default=['data/continual_trajcrowd/paisley_cluster/test'])
#parser.add_argument("--train", nargs='+')
#parser.add_argument("--train", nargs='+', default=['data/fullreplay/train'])
#parser.add_argument("--test", nargs='+', default=['data/fullreplay/test'])
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--config", type=str, default='config/configsandbox.py')
#parser.add_argument("--config", type=str, default='config/mot.py')
#parser.add_argument("--ckpt", type=str, default='log_eth/logExRecplayoctber28th6/logtask2')
#parser.add_argument("--ckpt", type=str, default='log_eth/logExRecplayfixx/logdataMOT2122')
parser.add_argument("--ckpt", type=str, default='log_eth/logproposedtest')
#parser.add_argument("--ckpt", type=str, default='log_eth/testRel15')
#parser.add_argument("--ckpt", type=str, default='log_eth/testprevious')
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--no-fpc", action="store_true", default=True)
parser.add_argument("--fpc-finetune", action="store_true", default=False)
#parser.add_argument("--prev_data",type=str,default="log_eth/logproposedtest")
#parser.add_argument("--prev_data",default='log_eth/logcontinualproposed_20260428_155403/logduri_morning_cluster')
parser.add_argument("--prev_data",type=str)
parser.add_argument("--datacapacity",type=str,default="15")#data capacity in percentage for previous data to be merged into current training data, default is 10 percent
parser.add_argument("--method",type=str,default="REL")
parser.add_argument("--taskno",type=str,default="1")#should be more than 1
parser.add_argument("--proxy_lr",type=float,default=1e-4)
parser.add_argument("--weight_lr",type=float,default=1e-10)
parser.add_argument("--selectionparam", type=float, default=25.0, help="percentage of training samples to keep during reduction")
#parser.add_argument("--buffersize", type=float, default=50.0, help="percentage of training samples to keep during reduction")
if __name__ == "__main__":
    countstop=0
    skip_to_last_epoch=False
    early_stop=50#early stop if there is no improvement in ADE for 50 consecutive epochs, this is to prevent overfitting and save time, but still execute the last epoch to save the best model and best XY data
    #pd=previousdata()
    max_mem = torch.cuda.max_memory_allocated(0)
    time_start = time.time()
    print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_start)))
    def loadallpreviousmemories(folder_path):
        #mengambil data terakhir saja, karena data sebelumnya sudah di merge ke data training, jadi tidak perlu diambil lagi, cukup yang terakhir saja, tapi kalau mau diambil semua juga bisa, nanti tinggal di filter aja per scene nya
        alldata=[]
        #pathall=folder
        pathall=os.path.join(folder_path,"*")
        pt_folder=glob.glob(pathall)
        #for folder in pt_folder:
        folder=folder_path
        test_files = glob.glob(os.path.join(folder, "bestXY.pt"))
        train_files = glob.glob(os.path.join(folder, "bestCandidate.pt"))
        #print(pt_files)
        
        
            #torch.serialization.add_safe_globals([previousdata])
            # check whether this file can be loaded; then fix the previous-data handling
            # and create the filter there instead of in the merge function
        if(len(test_files)>0):
            test=torch.load(test_files[0],weights_only=False)
            train=torch.load(train_files[0],weights_only=False)#ini hanya diambil 
            alldata.append([test,train])
        return alldata 
    torch.cuda.reset_peak_memory_stats()
    settings = parser.parse_args()
    print("data capacity:", settings.datacapacity)
    print("selection param:", settings.selectionparam)
    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    XYbest=[]
    lossbest=[]
    if settings.device is None:
        settings.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    settings.device = torch.device(settings.device)
   # print("ok1")
    seed(settings.seed)
    init_rng_state = get_rng_state(settings.device)
    rng_state = init_rng_state
    #prev_folder=settings.prev_data.split(",")
    ###############################################################################
    #####                                                                    ######
    ##### prepare datasets                                                   ######
    #####                                                                    ######
    ###############################################################################
    kwargs = dict(
            batch_first=False, frameskip=settings.frameskip,
            ob_horizon=config.OB_HORIZON, pred_horizon=config.PRED_HORIZON,
            device=settings.device, seed=settings.seed)
    train_data, test_data = None, None

    if settings.test:
        print('Testing!!!!!!!!!!!!!!!1')
        print(settings.test)
 
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.test))]
        else:
            inclusive = None
        test_dataset= Dataloader(
            settings.test, **kwargs,inclusive_groups=inclusive,idtask=int(settings.taskno),
            batch_size=config.BATCH_SIZE, shuffle=False, kneighbour=config.MAX_NEIGHBORS
        )
        if settings.prev_data!=None:
            sample=int(settings.datacapacity)#iniiiii pentingggg!!!!
            allprevdata=loadallpreviousmemories(settings.prev_data)
            #sample=sample//(len(allprevdata[0])/2)
            for prev_data in allprevdata:#looping per scene
                filterpreviousdata=previousdata(sample,settings.method)
                filterpreviousdata.getProposedData(prev_data[1],prev_data[0])
                prevtestdata=filterpreviousdata.dataprevvalid
                test_dataset=mergeprevioustestdata(test_dataset,prevtestdata)
            #test_dataset=test_dataset.extend(prevtestdata)
            #harusnya ditambahkan ke test_data karena dah disampler, ternyta ga usah karena dipakai semua
        test_data = torch.utils.data.DataLoader(test_dataset, 
            collate_fn=test_dataset.collate_fnscore,
            batch_sampler=test_dataset.batch_sampler
        )
        max_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"Peak memory usage: {max_memory_kb / 1024} MiB")
        def test(model, fpc=1):
            sys.stdout.write("\r\033[K Evaluating...{}/{}".format(
               0, len(test_dataset)
            ))
            tic = time.time()
            model.eval()
            ADE, FDE, XY = [], [], []
            set_rng_state(init_rng_state, settings.device)
            batch = 0
            fpc = int(fpc) if fpc else 1
            fpc_config = "FPC: {}".format(fpc) if fpc > 1 else "w/o FPC"
            with torch.no_grad():
                #print("test ui")
                #print(len(test_dataset))
                #for x, y, neighbor,err,kl,nclusters,id_track,id_tasks in test_data:
                
                for data in test_data:
                    x, y, neighbor,err,kl,nclusters,id_track,id_tasks,scores=data
                    batch += x.size(1)
                    sys.stdout.write("\r\033[K Evaluating...{}/{} ({}) -- time: {}s".format(
                       batch, len(test_dataset), fpc_config, int(time.time()-tic)
                    ))
                    
                    if config.PRED_SAMPLES > 0 and fpc > 1:
                        # disable fpc testing during training
                        y_ = []
                        for _ in range(fpc):
                            y_.append(model(x, neighbor, n_predictions=config.PRED_SAMPLES))
                        y_ = torch.cat(y_, 0)
                        cand = []
                        for i in range(y_.size(-2)):
                            cand.append(FPC(y_[..., i, :].cpu().numpy(), n_samples=config.PRED_SAMPLES))
                        # n_samples x PRED_HORIZON x N x 2
                        y_ = torch.stack([y_[_,:,i] for i, _ in enumerate(cand)], 2)
                    else:
                        # n_samples x PRED_HORIZON x N x 2
                        y_ = model(x, neighbor, n_predictions=config.PRED_SAMPLES)
                    ade, fde = ADE_FDE(y_, y)
                    if config.PRED_SAMPLES > 0:
                        ade = torch.min(ade, dim=0)[0]
                        fde = torch.min(fde, dim=0)[0]
                    ADE.append(ade)
                    FDE.append(fde)
                    XY.append([x,y,neighbor,ade,fde,nclusters,id_track,id_tasks])#salahnya disini harusnya cuma 1 saja ini diambil 9
            ADE = torch.cat(ADE)
            FDE = torch.cat(FDE)
            #XY=torch.cat(XY)
            #ADEnFDE di cat brati di gabungkan seluruh listnya ex:56x17 sehingga semua keluar dan bisa di mean
            if torch.is_tensor(config.WORLD_SCALE) or config.WORLD_SCALE != 1:
                if not torch.is_tensor(config.WORLD_SCALE):
                    config.WORLD_SCALE = torch.as_tensor(config.WORLD_SCALE, device=ADE.device, dtype=ADE.dtype)
                ADE *= config.WORLD_SCALE
                FDE *= config.WORLD_SCALE
            ade = ADE.mean()
            fde = FDE.mean()
            sys.stdout.write("\r\033[K ADE: {:.4f}; FDE: {:.4f} ({}) -- time: {}s".format(
               ade, fde, fpc_config, 
               int(time.time()-tic))
            )
            #print()
            
            return ade, fde, XY
    
    max_mem = torch.cuda.max_memory_allocated(0)
    train=False
    if settings.train:
        print('Training !!!!!!!!!!!!!!!!!!!1')
        print(settings.train)
        train=True
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.train))]
        else:
            inclusive = None
       
        train_dataset = Dataloader(
            settings.train, **kwargs, inclusive_groups=inclusive,
            flip=True, rotate=True, scale=True,
            batch_size=config.BATCH_SIZE, shuffle=False, batches_per_epoch=config.EPOCH_BATCHES,idtask=int(settings.taskno), kneighbour=config.MAX_NEIGHBORS
        )
        batches = train_dataset.batches_per_epoch
        #if settings.prev_data!=None:
           # prev_data=
            #prev_data=previousdata()
            #prev_data = torch.load("/home/USER/Documents/projects/MIND_VAE/log_eth/testCL2050top10/longterm.pt")# check here
          #  train_dataset=train_dataset.extend(prev_data)
        #train_dataset=Dataloader()
        train_data= torch.utils.data.DataLoader(train_dataset,
            collate_fn=train_dataset.collate_fnscore,
            batch_sampler=train_dataset.batch_sampler
        )
        if settings.prev_data!=None:
            #merge both train data and previous data
            #prevtrainingdata=prev_data[0].dataprevtrain
            sample=int(settings.datacapacity)#iniiiii pentingggg!!!!
            #test_dataset=mergepreviousdatatraining(train_data,prevtrainingdata,sample)
        #batches = train_dataset.batches_per_epoch
        max_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"Training Peak memory usage: {max_memory_kb / 1024} MiB")
    ###############################################################################
    #####                                                                    ######
    ##### load model                                                         ######
    #####                                                                    ######
    ###############################################################################
    model = SocialVAE(horizon=config.PRED_HORIZON, ob_radius=config.OB_RADIUS, hidden_dim=config.RNN_HIDDEN_DIM)
    model.to(settings.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    start_epoch = 0
    if settings.ckpt:
        ckpt = os.path.join(settings.ckpt, "ckpt-last")
        ckpt_best = os.path.join(settings.ckpt, "ckpt-best")
        if os.path.exists(ckpt_best):
            print('Use previous model!')
            state_dict = torch.load(ckpt_best, map_location=settings.device)
            ade_best = 10000#state_dict["ade"]
            fde_best = 10000#state_dict["fde"]
            fpc_best = state_dict["fpc"] if "fpc" in state_dict else 1
        else:
            print('New Model')
            ade_best = 100000
            fde_best = 100000
            fpc_best = 1
        if train_data is None: # testing mode
            ckpt = ckpt_best
        if os.path.exists(ckpt_best):
            print("Load from ckpt:", ckpt_best)
            #torch.serialization.add_safe_globals([_reconstruct])
            state_dict = torch.load(ckpt_best, map_location=settings.device, weights_only=False)
            model.load_state_dict(state_dict["model"])
            if "optimizer" in state_dict:
                optimizer.load_state_dict(state_dict["optimizer"])
                rng_state = [r.to("cpu") if torch.is_tensor(r) else r for r in state_dict["rng_state"]]
            start_epoch = 0#state_dict["epoch"]
    end_epoch = start_epoch+1 if train_data is None or start_epoch >= config.EPOCHS else config.EPOCHS

    if settings.train and settings.ckpt:
        logger = SummaryWriter(log_dir=settings.ckpt)
    else:
        logger = None
    #print("get 1")
    if train_data is not None:
        #print("get 2")
        log_str = "\r\033[K {cur_batch:>"+str(len(str(batches)))+"}/"+str(batches)+" [{done}{remain}] -- time: {time}s - {comment}"    
        progress = 20/batches if batches > 20 else 1
        optimizer.zero_grad()
    #print("get 3")
    #k=20
    buffersize=int((int(settings.datacapacity)/100)*len(train_dataset.data))#int(settings.datacapacity)#harusnya presentase dari data training
    #buffersize=32
    #batch_size=config.BATCH_SIZE//(2**2)
    batch_size=int(config.BATCH_SIZE*0.7)
    #selectionparam=50
    candidates_indices=[]
    candidates_weights=[]
    proxy_model = SocialVAE(horizon=config.PRED_HORIZON, ob_radius=config.OB_RADIUS, hidden_dim=config.RNN_HIDDEN_DIM)
    proxy_model.to(settings.device)
    bc = BCSR_Coreset(proxy_model=proxy_model, lr_proxy_model=settings.proxy_lr ,beta=10, out_dim=100,
            max_outer_it=5, max_inner_it=1,
            weight_lr=settings.weight_lr, logging_period=1000,device=settings.device)
    lastbatch=[]
    def reducethetrainingdata(train_data,train_dataset):
        if train_data is None:
            return None

        dataset = getattr(train_data, "dataset", None)
        if dataset is None:
            return train_data

        keep_ratio = max(0.0, min(1.0, settings.selectionparam / 100.0))
        if keep_ratio >= 1.0:
            return train_data

        original_count = len(dataset)
        print(f"Reducing training data: keeping {keep_ratio * 100:.1f}% of {original_count} samples")

        # Collect all error-based scores and track actual dataset indices
        all_scores = []
        all_dataset_indices = []  # Track which indices from the actual dataset were in the batches
        eps = 1e-8
        model.eval()
        with torch.no_grad():
            # Get the batch sampler to know which samples were actually included
            batch_sampler = train_data.batch_sampler
            for batch_indices, data in zip(batch_sampler, train_data):
                x, y, neighbor, err, kl, nclusters, id_track, id_tasks, scores = data
                current_batch_size = x.shape[0]  # Get batch size from input tensor
                
                # Get the actual dataset indices for this batch
                all_dataset_indices.extend(batch_indices)
                
                if config.PRED_SAMPLES > 0:
                    y_ = model(x, neighbor, n_predictions=config.PRED_SAMPLES)
                else:
                    y_ = model(x, neighbor, n_predictions=config.PRED_SAMPLES)

                ade, fde = ADE_FDE(y_, y)
                if config.PRED_SAMPLES > 0:
                    ade = torch.min(ade, dim=0)[0]
                    fde = torch.min(fde, dim=0)[0]

                # Reduce ade and fde to per-sample values (average over all dimensions except batch)
                if ade.dim() > 1:
                    ade = ade.view(current_batch_size, -1).mean(dim=1)
                if fde.dim() > 1:
                    fde = fde.view(current_batch_size, -1).mean(dim=1)

                # Higher ADE/FDE should make a sample more likely to be selected.
                score = ade + fde
                score = torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)
                if score.sum() <= 0:
                    score = torch.ones(current_batch_size, dtype=torch.float32, device=score.device)

                all_scores.append(score)

        if not all_scores:
            return train_data

        # Concatenate all scores and compute global probabilities
        all_scores = torch.cat(all_scores, dim=0)
        total_samples = all_scores.shape[0]
        
        # Calculate exact target number of samples to keep
        target_keep_count = max(1, int(round(total_samples * keep_ratio)))
        
        # Global multinomial sampling based on error scores
        probabilities = all_scores / (all_scores.sum() + eps)
        #selected_indices_relative = torch.multinomial(probabilities, target_keep_count, replacement=False).cpu().numpy()
        selected_indices_relative = torch.topk(probabilities, target_keep_count, largest=True).indices.cpu().numpy()
        
        # Map relative indices back to actual dataset indices
        all_dataset_indices = np.asarray(all_dataset_indices, dtype=np.int64)
        selected_indices = all_dataset_indices[selected_indices_relative]
        
        train_dataset.data = train_dataset.data[selected_indices]
        train_dataset.ncluster = [train_dataset.ncluster[i] for i in selected_indices]
        train_dataset.idtrack = [train_dataset.idtrack[i] for i in selected_indices]
        train_dataset.idtasks = [train_dataset.idtasks[i] for i in selected_indices]
        train_dataset.scores = [train_dataset.scores[i] for i in selected_indices]

        print(f"Training data reduced from {original_count} to {len(train_dataset)} samples")

        # Rebuild the batch sampler on the parent Dataloader so indices reflect the new dataset size
        #train_dataset=Dataloader
        train_dataset = Dataloader(
            settings.train, **kwargs, inclusive_groups=inclusive,
            flip=True, rotate=True, scale=True,
            batch_size=config.BATCH_SIZE, shuffle=False, batches_per_epoch=config.EPOCH_BATCHES,idtask=int(settings.taskno),
            kneighbour=config.MAX_NEIGHBORS,reduceddata=train_dataset
        )
        train_data = torch.utils.data.DataLoader(train_dataset,
            collate_fn=train_dataset.collate_fnscore,
            batch_sampler=train_dataset.batch_sampler
        )
        return train_data
    if(settings.prev_data!=None):
        train_data = reducethetrainingdata(train_data,train_dataset)
    batches = len(train_data)
    log_str = "\r\033[K {cur_batch:>" + str(len(str(batches))) + "}/" + str(batches) + " [{done}{remain}] -- time: {time}s - {comment}"
    progress = 20 / batches if batches > 20 else 1
    optimizer.zero_grad()

    #rl=ReducibleLoss(train_data,settings,config,buffersize,batch_size,selectionparam)
    epoch=start_epoch
    while epoch < end_epoch:
        epoch+=1
        if skip_to_last_epoch and epoch != end_epoch:
            end_epoch=epoch
        else:
            print("get", epoch)
        #print("get start:",start_epoch+1)
        #print("get end:",end_epoch+1)
        ###############################################################################
        #####                                                                    ######
        ##### train                                                              ######
        #####                                                                    ######
        ###############################################################################
        losses = None
        losslist=[]
        #print("epoch",epoch)
        #print("config epoch",config.EPOCHS)
        if train_data is not None and epoch <= config.EPOCHS:
            #print("get 5")
            #print("Epoch {}/{}".format(epoch, config.EPOCHS))
            tic = time.time()
            set_rng_state(rng_state, settings.device)
            losses = {}
            model.train()
            sys.stdout.write(log_str.format(
               cur_batch=0, done="", remain="."*int(batches*progress),
               time=round(time.time()-tic), comment=""))
            #realtraindata=[x for x in train_data]
            #if settings.prev_data!=None:
            #    realtraindata=mergepreviousdatatraining(realtraindata,prevtrainingdata,sample)
            lossarray=[]
            nbatch=len(train_data)
            errindividuallist=[]
            klindividuallist=[]
            trainedindex=[]
            for batch, items in enumerate(train_data):
                #multiplier = [1000000] * len(items[-1])
                idtracks = items[-3]#[m + val for m, val in zip(multiplier, items[-2])]
                ncluster=items[5]
                taskno=items[-2]
                s=np.array(items[-1], dtype=np.float64)
                item=items[:-3] + (s,)
                idtask=items[-2]
                #a=ncluster[0]
                #print(a)
                if settings.prev_data!=None:
                    #sample=sample//nbatch#ini membagi sample per batch
                    #masukan samplenya per batch disini
                    for prev_data in allprevdata:
                        sample=sample//len(allprevdata)
                        prevtrainingdata=filterpreviousdata.dataprevtrain
                        # Move prevtrainingdata to the same device as settings.device
                        if isinstance(prevtrainingdata, (list, tuple)):
                            prevtrainingdata = tuple(t.to(settings.device) if torch.is_tensor(t) else t for t in prevtrainingdata)
                        #prevtrainingdata=prev_data[0].dataprevtrain
                        #print("rise!!!")
                        items,trainedindex=mergepreviousdatatrainingproposed(items,prevtrainingdata,batches,trainedindex)#harusnya diambil 20 persen jg dari batch nya, tapi sudah
                        item=items[:-3] + [items[-1]]
                        idtracks=items[-3]
                        ncluster=items[5]#bagaimana ini apakah mau di kasih 0 untuk score diawalnya atau lgs ambil dari cluster score atau gimana cek cek.
                        idtask=items[-1]
                       # print("done!!")
                res = model(*item)#ini returnnya err, kl , err itu selisih y sama y' cuman bukan loss total, model loss nya berbeda dengan err
                free, total = torch.cuda.mem_get_info("cuda:0")
                mem_used_MB = (total - free) / 1024 ** 2
                #print('after model',mem_used_MB)
                errindividual=res[0].mean(dim=(0,2))#ini loss per data, mean di ambil dari sample prediksi
                klindividual=res[1].mean(dim=(0,2))
                loss_err=errindividual+klindividual
                loss = model.loss(*res)#err,kl di masukkan ke loss, dimana loss itu untuk 16 data per batch, jadi individual lossnya harus diambil dari res
                item_cpu = tuple(t.clone().cpu() if torch.is_tensor(t) else t for t in item)
                lossarray.append([item_cpu,errindividual.clone().cpu(),klindividual.clone().cpu(),torch.as_tensor(idtracks, dtype=torch.int, device='cpu'),torch.as_tensor(ncluster, dtype=torch.double, device='cpu'),torch.as_tensor(idtask, dtype=torch.int, device='cpu')])
               # lossarray.append([item,errindividual,klindividual,torch.as_tensor(idtracks, dtype=torch.int, device=settings.device),torch.as_tensor(ncluster, dtype=torch.double, device=settings.device),torch.as_tensor(taskno, dtype=torch.int, device=settings.device)])
                #lossaray isinnya item,errindividual,klindividual,idtracks,ncluster,taskno
                #errindividuallist.append(errindividual)
                #klindividuallist.append(klindividual)
                loss["loss"].backward()
                optimizer.step()
                optimizer.zero_grad()
                for k, v in loss.items():
                    #print(loss)
                    if k not in losses: 
                        losses[k] = v.item()
                    else:
                        losses[k] = (losses[k]*batch+v.item())/(batch+1)#ini akumulatif loss tiap batch untuk 1 epoch
                        #cari average loss per datanya untuk kemudian dicari weightnya

                if(epoch==end_epoch):#ini tentukan kapan training coresetnya
                    print("run BCSR coreset selection!")
                    if(batch==0):
                        starttimebcsr=time.time()
                    weights=bc.coreset_select_new(model, item, task_id=0,  topk=int(config.BATCH_SIZE*(int(settings.datacapacity)/100)),out_loss=[],idt=idtracks)
                    rng_state = get_rng_state(settings.device)
                    candidates_indices.append(idtracks)
                    candidates_weights.append(weights.detach().cpu().numpy())
                    item_cpu = tuple(t.detach().cpu() if torch.is_tensor(t) else t for t in item)
                    lastbatch.append([
                        item_cpu,
                        errindividual.detach().cpu(),
                        klindividual.detach().cpu(),
                        torch.as_tensor(idtracks, dtype=torch.int, device='cpu'),
                        torch.as_tensor(ncluster, dtype=torch.double, device='cpu'),
                        torch.as_tensor(idtask, dtype=torch.int, device='cpu'),
                        torch.as_tensor(weights, dtype=torch.double, device='cpu')
                    ])
                else:
                    del item
                    del item_cpu
                    torch.cuda.empty_cache()
                    gc.collect()
                losslist.append(losses['loss'])
                #rl.updatebuffer(items,loss_err,settings.taskno,idtracks,batch,epoch)
                sys.stdout.write(log_str.format(
                    cur_batch=batch+1, done="="*int((batch+1)*progress),
                    remain="."*(int(batches*progress)-int((batch+1)*progress)),
                    time=round(time.time()-tic),
                    comment=" - ".join(["{}: {:.4f}".format(k, v) for k, v in losses.items()])
                ))
                
            rng_state = get_rng_state(settings.device)
            if(epoch==end_epoch):
                endtimebcsr=time.time()
                
            # Flatten candidate index arrays into a single 1D array
                flat_candidates = np.concatenate([np.ravel(c) for c in candidates_indices]) if len(candidates_indices) > 0 else np.array([], dtype=int)
                flat_candidates=torch.asarray(flat_candidates   ,device=settings.device)
                addresslast=os.path.join(settings.ckpt, "lastbatch.pt")
                torch.save(lastbatch, addresslast)
                lastbatch.clear()      # optional but explicit
                del lastbatch
                gc.collect()
                torch.cuda.empty_cache()
            #print()
            
        ###############################################################################
        #####                                                                    ######
        ##### test                                                               ######
        #####                                                                    ######
        ###############################################################################
        ade, fde = 10000, 10000
        XY=[]
        perform_test = (train_data is None or epoch >= config.TEST_SINCE) and test_data is not None
        if perform_test:
            if not settings.no_fpc and not settings.fpc_finetune and losses is None and fpc_best > 1:
                fpc = fpc_best
            else:
                fpc = 1
            ade, fde, XY = test(model, fpc)
        ###############################################################################
        #####                                                                    ######
        ##### log                                                                ######
        #####                                                                    ######
        ###############################################################################
        if losses is not None and settings.ckpt:
            if logger is not None:
                for k, v in losses.items():
                    logger.add_scalar("train/{}".format(k), v, epoch)
                if perform_test:
                    logger.add_scalar("eval/ADE", ade, epoch)
                    logger.add_scalar("eval/FDE", fde, epoch)
            state = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                ade=ade, fde=fde, epoch=epoch, rng_state=rng_state
            )
            torch.save(state, ckpt)
            finalXY=[]
            finalloss=[]
            finalXY_cpu=[]
            neighbdata=[]
            futdata=[]
            obvdata=[]
            if(epoch >= config.TEST_SINCE):
                countstop+=1
            if(countstop>=early_stop):
                print("early stop at epoch",epoch)
                skip_to_last_epoch=True
            if ade < ade_best:
                countstop=0
                ade_best = ade
                fde_best = fde
                state = dict(
                    model=state["model"],
                    ade=ade, fde=fde, epoch=epoch
                )
                torch.save(state, ckpt_best)
                # Create a trainable copy of the model
                time_best = time.time()
                #XYbest=[XY.copy(),ade_best,fde_best]
                losslist=np.array(losslist)
                meanloss=np.mean(losslist, dtype=np.float64)
                #ka=10
                #topk_unsorted_idx = np.argpartition(losslist, -ka)[-ka:]
                # Step 2: Sort the top 5 values descending
                #topk_sorted_idx = topk_unsorted_idx[np.argsort(losslist[topk_unsorted_idx])[::-1]]
                #topk_values = losslist[topk_sorted_idx]
                #losslist=topk_values
                #lossarray=lossarray[topk_sorted_idx]
                #lossbest=lossarray.copy()
                #buat filter berdasarkan individual loss untuk loss arraynya
                #filter berdasarkan mean loss tiap individunya, cek perhitungan losslistnya
                finalloss=[]
                finalXY=[]
                isnotempty=False
                ######### select the validation data #########
                for i in range(len(XY)): 
                    maskADE=XY[i][3]<ade
                    adeindices=torch.nonzero(maskADE).squeeze()
                    temp=[XY[i][0][:,adeindices,:],XY[i][1][:,adeindices,:],XY[i][2][:,adeindices,:,:],XY[i][3][adeindices],torch.as_tensor(XY[i][4],device=XY[i][3].device)[adeindices],torch.as_tensor(XY[i][5],device=XY[i][3].device)[adeindices],torch.as_tensor(XY[i][6],device=XY[i][3].device) [adeindices],torch.as_tensor(XY[i][7],device=XY[i][3].device)[adeindices]]
                    if(isnotempty==False and adeindices.dim()!=0):
                        finalXY=temp
                        isnotempty=True
                    else:
                        if(adeindices.dim()==0 ):
                            lenade=1
                            if(isnotempty==False):
                                continue
                        else:
                            lenade=adeindices.size(0)
                        if(lenade==0):
                            continue
                        elif(lenade==1):
                            obvdata=temp[0].unsqueeze(1)
                            futdata=temp[1].unsqueeze(1)
                            neighbdata=temp[2].unsqueeze(1)
                            adedata=temp[3].unsqueeze(0)
                            fdedata=temp[4].unsqueeze(0)
                            clusterdata=temp[5].unsqueeze(0)
                            idtrajectory=temp[6].unsqueeze(0)
                            idtask=temp[7].unsqueeze(0)
                        else:
                            obvdata=temp[0]
                            futdata=temp[1]
                            neighbdata=temp[2]
                            adedata=temp[3]
                            fdedata=temp[4]
                            clusterdata=temp[5]
                            idtrajectory=temp[6]
                            idtask=temp[7]
                        obvdata=torch.cat((finalXY[0],obvdata),dim=1)
                        futdata=torch.cat((finalXY[1],futdata),dim=1)
                        neighbdata=torch.cat((finalXY[2],neighbdata),dim=1)
                        adedata=torch.cat((finalXY[3],adedata),dim=0)
                        fdedata=torch.cat((finalXY[4],fdedata),dim=0)
                        clusterdata=torch.cat((finalXY[5],clusterdata),dim=0)
                        idtrajectory=torch.cat((finalXY[6],idtrajectory),dim=0)
                        idtask=torch.cat((finalXY[7],idtask),dim=0)
                        sample_count = adedata.shape[0]
                        weightbcsr = torch.ones(sample_count, dtype=torch.float32, device=adedata.device) 
                        weightREL = torch.ones(sample_count, dtype=torch.float32, device=adedata.device) 
                        finalweight = torch.ones(sample_count, dtype=torch.float32, device=adedata.device) 
                        finalXY=[obvdata,futdata,neighbdata,adedata,fdedata,clusterdata,idtrajectory,idtask,weightbcsr,weightREL,finalweight]
                finalXY_cpu = [
                    t.detach().cpu().clone() if torch.is_tensor(t) else t
                    for t in finalXY
                ]
                ade_cpu = ade.detach().cpu() if torch.is_tensor(ade) else ade
                fde_cpu = fde.detach().cpu() if torch.is_tensor(fde) else fde
                XYbest = [finalXY_cpu, ade_cpu, fde_cpu]
                bestaddressXY=os.path.join(settings.ckpt, "bestXY.pt")
                torch.save(XYbest, bestaddressXY)
                XYbest.clear()      # optional but explicit
                del XYbest
                print("select validation data done!!") 
                print("run merge function!!")
                bestaddressloss=os.path.join(settings.ckpt, "bestloss.pt")
                torch.save(lossarray, bestaddressloss)
                lossarray.clear()
                del lossarray
                gc.collect()
                torch.cuda.empty_cache()
                # optional but explicit
    del train_data
    XY.clear()
    finalXY.clear()
    finalXY_cpu.clear()
    #item_cpu.clear()
    #neighbdata.clear()
    #futdata.clear()
    #obvdata.clear()
    train_dataset=[]
    train_data=[]
    test_data=[]
    test_dataset=[]
    del test_dataset
    del test_data
    del train_data
    del train_dataset
    del neighbdata
    del futdata
    del obvdata
    del item_cpu
    del finalXY_cpu
    del finalXY
    del XY
    gc.collect()
    torch.cuda.empty_cache()
    print("kill all data and variables to free memory!!")
    ram_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ram_mb = ram_kb / 1024
    if torch.cuda.is_available():
        gpu_alloc_mb = torch.cuda.memory_allocated(settings.device) / 1024**2
        gpu_reserved_mb = torch.cuda.memory_reserved(settings.device) / 1024**2
        print(f"Memory before lastbatch loop - RAM peak: {ram_mb:.2f} MiB; GPU allocated: {gpu_alloc_mb:.2f} MiB; GPU reserved: {gpu_reserved_mb:.2f} MiB; ")
    else:
        print(f"Memory before lastbatch loop - RAM peak: {ram_mb:.2f} MiB; GPU unavailable;")
    lossarray_rest=torch.load(os.path.join(settings.ckpt, "bestloss.pt"), weights_only=False)
    lossarray_len = len(lossarray_rest)
    #max_memtemp = torch.cuda.max_memory_allocated(0)
    for i in range(lossarray_len):
        max_memtemp = torch.cuda.max_memory_allocated(0)
        if(max_memtemp>max_mem):
            max_mem=max_memtemp
        # Consume entries from the end so processed items can be freed immediately.
        loss_item = lossarray_rest.pop()
        #ini digunakan untuk filter data training yang bagus dan yang enggak berdasarkan mean valuenya atau kalau saat ini msh menggunakan losslist-1
        #print('loss individu',lossarray[i][1])[i for i, val in enumerate(arr) if val < 5]
        #lossfinal=lossarray[i][1]<losslist[-1]
        #lossaray isinnya item,errindividual,klindividual,idtracks,ncluster,taskno
        sys.stdout.write("\r\033[K Processing lossarray...{}/{}".format(i+1, lossarray_len))
        sys.stdout.flush()
        #a=loss_item[1]
        lossfinal = torch.ones_like(loss_item[1], dtype=torch.bool)
        lossindices = torch.nonzero(lossfinal).squeeze()
        temp=[loss_item[0][0][:,lossindices,:], loss_item[0][1][:,lossindices,:], loss_item[0][2][:, lossindices,:,:],loss_item[1][lossindices],loss_item[2][lossindices],loss_item[3][lossindices], loss_item[4][lossindices],loss_item[5][lossindices]]
        del loss_item
        if(i==0):
            #temp.append(weights[i])
            finalloss=temp
        else:
            if(temp[3].dim()==0 or temp[3].shape[0] == 1):
                obv = torch.cat((finalloss[0], temp[0].unsqueeze(1)), dim=1)
                fut = torch.cat((finalloss[1], temp[1].unsqueeze(1)), dim=1)
                neighb = torch.cat((finalloss[2], temp[2].unsqueeze(1)), dim=1)
                yt = torch.cat((finalloss[3], temp[3].unsqueeze(0)), dim=0)
                ft = torch.cat((finalloss[4], temp[4].unsqueeze(0)), dim=0)
                idt = torch.cat((finalloss[5], temp[5].unsqueeze(0)), dim=0)
                ncluster=torch.cat((finalloss[6], temp[6].unsqueeze(0)), dim=0)
                idtask=torch.cat((finalloss[7], temp[7].unsqueeze(0)), dim=0)
                #weights=torch.cat((finalloss[8], weights[i].unsqueeze(0)), dim=0)
            else:
            #t1_cat = torch.cat([t1a, t1b], dim=1) 
                obv=torch.cat((finalloss[0],temp[0]),dim=1)
                fut=torch.cat((finalloss[1],temp[1]),dim=1)
                neighb=torch.cat((finalloss[2],temp[2]),dim=1)
                yt=torch.cat((finalloss[3],temp[3]),dim=0)
                ft=torch.cat((finalloss[4],temp[4]),dim=0)
                idt=torch.cat((finalloss[5],temp[5]),dim=0)
                ncluster=torch.cat((finalloss[6], temp[6]), dim=0)
                idtask=torch.cat((finalloss[7], temp[7]), dim=0)
                #weights=torch.cat((finalloss[8], weights[i]), dim=0)
            del temp
            loss_item=[]
            del loss_item
            torch.cuda.empty_cache()
            gc.collect()
            finalloss=[obv,fut,neighb,yt,ft,idt,ncluster,idtask]
            
    # finalloss_cpu = [
    #     t.detach().cpu().clone() if torch.is_tensor(t) else t
    #     for t in finalloss
    # ]
    # print("copy finalloss to cpu!!")
    # finalloss_cpu = [
    # t.detach().cpu() if torch.is_tensor(t) else t
    # for t in finalloss
    # ]
    finalloss_cpu=finalloss
    ram_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ram_mb = ram_kb / 1024
    if torch.cuda.is_available():
        gpu_alloc_mb = torch.cuda.memory_allocated(settings.device) / 1024**2
        gpu_reserved_mb = torch.cuda.memory_reserved(settings.device) / 1024**2
        print(f"Memory after lastbatch loop - RAM peak: {ram_mb:.2f} MiB; GPU allocated: {gpu_alloc_mb:.2f} MiB; GPU reserved: {gpu_reserved_mb:.2f} MiB; ")
    else:
        print(f"Memory after lastbatch loop - RAM peak: {ram_mb:.2f} MiB; GPU unavailable;")
    # pass 1: total N
    # total_n = 0
    # for loss_item in lossarray:
    #     total_n += loss_item[1].numel()  # or selected count after masking

    # # infer dims from first item
    # first = lossarray[-1]
    # ob_h, _, ob_d = first[0][0].shape
    # fu_h, _, fu_d = first[0][1].shape
    # nb_h, _, nb_n, nb_d = first[0][2].shape

    # # allocate once (CPU)
    # obv_all = torch.empty((ob_h, total_n, ob_d), dtype=first[0][0].dtype)
    # fut_all = torch.empty((fu_h, total_n, fu_d), dtype=first[0][1].dtype)
    # nei_all = torch.empty((nb_h, total_n, nb_n, nb_d), dtype=first[0][2].dtype)
    # yt_all  = torch.empty((total_n,), dtype=first[1].dtype)
    # ft_all  = torch.empty((total_n,), dtype=first[2].dtype)
    # idt_all = torch.empty((total_n,), dtype=first[3].dtype)
    # ncl_all = torch.empty((total_n, first[4].shape[-1]), dtype=first[4].dtype)  # adjust if scalar
    # tsk_all = torch.empty((total_n,), dtype=first[5].dtype)

    # # pass 2: fill slices
    # p = 0
    # while lossarray:
    #     loss_item = lossarray.pop()
    #     lossfinal = torch.ones_like(loss_item[1], dtype=torch.bool)
    #     idx = torch.nonzero(lossfinal).squeeze()
    #     if idx.dim() == 0:
    #         idx = idx.unsqueeze(0)
    #     n = idx.numel()

    #     obv_all[:, p:p+n, :] = loss_item[0][0][:, idx, :]
    #     fut_all[:, p:p+n, :] = loss_item[0][1][:, idx, :]
    #     nei_all[:, p:p+n, :, :] = loss_item[0][2][:, idx, :, :]
    #     yt_all[p:p+n]  = loss_item[1][idx]
    #     ft_all[p:p+n]  = loss_item[2][idx]
    #     idt_all[p:p+n] = loss_item[3][idx]
    #     ncl_all[p:p+n] = loss_item[4][idx]
    #     tsk_all[p:p+n] = loss_item[5][idx]
    #     p += n

    # finalloss = [obv_all, fut_all, nei_all, yt_all, ft_all, idt_all, ncl_all, tsk_all]
    lossbest = finalloss_cpu
    print("run offline RE-L!")
    beststate = SocialVAE(horizon=config.PRED_HORIZON, ob_radius=config.OB_RADIUS, hidden_dim=config.RNN_HIDDEN_DIM)
    beststate.to(settings.device)
    beststate.load_state_dict(state.get("model"))
    endtimemain=time.time()
    train_data=[]
    del train_data
    
    gc.collect()
    torch.cuda.empty_cache()
    #n_best=int(buffersize//batch_size)
    buffersize=10
    n_best=int(buffersize*0.1)
    offline_REL=offREL(settings=settings, config=config, model=beststate, lossmain=lossbest, buffer_size=buffersize, batch_size=batch_size, epoch=1,nbest=n_best)
    id_selected, selected_batch_indices, loss_diff_np=offline_REL.update_buffer_new_noduplicate()
    offline_REL.model = None
    del beststate
    del offline_REL
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    lossbest.clear()
    del lossbest
    gc.collect()
    torch.cuda.empty_cache()
    if(epoch==end_epoch):
        isnotempty=False
        finalcand=None
        endbcsrselect=time.time()
        all_idtracks=[]
        candidatesweights=candidates_weights
        flatted_weights=flat_candidates
        #idtrack_REL=lossbest[5][selected_batch_indices]
        
            #all_indices=selected_batch_indices[i]
        ram_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ram_mb = ram_kb / 1024
        if torch.cuda.is_available():
            gpu_alloc_mb = torch.cuda.memory_allocated(settings.device) / 1024**2
            gpu_reserved_mb = torch.cuda.memory_reserved(settings.device) / 1024**2
            print(f"Memory before lastbatch loop - RAM peak: {ram_mb:.2f} MiB; GPU allocated: {gpu_alloc_mb:.2f} MiB; GPU reserved: {gpu_reserved_mb:.2f} MiB; ")
        else:
            print(f"Memory before lastbatch loop - RAM peak: {ram_mb:.2f} MiB; GPU unavailable;")
        lastbatchrest=torch.load(addresslast, weights_only=False)
        lastbatch_len = len(lastbatchrest)
        
        for i in range(lastbatch_len):
            # Consume entries from the end so processed items can be freed immediately.
            last_item = lastbatchrest.pop()
            #ini lossarray terakhir yang diambil
            # Use candidates_indices[i] for this specific batch, not flat_candidates (which has ALL batches)
            #batch_indices = torch.as_tensor(candidates_indices[i], device=settings.device)
            #batch_indices=candidates_indices[i]
            sys.stdout.write("\r\033[K Processing lastbatch...{}/{}".format(i+1, lossarray_len))
            sys.stdout.flush()
            tempbatch=[last_item[0][0], last_item[0][1], last_item[0][2],last_item[1],last_item[2],last_item[3],last_item[4],last_item[5],last_item[6]]
            
            # Match loss_diff_np based on track IDs
            batch_track_ids = last_item[3]  # track IDs in this batch
            loss_diffs_batch = []
            for track_id in batch_track_ids:
                # Find matching index in id_selected
                matching_idx = np.where(np.array(id_selected) == track_id.item() if torch.is_tensor(track_id) else track_id)[0]
                if len(matching_idx) > 0:
                    loss_diffs_batch.append(loss_diff_np[matching_idx[0]])
                else:
                    loss_diffs_batch.append(0.0)  # default if no match found
            tempbatch.append(torch.as_tensor(loss_diffs_batch, device=settings.device, dtype=torch.float))
            
            if(isnotempty==False):
                #tempbatch.append(candidates_weights[i])
                finalcand=tempbatch
                isnotempty=True
            else:
                #t1_cat = torch.cat([t1a, t1b], dim=1)
                if(len(adeindices)==0 ):                       
                    continue
                else:
                    obv=torch.cat((finalcand[0],tempbatch[0]),dim=1)
                    fut=torch.cat((finalcand[1],tempbatch[1]),dim=1)
                    neighb=torch.cat((finalcand[2],tempbatch[2]),dim=1)
                    yt=torch.cat((finalcand[3],tempbatch[3]),dim=0)
                    ft=torch.cat((finalcand[4],tempbatch[4]),dim=0)
                    idtracks=torch.cat((finalcand[5],tempbatch[5]),dim=0)
                    ncluster=torch.cat((finalcand[6],tempbatch[6]),dim=0)
                    idtask=torch.cat((finalcand[7],tempbatch[7]),dim=0)
                    weightbcsr=torch.cat((finalcand[8], tempbatch[8]), dim=0)
                    weightslast=torch.cat((finalcand[9], tempbatch[9]), dim=0)
                    # Normalize components before combining to keep scales comparable
                   
                    finalcand=[obv,fut,neighb,yt,ft,idtracks,ncluster,idtask,weightbcsr,weightslast]                        
            last_item=[]
            tempbatch=[]
            del last_item
            del tempbatch
            gc.collect()
            torch.cuda.empty_cache()
        # Convert idtracks, ncluster, idtask to array lists after loop
        #ncluster consist of history cluster and future cluster
        # Compute finalweight once after aggregating all batches to use global max/min
        if finalcand:
            eps = 1e-8
            # Keep all weighting components on one device to avoid cuda/cpu mixing at combine time.
            #weightslast = torch.as_tensor(finalcand[9], dtype=torch.float32, device=settings.device)
            weightbcsr = torch.as_tensor(finalcand[8], dtype=torch.float32, device=settings.device)
            nc_comp = torch.as_tensor(finalcand[6][:,1], dtype=torch.float32, device=settings.device)
            # Higher loss is worse, so invert: (max - value) / range
            #wl_norm = (weightslast.max() - weightslast) / (weightslast.max() - weightslast.min() + eps)
            wb_norm = (weightbcsr - weightbcsr.min()) / (weightbcsr.max() - weightbcsr.min() + eps)
            nc_norm = (nc_comp - nc_comp.min()) / (nc_comp.max() - nc_comp.min() + eps)
            #wl_norm wb_norma dan nc_norm itu sudah dinormalisasi antara 0 dan 1, dimana 1 itu adalah yang paling penting untuk dipilih sebagai coreset, jadi bisa dijadikan salah satu komponen untuk menentukan weightnya
            finalweight = (wb_norm + nc_norm) / 2
            finalcand.append(finalweight)
            torch.cuda.empty_cache()
            # Select samples using multinomial sampling based on finalweight
            # Normalize finalweight to probability distribution
            #####################our proposed selection method########################
            finalweight_prob = finalweight / (finalweight.sum() + eps)
            num_samples = buffersize #min(len(finalweight), int(len(finalweight) * 0.8))  # Select 80% of candidates
            selected_indices = torch.multinomial(finalweight_prob, num_samples, replacement=False)
            
            # Filter finalcand based on selected indices
            selected_indices_np = selected_indices.cpu().numpy()
            finalcand = [
                finalcand[0][:, selected_indices_np, :],  # observations
                finalcand[1][:, selected_indices_np, :],  # future
                finalcand[2][:, selected_indices_np, :, :],  # neighbors
                finalcand[3][selected_indices_np],  # ade
                finalcand[4][selected_indices_np],  # fde
                finalcand[5][selected_indices_np],  # idtracks
                finalcand[6][selected_indices_np],  # ncluster
                finalcand[7][selected_indices_np],  # idtask
                finalcand[8][selected_indices_np],  # weightbcsr
                finalcand[9][selected_indices_np],  # weightslast
                finalcand[10][selected_indices_np],  # finalweight
            ]
        
        if finalcand:
            finalcand[5] = finalcand[5].cpu().numpy() if torch.is_tensor(finalcand[5]) else finalcand[5]
            finalcand[6] = finalcand[6].cpu().numpy() if torch.is_tensor(finalcand[6]) else finalcand[6]
            finalcand[7] = finalcand[7].cpu().numpy() if torch.is_tensor(finalcand[7]) else finalcand[7]
        candbest=finalcand.copy()

    if train:    
        #bestaddressRL=os.path.join(settings.ckpt, "bestRLB.pt")
        bestaddresscandidate=os.path.join(settings.ckpt, "bestCandidate.pt")
        bestfinalloss=os.path.join(settings.ckpt, "bestlossfinal.pt")
        
        torch.save(candbest, bestaddresscandidate)
        #torch.save(finalloss, bestfinalloss)
        #torch.save(offline_REL.buffer, bestaddressRL)
        #XYprint=[]
        # for rows in XYbest:
        #     XYrow=[]
        #     for row in rows:
        #         XYrow.append(row.cpu().numpy())
        #     XYprint.ap
        # np.savetxt(bestaddress, XYbest, delimiter=",", fmt="%d")
    if settings.fpc_finetune:# or losses is not None:
        # FPC finetune if it is specified or after training
        precision = 2
        trunc = lambda v: np.trunc(v*10**precision)/10**precision
        ade_, fde_, fpc_ = [], [], []
        for fpc in config.FPC_SEARCH_RANGE:
            ade, fde, XY = test(model, fpc)
            ade_.append(trunc(ade.item()))
            fde_.append(trunc(fde.item()))
            fpc_.append(fpc)
        i = np.argmin(np.add(ade_, fde_))
        ade, fde, fpc = ade_[i], fde_[i], fpc_[i]
        if settings.ckpt:
            ckpt_best = os.path.join(settings.ckpt, "ckpt-best")
            if os.path.exists(ckpt_best):
                state_dict = torch.load(ckpt_best, map_location=settings.device,weights_only=False)
                state_dict["ade_fpc"] = ade
                state_dict["fde_fpc"] = fde
                state_dict["fpc"] = fpc
                torch.save(state_dict, ckpt_best)
        print(" ADE: {:.2f}; FDE: {:.2f} ({})".format(
            ade, fde, "FPC: {}".format(fpc) if fpc > 1 else "w/o FPC", 
        ))
    endtimebcsr=time.time()
    print("Total time taken (s):",endtimebcsr-time_start)
    print("total time reducible Loss (s):",endtimebcsr-endtimemain)
    print("total time BCSR:",endtimebcsr-starttimebcsr)
    print("ckpt-best time taken (s):",endtimemain-time_best)
    
    print(f"Peak GPU memory allocated: {max_mem / 1024**2:.2f} MB")
