import os, sys, time
import importlib
import torch
import numpy as np
import random
#import ContinualLearning.PreviousData
from ContinualLearning.previousdata import previousdata
from torch.utils.tensorboard import SummaryWriter

from social_vae import SocialVAE
from data import Dataloader
from utils import *
from ContinualLearning.previousdata import previousdata
#from memory_profiler import memory_usage
import resource
import argparse
import glob
import time
#from reducibleloss import *
from bcsr_coreset import *
from bcsr_training import *
from offREL import *
parser = argparse.ArgumentParser()#1.21.2
parser.add_argument("--train", nargs='+', default=['data/continualdata_sandbox/task3/train'])
parser.add_argument("--test", nargs='+', default=['data/continualdata_sandbox/task3/test'])#/home/USER/Documents/projects/MIND_VAE/data/continualdata/data90d7/
#parser.add_argument("--train", nargs='+', default=['data/continualdata/data120b2/train'])
#parser.add_argument("--test", nargs='+', default=['data/continualdata/data120b2/test'])
#parser.add_argument("--train", nargs='+')
#parser.add_argument("--train", nargs='+', default=['data/fullreplay/train'])
#parser.add_argument("--test", nargs='+', default=['data/fullreplay/test'])
#parser.add_argument("--train", nargs='+', default=['data/crowdTraj World/dataset cluster world CL/duri_morning_cluster_world/train'])
#parser.add_argument("--test", nargs='+', default=['data/crowdTraj World/dataset cluster world CL/duri_morning_cluster_world/test'])
#
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--config", type=str, default='config/configsandbox.py')
#parser.add_argument("--config", type=str, default='config/mot.py')
#parser.add_argument("--ckpt", type=str, default='log_eth/logExRecplayoctber28th6/logtask2')
#parser.add_argument("--ckpt", type=str, default='log_eth/logExRecplayfixx/logdataMOT2122')
#parser.add_argument("--ckpt", type=str, default='log_eth/testlogExRecplay/logtask3')/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsandboxoffrel_20260513_193955/logtask2
#parser.add_argument("--ckpt", type=str, default='log_eth/logcontinualsandboxoffrel_20260513_193955/logtask3')
parser.add_argument("--ckpt", type=str, default='log_eth/testbcsr4')
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--no-fpc", action="store_true", default=True)
parser.add_argument("--fpc-finetune", action="store_true", default=False)
#parser.add_argument("--prev_data",type=str,default="log_eth/logcontinualsandboxoffrel_20260513_193955/logtask3")
#parser.add_argument("--prev_data",default='log_eth/testbcsr2')
parser.add_argument("--prev_data",type=str)
parser.add_argument("--datacapacity",type=str,default="15")
parser.add_argument("--method",type=str,default="filteredexperiencereplay")
parser.add_argument("--proxy_lr",type=float,default=1e-4)
parser.add_argument("--weight_lr",type=float,default=1e-10)
parser.add_argument("--selectionparam", type=float, default=25.0, help="percentage of training samples to keep during reduction")
parser.add_argument("--taskno",type=str,default="1")#should be more than 1
if __name__ == "__main__":
    start=time.time()
    countstop=0
    skip_to_last_epoch=False
    early_stop=50#early stop if there is no improvement in ADE for 50 consecutive epochs, this is to prevent overfitting and save time, but still execute the last epoch to save the best model and best XY data
    max_mem = torch.cuda.max_memory_allocated(0)
    time_start = time.time()
    print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_start)))
    #pd=previousdata()
    # def move_batch_to_device(batch, device):
    #     """Move all tensors in batch to specified device"""
    #     for i in range(len(batch)):
    #         if isinstance(batch[i], (list, tuple)):
    #             for j in range(len(batch[i])):
    #                 if torch.is_tensor(batch[i][j]):
    #                     batch[i][j] = batch[i][j].to(device)
    #         elif torch.is_tensor(batch[i]):
    #             batch[i] = batch[i].to(device)
    def loadallpreviousmemories(folder_path):
        alldata=[]
        #pathall=folder
        pathall=os.path.join(folder_path,"*")
        folder=folder_path
        test_files = glob.glob(os.path.join(folder, "bestXY.pt"))
        train_files = glob.glob(os.path.join(folder, "bestBCSR.pt"))
        if(len(test_files)>0):
            test=torch.load(test_files[0], weights_only=False)
            train=torch.load(train_files[0], weights_only=False)
            alldata.append([test,train])
        return alldata 
    torch.cuda.reset_peak_memory_stats()
    settings = parser.parse_args()
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
            device=settings.device, seed=settings.seed,kneighbour=10)
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
            batch_size=config.BATCH_SIZE, shuffle=False
        )#disini tidak spesifik ditentukan batch per epochnya makanya di evaluasi semuanya
        #sekarang tambahkan di data XY yang disimpan idtrack dan idtask, clusternya jg
        if settings.prev_data!=None:
            sample=int(settings.datacapacity)#iniiiii pentingggg!!!!
            allprevdata=loadallpreviousmemories(settings.prev_data)
            sample=sample//(len(allprevdata)/2)
            for prev_data in allprevdata:#looping per scene
                filterpreviousdata=previousdata(sample,settings.method)
                filterpreviousdata.getRelData(prev_data[1],prev_data[0])
                prevtestdata=filterpreviousdata.dataprevvalid
                test_dataset=mergeprevioustestdataREL(test_dataset,prevtestdata)#brati prevtestdata isinya per scene
                #coba langsung cek bagian ini lgs run di mainreducible dengan ditambahin bagian prev data
            #test_dataset=test_dataset.extend(prevtestdata)
            #harusnya ditambahkan ke test_data karena dah disampler, ternyta ga usah karena dipakai semua
        test_data = torch.utils.data.DataLoader(test_dataset, 
            collate_fn=test_dataset.collate_fn,
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
                print("test ui")
                #print(len(test_dataset))
                for x, y, neighbor,err,kl,nclusters,id_track,id_tasks in test_data:
                    batch += x.size(1)
                    #sys.stdout.write("\r\033[K Evaluating...{}/{} ({}) -- time: {}s".format(
                    #    batch, len(test_dataset), fpc_config, int(time.time()-tic)
                    #))
                    
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
                    xy_entry = [t.clone().cpu() if torch.is_tensor(t) else t for t in [x,y,neighbor,ade,fde,nclusters,id_track,id_tasks]]
                    XY.append(xy_entry)#salahnya disini harusnya cuma 1 saja ini diambil 9
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
            #sys.stdout.write("\r\033[K ADE: {:.4f}; FDE: {:.4f} ({}) -- time: {}s".format(
            #    ade, fde, fpc_config, 
            #    int(time.time()-tic))
            #)
            #print()
            return ade, fde, XY
    
    
    
    if settings.train:
        print('Training !!!!!!!!!!!!!!!!!!!1')
        print(settings.train)
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.train))]
        else:
            inclusive = None
       
        train_dataset = Dataloader(
            settings.train, **kwargs, inclusive_groups=inclusive,
            flip=True, rotate=True, scale=True,
            batch_size=config.BATCH_SIZE, shuffle=True, batches_per_epoch=config.EPOCH_BATCHES,idtask=int(settings.taskno)
        )
        batches = train_dataset.batches_per_epoch
        #if settings.prev_data!=None:
           # prev_data=
            #prev_data=previousdata()
            #prev_data = torch.load("/home/USER/Documents/projects/MIND_VAE/log_eth/testCL2050top10/longterm.pt")# check here
          #  train_dataset=train_dataset.extend(prev_data)
        train_data= torch.utils.data.DataLoader(train_dataset,
            collate_fn=train_dataset.collate_fn,
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
            state_dict = torch.load(ckpt_best, map_location=settings.device)#, weights_only=False)
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
        if os.path.exists(ckpt):
            print("Load from ckpt:", ckpt)
            #torch.serialization.add_safe_globals([_reconstruct])
            state_dict = torch.load(ckpt, map_location=settings.device,weights_only=False)
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
    batch_size=config.BATCH_SIZE//(2**2)
    #selectionparam=50
    candidates_indices=[]
    lastbatch=[]
    #rl=ReducibleLoss(train_data,settings,config,buffersize,batch_size,selectionparam)
    proxy_model = SocialVAE(horizon=config.PRED_HORIZON, ob_radius=config.OB_RADIUS, hidden_dim=config.RNN_HIDDEN_DIM)
    proxy_model.to(settings.device)
    bc = BCSR_Coreset(proxy_model, settings.proxy_lr , 10, out_dim=100,
            max_outer_it=10, max_inner_it=10,
            weight_lr=settings.weight_lr, logging_period=1000,device=settings.device)
    def reducethetrainingdata(train_data,train_dataset):
        # fungsi untuk mengurangi data training dengan cara mengevaluasi data training dengan model yang sudah diload,
        # data training akan diseleksi dengan multinomial berdasarkan nilai akurasi yang diturunkan dari error test.
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
                x, y, neighbor, err, kl, nclusters, id_track, id_tasks = data
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
        selected_indices_relative = torch.multinomial(probabilities, target_keep_count, replacement=False).cpu().numpy()
        
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
            reduceddata=train_dataset
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
            #sys.stdout.write(log_str.format(
            #    cur_batch=0, done="", remain="."*int(batches*progress),
            #    time=round(time.time()-tic), comment=""))
            #realtraindata=[x for x in train_data]
            #if settings.prev_data!=None:
            #    realtraindata=mergepreviousdatatraining(realtraindata,prevtrainingdata,sample)
            lossarray=[]
            #dapat semua data, nah sekarang tinggal ditambahkan ke real traindata cuyyy uhuyyy!!! tambahkan aja di dataloadnya biar sama kayak test, tapi nanti jadi tidak tercontrol
            nbatch=len(train_data)
            errindividuallist=[]
            klindividuallist=[]
            trainedindex=[]
            for batch, items in enumerate(train_data):
                idtracks = items[-2]#[m + val for m, val in zip(multiplier, items[-2])]
                ncluster=items[5]
                idtask=items[-1]
                item=items[:-2]
                #a=ncluster[0]
                #print(a)
                if settings.prev_data!=None:
                    #sample=sample//nbatch#ini membagi sample per batch
                    #masukan samplenya per batch disini
                    for prev_data in allprevdata:
                        sample=sample//len(allprevdata)
                        prevtrainingdata=filterpreviousdata.dataprevtrain
                        #prevtrainingdata=prev_data[0].dataprevtrain
                        #print("rise!!!")
                        item,trainedindex=mergepreviousdatatraining(items,prevtrainingdata,config.BATCH_SIZE,trainedindex)#harusnya diambil 20 persen jg dari batch nya, tapi sudah
                        item=items[:-2]
                        idtracks=items[-3]
                        ncluster=items[5]
                        idtask=items[-2]
                
               
                res = model(*item)#ini returnnya err, kl , err itu selisih y sama y' cuman bukan loss total, model loss nya berbeda dengan err
                
                free, total = torch.cuda.mem_get_info(settings.device)
                mem_used_MB = (total - free) / 1024 ** 2
                #print('after model',mem_used_MB)
                errindividual=res[0].mean(dim=(0,2))#ini loss per data, mean di ambil dari sample prediksi
                klindividual=res[1].mean(dim=(0,2))
                loss_err=errindividual+klindividual
                loss = model.loss(*res)#err,kl di masukkan ke loss, dimana loss itu untuk 16 data per batch, jadi individual lossnya harus diambil dari res
                item_cpu = tuple(t.clone().cpu() if torch.is_tensor(t) else t for t in item)
                lossarray.append([item_cpu,errindividual.clone().cpu(),klindividual.clone().cpu(),torch.as_tensor(idtracks, dtype=torch.int, device='cpu'),torch.as_tensor(ncluster, dtype=torch.double, device='cpu'),torch.as_tensor(idtask, dtype=torch.int, device='cpu')])
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
                    if(batch==0):
                        endtimebcsr=time.time()
                    print('start coreset selection for batch ',batch+1,'/',int(nbatch))
                    #ids,loss,weights=bc.coreset_select(model, item, task_id=0,  topk=int(config.BATCH_SIZE*(int(settings.datacapacity)/100)),out_loss=[],idt=idtracks)
                    ids,loss,weights=bc.coreset_select(model, item, task_id=0,  topk=int(buffersize/batch_size),out_loss=[],idt=idtracks)
                    rng_state = get_rng_state(settings.device)
                    candidates_indices.append(ids.cpu().numpy())
                    item_cpu = tuple(t.clone().cpu() if torch.is_tensor(t) else t for t in item)
                    lastbatch.append([item_cpu,errindividual.clone().cpu(),klindividual.clone().cpu(),torch.as_tensor(idtracks, dtype=torch.int, device='cpu'),torch.as_tensor(ncluster, dtype=torch.double, device='cpu'),torch.as_tensor(idtask, dtype=torch.int, device='cpu'),weights.clone().cpu()])
                    #print('coreset selection done for batch',batch )

                sys.stdout.write(log_str.format(
                   cur_batch=batch+1, done="="*int((batch+1)*progress),
                   remain="."*(int(batches*progress)-int((batch+1)*progress)),
                   time=round(time.time()-tic),
                   comment=" - ".join(["{}: {:.4f}".format(k, v) for k, v in losses.items()])
                ))
                ####calculate reducible loss score here!!!#####
                #reducibleloss=ReducibleLoss()
                losslist.append(losses['loss'])#ini loss per epochnya, dimana batch itu isinya berbeda2
                #rl.updatebuffer(items,loss_err,settings.taskno,idt,batch,epoch)
                ids=torch.randperm(len(idtracks))
                    #rl.selectcoresetdata(settings.taskno,model,bc,epoch)
                    #print('selected coreset data for task',settings.taskno)
            #print()
            if(epoch==end_epoch):
            # Flatten candidate index arrays into a single 1D array
                del bc.training_model_op.proxy_model
                import gc, torch
                gc.collect()
                torch.cuda.empty_cache()
                flat_candidates = np.concatenate([np.ravel(c) for c in candidates_indices]) if len(candidates_indices) > 0 else np.array([], dtype=int)
                flat_candidates=torch.asarray(flat_candidates   ,device=settings.device)
                #select_coreset(loader, task, model, flat_candidates, args, bc=our_bc)
            
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
                finalloss=[]
                finalADE=[]
                finalFDE=[]
                finalXY=[]
                isnotempty=False
                ######### select the validation data #########
                #move_batch_to_device(XY, "cpu")
                ade_cpu = ade.clone().cpu() if torch.is_tensor(ade) else ade
                for i in range(len(XY)): 
                    maskADE=XY[i][3]<ade_cpu
                    adeindices=torch.nonzero(maskADE).squeeze()
                    temp=[XY[i][0][:,adeindices,:],XY[i][1][:,adeindices,:],XY[i][2][:,adeindices,:,:],XY[i][3][adeindices],torch.as_tensor(XY[i][4],device=XY[i][3].device)[adeindices],torch.as_tensor(XY[i][5],device=XY[i][3].device)[adeindices],torch.as_tensor(XY[i][6],device=XY[i][3].device) [adeindices],torch.as_tensor(XY[i][7],device=XY[i][3].device)[adeindices]]
                    if(isnotempty==False and adeindices.dim()!=0):#jika tidak ada isinya dan adeindicesnya tidak kosong
                        finalXY=temp
                        isnotempty=True
                    else:
                        if(adeindices.dim()==0 ):#jika adeindicesnya kosong                            
                            lenade=0
                            # if(isnotempty==False):#jika belum ada isinya maka next
                            #     continue
                        else:#]jika adeindicesnya tidak kosong
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
                        finalXY=[obvdata,futdata,neighbdata,adedata,fdedata,clusterdata,idtrajectory,idtask]
                XYbest=[finalXY.copy(),ade,fde]
                #move_batch_to_device(lossarray, "cpu")
                for i in range(len(lossarray)):
                     #ini digunakan untuk filter data training yang bagus dan yang enggak berdasarkan mean valuenya atau kalau saat ini msh menggunakan losslist-1
                    #print('loss individu',lossarray[i][1])[i for i, val in enumerate(arr) if val < 5]
                    #lossfinal=lossarray[i][1]<losslist[-1]
                    #masalahnya adalah filteringnya kadang kosong jadi kalau mau pakai ini makesure dibuat seperti diatas
                    lossfinal = torch.ones_like(lossarray[i][1], dtype=torch.bool)
                    lossindices = torch.nonzero(lossfinal).squeeze()
                    temp=[lossarray[i][0][0][:,lossindices,:], lossarray[i][0][1][:,lossindices,:], lossarray[i][0][2][:, lossindices,:,:],lossarray[i][1][lossindices],lossarray[i][2][lossindices],lossarray[i][3][lossindices],lossarray[i][4][lossindices],lossarray[i][5][lossindices]]
                    if(i==0):
                        finalloss=temp
                    else:
                        
                        #t1_cat = torch.cat([t1a, t1b], dim=1) 
                        obv=torch.cat((finalloss[0],temp[0]),dim=1)
                        fut=torch.cat((finalloss[1],temp[1]),dim=1)
                        neighb=torch.cat((finalloss[2],temp[2]),dim=1)
                        yt=torch.cat((finalloss[3],temp[3]),dim=0)
                        ft=torch.cat((finalloss[4],temp[4]),dim=0)
                        idtracks=torch.cat((finalloss[5],temp[5]),dim=0)
                        ncluster=torch.cat((finalloss[6],temp[6]),dim=0)
                        idtask=torch.cat((finalloss[7],temp[7]),dim=0)
                        finalloss=[obv,fut,neighb,yt,ft,idtracks,ncluster,idtask]                        
                lossbest=finalloss.copy()
                #RLB=rl.buffer.copy()#coba cek bagian timingnya apakah sudah sesuai atau belum, perbaiki loss final juga untuk ngecek apakah mau pakai buffersize atau enggak
                
                #a=0
    print("BCSR Coreset Selection Phase Ended")
   
    endtimemain=time.time()
    print("total time epoch training (s):",endtimebcsr-start)
    print("total time main training BCSR (s) Only:",endtimemain-endtimebcsr)
    #nbest=int(buffersize*0.05)
    # print("nbest:",nbest)
    # offline_REL=offREL(settings=settings,model=model,data_train=train_dataset, config=config, lossmain=lossbest, buffer_size=buffersize, batch_size=batch_size, epoch=100,nbest=nbest)
    # offline_REL.update_buffer_new()
    # endtimemainREL=time.time()
    
    
    
    #move_batch_to_device(lastbatch, 'cpu')
    
   # print("total time main training REL (s) Only:",endtimemainREL-endtimemain)
    if(epoch==end_epoch):
        isnotempty=False
        endbcsrselect=time.time()
        for i in range(len(lastbatch)):
            #ini lossarray terakhir yang diambil
            # Use candidates_indices[i] for this specific batch, not flat_candidates (which has ALL batches)
            #batch_indices = torch.as_tensor(candidates_indices[i], device=settings.device)
            batch_indices=candidates_indices[i]
            tempbatch=[lastbatch[i][0][0][:,batch_indices,:], lastbatch[i][0][1][:,batch_indices,:], lastbatch[i][0][2][:, batch_indices,:,:],lastbatch[i][1][batch_indices],lastbatch[i][2][batch_indices],lastbatch[i][3][batch_indices],lastbatch[i][4][batch_indices],lastbatch[i][5][batch_indices],lastbatch[i][6][batch_indices]]
            if(isnotempty==False and len(batch_indices)!=0):
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
                    weights=torch.cat((finalcand[8],tempbatch[8]),dim=0)
                    finalcand=[obv,fut,neighb,yt,ft,idtracks,ncluster,idtask,weights]                        
        # Convert idtracks, ncluster, idtask to array lists after loop
        if finalcand:
            finalcand[5] = finalcand[5].detach().cpu().numpy() if torch.is_tensor(finalcand[5]) else finalcand[5]
            finalcand[6] = finalcand[6].detach().cpu().numpy() if torch.is_tensor(finalcand[6]) else finalcand[6]
            finalcand[7] = finalcand[7].detach().cpu().numpy() if torch.is_tensor(finalcand[7]) else finalcand[7]
            finalcand[8] = finalcand[8].detach().cpu().numpy() if torch.is_tensor(finalcand[8]) else finalcand[8]
        candbest=finalcand.copy()  
    if train_data is not None:
        bestaddressXY=os.path.join(settings.ckpt, "bestXY.pt") 
        #bestaddressloss=os.path.join(settings.ckpt, "bestloss.pt")
        bestaddressbcsr=os.path.join(settings.ckpt, "bestBCSR.pt")
        bestaddressloss=os.path.join(settings.ckpt, "bestloss.pt")
        #torch.cpu()
        torch.save(XYbest, bestaddressXY)
        torch.save(candbest, bestaddressbcsr)
        torch.save(lossbest, bestaddressloss)
        #torch.save(offline_REL.buffer, os.path.join(settings.ckpt, "bestREL.pt"))
        #torch.save(RLB, bestaddressRL)
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
                state_dict = torch.load(ckpt_best, map_location=settings.device)#_only=False)
                state_dict["ade_fpc"] = ade
                state_dict["fde_fpc"] = fde
                state_dict["fpc"] = fpc
                torch.save(state_dict, ckpt_best)
        print(" ADE: {:.2f}; FDE: {:.2f} ({})".format(
            ade, fde, "FPC: {}".format(fpc) if fpc > 1 else "w/o FPC", 
        ))

    max_mem = torch.cuda.max_memory_allocated(0)
    endtotal=time.time()
    print("Total time taken (s):",endtotal-start)
    print("total time bcsr (s):",endtotal-endtimebcsr)
    print("total time bcsr selection (s):",endtotal-endbcsrselect)
    print(f"Peak GPU memory allocated: {max_mem / 1024**2:.2f} MB")
    ##berikutnya kerjakan multi tasknya yaa!!!!!