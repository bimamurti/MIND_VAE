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
from utils import ADE_FDE, FPC, seed, get_rng_state, set_rng_state
from ContinualLearning.previousdata import previousdata
#from memory_profiler import memory_usage
import resource
import argparse
import glob
import time

parser = argparse.ArgumentParser()#1.21.2
parser.add_argument("--train", nargs='+', default=['data/continualdata_sandbox/task3/train'])
parser.add_argument("--test", nargs='+', default=['data/continualdata_sandbox/task3/eval'])#/home/USER/Documents/projects/MIND_VAE/data/continualdata/data90d7/
#parser.add_argument("--train", nargs='+', default=['data/continualdata/dataMOT212/train'])
#parser.add_argument("--test", nargs='+', default=['data/continualdata/dataMOT212/test'])
#parser.add_argument("--train", nargs='+')
#parser.add_argument("--train", nargs='+', default=['data/fullreplay/train'])
#parser.add_argument("--test", nargs='+', default=['data/fullreplay/test'])
parser.add_argument("--frameskip", type=int, default=1)
#parser.add_argument("--config", type=str, default='config/configsandbox.py')
parser.add_argument("--config", type=str, default='config/eth.py')
#parser.add_argument("--ckpt", type=str, default='log_eth/testCL2050top10_no')
parser.add_argument("--ckpt", type=str, default='log_eth/coba1')
#parser.add_argument("--ckpt", type=str, default='log_eth/testHT')
#parser.add_argument("--ckpt", type=str, default='log_eth/testprevious')
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--no-fpc", action="store_true", default=True)
parser.add_argument("--fpc-finetune", action="store_true", default=False)
#parser.add_argument("--prev_data",type=str,default="/home/USER/Documents/projects/MIND_VAE/log_eth/logExReplay1/previous")
parser.add_argument("--prev_data",type=str)
parser.add_argument("--datacapacity",type=str,default="100")
parser.add_argument("--method",type=str,default="experiencereplay")
if __name__ == "__main__":
    #pd=previousdata()
    settings = parser.parse_args()
    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    XYbest=[]
    lossbest=[]
    if settings.device is None:
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
    settings.device = torch.device(settings.device)
    print("ok1")
    seed(settings.seed)
    init_rng_state = get_rng_state(settings.device)
    rng_state = init_rng_state

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
    if settings.prev_data!=None:
        #inisialisasi class previous data
        prev_data=previousdata()
    print("ok2")
    def mergeprevioustestdata(test_dataset,prevtestdata,sample):
        #merge prev dataset to test dataset based on distribution on ADE, for nsample
        yt= [y[4].mean() for y in prevtestdata]
        if(settings.method!="experiencereplay"):
            print('ex replay!!!')
            sample=int((sample/100)*len(prevtestdata))
            print('sample=',sample)
            random_samples = random.choices(prevtestdata,weights=yt,k=sample)#salahnya disini ketmu, jadi harusnya yang dipilih itu dari anggota prevtestdata
        else:
            print('filtered replay!!!')
            random_samples=prevtestdata#take all choosen previous data
        temporary=[]
        for idx,rs in enumerate(random_samples):
            temp=rs[:3]
            temp=np.array([t.cpu().numpy() for t in temp])
            temptransposex=np.transpose(temp[0], (1, 0, 2))
            temptransposey=np.transpose(temp[1], (1, 0, 2))
            temptransposen=np.transpose(temp[2], (1, 0, 2, 3))
            if(idx==0):
                tempx=temptransposex
                tempy=temptransposey
                tempn=temptransposen
            else:
                tempx=np.concatenate((tempx,temptransposex), axis=0)
                tempy=np.concatenate((tempy,temptransposey), axis=0)
                tempn=np.concatenate((tempn,temptransposen), axis=0)
        #temporary=[tempx,tempy,tempn]
        #test_dataset.data[0]=np.concatenate((test_dataset.data[0],tempx),axis=0)
        #test_dataset.data[1]=np.concatenate((test_dataset.data[1],tempy),axis=0)
        #test_dataset.data[2]=np.concatenate((test_dataset.data[2],tempn),axis=0)
        for i in range (tempx.shape[0]):
            perdata=np.array([np.array([tempx[i],tempy[i],tempn[i]])], dtype=object)
            test_dataset.data=np.concatenate((  test_dataset.data, perdata), axis=0)
            #test_dataset.data[1].append(tempy[i])
            #test_dataset.data[2].append(tempn[i])
        return test_dataset
    
    def loadallpreviousmemorie(folder_path):
        pt_files = glob.glob(os.path.join(folder_path, "*.pt"))
        print(pt_files)
        alldata=[]
        for file in pt_files:
            temp=previousdata()
            #torch.serialization.add_safe_globals([previousdata])
            
            temp=torch.load(file,weights_only=False)
            alldata.append(temp)
        return alldata
    if settings.test:
        print('Testing!!!!!!!!!!!!!!!1')
        print(settings.test)
 
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.test))]
        else:
            inclusive = None
        test_dataset= Dataloader(
            settings.test, **kwargs, inclusive_groups=inclusive,
            batch_size=config.BATCH_SIZE, shuffle=False
        )#disini tidak spesifik ditentukan batch per epochnya makanya di evaluasi semuanya
        if settings.prev_data!=None:
            sample=int(settings.datacapacity)#iniiiii pentingggg!!!!
            allprevdata=loadallpreviousmemorie(settings.prev_data)
            sample=sample//len(allprevdata)
            #filter new similar data            
            #prev_data = torch.load("/home/USER/Documents/projects/MIND_VAE/log_eth/testCL2050top10/longterm.pt")
            
            for prev_data in allprevdata:
                prevtestdata=prev_data[0].dataprevvalid
                #start = time.time()
                #test_dataset.filtersimilarity(prevtestdata)#previous data should store a similarity score
                #end=time.time()
                #total=end-start
                #print('time exect',total)
                test_dataset=mergeprevioustestdata(test_dataset,prevtestdata,sample)
            
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

                #print(len(test_dataset))
                for x, y, neighbor in test_data:
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
                    XY.append([x,y,neighbor,ade,fde])#salahnya disini harusnya cuma 1 saja ini diambil 9
            ADE = torch.cat(ADE)
            FDE = torch.cat(FDE)
            #XY=torch.cat(XY)

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
            print()
            return ade, fde, XY
    def mergepreviousdatatraining(traindata,prevtrainingdata,sample):
        #include previous data training to train data as much as sample
        yt= [y[3] for y in prevtrainingdata]
        if(settings.method!="experiencereplay"):
            sample=int((sample/100)*len(prevtrainingdata))
            random_samples = random.choices(prevtrainingdata,weights=yt,k=sample)
        else:
            random_samples=prevtrainingdata
        temporary=[]#random_samples[:,:2]
        traindata0=traindata[0]
        traindata1=traindata[1]
        traindata2=traindata[2]
        for rs in random_samples:
            temp=rs[:3]
            #temp=np.array([t.cpu().numpy() for t in temp])
            #temporary.append(temp)
            traindata0=torch.cat((traindata0,temp[0].unsqueeze(1)),dim=1)
            traindata1=torch.cat((traindata1,temp[1].unsqueeze(1)),dim=1)
            traindata2=torch.cat((traindata2,temp[2].permute(1, 0, 2).unsqueeze(1)),dim=1)

        #traindata.append(temporary)
        return [traindata0,traindata1,traindata2,yt]
    
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
            batch_size=config.BATCH_SIZE, shuffle=True, batches_per_epoch=config.EPOCH_BATCHES
        )
        if settings.prev_data!=None:
            sample=int(settings.datacapacity)
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
        if os.path.exists(ckpt):
            print("Load from ckpt:", ckpt)
            #torch.serialization.add_safe_globals([_reconstruct])
            state_dict = torch.load(ckpt, map_location=settings.device, weights_only=False)
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
    print("get 1")
    if train_data is not None:
        print("get 2")
        log_str = "\r\033[K {cur_batch:>"+str(len(str(batches)))+"}/"+str(batches)+" [{done}{remain}] -- time: {time}s - {comment}"    
        progress = 20/batches if batches > 20 else 1
        optimizer.zero_grad()
    print("get 3")
    
    for epoch in range(start_epoch+1, end_epoch+1):
        print("get 4")
        print("get start:",start_epoch+1)
        print("get end:",end_epoch+1)
        ###############################################################################
        #####                                                                    ######
        ##### train                                                              ######
        #####                                                                    ######
        ###############################################################################
        losses = None
        losslist=[]
        print("epoch",epoch)
        print("config epoch",config.EPOCHS)
        if train_data is not None and epoch <= config.EPOCHS:
            print("get 5")
            print("Epoch {}/{}".format(epoch, config.EPOCHS))
            tic = time.time()
            set_rng_state(rng_state, settings.device)
            losses = {}
            model.train()
            sys.stdout.write(log_str.format(
                cur_batch=0, done="", remain="."*int(batches*progress),
                time=round(time.time()-tic), comment=""))
            realtraindata=[x for x in train_data]
            #if settings.prev_data!=None:
            #    realtraindata=mergepreviousdatatraining(realtraindata,prevtrainingdata,sample)
            lossarray=[]
            #dapat semua data, nah sekarang tinggal ditambahkan ke real traindata cuyyy uhuyyy!!!
            for batch, item in enumerate(train_data):
                if settings.prev_data!=None:
                    for prev_data in allprevdata:
                        sample=sample//len(allprevdata)
                        prevtrainingdata=prev_data[0].dataprevtrain
                        item,lo_val=mergepreviousdatatraining(item,prevtrainingdata,sample)
                res = model(*item)
                lossarray.append([item,res])
                loss = model.loss(*res)
                loss["loss"].backward()
                optimizer.step()
                optimizer.zero_grad()
                for k, v in loss.items():
                    #print(loss)
                    if k not in losses: 
                        losses[k] = v.item()
                    else:
                        losses[k] = (losses[k]*batch+v.item())/(batch+1)
                sys.stdout.write(log_str.format(
                    cur_batch=batch+1, done="="*int((batch+1)*progress),
                    remain="."*(int(batches*progress)-int((batch+1)*progress)),
                    time=round(time.time()-tic),
                    comment=" - ".join(["{}: {:.4f}".format(k, v) for k, v in losses.items()])
                ))
                losslist.append(losses['loss'])
            rng_state = get_rng_state(settings.device)
            
            print()

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
            if ade < ade_best:
                ade_best = ade
                fde_best = fde
                state = dict(
                    model=state["model"],
                    ade=ade, fde=fde, epoch=epoch
                )
                torch.save(state, ckpt_best)
                XYbest=[XY.copy(),ade_best,fde_best]
                losslist=np.array(losslist)
                lossbest=[lossarray.copy(),np.mean(losslist, dtype=np.float64)]
                #a=0

    if train_data is not None:
        bestaddressXY=os.path.join(settings.ckpt, "bestXY.pt") 
        bestaddressloss=os.path.join(settings.ckpt, "bestloss.pt")

        torch.save(XYbest, bestaddressXY)
        torch.save(lossbest, bestaddressloss)
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

