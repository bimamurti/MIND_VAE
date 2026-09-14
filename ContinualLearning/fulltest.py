import torch
import os
import subprocess
import time
import torch
import os, sys, time
import numpy as np
sys.path.append('../MIND_VAE/')
from utils import ADE_FDE, FPC, get_rng_state, set_rng_state,seed
from data import Dataloader
from social_vae import SocialVAE
import importlib
import shutil
import tracemalloc
import pickle
import gc
from ContinualLearning.previousdata import previousdata


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

#dibuat arrangement untuk pengujiannya mana yang dieksekusi terlebih dahulu dengan declare listnya secara manual untuk menentukan mana yang diekesuksi terlebih dahulu
#untuk foldernya outputnya dibuat sebanyak 3, yang pertama berisi task1 yang kedua berisi training 1 dan 2, yang ketiga berisi training 1,2,3
#berikutnya untuk proses ujinya bisa dilakukan evaluasi langsung dengan: menguji dari sekarang kebelakang trus buat file hasil ujinya, atau uji terpisah uji perfoldernya untuk file sebelumnya
#taskarrangement=["data90d7","data90g3","data120b2","dataMOT212"]
#taskarrangement=["task2","task3","task4"]
#taskarrangement=["dataMOT212","data120b2","data90g3","data90d7"]
#capacity buffernya harusnya di kali dari total data trraining data sebelumnya
#taskarrangement=["duri_morning_cluster_world","duri_evening_cluster_world","vredeburg_cluster_world","duri_platform1_cluster_world","paisley_cluster_world"]
#taskarrangement=["duri_morning_world","duri_evening_world","vredeburg_world","duri_platform1_world","paisley_world"]
taskarrangement=["duri_platform1_world","vredeburg_world","duri_morning_world","paisley_world","duri_evening_world"]
#taskarrangement=["duri_platform1_cluster_world","vredeburg_cluster_world","duri_morning_cluster_world","paisley_cluster_world","duri_evening_cluster_world"]
# taskarrangement=["paisley_cluster_world","duri_platform1_cluster_world","vredeburg_cluster_world","duri_evening_cluster_world","duri_morning_cluster_world"]
#taskarrangement=["paisley_cluster_world","duri_platform1_cluster_world","vredeburg_cluster_world","duri_evening_cluster_world","duri_morning_cluster_world"]
#taskarrangement=["duri_evening_cluster_world","vredeburg_cluster_world","duri_platform1_cluster_world"]
# taskarrangement=[
#     "data120b2",
#     "data90d7",
#     "data90g3",
#     "dataMOT212",
#     "duri_evening_cluster_pixel",
#     "duri_morning_cluster_pixel",
#     "duri_platform1_cluster_pixel",
#     "paisley_cluster_pixel",
#     "vredeburg_cluster_pixel",
# ]
# taskarrangement=[
#     "crossing_90_g_3",
#     "crossing_90_d_7",
#     "MOT212",
#     "crossing_120_b_02",
# ]
# taskarrangement=[
#     "data90g3",
#     "data90d7",
#     "dataMOT212",
#     "data120b2",
# ]
# taskarrangement=[
#     "data120b2",
#     "data90d7",
#     "data90g3",
#     "dataMOT212",
# ]
# taskarrangement=[
#     "crossing_120_b_02",
#     "crossing_90_d_7",
#     "crossing_90_g_3",
#     "MOT212",
# ]

homefolder='/home/USER/Documents/projects/MIND_VAE/'
#datafolder=os.path.join(homefolder,'data/continualdata_sandbox')
#datafolder=os.path.join(homefolder,'data/crowdTraj World/dataset cluster world CL/')
datafolder=os.path.join(homefolder,'data/crowdTraj World/dataset GT world CL/')
#datafolder=os.path.join(homefolder,'data/Long_seq')
#datafolder=os.path.join(homefolder,'data/julichmotfull')
#datafolder=os.path.join(homefolder,'data/continualdata')
#ckptpath=os.path.join(homefolder,'log_eth/logcontinualofflineRELjan5th2')
ckptpath=os.path.join(homefolder, f"log_eth/logcontinualsEXuniform_{time.strftime('%Y%m%d_%H%M%S')}")
#configpath=os.path.join(homefolder,'config/eth.py')
configpath=os.path.join(homefolder,'config/mot.py')
#configpath=os.path.join(homefolder,'config/configsandbox.py')
configpathpaisley=os.path.join(homefolder,'config/mot_paisley.py')
#EVAL_CKPTPATH="/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinuals_longSeq_EXuniform_20260609_184408"
#EVAL_CKPTPATH="/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinuals_longSeq_EXuniform_20260609_184408/"
#EVAL_CKPTPATH="/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsproposednobcsr_Crowdtraj120260802_022351" 

##120b2Cl##
# EVAL_CKPTPATH=[""/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsproposednobcsr_20260730_181934"
# ,"/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsproposednoCL_20260729_163903"
# ,"/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsproposednorel_20260731_060952"
# ,"/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinuals_proposed_long_rev1_20260622_213347"
# ,"x/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinuals_proposed_long_rev2_202606"
# ,"/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinuals_REL_desc_longseq20260622_191840"
# ,"/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinuals_BCSR_rev_20260628_000137"
# ,"/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinuals_longSeq_EXuniform_20260609_184408"
# ,"/home/USER/Documents/projects/MIND_VAE/log_eth/log_noreplay_longseq20260610_173848"]
##90d3CL##
# EVAL_CKPTPATH=["/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsproposednobcsr_Longseq220260801_055544",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsproposednoCL_Longseq220260731_160852",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsproposednorel_Longseq220260801_185705",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinuals_proposed_longseq_20260614_173557",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinuals_REL_longseq_20260615_111211",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinuals_BCSR_longseq_20260615_213348",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinuals_exreplay_longseq_20260616_132301",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinuals_noreply_longseq_20260616_072346"]
##durimorningCL##
# EVAL_CKPTPATH=["/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsproposednobcsr_Crowdtraj120260802_022351",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsproposednoCL_Crowdtraj120260731_160221",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsproposednorel_Crowdtraj120260802_104622",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinuals_proposedrev_first20260628_173036",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinuals_REL_rev_firstorder20260629_061710",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinuals_BCSR_rev_firstorder20260628_133632",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsEXuniform_20260605_152400_done",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualnoreplayCL_20260511_140628_done"
#                ]
##duriplatform1CL##
# EVAL_CKPTPATH=["/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsproposednoCL_crowdtraj220260731_163934",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsproposednorel_crowdtraj220260802_090737",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsproposednobcsr_crowdtraj220260801_121424",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualProposed_dpvdmpde_20260518_102921_done",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logBCSR_dpvdmpde_20260518_151917",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualsEXuniform_20260606_012519_dpvdmpde",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logREL_dpvdmpde_20260517_181012",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/lognoCL_dpvdmpde_20260519_071108"
#                 ]
#FULL CROWDTRAJ1
# EVAL_CKPTPATH=["/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualRELFULL_20260707_120032",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualBCSRFULL_20260706_141612",
# ]
# EVAL_CKPTPATH=["/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualRELFULL_order220260708_202733",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualBCSRFULL_order220260709_103854",
# ]
#FUll longseq1
# EVAL_CKPTPATH=["/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualRELFULL_longseq120260709_143809",
#                "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualBCSRFULL_longseq120260709_172850",
#                #"/home/USER/Documents/projects/MIND_VAE/log_eth/logindividual_long_20260608_144750"
# ]
# EVAL_CKPTPATH=["/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualBCSRFULL_longseq220260710_095156",
#                 "/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualRELFULL_longseq220260710_080044",
#                   #"/home/USER/Documents/projects/MIND_VAE/log_eth/logindividual_long_20260608_144750"
#     ]
#individual longseq
#EVAL_CKPTPATH=["/home/USER/Documents/projects/MIND_VAE/log_eth/logindividual_long_20260608_144750"]
#full indidvual
#EVAL_CKPTPATH=["log_eth/logcontinuals_indvidual_julichfull_20260617_102843"]
EVAL_CKPTPATH=["/home/USER/Documents/projects/MIND_VAE/log_eth/logIndependent_noCluster_20260523_185820"]
def test(model, test_dataset, device, fpc=1):
    PRED_SAMPLES=100
    WORLD_SCALE=1
    test_data = torch.utils.data.DataLoader(test_dataset, 
            collate_fn=test_dataset.collate_fn,
            batch_sampler=test_dataset.batch_sampler
        )
    init_rng_state = get_rng_state(device)
    sys.stdout.write("\r\033[K Evaluating...{}/{}".format(
        0, len(test_dataset)
    ))
    tic = time.time()
    model.eval()
    ADE, FDE, XY = [], [], []
    set_rng_state(init_rng_state, device)
    batch = 0
    fpc = int(fpc) if fpc else 1
    fpc_config = "FPC: {}".format(fpc) if fpc > 1 else "w/o FPC"
    with torch.no_grad():
         for data in test_data:
            x, y, neighbor,err,kl,nclusters,id_track,id_tasks=data#kalau selain main tambahkan scores
            batch += x.size(1)
            sys.stdout.write("\r\033[K Evaluating...{}/{} ({}) -- time: {}s".format(
                batch, len(test_dataset), fpc_config, int(time.time()-tic)
            ))
            
            if PRED_SAMPLES > 0 and fpc > 1:
                # disable fpc testing during training
                y_ = []
                for _ in range(fpc):
                    y_.append(model(x, neighbor, n_predictions=PRED_SAMPLES))
                y_ = torch.cat(y_, 0)
                cand = []
                for i in range(y_.size(-2)):
                    cand.append(FPC(y_[..., i, :].cpu().numpy(), n_samples=PRED_SAMPLES))
                # n_samples x PRED_HORIZON x N x 2
                y_ = torch.stack([y_[_,:,i] for i, _ in enumerate(cand)], 2)
            else:
                # n_samples x PRED_HORIZON x N x 2
                y_ = model(x, neighbor, n_predictions=PRED_SAMPLES)
            ade, fde = ADE_FDE(y_, y)
            if PRED_SAMPLES > 0:
                ade = torch.min(ade, dim=0)[0]
                fde = torch.min(fde, dim=0)[0]
            ADE.append(ade)
            FDE.append(fde)
            XY.append([x,y,neighbor,ade,fde])
    ADE = torch.cat(ADE)
    FDE = torch.cat(FDE)
    #XY=torch.cat(XY)

    if torch.is_tensor(WORLD_SCALE) or WORLD_SCALE != 1:
        if not torch.is_tensor(WORLD_SCALE):
            WORLD_SCALE = torch.as_tensor(WORLD_SCALE, device=ADE.device, dtype=ADE.dtype)
        ADE *= WORLD_SCALE
        FDE *= WORLD_SCALE
    ade = ADE.mean()
    fde = FDE.mean()
    adeq05= torch.quantile(ADE, 0.05)
    adeq95 = torch.quantile(ADE, 0.95)
    fdeq05= torch.quantile(FDE, 0.05)
    fdeq95 = torch.quantile(FDE, 0.95)

    sys.stdout.write("\r\033[K ADE: {:.4f}; FDE: {:.4f} ({}) ; ADE Quantiles: [{:.4f}, {:.4f}]; FDE Quantiles: [{:.4f}, {:.4f}] -- time: {}s".format(
        ade, fde, fpc_config, 
        adeq05, adeq95, fdeq05, fdeq95,
        int(time.time()-tic))
    )
    print()
    return ade, fde, adeq05, adeq95, fdeq05, fdeq95

def evaluation():
    #ckptpath="/home/USER/Documents/projects/MIND_VAE/log_eth/logcontinualproposed_20260503_185330/"
    #ckptpath="/home/USER/Documents/projects/MIND_VAE/log_eth/logIndependent_noCluster_20260523_185820/"
    ckptpath=EVAL_CKPTPATH
    spec = importlib.util.spec_from_file_location("config", configpath)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    device="cuda"
    evalfinal={}
    evaldata_cache_paths=[]
    cache_dir=os.path.join(EVAL_CKPTPATH, "eval_dataset_cache")
    os.makedirs(cache_dir, exist_ok=True)
    print(len(taskarrangement))
    #for root, dirs, files in os.walk(datafolder):
    for dir_name in taskarrangement:
        #dir_name=taskarrangement[idtask]
        folder_path = os.path.join(datafolder, dir_name)
        print("Evaluating folder:", folder_path)
        evaluationpath=os.path.join(folder_path,'eval')
        kwargs = dict(
            batch_first=False, frameskip=1,
            ob_horizon=config.OB_HORIZON, pred_horizon=config.PRED_HORIZON,
            device=device, seed=2)
        evallist=[evaluationpath]
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(evallist))]
        else:
            inclusive = None
        test_dataset = Dataloader(
                    evallist, **kwargs, inclusive_groups=inclusive,
                    batch_size=200, shuffle=False
                )
        cache_path=os.path.join(cache_dir, f"test_dataset_{dir_name}.pkl")
        with open(cache_path, "wb") as cache_file:
            pickle.dump(test_dataset, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
        evaldata_cache_paths.append(cache_path)
        del test_dataset
        gc.collect()
    AVGerror=0
    countdata=0
    for i in range(0,len(taskarrangement)):
        print('model for:',taskarrangement[i])
        evalpertask={}
        for idtask in range (i,-1,-1):
            with open(evaldata_cache_paths[idtask], "rb") as cache_file:
                test_dataset=pickle.load(cache_file)
            model = SocialVAE(horizon=config.PRED_HORIZON, ob_radius=config.OB_RADIUS, hidden_dim=config.RNN_HIDDEN_DIM)
            model.to(device)
            logfolder= "log"+taskarrangement[i]
            ckptpertask=os.path.join(ckptpath,logfolder)
            ckpt_best = os.path.join(ckptpertask, "ckpt-best")
            state_dict = torch.load(ckpt_best, map_location=device)
            model.load_state_dict(state_dict["model"])
            ADE,FDE,ADEQ05,ADEQ95,FDEQ05,FDEQ95=test(model,test_dataset,device)
            AVGerror+=ADE.cpu().item()
            countdata += 1
            del test_dataset
            gc.collect()
            dataname="evdata_"+taskarrangement[idtask]
            evalpertask[dataname]={'ADE':ADE.cpu().item(),'FDE':FDE.cpu().item(),'ADEQ05':ADEQ05.cpu().item(),'ADEQ95':ADEQ95.cpu().item(),'FDEQ05':FDEQ05.cpu().item(),'FDEQ95':FDEQ95.cpu().item()}
            print('evaluation task-',taskarrangement[idtask])
            print("ADE",ADE)
            print("FDE",FDE)
            print("ADEQ05",ADEQ05)
            print("ADEQ95",ADEQ95)
            print("FDEQ05",FDEQ05)
            print("FDEQ95",FDEQ95)
          
        evalfinal[taskarrangement[i]]={'evaluation':evalpertask.copy()}
        #evalfinal['evaluation']=evalpertask
    AVGerror = AVGerror / countdata if countdata else None
    
    print('Average ADE:',AVGerror)  
    print(evalfinal)
    print('ADE summary:')
    all_datanames = []
    for task_result in evalfinal.values():
        evaluation_result = task_result.get('evaluation', {})
        for dataname in evaluation_result.keys():
            if dataname not in all_datanames:
                all_datanames.append(dataname)
    res = []
    for task_name, task_result in evalfinal.items():
        evaluation_result = task_result.get('evaluation', {})
        row = []
        for dataname in all_datanames:
            metrics = evaluation_result.get(dataname)
            ade_value = metrics.get('ADE') if metrics else None
            row.append(str(ade_value))
        res.append(row)
        print('\t'.join(row))

    # Transpose the ADE matrix so rows correspond to datasets and columns to models
    if res:
        res_t = list(map(list, zip(*res)))
        print('Transposed ADE matrix (rows=datasets, cols=models):')
        total_forgetting = 0
        lastforgetting=0
        total_plasticity=0
        datacount=0
        for dataname, row in zip(all_datanames, res_t):
            i=0
            forgeting=0
            plasticcells=0
            print('\t'.join(row))
            for j,r in enumerate(row):
                if(i>0):
                    forgeting+=(float(row[j])-float(row[j-1]))
                    print('row[j]:',row[j])
                    print('row[j-1]:',row[j-1])
                    print('forgeting:',forgeting)
                if(j==len(row)-1) and (datacount!=len(all_datanames)-1):
                    lastforgetting+=float(row[j])-plasticcells
                   # print('row[j]:',row[j])
                   # print('plasticcells:',plasticcells)
                   # print('lastforgetting:', lastforgetting)
                if(r!="None"):
                    if(i==0):
                        plasticcells=float(r)
                        total_plasticity+=plasticcells
                    i+=1            
            forgeting=forgeting/(i-1) if i > 1 else 0
            
            total_forgetting += forgeting
            print(f'Total Forgetteness: {forgeting}')
            datacount+=1
        average_last_forgetting = lastforgetting / (len(all_datanames)-1) if all_datanames else 0
        average_forgetting = total_forgetting / (len(all_datanames)-1) if all_datanames else 0
        average_plasticity = total_plasticity / len(all_datanames) if all_datanames else 0
        
        print(f'Average Forgetting across all datasets: {average_forgetting}')
        print(f'average error across all datasets: {AVGerror}')
        print(f'Average Last Forgetting across all datasets: {average_last_forgetting}')
        print(f'Average Plasticity across all datasets: {average_plasticity}')
    else:
        res_t = []
    
    #tinggal buat untuk forgetting, last forgetting, plasticity and the trade-off
    return evalfinal,average_forgetting,AVGerror,average_last_forgetting,average_plasticity


def average_evalresults(evalresults):
    if not evalresults:
        return {}

    all_model_names = []
    all_dataset_names = []
    for evalresult in evalresults:
        for model_name, task_result in evalresult.items():
            if model_name not in all_model_names:
                all_model_names.append(model_name)
            for dataset_name in task_result.get('evaluation', {}).keys():
                if dataset_name not in all_dataset_names:
                    all_dataset_names.append(dataset_name)

    averaged = {}
    for model_name in all_model_names:
        model_runs = [evalresult[model_name] for evalresult in evalresults if model_name in evalresult]
        averaged_evaluation = {}
        for dataset_name in all_dataset_names:
            metric_names = set()
            for model_run in model_runs:
                dataset_metrics = model_run.get('evaluation', {}).get(dataset_name)
                if dataset_metrics:
                    metric_names.update(dataset_metrics.keys())

            averaged_metrics = {}
            for metric_name in metric_names:
                metric_values = []
                for model_run in model_runs:
                    dataset_metrics = model_run.get('evaluation', {}).get(dataset_name)
                    if not dataset_metrics:
                        continue
                    metric_value = dataset_metrics.get(metric_name)
                    if metric_value is not None:
                        metric_values.append(float(metric_value))
                averaged_metrics[metric_name] = sum(metric_values) / len(metric_values) if metric_values else None

            averaged_evaluation[dataset_name] = averaged_metrics
        averaged[model_name] = {'evaluation': averaged_evaluation}

    return averaged


def print_transposed_metric_matrix(evalresult, metric_name, title=None):
    all_datanames = []
    for task_result in evalresult.values():
        evaluation_result = task_result.get('evaluation', {})
        for dataname in evaluation_result.keys():
            if dataname not in all_datanames:
                all_datanames.append(dataname)

    res = []
    for task_name, task_result in evalresult.items():
        evaluation_result = task_result.get('evaluation', {})
        row = []
        for dataname in all_datanames:
            metrics = evaluation_result.get(dataname)
            metric_value = metrics.get(metric_name) if metrics else None
            row.append(str(metric_value))
        res.append(row)
        print('\t'.join(row))

    if res:
        res_t = list(map(list, zip(*res)))
        heading = title or f'Transposed {metric_name} matrix (rows=datasets, cols=models):'
        print(heading)
        for dataname, row in zip(all_datanames, res_t):
            print(dataname + '\t' + '\t'.join(row))

    return all_datanames, res
#noreplystrategy()    
if __name__ == '__main__':
    #independentlearning()
   # noreplystrategy()
   #exreplay()
    # Check if EVAL_CKPTPATH is a list or single path
    if isinstance(EVAL_CKPTPATH, list):
        eval_paths = EVAL_CKPTPATH
    else:
        eval_paths = [EVAL_CKPTPATH]
    
    # Filter and verify paths exist
    valid_paths = []
    for path in eval_paths:
        if os.path.exists(path):
            valid_paths.append(path)
            print(f'Path exists: {path}')
        else:
            print(f'Path does NOT exist: {path}')
    
    if not valid_paths:
        print('ERROR: No valid EVAL_CKPTPATH found. Please check the paths.')
        sys.exit(1)
    final_forgetting=[]
    final_ADE_error=[]
    final_last_forgetting=[]
    final_plasticity=[]
    final_paths=[]
    final_times=[]
    overall_start = time.time()
    # Run evaluation for each valid path
    for EVAL_CKPTPATH in valid_paths:
        path_start = time.time()
        print(f'\n\n========== Processing: {EVAL_CKPTPATH} ==========\n')
        outputpath=os.path.join(EVAL_CKPTPATH, f'output5times_{time.strftime("%Y%m%d_%H%M%S")}.txt')
        log_file = open(outputpath, 'w')
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)
        forgetting_runs = []
        avgerror_runs = []
        last_forgetting_runs = []
        plasticity_runs = []
        evalresults = []
        try:
            for i in range(0,5):
                seed(i)
                evalresult,average_forgetting,AVGerror,average_last_forgetting,average_plasticity=evaluation()
                evalresults.append(evalresult)
                forgetting_runs.append(average_forgetting)
                avgerror_runs.append(AVGerror)
                last_forgetting_runs.append(average_last_forgetting)
                plasticity_runs.append(average_plasticity)

                for key, value in evalresult.items():
                    print(f'{key}: {value}')
            averaged_evalresult = average_evalresults(evalresults)
            print('Averaged evalresult over 5 runs:')
            for metric_name in ['ADE', 'FDE', 'ADEQ05', 'ADEQ95', 'FDEQ05', 'FDEQ95']:
                print_transposed_metric_matrix(
                    averaged_evalresult,
                    metric_name,
                    title=f'Transposed averaged {metric_name} matrix (rows=datasets, cols=models):'
                )
            print('5-run summary:')
            print(f'Average forgetting: mean={np.mean(forgetting_runs)}, std={np.std(forgetting_runs)}')
            print(f'Average ADE error: mean={np.mean(avgerror_runs)}, std={np.std(avgerror_runs)}')
            print(f'Average last forgetting: mean={np.mean(last_forgetting_runs)}, std={np.std(last_forgetting_runs)}')
            print(f'Average plasticity: mean={np.mean(plasticity_runs)}, std={np.std(plasticity_runs)}')
            final_forgetting.append(np.mean(forgetting_runs))
            final_ADE_error.append(np.mean(avgerror_runs))
            final_last_forgetting.append(np.mean(last_forgetting_runs))
            final_plasticity.append(np.mean(plasticity_runs))
            # collect stds as lists so final zip iterates properly
            try:
                final_std_forgetting.append(np.std(forgetting_runs))
                final_std_ADE_error.append(np.std(avgerror_runs))
                final_std_last_forgetting.append(np.std(last_forgetting_runs))
                final_std_plasticity.append(np.std(plasticity_runs))
            except NameError:
                final_std_forgetting = [np.std(forgetting_runs)]
                final_std_ADE_error = [np.std(avgerror_runs)]
                final_std_last_forgetting = [np.std(last_forgetting_runs)]
                final_std_plasticity = [np.std(plasticity_runs)]
            final_paths.append(EVAL_CKPTPATH)
        finally:
            sys.stdout = original_stdout
            log_file.close()
            path_elapsed = time.time() - path_start
            final_times.append(path_elapsed)
            print(f'Finished processing {EVAL_CKPTPATH} in {path_elapsed:.2f}s')
    print('\n\n========== Final Summary Across All Paths ==========\n')
    for path, forgetting, ade_error, last_forgetting, plasticity, std_forgetting, std_ADE_error, std_last_forgetting, std_plasticity, elapsed in zip(final_paths, final_forgetting, final_ADE_error, final_last_forgetting, final_plasticity, final_std_forgetting, final_std_ADE_error, final_std_last_forgetting, final_std_plasticity, final_times):
        print(f'Path: {path}')
        print(f'{forgetting}\t{ade_error}\t{last_forgetting}\t{plasticity}\t{std_forgetting}\t{std_ADE_error}\t{std_last_forgetting}\t{std_plasticity}\t{elapsed:.2f}s')
    total_elapsed = time.time() - overall_start
    print(f'Total elapsed time for all paths: {total_elapsed:.2f}s')
    
    #noreplystrategy()
    #fullreplaystrategy()
    #experienceReplay(20)
    #BCSRexperienceReplay()
    #RELexperienceReplay()
    #ProposedexperienceReplay()
    #filteredexperienceReplay()
    #print(evaluation())
    #prepare the dataset and just kill them all!!
    #pakai dataset dari german sama buat MOT214 yang durasi panjang!
    #kemudian rocknroll pakai data cluster buat : 
    #buatscenebaseexperience replay tanpa ada embel embel filter!