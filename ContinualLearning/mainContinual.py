import torch
import os
import subprocess
import time
import torch
import os, sys, time
sys.path.append('..')
from utils import ADE_FDE, FPC, get_rng_state, set_rng_state
from data import Dataloader
from social_vae import SocialVAE
import importlib
import shutil
import tracemalloc
import pickle
import gc
from ContinualLearning.previousdata import previousdata

#taskarrangement=["data90d7","data90g3","data120b2","dataMOT212"]
#taskarrangement=["task2","task3","task4"]
#taskarrangement=["dataMOT212","data120b2","data90g3","data90d7"]
taskarrangement=["duri_morning_cluster_world","duri_evening_cluster_world","vredeburg_cluster_world","duri_platform1_cluster_world","paisley_cluster_world"]
#taskarrangement=["duri_morning_world","duri_evening_world","vredeburg_world","duri_platform1_world","paisley_world"]
#taskarrangement=["duri_platform1_world","vredeburg_world","duri_morning_world","paisley_world","duri_evening_world"]
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

homefolder='/home/Documents/projects/MIND_VAE/'
#datafolder=os.path.join(homefolder,'data/continualdata_sandbox')
datafolder=os.path.join(homefolder,'data/crowdTraj World/dataset cluster world CL/')
# datafolder=os.path.join(homefolder,'data/crowdTraj World/dataset GT world CL/')
#datafolder=os.path.join(homefolder,'data/Long_seq')
#datafolder=os.path.join(homefolder,'data/julichmotfull')
#datafolder=os.path.join(homefolder,'data/continualdata')
#ckptpath=os.path.join(homefolder,'log_eth/logcontinualofflineRELjan5th2')
ckptpath=os.path.join(homefolder, f"log_eth/logcontinualsEXuniform_{time.strftime('%Y%m%d_%H%M%S')}")
configpath=os.path.join(homefolder,'config/mot.py')
configpathpaisley=os.path.join(homefolder,'config/mot_paisley.py')
EVAL_CKPTPATH="/home/Documents/projects/MIND_VAE/log_eth/logcontinualsproposednobcsr_Crowdtraj120260802_022351"
def independentlearning():
    #training setiap task secara independen tanpa membawa data atau model sebelumnya, jadi setiap task itu seperti training dari awal
    start=time.time()
    print("start")
    #for root, dirs, files in os.walk(datafolder):
    for iddir,dir_name in enumerate(taskarrangement):
            tracemalloc.start()
            folder_path = os.path.join(datafolder, dir_name)
            print("Processing folder:", folder_path)
            trainpath=os.path.join(folder_path,'train')
            validationpath=os.path.join(folder_path,'test')
            logpath="log"+dir_name
            ckptpathcl=os.path.join(ckptpath,logpath)
            os.makedirs(ckptpathcl, exist_ok=True)
            processlogpath=os.path.join(ckptpathcl, "process.log")
            startpertask=time.time()
            if dir_name == "paisley_cluster_world":
                    current_configpath = configpathpaisley
            else:
                    current_configpath = configpath
            cmd = [sys.executable, "main.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',current_configpath]
            result = run_with_log(cmd, processlogpath)
            print("Saved process log:", processlogpath)
            if result.returncode != 0:
                print(f"Training process failed for {dir_name} with return code {result.returncode}.")
                print(f"Check log file: {processlogpath}")
                raise RuntimeError(f"main.py failed for {dir_name} (code {result.returncode})")
            #print("Output:", result.stdout)
            
            endpertask=time.time()
            current, peak = tracemalloc.get_traced_memory()
            print(f"Peak memory usage: {peak / 1024**2:.2f} MB")
            print("Script finished running! for :", dir_name)
            pertask=endpertask-startpertask
            print('time execution on task:',pertask)
    end=time.time()
    total=end-start
    print('total execution time:', total)

def noreplystrategy():
    start=time.time()
    print("start")
    #for root, dirs, files in os.walk(datafolder):
    for iddir,dir_name in enumerate(taskarrangement):
            tracemalloc.start()
            folder_path = os.path.join(datafolder, dir_name)
            print("Processing folder:", folder_path)
            trainpath=os.path.join(folder_path,'train')
            validationpath=os.path.join(folder_path,'test')
            logpath="log"+dir_name
            ckptpathcl=os.path.join(ckptpath,logpath)
            os.makedirs(ckptpathcl, exist_ok=True)
            processlogpath=os.path.join(ckptpathcl, "process.log")
            startpertask=time.time()
            #cmd = [sys.executable, "mainproposed.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',current_configpath,'--method','FilteredReplay','--datacapacity',sample,'--taskno',idstr]
            if dir_name == "paisley_cluster_world":
                    current_configpath = configpathpaisley
            else:
                    current_configpath = configpath
            cmd = [sys.executable, "reducedmain.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',current_configpath,'--taskno',str(iddir+1)]
            result = run_with_log(cmd, processlogpath)
            prevdata=ckptpathcl
            print("Saved process log:", processlogpath)
            if result.returncode != 0:
                print(f"Training process failed for {dir_name} with return code {result.returncode}.")
                print(f"Check log file: {processlogpath}")
                raise RuntimeError(f"main.py failed for {dir_name} (code {result.returncode})")
            #print("Output:", result.stdout)
            if(iddir<len(taskarrangement)-1):#take previous log
                logpath="log"+taskarrangement[iddir+1]
                ckptpathnext=os.path.join(ckptpath,logpath)
                shutil.copytree(ckptpathcl, ckptpathnext)
            
            endpertask=time.time()
            current, peak = tracemalloc.get_traced_memory()
            print(f"Peak memory usage: {peak / 1024**2:.2f} MB")
            print("Script finished running! for :", dir_name)
            pertask=endpertask-startpertask
            print('time execution on task:',pertask)
            #print("Output:", result.stdout)
    #    break
    end=time.time()
    total=end-start
    print('total execution time:', total)
def fullreplaystrategy():
    #training all of the dataset: need more storage, make the batch for neighbourhood
    taskarrangement=["dataMOT212"]
    start=time.time()
    print("start")
    datafolder=os.path.join(homefolder,'data/continualfullreplay')
    #for root, dirs, files in os.walk(datafolder):
    for iddir,dir_name in enumerate(taskarrangement):
            tracemalloc.start()
            folder_path = os.path.join(datafolder, dir_name)
            print("Processing folder:", folder_path)
            trainpath=os.path.join(folder_path,'train')
            validationpath=os.path.join(folder_path,'test')
            logpath="log"+dir_name
            ckptpathcl=os.path.join(ckptpath,logpath)
            startpertask=time.time()
            result = subprocess.run(["python", "main.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',configpath], capture_output=True, text=True)
            #print(result)
            endpertask=time.time()
            current, peak = tracemalloc.get_traced_memory()
            print(f"Peak memory usage: {peak / 1024**2:.2f} MB")
            print("Script finished running! for :", dir_name)
            pertask=endpertask-startpertask
            print('time execution on task:',pertask)
            # print("Output:", result.stdout)
    #    break
    end=time.time()
    total=end-start
    print('total execution time:', total)
def experienceReplay(replaysample):
    start =time.time()
    print("starting experience replay")
    sample="20"
    #start=time.time()
    #print("start")
    #for root, dirs, files in os.walk(datafolder):
    for iddir,dir_name in enumerate(taskarrangement):
            tracemalloc.start()
            folder_path = os.path.join(datafolder, dir_name)
            print("Processing folder:", folder_path)
            trainpath=os.path.join(folder_path,'train')
            validationpath=os.path.join(folder_path,'test')
            logpath="log"+dir_name
            ckptpathcl=os.path.join(ckptpath,logpath)
            startpertask=time.time()
            
            if(iddir==0):#if this the first scene then run and create the memory buffer
                result = subprocess.run([sys.executable, "main.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',configpath], capture_output=True, text=True)
                memorybufferfolder=os.path.join(ckptpath, "memorybuffer")
                print(result)
                try:
                    os.mkdir(memorybufferfolder)
                    print(memorybufferfolder+'is created')
                except FileExistsError:
                    print(f"Directory '{memorybufferfolder}' already exists, we will use this one.")
                except PermissionError:
                    print(f"Permission denied: Unable to create '{memorybufferfolder}'.")
                except Exception as e:
                    print(f"An error occurred: {e}")
            else:
                prevdata= memorybufferfolder
                result = subprocess.run([sys.executable, "main.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',configpath,'--prev_data',prevdata,'--datacapacity',sample], capture_output=True, text=True)
                print('done ',iddir,'!')
                print(result)
            XYvalid=os.path.join(ckptpathcl,"bestXY.pt")
            bestloss=os.path.join(ckptpathcl,"bestloss.pt")
            loaded_tensorXYvalid = torch.load(XYvalid)
            loaded_tensorlosstraining = torch.load(bestloss)
            p=previousdata()
            #sample=int(sample)
            p.getxreplaydata(loaded_tensorlosstraining,loaded_tensorXYvalid,int(sample))   
            longterm_memory=[]
            longterm_memory.append(p)
            
            memoryname="memory"+dir_name+".pt"
            bestaddress=os.path.join(memorybufferfolder, memoryname)
            torch.save(longterm_memory, bestaddress)
            if(iddir<len(taskarrangement)-1):#take previous log
                logpath="log"+taskarrangement[iddir+1]
                ckptpathnext=os.path.join(ckptpath,logpath)
                shutil.copytree(ckptpathcl, ckptpathnext)
            endpertask=time.time()
            current, peak = tracemalloc.get_traced_memory()
            
            print(f"Peak RAM usage: {peak / 1024**2:.2f} MB")
            print("Script finished running! for :", dir_name)
            pertask=endpertask-startpertask
            print('time execution on task:',pertask)
            # print("Output:", result.stdout)
    #    break
    end=time.time()
    total=end-start
    print('total execution time:', total) 
def filteredexperienceReplay():

    start=time.time()
    print("start")
    sample='20'
    
    #for root, dirs, files in os.walk(datafolder):
    for iddir,dir_name in enumerate(taskarrangement):
            tracemalloc.start()
            folder_path = os.path.join(datafolder, dir_name)
            print("Processing folder:", folder_path)
            trainpath=os.path.join(folder_path,'train')
            validationpath=os.path.join(folder_path,'test')
            logpath="log"+dir_name
            ckptpathcl=os.path.join(ckptpath,logpath)
            startpertask=time.time()
            if(iddir==0):#if this the first scene then run and create the memory buffer
                result = subprocess.run(["python", "main.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',configpath,'--method','FilteredReplay','--datacapacity',sample], capture_output=True, text=True)
                #memorybufferfolder=os.path.join(ckptpath, "memorybuffer")
                #print('done 1!')
                print(result)
                #prevdata=logpath
                #try:
                #    os.mkdir(memorybufferfolder)
                #    print(memorybufferfolder+'is created')
                #except FileExistsError:
                #except PermissionError:
                #    print(f"Directory '{memorybufferfolder}' already exists, we will use this one.")
                #    print(f"Permission denied: Unable to create '{memorybufferfolder}'.")
                #except Exception as e:
                #    print(f"An error occurred: {e}")
            else:
                prevdata=ckptpath
                result = subprocess.run(["python", "main.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',configpath,'--prev_data',prevdata], capture_output=True, text=True)
                prevdata+=","+logpath
                print('done ',iddir,'!')
                print(result)
            #XYvalid=os.path.join(ckptpathcl,"bestXY.pt")
            #bestloss=os.path.join(ckptpathcl,"bestloss.pt")
            #loaded_tensorXYvalid = torch.load(XYvalid)#"/home/USER/Documents/projects/MIND_VAE/log_eth/testCL2050top10/bestXY.pt")
            #loaded_tensorlosstraining = torch.load(bestloss)#"/home/USER/Documents/projects/MIND_VAE/log_eth/testCL2050top10/bestloss.pt")
            #p=previousdata()
            #p.getFilteredData(loaded_tensorlosstraining,loaded_tensorXYvalid)   
            #longterm_memory=[]
            #longterm_memory.append(p)
            
            #memoryname="memory"+dir_name+".pt"
            #bestaddress=os.path.join(memorybufferfolder, memoryname)
            #torch.save(longterm_memory, bestaddress)
            #if(iddir<len(taskarrangement)-1):#take previous log
            #    logpath="log"+taskarrangement[iddir+1]
            #    ckptpathnext=os.path.join(ckptpath,logpath)
            #    shutil.copytree(ckptpathcl, ckptpathnext)
            endpertask=time.time()
            current, peak = tracemalloc.get_traced_memory()
            print(f"Peak memory usage: {peak / 1024**2:.2f} MB")
            print("Script finished running! for :", dir_name)
            pertask=endpertask-startpertask
            print('time execution on task:',pertask)
            # print("Output:", result.stdout)
    #    break
    end=time.time()
    total=end-start
    print('total execution time:', total) 
def exreplay():
    start=time.time()
    print("start")
    sample='10'
   
    #for root, dirs, files in os.walk(datafolder):
    for iddir,dir_name in enumerate(taskarrangement):
            tracemalloc.start()
            idstr=str(iddir+1)
            folder_path = os.path.join(datafolder, dir_name)
            print("Processing folder:", folder_path)
            trainpath=os.path.join(folder_path,'train')
            validationpath=os.path.join(folder_path,'test')
            logpath="log"+dir_name
            ckptpathcl=os.path.join(ckptpath,logpath)
            os.makedirs(ckptpathcl, exist_ok=True)
            processlogpath=os.path.join(ckptpathcl, "process.log")
            startpertask=time.time()
            if dir_name == "paisley_cluster_world":
                    current_configpath = configpathpaisley
            else:
                    current_configpath = configpath
            if(iddir==0):#if this the first scene then run and create the memory buffer
                cmd = [sys.executable, "mainexreplay.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',current_configpath,'--method','FilteredReplay','--datacapacity',sample,'--taskno',idstr]
                result = run_with_log(cmd, processlogpath)
                prevdata=ckptpathcl
                print("Saved process log:", processlogpath)
            else:
                #prevdata=ckptpath
                print("prev data:",prevdata)
                cmd = [sys.executable, "mainexreplay.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',current_configpath,'--prev_data',prevdata,'--datacapacity',sample,'--method','FilteredReplay','--taskno',idstr]
                result = run_with_log(cmd, processlogpath)
                #prevdata+=","+logpath
                prevdata=ckptpathcl
                print('done ',iddir,'!')
                print("Saved process log:", processlogpath)
            if result.returncode != 0:
                print(f"Training process failed for {dir_name} with return code {result.returncode}.")
                print(f"Check log file: {processlogpath}")
                raise RuntimeError(f"mainexreplay.py failed for {dir_name} (code {result.returncode})")
            endpertask=time.time()
            if(iddir<len(taskarrangement)-1):#take previous log
                logpath="log"+taskarrangement[iddir+1]
                ckptpathnext=os.path.join(ckptpath,logpath)
                shutil.copytree(ckptpathcl, ckptpathnext)
            current, peak = tracemalloc.get_traced_memory()
            print(f"Peak memory usage: {peak / 1024**2:.2f} MB")
            print("Script finished running! for :", dir_name)
            pertask=endpertask-startpertask
            print('time execution on task:',pertask)
            # print("Output:", result.stdout)
    #    break
    end=time.time()
    total=end-start
    print('total execution time:', total)     
def RELexperienceReplay():
    start=time.time()
    print("start")
    sample='10'
   
    #for root, dirs, files in os.walk(datafolder):
    for iddir,dir_name in enumerate(taskarrangement):
            tracemalloc.start()
            idstr=str(iddir+1)
            folder_path = os.path.join(datafolder, dir_name)
            print("Processing folder:", folder_path)
            trainpath=os.path.join(folder_path,'train')
            validationpath=os.path.join(folder_path,'test')
            logpath="log"+dir_name
            ckptpathcl=os.path.join(ckptpath,logpath)
            os.makedirs(ckptpathcl, exist_ok=True)
            processlogpath=os.path.join(ckptpathcl, "process.log")
            startpertask=time.time()
            if dir_name == "paisley_cluster_world":
                    current_configpath = configpathpaisley
            else:
                    current_configpath = configpath
            if(iddir==0):#if this the first scene then run and create the memory buffer
                cmd = [sys.executable, "mainofflinereducible.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',current_configpath,'--method','FilteredReplay','--datacapacity',sample,'--taskno',idstr]
                result = run_with_log(cmd, processlogpath)
                prevdata=ckptpathcl
                print("Saved process log:", processlogpath)
            else:
                #prevdata=ckptpath
                print("prev data:",prevdata)
                cmd = [sys.executable, "mainofflinereducible.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',current_configpath,'--prev_data',prevdata,'--datacapacity',sample,'--method','FilteredReplay','--taskno',idstr]
                result = run_with_log(cmd, processlogpath)
                #prevdata+=","+logpath
                prevdata=ckptpathcl
                print('done ',iddir,'!')
                print("Saved process log:", processlogpath)
            if result.returncode != 0:
                print(f"Training process failed for {dir_name} with return code {result.returncode}.")
                print(f"Check log file: {processlogpath}")
                raise RuntimeError(f"mainofflinereducible.py failed for {dir_name} (code {result.returncode})")
            endpertask=time.time()
            if(iddir<len(taskarrangement)-1):#take previous log
                logpath="log"+taskarrangement[iddir+1]
                ckptpathnext=os.path.join(ckptpath,logpath)
                shutil.copytree(ckptpathcl, ckptpathnext)
            current, peak = tracemalloc.get_traced_memory()
            print(f"Peak memory usage: {peak / 1024**2:.2f} MB")
            print("Script finished running! for :", dir_name)
            pertask=endpertask-startpertask
            print('time execution on task:',pertask)
            # print("Output:", result.stdout)
    #    break
    end=time.time()
    total=end-start
    print('total execution time:', total)     

def BCSRexperienceReplay():
    start=time.time()
    print("start")
    sample='10'
    #for root, dirs, files in os.walk(datafolder):
    for iddir,dir_name in enumerate(taskarrangement):
            tracemalloc.start()
            idstr=str(iddir)
            folder_path = os.path.join(datafolder, dir_name)
            print("Processing folder:", folder_path)
            trainpath=os.path.join(folder_path,'train')
            validationpath=os.path.join(folder_path,'test')
            logpath="log"+dir_name
            ckptpathcl=os.path.join(ckptpath,logpath)
            os.makedirs(ckptpathcl, exist_ok=True)
            processlogpath=os.path.join(ckptpathcl, "process.log")
            startpertask=time.time()
            if dir_name == "paisley_cluster_world":
                    current_configpath = configpathpaisley
            else:
                    current_configpath = configpath
            if(iddir==0):#if this the first scene then run and create the memory buffer
                cmd = [sys.executable, "mainbilevelreg.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',current_configpath,'--method','FilteredReplay','--datacapacity',sample,'--taskno',idstr]
                result = run_with_log(cmd, processlogpath)
                prevdata=ckptpathcl
                print("Saved process log:", processlogpath)
            else:
                cmd = [sys.executable, "mainbilevelreg.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',current_configpath,'--prev_data',prevdata,'--datacapacity',sample,'--method','FilteredReplay','--taskno',idstr]
                result = run_with_log(cmd, processlogpath)
                prevdata=ckptpathcl
                print('done ',iddir,'!')
                print("Saved process log:", processlogpath)
            if result.returncode != 0:
                print(f"Training process failed for {dir_name} with return code {result.returncode}.")
                print(f"Check log file: {processlogpath}")
                raise RuntimeError(f"mainbilevelreg.py failed for {dir_name} (code {result.returncode})")
            endpertask=time.time()
            current, peak = tracemalloc.get_traced_memory()
            if(iddir<len(taskarrangement)-1):#take previous log
                logpath="log"+taskarrangement[iddir+1]
                ckptpathnext=os.path.join(ckptpath,logpath)
                shutil.copytree(ckptpathcl, ckptpathnext)
            print(f"Peak memory usage: {peak / 1024**2:.2f} MB")
            print("Script finished running! for :", dir_name)
            pertask=endpertask-startpertask
            print('time execution on task:',pertask)
            # print("Output:", result.stdout)
    #    break
    end=time.time()
    total=end-start
    print('total execution time:', total)
    #cek pengaruhnya capacity disini yaaa
def run_with_log(cmd, log_file_path):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n" + "=" * 80 + "\n")
            log_file.write(f"[{timestamp}] Running command:\n")
            log_file.write(" ".join(cmd) + "\n\n")
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_file.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Return code: {result.returncode}\n")
        return result
def ProposedexperienceReplay():
    start=time.time()
    print("start Proposed experience replay")
    sample='10'
    current_configpath = configpath
    #for root, dirs, files in os.walk(datafolder):
    for iddir,dir_name in enumerate(taskarrangement):
            tracemalloc.start()
            idstr=str(iddir+1)
            folder_path = os.path.join(datafolder, dir_name)
            print("Processing folder:", folder_path)
            trainpath=os.path.join(folder_path,'train')
            validationpath=os.path.join(folder_path,'test')
            logpath="log"+dir_name
            ckptpathcl=os.path.join(ckptpath,logpath)
            os.makedirs(ckptpathcl, exist_ok=True)
            if(iddir!=0):
                
                processlogpath=os.path.join(ckptpathcl, "process1.log")
            else:
                processlogpath=os.path.join(ckptpathcl, "process.log")
            startpertask=time.time()
            if dir_name == "paisley_cluster_world":
                    current_configpath = configpathpaisley
            else:
                    current_configpath = configpath
            if(iddir==0):#if this the first scene then run and create the memory buffer
                cmd = [sys.executable, "mainproposed.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',current_configpath,'--method','FilteredReplay','--datacapacity',sample,'--taskno',idstr]
                result = run_with_log(cmd, processlogpath)
                prevdata=ckptpathcl
                print("Saved process log:", processlogpath)
            else:
                #prevdata=ckptpath
                print("prev data:",prevdata)
                
                cmd = [sys.executable, "mainproposed.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',current_configpath,'--prev_data',prevdata,'--datacapacity',sample,'--method','FilteredReplay','--taskno',idstr]
                result = run_with_log(cmd, processlogpath)
                #prevdata+=","+logpath
                prevdata=ckptpathcl
                print('done ',iddir,'!')
                print("Saved process log:", processlogpath)
                if result.returncode != 0:
                    print(f"Training process failed for {dir_name} with return code {result.returncode}.")
                    print(f"Check log file: {processlogpath}")
                    raise RuntimeError(f"mainproposed.py failed for {dir_name} (code {result.returncode})")
            if(iddir<len(taskarrangement)-1):#take previous log
                logpath="log"+taskarrangement[iddir+1]
                ckptpathnext=os.path.join(ckptpath,logpath)
                shutil.copytree(ckptpathcl, ckptpathnext)
            endpertask=time.time()
            current, peak = tracemalloc.get_traced_memory()
            print(f"Peak memory usage: {peak / 1024**2:.2f} MB")
            print("Script finished running! for :", dir_name)
            pertask=endpertask-startpertask
            print('time execution on task:',pertask)
            # print("Output:", result.stdout)
    #    break
    end=time.time()
    total=end-start
    print('total execution time:', total)    
def ProposedexperienceReplayFULL():
    start=time.time()
    print("start Proposed experience replay")
    sample='10'
    current_configpath = configpath
    #for root, dirs, files in os.walk(datafolder):
    for iddir,dir_name in enumerate(taskarrangement):
            tracemalloc.start()
            idstr=str(iddir+1)
            folder_path = os.path.join(datafolder, dir_name)
            print("Processing folder:", folder_path)
            trainpath=os.path.join(folder_path,'train')
            validationpath=os.path.join(folder_path,'test')
            logpath="log"+dir_name
            ckptpathcl=os.path.join(ckptpath,logpath)
            os.makedirs(ckptpathcl, exist_ok=True)
            processlogpath=os.path.join(ckptpathcl, "process.log")
            startpertask=time.time()
            if dir_name == "paisley_cluster_world":
                    current_configpath = configpathpaisley
            else:
                    current_configpath = configpath
            if(iddir==0):#if this the first scene then run and create the memory buffer
                cmd = [sys.executable, "mainproposed.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',current_configpath,'--method','FilteredReplay','--datacapacity',sample,'--taskno',idstr]
                result = run_with_log(cmd, processlogpath)
                prevdata=ckptpathcl
                print("Saved process log:", processlogpath)
            else:
                #prevdata=ckptpath
                print("prev data:",prevdata)
                
                cmd = [sys.executable, "mainproposed.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',current_configpath,'--prev_data',prevdata,'--datacapacity',sample,'--method','FilteredReplay','--taskno',idstr]
                result = run_with_log(cmd, processlogpath)
                #prevdata+=","+logpath
                prevdata=ckptpathcl
                print('done ',iddir,'!')
                print("Saved process log:", processlogpath)
            if result.returncode != 0:
                print(f"Training process failed for {dir_name} with return code {result.returncode}.")
                print(f"Check log file: {processlogpath}")
                raise RuntimeError(f"mainproposed.py failed for {dir_name} (code {result.returncode})")
            if(iddir<len(taskarrangement)-1):#take previous log
                logpath="log"+taskarrangement[iddir+1]
                ckptpathnext=os.path.join(ckptpath,logpath)
                shutil.copytree(ckptpathcl, ckptpathnext)
            endpertask=time.time()
            current, peak = tracemalloc.get_traced_memory()
            print(f"Peak memory usage: {peak / 1024**2:.2f} MB")
            print("Script finished running! for :", dir_name)
            pertask=endpertask-startpertask
            print('time execution on task:',pertask)
            # print("Output:", result.stdout)
    #    break
    end=time.time()
    total=end-start
    print('total execution time:', total)     

def Proposedablation(ablationmain):
    #ablationmain can be "mainablationnocluster.py" or "mainablationnobcsr.py" or "mainablationnorel.py"
    start=time.time()
    print("start Proposed experience replay")
    sample='10'
    current_configpath = configpath
    #for root, dirs, files in os.walk(datafolder):
    for iddir,dir_name in enumerate(taskarrangement):
            tracemalloc.start()
            idstr=str(iddir+1)
            folder_path = os.path.join(datafolder, dir_name)
            print("Processing folder:", folder_path)
            trainpath=os.path.join(folder_path,'train')
            validationpath=os.path.join(folder_path,'test')
            logpath="log"+dir_name
            ckptpathcl=os.path.join(ckptpath,logpath)
            os.makedirs(ckptpathcl, exist_ok=True)
            processlogpath=os.path.join(ckptpathcl, "process.log")
            startpertask=time.time()
            if dir_name == "paisley_cluster_world":
                    current_configpath = configpathpaisley
            else:
                    current_configpath = configpath
            if(iddir==0):#if this the first scene then run and create the memory buffer
                cmd = [sys.executable, ablationmain,'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',current_configpath,'--method','FilteredReplay','--datacapacity',sample,'--taskno',idstr]
                result = run_with_log(cmd, processlogpath)
                prevdata=ckptpathcl
                print("Saved process log:", processlogpath)
            else:
                #prevdata=ckptpath
                print("prev data:",prevdata)
                
                cmd = [sys.executable, ablationmain,'--train',trainpath,'--test',validationpath,'--ckpt',ckptpathcl,'--config',current_configpath,'--prev_data',prevdata,'--datacapacity',sample,'--method','FilteredReplay','--taskno',idstr]
                result = run_with_log(cmd, processlogpath)
                #prevdata+=","+logpath
                prevdata=ckptpathcl
                print('done ',iddir,'!')
                print("Saved process log:", processlogpath)
            if result.returncode != 0:
                print(f"Training process failed for {dir_name} with return code {result.returncode}.")
                print(f"Check log file: {processlogpath}")
                raise RuntimeError(f"{ablationmain} failed for {dir_name} (code {result.returncode})")
            if(iddir<len(taskarrangement)-1):#take previous log
                logpath="log"+taskarrangement[iddir+1]
                ckptpathnext=os.path.join(ckptpath,logpath)
                shutil.copytree(ckptpathcl, ckptpathnext)
            endpertask=time.time()
            current, peak = tracemalloc.get_traced_memory()
            print(f"Peak memory usage: {peak / 1024**2:.2f} MB")
            print("Script finished running! for :", dir_name)
            pertask=endpertask-startpertask
            print('time execution on task:',pertask)
            # print("Output:", result.stdout)
    #    break
    end=time.time()
    total=end-start
    print('total execution time:', total)     

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
            del test_dataset
            gc.collect()
            dataname="evdata_"+taskarrangement[idtask]
            evalpertask[dataname]={'ADE':ADE.cpu().item(),'FDE:':FDE.cpu().item(),'ADEQ05':ADEQ05.cpu().item(),'ADEQ95':ADEQ95.cpu().item(),'FDEQ05':FDEQ05.cpu().item(),'FDEQ95':FDEQ95.cpu().item()}
            print('evaluation task-',taskarrangement[idtask])
            print("ADE",ADE)
            print("FDE",FDE)
            print("ADEQ05",ADEQ05)
            print("ADEQ95",ADEQ95)
            print("FDEQ05",FDEQ05)
            print("FDEQ95",FDEQ95)

        evalfinal[taskarrangement[i]]={'evaluation':evalpertask.copy()}
        #evalfinal['evaluation']=evalpertask
    print(evalfinal)
    print('ADE summary:')
    all_datanames = []
    for task_result in evalfinal.values():
        evaluation_result = task_result.get('evaluation', {})
        for dataname in evaluation_result.keys():
            if dataname not in all_datanames:
                all_datanames.append(dataname)

    for task_name, task_result in evalfinal.items():
        evaluation_result = task_result.get('evaluation', {})
        row = []
        for dataname in all_datanames:
            metrics = evaluation_result.get(dataname)
            ade_value = metrics.get('ADE') if metrics else None
            row.append(str(ade_value))
        print('\t'.join(row))
    return evalfinal
#noreplystrategy()
def evaluation():
    evalresult=evaluation()
        #evalresult={'coba':'1'}
    outputpath=os.path.join(EVAL_CKPTPATH, f'output_{time.strftime("%Y%m%d_%H%M%S")}.txt')
    with open(outputpath, 'w') as f:
        for key, value in evalresult.items():
            f.write(f'{key}: {value}\n')    
if __name__ == '__main__':
    #independentlearning()
   # noreplystrategy()
   #exreplay()
    evaluation()
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