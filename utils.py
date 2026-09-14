import torch
import random
import numpy as np 
def get_random_indexes(arr_length, excluded_indexes=[], count=5):
    flattened_excluded_indexes = []
    for item in excluded_indexes:
        if isinstance(item, (list, tuple, set)):
            flattened_excluded_indexes.extend(item)
        else:
            flattened_excluded_indexes.append(item)
    excluded_set = set(flattened_excluded_indexes)
    allowed = [i for i in range(arr_length) if i not in excluded_set]
    return random.sample(allowed, min(count, len(allowed)))
def mergepreviousdatatraining(traindata,prevtrainingdata,batches,trainedindex):
    #include previous data training to train data as much as sample
    count=len(prevtrainingdata[3])//batches
    #temporary=[]
    #tambahakan weight data dari awal dengan inisialisasi dengan nilai 1, atau gunakan weight vector, sebagai inputan tambahan untuk model, jadi di model ditambahkan weight diawal 
    #  free, total = torch.cuda.mem_get_info("cuda:2")
    #  mem_used_MB = (total - free) / 1024 ** 2
    #  print('before',mem_used_MB)
    #for rs in prevtrainingdata:
    temp=prevtrainingdata
    #initialscore=np.zeros((temp[8].shape[0],1))
    #prevtrainingdata.append(initialscore)
    #temp=prevtrainingdata
    #temp=np.array([t.cpu().numpy() for t in temp])
    #temporary.append(temp)
    random_indices=get_random_indexes(temp[0].shape[1], trainedindex, count)#mengambil sejumlah batch dari prevtrainingdata
    ############################sini nih masih jadi masalah gmn caranya ngambil sebagian batch aja##############
    traindataobs=torch.cat((traindata[0],temp[0][:,random_indices,:].to(traindata[0].device)),dim=1)
    traindatafut=torch.cat((traindata[1],temp[1][:,random_indices,:].to(traindata[1].device)),dim=1)
    traindataneig=torch.cat((traindata[2],temp[2][:,random_indices,:,:].to(traindata[2].device)),dim=1)
    traindataerr=torch.cat((traindata[3],temp[3][random_indices].to(traindata[3].device)),dim=0)
    traindatakl=torch.cat((traindata[4],temp[4][random_indices].to(traindata[4].device)),dim=0)
    traindataidtrack = np.concatenate([np.asarray(traindata[6]), np.asarray(temp[5][random_indices])], axis=0)
    traindatacluster=np.concatenate([np.asarray(traindata[5]), np.asarray(temp[6][random_indices])], axis=0)
    traindatataskno=np.concatenate([np.asarray(traindata[7]), np.asarray(temp[7][random_indices])], axis=0)
    taraindatascore=np.concatenate([np.asarray(traindata[8]), np.asarray(temp[8][random_indices])], axis=0)
    # temp_cluster = torch.as_tensor( temp[6], device=traindata[0].device)
    # traindatacluster=torch.cat((traindata[6],temp_cluster[random_indices]))
    # temp_taskno = torch.as_tensor( temp[7], device=traindata[0].device) 
    # traindatataskno=torch.cat((traindata[7],temp_taskno[random_indices]))
    trainedindex.append(random_indices)
    #traindata.append(temporary)
    return [traindataobs,traindatafut,traindataneig,traindataerr,traindatakl,traindatacluster,traindataidtrack,traindatataskno, taraindatascore],trainedindex
def mergepreviousdatatrainingproposed(traindata,prevtrainingdata,batches,trainedindex):
    #include previous data training to train data as much as sample
    count=len(prevtrainingdata[3])//batches
    #temporary=[]
    #tambahakan weight data dari awal dengan inisialisasi dengan nilai 1, atau gunakan weight vector, sebagai inputan tambahan untuk model, jadi di model ditambahkan weight diawal 
    #  free, total = torch.cuda.mem_get_info("cuda:2")
    #  mem_used_MB = (total - free) / 1024 ** 2
    #  print('before',mem_used_MB)
    #for rs in prevtrainingdata:
    temp=prevtrainingdata
    # Initialize zeros as torch tensor matching temp[8] shape
    #initialscore = np.ones_like(traindata[7])
 

    #temp=prevtrainingdata
    #temp=np.array([t.cpu().numpy() for t in temp])
    #temporary.append(temp)
    random_indices=get_random_indexes(temp[0].shape[1], trainedindex, count)#mengambil sejumlah batch dari prevtrainingdata
    ############################sini nih masih jadi masalah gmn caranya ngambil sebagian batch aja##############
    traindataobs=torch.cat((traindata[0],temp[0][:,random_indices,:]),dim=1)
    traindatafut=torch.cat((traindata[1],temp[1][:,random_indices,:]),dim=1)
    traindataneig=torch.cat((traindata[2],temp[2][:,random_indices,:,:]),dim=1)
    traindataerr=torch.cat((traindata[3],temp[3][random_indices]),dim=0)
    traindatakl=torch.cat((traindata[4],temp[4][random_indices]),dim=0)
    traindataidtrack = np.concatenate([np.asarray(traindata[6]), np.asarray(temp[5][random_indices])], axis=0)
    traindatacluster=np.concatenate([np.asarray(traindata[5]), np.asarray(temp[6][random_indices])], axis=0)
    traindatataskno=np.concatenate([np.asarray(traindata[7]), np.asarray(temp[7][random_indices])], axis=0)
    traindatascore=np.concatenate([np.asarray(traindata[8]), np.asarray(temp[10].cpu().detach().numpy()[random_indices])], axis=0)
    # temp_cluster = torch.as_tensor( temp[6], device=traindata[0].device)
    # traindatacluster=torch.cat((traindata[6],temp_cluster[random_indices]))
    # temp_taskno = torch.as_tensor( temp[7], device=traindata[0].device) 
    # traindatataskno=torch.cat((traindata[7],temp_taskno[random_indices]))
    trainedindex.append(random_indices)
    #traindata.append(temporary)
    return [traindataobs,traindatafut,traindataneig,traindataerr,traindatakl,traindatacluster,traindataidtrack,traindatataskno,traindatascore],trainedindex
def mergeprevioustestdata(test_dataset,prevtestdata):
    #merge prev dataset to test dataset based on distribution on ADE, for nsample
    #fungsi ini mengambil data dari masing masing batch, sebab prevtestdata berisi
    #yt= [y[4].mean() for y in prevtestdata[0]]
        #random_samples=prevtestdata#take all choosen previous data
    temporary=[]
    #for idx,rs in range(prevtestdata):
    #    temp=rs[:3]
    temp=prevtestdata
    temp=[t.cpu().numpy() for t in temp]#numpy conversion
    temptransposex= np.transpose(temp[0], (1, 0, 2))
    temptransposey=np.transpose(temp[1], (1, 0, 2))
    temptransposen=np.transpose(temp[2], (1, 0, 2, 3))#shape conversion
    
    for iddata in range(temptransposex.shape[0]):
        #if(iddata==0):    
        tempx=temptransposex[iddata]#ini isinya masih banyak jadi mbending di loop aja
        tempy=temptransposey[iddata]
        tempn=temptransposen[iddata]
       # else:
       #     tempx=np.concatenate((tempx,temptransposex), axis=0)
       #     tempy=np.concatenate((tempy,temptransposey), axis=0)
       #     tempn=np.concatenate((tempn,temptransposen), axis=0)
# if(method!="experiencereplay"):
#     print('filtered replay!!!')
#     sample=int((sample/100)*tempx.shape[0])
#     print('sample=',sample)
#     #random_samples = random.choices(prevtestdata,weights=yt,k=sample)#salahnya disini ketmu, jadi harusnya yang dipilih itu dari anggota prevtestdata
#     #random_indices = np.random.choice(tempx.shape[0],size=sample,replace=False)#harusnya diambil berdasarkan distribusi
#     selecteddata,random_indices=randomsampletensor(tempx,weights=yt,k=sample)
# else:
#     print('ex replay!!!')
#     random_indices=range(tempx.shape[0])

        hist=tempx
        fut=tempy
        nei=tempn
        perdata=[]
        perdata.append(hist)
        perdata.append(fut)
        perdata.append(nei)
        perdata.append(prevtestdata[3][iddata].cpu().numpy().item())
        perdata=np.array(perdata, dtype=object)
        datainsert=[perdata]
        test_dataset.data=np.concatenate((test_dataset.data, datainsert), axis=0)
        ncl=prevtestdata[5][iddata].cpu().numpy()
        test_dataset.ncluster=np.concatenate((test_dataset.ncluster, np.array([ncl])), axis=0)
        test_dataset.idtrack=np.concatenate((test_dataset.idtrack, np.array([prevtestdata[6][iddata].cpu().numpy().item()])), axis=0)
        test_dataset.idtasks=np.concatenate((test_dataset.idtasks, np.array([prevtestdata[7][iddata].cpu().numpy().item()])), axis=0)
        test_dataset.scores=np.concatenate((test_dataset.scores, np.array([prevtestdata[10][iddata].cpu().numpy().item()])), axis=0)
    return test_dataset
def mergeprevioustestdataREL(test_dataset,prevtestdata):
    temp=prevtestdata
    temp=[t.cpu().numpy() for t in temp]#numpy conversion
    temptransposex= np.transpose(temp[0], (1, 0, 2))
    temptransposey=np.transpose(temp[1], (1, 0, 2))
    temptransposen=np.transpose(temp[2], (1, 0, 2, 3))#shape conversion
    
    for iddata in range(temptransposex.shape[0]):
        #if(iddata==0):    
        tempx=temptransposex[iddata]#ini isinya masih banyak jadi mbending di loop aja
        tempy=temptransposey[iddata]
        tempn=temptransposen[iddata]
        hist=tempx
        fut=tempy
        nei=tempn
        perdata=[]
        perdata.append(hist)
        perdata.append(fut)
        perdata.append(nei)
        perdata.append(prevtestdata[3][iddata].cpu().numpy().item())
        perdata=np.array(perdata, dtype=object)
        datainsert=[perdata]
        test_dataset.data=np.concatenate((test_dataset.data, datainsert), axis=0)
        ncl=prevtestdata[5][iddata].cpu().numpy()
        test_dataset.ncluster=np.concatenate((test_dataset.ncluster, np.array([ncl])), axis=0)
        test_dataset.idtrack=np.concatenate((test_dataset.idtrack, np.array([prevtestdata[6][iddata].cpu().numpy().item()])), axis=0)
        test_dataset.idtasks=np.concatenate((test_dataset.idtasks, np.array([prevtestdata[7][iddata].cpu().numpy().item()])), axis=0)
        test_dataset.scores=np.concatenate((test_dataset.scores, np.array([prevtestdata[7][iddata].cpu().numpy().item()])), axis=0)
    return test_dataset
def mergeprevioustestdata2(test_dataset,prevtestdata):
    #merge prev dataset to test dataset based on distribution on ADE, for nsample
    #fungsi ini mengambil data dari masing masing batch, sebab prevtestdata berisi
    for perdata in prevtestdata:
        test_dataset.data=np.concatenate((  test_dataset.data, perdata), axis=0)
    return test_dataset

def randomsampletensor(tensor_list, weights, k):
    #weighted_tensor_sample_without_replacement
    assert len(tensor_list) == len(weights), "Tensor list and weights must be same length"
    assert k <= len(tensor_list), "Sample size can't be larger than data size"

    weights = [float(w) for w in weights]
    total = sum(weights)
    probabilities = [w / total for w in weights]

    data_copy = tensor_list.clone().tolist()
    index_copy = list(range(len(tensor_list)))
    prob_copy = probabilities[:]

    sampled = []
    sampled_indices = []

    for _ in range(k):
        idx= random.choices(range(len(data_copy)), weights=prob_copy, k=1)[0]
        choice=data_copy[idx]
        #idx = next(i for i, t in enumerate(data_copy) if t is choice)

        sampled.append(choice)
        sampled_indices.append(index_copy[idx])

        # Remove chosen entry
        del data_copy[idx]
        del index_copy[idx]
        del prob_copy[idx]

        # Re-normalize probabilities
        if prob_copy:
            total = sum(prob_copy)
            prob_copy = [w / total for w in prob_copy]

    return sampled, sampled_indices
def ADE_FDE(y_, y, batch_first=False):
    # average displacement error
    # final displacement error
    # y_, y: S x L x N x 2
    if torch.is_tensor(y):
        err = (y_ - y).norm(dim=-1)
    else:
        err = np.linalg.norm(np.subtract(y_, y), axis=-1)
    if len(err.shape) == 1:
        fde = err[-1]
        ade = err.mean()
    elif batch_first:
        fde = err[..., -1]
        ade = err.mean(-1)
    else:
        fde = err[..., -1, :]
        ade = err.mean(-2)
    return ade, fde

def kmeans(k, data, iters=None):
    centroids = data.copy()
    np.random.shuffle(centroids)
    centroids = centroids[:k]

    if iters is None: iters = 100000
    for _ in range(iters):
    # while True:
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        closest = np.argmin(distances, axis=0)
        centroids_ = []
        for k in range(len(centroids)):
            cand = data[closest==k]
            if len(cand) > 0:
                centroids_.append(cand.mean(axis=0))
            else:
                centroids_.append(data[np.random.randint(len(data))])
        centroids_ = np.array(centroids_)
        if np.linalg.norm(centroids_ - centroids) < 0.0001:
            break
        centroids = centroids_
    return centroids

def FPC(y, n_samples):
    # y: S x L x 2
    goal = y[...,-1,:2]
    goal_ = kmeans(n_samples, goal)
    dist = np.linalg.norm(goal_[:,np.newaxis,:2] - goal[np.newaxis,:,:2], axis=-1)
    chosen = np.argmin(dist, axis=1)
    return chosen
    
def seed(seed: int):
    rand = seed is None
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = not rand
    torch.backends.cudnn.benchmark = rand

def get_rng_state(device):
    return (
        torch.get_rng_state(), 
        torch.cuda.get_rng_state(device) if torch.cuda.is_available and "cuda" in str(device) else None,
        np.random.get_state(),
        random.getstate(),
        )

def set_rng_state(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None: torch.cuda.set_rng_state(state[1], device)
    np.random.set_state(state[2])
    random.setstate(state[3])
