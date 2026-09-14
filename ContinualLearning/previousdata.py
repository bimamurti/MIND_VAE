import torch
import os
import random
import numpy as np
from utils import randomsampletensor

class previousdata:
    def __init__(self,sample,method) -> None:
        #self.prevtraining=[]
       # self.prevvalid=[]
        self.p=0
        self.id=0
        self.adethreshold=10 #ADE theshold should be measured from the best ADE
        self.lossthreshold=15 #loss threshold should be measured from the best loss
        self.sample=sample #percentage of data to be sampled from previous data
        self.method=method
        
    def getFilteredData(self,prevtrainingpath,prevvalidpath):
        self.dataprevtrain=self.__filterLoss(prevtrainingpath)
        self.dataprevvalid=self.__filterADE(prevvalidpath)
    def getRelData(self,prevtrainingpath,prevvalidpath):
        self.dataprevtrain=self.__filterReLoss(prevtrainingpath)
        self.dataprevvalid=self.__filterADEREL(prevvalidpath)
    def getexData(self,prevtrainingpath,prevvalidpath):
        self.dataprevtrain=self.__filterexLoss(prevtrainingpath)
        self.dataprevvalid=self.__filterADEREL(prevvalidpath)    
    def getProposedData(self,prevtrainingpath,prevvalidpath):
        self.dataprevvalid=self.__filterADE(prevvalidpath)
        self.dataprevtrain=self.__filterProposedLoss(prevtrainingpath)
    def getxreplaydata(self,prevtrainingpath,prevvalidpath,sample):
        self.dataprevtrain=self.__filterlossxreplay(prevtrainingpath,sample)
        self.dataprevvalid=self.__filterADE(prevvalidpath)
        #self.dataprevvalid=self.__filterADExreplay2(prevvalidpath,sample)
    def __filterlossxreplay(self, prevtraining,sample):
        dataLoss=[]
        prevtraining,meanloss=prevtraining
        print(len(prevtraining))
        
        sample=int((sample/100)*len(prevtraining))
        print('sample loss=',sample)
        prevtraining = random.choices(prevtraining,k=sample)
        for indexbatch,batchitem in enumerate(prevtraining):
            #print(indexbatch)
            lossdata=batchitem[1][0].permute(1,0,2).squeeze(1)
            dataxy=batchitem[0][0].permute(1,0,2)
            datafu=batchitem[0][1].permute(1,0,2)
            dataneigh=batchitem[0][2].permute(1, 2, 0, 3)
            for indexitem,loss in enumerate(lossdata):#to the loss part
                #print(len(loss))
                #print(indexitem)
                floss=loss.mean()#************kalau mau update perhitungan lossnya
                #if(floss<meanloss):
                dataLoss.append([dataxy[indexitem],datafu[indexitem],dataneigh[indexitem],floss])
        return dataLoss
    
    def __filterADExreplay2(self,prevvalid,sample):
        #fungsi ini mengambil data dari masing masing batch, sebab prevtestdata berisi
        #yt= [y[4].mean() for y in prevtestdata]
        prevtestdata=[]
        prevvalid,bestade,besfde=prevvalid
        #sample=int((sample/100)*len(prevvalid))
      #  print('sample valid=',sample)
       # prevvalid = random.choices(prevvalid,k=sample)
        for batchitem in prevvalid:
            # if(batchitem[3].mean()<self.lossthreshold):
            #filtervalue=batchitem[3]<bestade*(1.5)#ubah disini untuk threshold
            x=batchitem[0]#variable, batch, path
            y=batchitem[1]
            neigh=batchitem[2]
            ade=batchitem[3]
            fde=batchitem[4]
            prevtestdata.append([x,y,neigh,ade,fde])

        for idx,rs in enumerate(prevtestdata):
            temp=rs[:3]
            temp=[t.cpu().numpy() for t in temp]
            temptransposex=np.transpose(temp[0], (1, 0, 2)).tolist()
            temptransposey=np.transpose(temp[1], (1, 0, 2)).tolist()
            temptransposen=np.transpose(temp[2], (1, 0, 2, 3)).tolist()
            if(idx==0):    
                tempx=temptransposex#ini isinya masih banyak jadi mbending di loop aja
                tempy=temptransposey
                tempn=temptransposen
            else:
                tempx=np.concatenate((tempx,temptransposex), axis=0)
                tempy=np.concatenate((tempy,temptransposey), axis=0)
                tempn=np.concatenate((tempn,temptransposen), axis=0)

        print('filtered replay!!!')
        sample=int((sample/100)*tempx.shape[0])
        print('sample=',sample)
         #random_samples = random.choices(prevtestdata,weights=yt,k=sample)#salahnya disini ketmu, jadi harusnya yang dipilih itu dari anggota prevtestdata
        random_indices = np.random.choice(tempx.shape[0], size=sample, replace=False)
        #random_indices=range(tempx.shape[0])
        result=[]
        for i in (random_indices):
            hist=tempx[i]
            fut=tempy[i]
            nei=tempn[i]
            perdata=[]
            perdata.append(hist)
            perdata.append(fut)
            perdata.append(nei)
            perdata=np.array([perdata], dtype=object)
            result.append(perdata)
        return result
    def __filterADExreplay(self, prevvalid,sample):
        #digunakan untuk filter validation data menggunakan random sebanyak sample
        #input : previous validation data
        #output: data validasinya yang sudah difilter diikuti dengan nilai ade dasarnya, [history,future,neighbour,losscore], untuk perbandingannya mungkin lebih mudah menggunakan nilai mean
        
        dataADEFDE=[]
        prevvalid,bestade,besfde=prevvalid
        #sample=int((sample/100)*len(prevvalid))
      #  print('sample valid=',sample)
       # prevvalid = random.choices(prevvalid,k=sample)
        for batchitem in prevvalid:
            # if(batchitem[3].mean()<self.lossthreshold):
            filtervalue=batchitem[3]<bestade*(1.5)#ubah disini untuk threshold
            x=batchitem[0][:,filtervalue,:]#variable, batch, path
            y=batchitem[1][:,filtervalue,:]
            neigh=batchitem[2][:,filtervalue,:]
            ade=batchitem[3][filtervalue]
            fde=batchitem[4][filtervalue]
            dataADEFDE.append([x,y,neigh,ade,fde])
            #     torch.vstack((dataTrainloss,batchitem))
            #for ade in batchitem[3]:
            #     if(ade<self.lossthreshold):
                     #dataTrainloss.append(batchitem)
        return dataADEFDE
    def __filterLoss(self, prevtraining):
        #digunakan untuk sampling nilai lossnya menggunakan minimum loss
        #input : previous training data with loss score
        #output: data trainingnya yang sudah difilter dalam bentuk numpy aja
        yt= prevtraining[3]
        #s=self.sample
        if(self.method!="experiencereplay"):
            sample=int((self.sample/100)*prevtraining[0].shape[1])
            random_samples,indices=randomsampletensor(prevtraining[3],weights=yt,k=sample)
            #random_samples = np.random.choice(prevtrainingdata,size=sample,replace=False,p=yt)
        else:
            #cpu_data = prevtrainingdata.cpu().numpy()
            sample=int((self.sample/100)*prevtraining[0].shape[1])
            traincount=prevtraining[0].shape[1]
            w = [1]*len(prevtraining)
            random_samples,idx=randomsampletensor(prevtraining,weights=w,k=sample)
            #random_samples=random.sample(prevtrainingdata,k=sample)
            #random.choices(prevtrainingdata,k=sample)
            # random_samples=prevtrainingdata
        temporary=[]#random_samples[:,:2]
        indices=torch.tensor(indices,dtype=torch.long)
        traindata0=prevtraining[0][:,indices,:]
        traindata1=prevtraining[1][:,indices,:]
        traindata2=prevtraining[2][:,indices,:,:]
        #indices=[1,2]
        traindata3=prevtraining[3][indices]
        traindata4=prevtraining[4][indices]
        traindata5=prevtraining[5][indices]
        traindata6=prevtraining[6][indices]
        traindata7=prevtraining[7][indices]
        dataLoss=[traindata0,traindata1,traindata2,traindata3,traindata4,traindata5,traindata6,traindata7]

        # for indexbatch,batchitem in enumerate(prevtraining):
        #     #print(indexbatch)
        #     #lossdata=batchitem[1][0].permute(1,0,2).squeeze(1)
        #     dataxy=batchitem[0].permute(1,0,2)
        #     datafu=batchitem[1].permute(1,0,2)
        #     dataneigh=batchitem[2].permute(1, 2, 0, 3)
        #     for indexitem,loss in enumerate(lossdata):#to the loss part
        #         #print(len(loss))
        #         #print(indexitem)
        #         floss=loss.mean()#************kalau mau update perhitungan lossnya
        #         if(floss<meanloss):
        #             dataLoss.append([dataxy[indexitem],datafu[indexitem],dataneigh[indexitem],floss])
        return dataLoss
    def __filterProposedLoss(self, prevtraining):
        dataLoss=prevtraining#[prevtraining[0],prevtraining[1],prevtraining[2],prevtraining[3],prevtraining[4],prevtraining[5],prevtraining[6],prevtraining[7]]
        return dataLoss
    def __filterADE(self, prevvalid):
        #digunakan untuk filter validation data menggunakan minimum ADE
        #input : previous validation data
        #output: data validasinya yang sudah difilter diikuti dengan nilai ade dasarnya, [history,future,neighbour,losscore], untuk perbandingannya mungkin lebih mudah menggunakan nilai mean
        
        dataADEFDE=[]
        prevvalid,bestade,besfde=prevvalid
            # if(batchitem[3].mean()<self.lossthreshold):
        prevdatavalid=prevvalid
        sample=int((self.sample/100)*prevdatavalid[0].shape[1])
        weights=1/(prevdatavalid[3] + 1e-8) # semakin kecil ade semakin besar bobotnya, jadi lebih besar kemungkinan untuk terpilih
        filtervalue,filteredid=randomsampletensor(prevdatavalid[3],weights=prevdatavalid[3],k=sample) #batchitem[3]<bestade*(1.5)#ubah disini untuk threshold
        x=prevdatavalid[0][:,filteredid,:]#variable, batch, path
        y=prevdatavalid[1][:,filteredid,:]
        neigh=prevdatavalid[2][:,filteredid,:,:]
        ade=prevdatavalid[3][filteredid]
        fde=prevdatavalid[4][filteredid]
        ncluster=prevdatavalid[5][filteredid]
        idtracks=prevdatavalid[6][filteredid]
        idtasks=prevdatavalid[7][filteredid]
        weightbcsr=prevdatavalid[8][filteredid]
        weightREL=prevdatavalid[9][filteredid]
        finalweight=prevdatavalid[10][filteredid]
        dataADEFDE=[x,y,neigh,ade,fde,ncluster,idtracks,idtasks,weightbcsr,weightREL,finalweight]
            #     torch.vstack((dataTrainloss,batchitem))
            #for ade in batchitem[3]:
            #     if(ade<self.lossthreshold):
                     #dataTrainloss.append(batchitem)
        return dataADEFDE
    def __filterADEREL(self, prevvalid):
        #digunakan untuk filter validation data menggunakan minimum ADE
        #input : previous validation data
        #output: data validasinya yang sudah difilter diikuti dengan nilai ade dasarnya, [history,future,neighbour,losscore], untuk perbandingannya mungkin lebih mudah menggunakan nilai mean
        
        dataADEFDE=[]
        prevvalid,bestade,besfde=prevvalid
            # if(batchitem[3].mean()<self.lossthreshold):
        prevdatavalid=prevvalid
        sample=int((self.sample/100)*prevdatavalid[0].shape[1])
        weights=1/(prevdatavalid[3] + 1e-8)  # semakin kecil ade semakin besar bobotnya, jadi lebih besar kemungkinan untuk terpilih
        filtervalue,filteredid=randomsampletensor(prevdatavalid[3],weights=prevdatavalid[3],k=sample) #batchitem[3]<bestade*(1.5)#ubah disini untuk threshold
        x=prevdatavalid[0][:,filteredid,:]#variable, batch, path
        y=prevdatavalid[1][:,filteredid,:]
        neigh=prevdatavalid[2][:,filteredid,:,:]
        ade=prevdatavalid[3][filteredid]
        fde=prevdatavalid[4][filteredid]
        ncluster=prevdatavalid[5][filteredid]
        idtracks=prevdatavalid[6][filteredid]
        idtasks=prevdatavalid[7][filteredid]
        #weightbcsr=prevdatavalid[8][filteredid]
        #weightREL=prevdatavalid[8][filteredid]#return [traj, future, neighbour, err, kl, idt, ncluster, taskno, lossscore]
        #finalweight=prevdatavalid[9][filteredid]
        dataADEFDE=[x,y,neigh,ade,fde,ncluster,idtracks,idtasks]
            #     torch.vstack((dataTrainloss,batchitem))
            #for ade in batchitem[3]:
            #     if(ade<self.lossthreshold):
                     #dataTrainloss.append(batchitem)
        return dataADEFDE
    def __filterexLoss(self, prevtraining):
        dataLoss=[prevtraining[0],prevtraining[1],prevtraining[2],prevtraining[3],prevtraining[4],prevtraining[5],prevtraining[6],prevtraining[7],prevtraining[8]]
        return dataLoss
    def __filterReLoss(self, prevtraining):
        #digunakan untuk sampling nilai lossnya menggunakan minimum loss
        #input : previous training data with loss score
        #output: data trainingnya yang sudah difilter dalam bentuk numpy aja
        dataLoss=[prevtraining[0],prevtraining[1],prevtraining[2],prevtraining[3],prevtraining[4],prevtraining[5],prevtraining[6],prevtraining[7],prevtraining[8]]

        # for indexbatch,batchitem in enumerate(prevtraining):
        #     #print(indexbatch)
        #     #lossdata=batchitem[1][0].permute(1,0,2).squeeze(1)
        #     dataxy=batchitem[0].permute(1,0,2)
        #     datafu=batchitem[1].permute(1,0,2)
        #     dataneigh=batchitem[2].permute(1, 2, 0, 3)
        #     for indexitem,loss in enumerate(lossdata):#to the loss part
        #         #print(len(loss))
        #         #print(indexitem)
        #         floss=loss.mean()#************kalau mau update perhitungan lossnya
        #         if(floss<meanloss):
        #             dataLoss.append([dataxy[indexitem],datafu[indexitem],dataneigh[indexitem],floss])
        return dataLoss
#moif get dataclusternya dulu ciiin buwat average score ajee
if __name__ == "__main__":
    loaded_tensorXYvalid = torch.load("log_eth/sandboxlogExReplay7/logtask2/bestXY.pt")
    loaded_tensorlosstraining = torch.load("log_eth/sandboxlogExReplay7/logtask2/bestloss.pt",weights_only=False)
    p=previousdata()
    p.getxreplaydata(loaded_tensorlosstraining,loaded_tensorXYvalid,20)   
    longterm_memory=[]
    longterm_memory.append(p)
    bestaddress=os.path.join("log_eth/sandboxlogExReplay7/memorybuffer", "longterm2.pt")  
    torch.save(longterm_memory, bestaddress)

#masalah dengan augmentasinya gmn hayoooo
#yuk filter yuk 
#print(loaded_tensorloss)
#hilangkan augmentasinya ajaa cuyyyyyyyyyyyyyyyy

