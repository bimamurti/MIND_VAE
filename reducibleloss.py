from social_vae import SocialVAE
import torch
import numpy as np
class ReducibleLoss: 
    def __init__(self,dataori,settings,config,buffersize,batch_size,selectionparam):
        self.dataori=dataori
        self.settings=settings
        self.config=config
        self.modelProxy=self.initiateModelProxy()
        self.buffersize=buffersize
        self.buffer=[]
        self.bufferid=[]
        self.upd_cnt=0
        self.batch_size=batch_size
        self.selectionparam=selectionparam
        self.seen_samples=0
    def initiateModelProxy(self):
        modelProxy = SocialVAE(horizon=self.config.PRED_HORIZON, ob_radius=self.config.OB_RADIUS, hidden_dim=self.config.RNN_HIDDEN_DIM)
        modelProxy.to(self.settings.device)
        self.optimizer = torch.optim.Adam(modelProxy.parameters(), lr=self.config.LEARNING_RATE)
        return modelProxy
    def calculateLoss(self,item):
        res = self.modelProxy(*item)#ini returnnya err, kl , err itu selisih y sama y' cuman bukan loss total, model loss nya berbeda dengan err
        loss = self.modelProxy.loss(*res)#err,kl di masukkan ke loss, dimana loss itu untuk 16 data per batch, jadi individual lossnya harus diambil dari res
        errindividual=res[0].mean(dim=(0,2))#ini loss per data, mean di ambil dari sample prediksi
        klindividual=res[1].mean(dim=(0,2))
        return errindividual+klindividual
    def convertIndextoBuffer(self,items,i):
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
        cluster=np.array(items[5])
        cluster=cluster[idx]
        items=np.array(items[6])
        idt=items[idx]
        return [traj,future,neighbour,err,kl,cluster,idt]
        
    def updatebuffer(self,items,loss_ori,taskno,idt,batch,epoch):
        loss_diffs=[]
        taskno=int(taskno)
        #idt=idt)
        softmax_fn = torch.nn.Softmax(dim=-1)
        lossproxy=self.calculateLoss(items)
        for i in range(items[0].shape[1]):
            loss_diff =lossproxy[i] - 0.01 * loss_ori[i]#calculate CE+MSE -CE (batch)
            loss_diffs.append(loss_diff)

        normed_scales = softmax_fn(torch.tensor(loss_diffs, dtype=torch.float32)).numpy() * items[0].shape[1]
        # update sample
        # if prediction is not None:
        #     logits_numpy = prediction.numpy()
        # else:
        #     logits_numpy = None
        num_upd = 0

        for i in range(items[0].shape[1]):##update the buffer
            if len(self.bufferid) < self.buffersize:
                if(idt[i] not in np.array(self.bufferid)[:,0]):  # prevent duplicated data######kerjakan bagian ID disini~~~~~~~~~~~~~~~
                 # only update logit
                   #if the data id not in buffer
                   self.bufferid[i][2]=loss_ori[i].item()
                else:
                    self.bufferid.append([idt[i],i,loss_ori[i].item()])
                num_upd += 1
            else:
                # compute different possibility, SELECTION HAPPEN IN HERE
                rand = np.random.randint(0, len(items) + 1)
                if rand < self.buffersize * normed_scales[i]:#if random number < buffer size * normalized scale
                    buffernp=np.array(self.bufferid)
                    if idt[rand] not in buffernp[:,0]:  # prevent duplicated data######kerjakan bagian ID disini~~~~~~~~~~~~~~~
                     # only update logit
                   #if the data id not in buffer
                        # randomly remove existing samples
                        r_pos = np.random.randint(0, self.buffersize)
                        #ori_id = self.buffer[r_pos][0]
                        del self.bufferid[r_pos]
                        #self.id2pos[int(d_ids[i])] = r_pos
                        self.bufferid.append([idt[i],i,loss_ori[i].item()])
                    else:
                        if self.bufferid[rand][2]<loss_ori[i]:
                            self.bufferid[rand][2]=loss_ori[i].item()#cek dulu benarkah no 3 itu loss function
                    num_upd += 1
            self.seen_samples += 1
        self.upd_cnt += num_upd
        self.buffer=self.convertIndextoBuffer(items,self.bufferid)
        if self.upd_cnt >= max(#ubah angka 1 ini dengan nilai bersesuaian#max(##train buffer model##this will determine when to update, maximum between buffersize/number of updates/selection step or 1
            (self.buffersize / (taskno + 1)) / self.selectionparam, 1):
            self.train_buffer_model()
            self.upd_cnt = 0
        return num_upd
    def getdataperbatch(self,i,batch,batchtrained):
        import random
        allowed = [a for a in range(len(self.buffer[3])) if a not in (batchtrained)]
        if(batch<len(allowed)):
            batchtrained =  random.sample(allowed, batch) 
        else:
             batchtrained =  random.sample(allowed, len(allowed))
        return self.convertIndextoBuffer(self.buffer,batchtrained),batchtrained
    def train_buffer_model(self):
        # extract current data
        #dataset = TensorDataset(self.buffer)
        #self.batch_size = 4
        batchtrained=[]
        print("Training buffer model with {} samples".format(len(self.buffer[3])))
        loop=len(self.buffer[3])//self.batch_size
        if(len(self.buffer[3])%self.batch_size)!=0:
            loop+=1


        if loop>=1:
        #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i in range(loop):
                data,batchtrained=self.getdataperbatch(i,self.batch_size,batchtrained)
                output=self.modelProxy(*data)
                loss=self.modelProxy.loss(*output)
                loss["loss"].backward()
                self.optimizer.step()
                self.optimizer.zero_grad()



        # ##end##
        # cur_data = []
        # for di in self.buffer:
        #     d_id = int(di[0])
        #     if d_id not in self.id2task:
        #         new_di = copy.deepcopy(di)
        #         if len(new_di) > 3:  # use new logit to train buffer model
        #             new_di[3] = self.id2logit[d_id]
        #         cur_data.append(new_di)
        # # make train loader
        # temp_train_file = os.path.join(self.local_path, 'temp_train.pkl')
        # with open(temp_train_file, 'wb') as fw:
        #     for di in cur_data:
        #         pickle.dump(di, fw)
        # temp_train_dataset = single_task_dataset.RandomDataset(
        #     seed=self.seed,
        #     data_path=temp_train_file,
        #     transforms=self.transforms,  # the transforms can be altered here
        #     extra_data=None
        # )
        # temp_train_loader = DataLoader(temp_train_dataset, batch_size=32, drop_last=False)
        # # init model and train model
        # init_model = utils.build_model(model_params=self.model_params)
        # trained_model = train_methods_for_selection.train_model(
        #     local_path=self.local_path,
        #     model=init_model,
        #     train_loader=temp_train_loader,
        #     train_params=self.selection_params['cur_train_params'],
        #     eval_loader=None,
        #     eval_mode='none',
        #     verbose=False,
        #     load_best=False,
        #     eval_steps=10
        # )
        # os.remove(temp_train_file)
        # if self.use_cuda:
        #     trained_model.cuda()
        # self.cur_model = trained_model
        # self.cur_model.eval()