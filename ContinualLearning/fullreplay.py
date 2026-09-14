import torch
import os
import subprocess


homefolder='/home/USER/Documents/projects/MIND_VAE/'
datafolder=os.path.join(homefolder,'data/fullreplay')
#for root, dirs, files in os.walk(datafolder):
#   for dir_name in dirs:
#        folder_path = os.path.join(root, dir_name)
#        print("Processing folder:", folder_path)
trainpath=os.path.join(datafolder,'train')
validationpath=os.path.join(datafolder,'test')
ckptpath=os.path.join(homefolder,'log_eth/fullreplay')
configpath=os.path.join(homefolder,'config/eth.py')
        #mulai dengan fungsi evaluasinya dulu cuy
        #langsung trainingkan dengan hajar training lagi.#ini disebut dengan no replay
        #kalau mau skenario full replay ini agak tricky karena harus menggabungkan dua skenario yang berbeda
        #jadi buat skema buat assign ke agennya dengan dua file
        #atau cek!!!!!!!!!!! bikin dua file dalam satu folder apakah bekerja????????
        #cek apakah bestlost and bestXY ada atau tidak,
        #jika ada load keduanya dengan memanggil previousdata.py
        #ambil data keduanya kemudian ambil juga data setelahnya, lalu di gabung dimasukkan dalam data agen socialVAEnya        
result = subprocess.run(["python", "/home/USER/Documents/projects/MIND_VAE/main.py",'--train',trainpath,'--test',validationpath,'--ckpt',ckptpath,'--config',configpath], capture_output=True, text=True)
print("Script finished running!")
        # print("Output:", result.stdout)
  #  break








































































