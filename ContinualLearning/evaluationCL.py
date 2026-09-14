import torch
import os, sys, time
sys.path.append('..')
from utils import ADE_FDE, FPC, get_rng_state, set_rng_state
from data import Dataloader
from social_vae import SocialVAE
import importlib
homeaddress="/home/USER/Documents/projects/MIND_VAE/"
testaddress=os.path.join(homeaddress,"data/continualdata_sandbox/eval")
ckpt=os.path.join(homeaddress,"log_eth/testCL22")
inclusive=None
spec = importlib.util.spec_from_file_location("config", '/home/USER/Documents/projects/MIND_VAE/config/eth.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
device="cuda"
kwargs = dict(
            batch_first=False, frameskip=1,
            ob_horizon=config.OB_HORIZON, pred_horizon=config.PRED_HORIZON,
            device=device, seed=1)
test_dataset = Dataloader(
            testaddress, **kwargs, inclusive_groups=inclusive,
            batch_size=16, shuffle=False
        )

model = SocialVAE(horizon=config.PRED_HORIZON, ob_radius=config.OB_RADIUS, hidden_dim=config.RNN_HIDDEN_DIM)
model.to(device)
state_dict = torch.load(ckpt, map_location=device)
model.load_state_dict(state_dict["model"])

def test(model, test_dataset, device, fpc=1):
    PRED_SAMPLES=20
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
        for x, y, neighbor in test_data:
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
    sys.stdout.write("\r\033[K ADE: {:.4f}; FDE: {:.4f} ({}) -- time: {}s".format(
        ade, fde, fpc_config, 
        int(time.time()-tic))
    )
    print()
    return ade, fde, XY

goTest=test(model,test_dataset,device)