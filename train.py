import os
import pickle
import torch
from torch.utils.data import TensorDataset
import numpy as np
import argparse
import random
from tqdm import tqdm
from model import deepLinear

class popTrainer():
    def __init__(self, model, beta, param_stream=False, device='cpu'):
        self.model = model
        #self.device = torch.device(device="cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = get_device(device)
        self.beta = beta.to(self.device)
        self.loss_fn = lambda layers: population_loss(layers,self.beta)
        self.param_stream = param_stream

        self.model.to(self.device)
    
    def makeSaveDict(self, epochs, save_freq):
        n_saves = (epochs-1)//save_freq + 2
        self.save_dict = {
            'loss_stream': torch.zeros((n_saves, self.model.instances)),
            'ge_stream':  torch.zeros((n_saves, self.model.instances)),
            #'param_stream': {l: torch.zeros(n_saves, *list(self.model.layers[l].shape)) for l in range(len(self.model.layers))}
        }
        if self.param_stream:
            self.save_dict['param_stream'] = {l: torch.zeros(n_saves, *list(self.model.layers[l].shape)) for l in range(len(self.model.layers))}
            for l in range(len(self.model.layers)):
                self.save_dict['param_stream'][l][0,:,:,:] = self.model.layers[l].data.clone().detach().cpu()
        

    def save(self, loss, ge, ind):
        with torch.no_grad():
            self.save_dict['loss_stream'][ind, :] = loss
            self.save_dict['ge_stream'][ind, :] = ge
            if self.param_stream:
                for l in range(len(self.model.layers)):
                    self.save_dict['param_stream'][l][ind+1,:,:,:] = self.model.layers[l].data.clone().detach().cpu()

    def formatDict(self, ind):
        self.save_dict['ge_stream'] = self.save_dict['ge_stream'][:ind+1].numpy().astype(np.float32)
        self.save_dict['loss_stream'] = self.save_dict['loss_stream'][:ind+1].numpy().astype(np.float32)
        if self.param_stream:
            for key, value in self.save_dict['param_stream'].items():
                self.save_dict['param_stream'][key] = value[:ind + 1].numpy().astype(np.float32)

    def trainStep(self):
        loss = self.loss_fn(self.model.layers)
        loss.sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def train(self, epochs, lr, save_freq=1, min_err=0, verbose=False, weight_decay=0): #added weight decay
        self.makeSaveDict(epochs, save_freq)
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr, weight_decay=weight_decay)
        min_err_flag = False
        for k in tqdm(range(epochs)):
                loss_val = self.trainStep()
                if k % save_freq == 0:
                    if verbose:
                        tqdm.write(f"step {k}: loss = {loss_val.mean():.5f}") #print average loss value over all the model instances 
                    if torch.all(loss_val <= min_err):
                        min_err_flag = True
                        break
                    else:
                        self.save(loss_val.detach().cpu(), loss_val.detach().cpu(), ind = k//save_freq) 
        self.param_stream = False
        self.save(loss_val.detach().cpu(), loss_val.detach().cpu(), ind=k//save_freq + int(not min_err_flag)) 
        self.formatDict(ind=k//save_freq + int(not min_err_flag))
        return self.save_dict

class empTrainer():
    def __init__(self, model, beta, param_stream=False, device='cpu'):
        self.model = model
        self.device = get_device(device)
        self.beta = beta.to(self.device)
        self.loss_fn =  torch.vmap(lambda yhat,y: ((yhat-y)**2).mean(), in_dims=(0,0))
        self.param_stream = param_stream

        self.model.to(self.device)
    
    def makeSaveDict(self, epochs, save_freq):
        n_saves = (epochs-1)//save_freq + 2
        self.save_dict = {
            'loss_stream': torch.zeros((n_saves, self.model.instances)),
            'ge_stream':  torch.zeros((n_saves, self.model.instances)),
            #'param_stream': {l: torch.zeros(n_saves, *list(self.model.layers[l].shape)) for l in range(len(self.model.layers))}
        }
        if self.param_stream:
            self.save_dict['param_stream'] = {l: torch.zeros(n_saves, *list(self.model.layers[l].shape)) for l in range(len(self.model.layers))}
            for l in range(len(self.model.layers)):
                self.save_dict['param_stream'][l][0,:,:,:] = self.model.layers[l].data.clone().detach().cpu()
        

    def save(self, loss, ge, ind):
        with torch.no_grad():
            self.save_dict['loss_stream'][ind, :] = loss
            self.save_dict['ge_stream'][ind, :] = ge
            if self.param_stream:
                for l in range(len(self.model.layers)):
                    self.save_dict['param_stream'][l][ind+1,:,:,:] = self.model.layers[l].data.clone().detach().cpu()

    def formatDict(self, ind):
        self.save_dict['ge_stream'] = self.save_dict['ge_stream'][:ind+1].numpy().astype(np.float32)
        self.save_dict['loss_stream'] = self.save_dict['loss_stream'][:ind+1].numpy().astype(np.float32)
        if self.param_stream:
            for key, value in self.save_dict['param_stream'].items():
                self.save_dict['param_stream'][key] = value[:ind + 1].numpy().astype(np.float32)

    def trainStep(self):
        self.model.train()
        for X,y in self.dataloader:
            X = X.to(self.device)
            y = y.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            loss.sum().backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    def train(self, epochs, lr, dataloader, save_freq=1, min_err=0, verbose=False, weight_decay=0): #added weight decay
        self.makeSaveDict(epochs, save_freq)
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.dataloader = dataloader
        min_err_flag = False
        for k in tqdm(range(epochs)):
                ge_val = population_loss(self.model.layers, self.beta)
                loss_val = self.trainStep()
                if k % save_freq == 0:
                    if verbose:
                        tqdm.write(f"step {k}: loss = {loss_val.mean():.5f} | ge = {ge_val.mean():.5f}") #print average loss value over all the model instances 
                    if torch.all(loss_val <= min_err):
                        min_err_flag = True
                        break
                    else:
                        self.save(loss_val.detach().cpu(), ge_val.detach().cpu(), ind = k//save_freq) 
        self.param_stream = False
        self.save(loss_val.detach().cpu(), ge_val.detach().cpu(), ind=k//save_freq + int(not min_err_flag)) 
        self.formatDict(ind=k//save_freq + int(not min_err_flag))
        return self.save_dict

def population_loss(layers, beta):
    beta_hat = layers[0]
    for l in range(len(layers)-1):
        beta_hat = beta_hat @ layers[l+1] 
    return torch.linalg.norm((beta - beta_hat).squeeze(), dim=-1)**2

def makeGaussianLinearDataset(size, beta, noise=0):
    X = torch.randn(size)
    #y = (X @ beta).squeeze((-1,-2)) + noise*torch.randn((size[0], size[1]))
    y = (X @ beta + noise * torch.randn((size[0], size[1], 1))).squeeze(-1)
    dataset = TensorDataset(X,y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=size[0])
    return X, y, dataset, dataloader

def minimum_loss(X, y, device='cpu'):
    _, n, d = X.shape
    if n < d:
        return 0
    else:
        proj = X @ torch.linalg.pinv(X)
        yvec = y.unsqueeze(-1)
        minimum = (torch.norm((yvec - proj @ yvec), dim=1)**2) / n 
        return minimum.to(device)

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_experiment_name(args):
    if all(key in vars(args).keys() for key in args.exp_name.split('-')):
        exp_name = '-'.join([f"{key}={vars(args)[key]}" for key in args.exp_name.split('-')])
    else:
        exp_name = args.exp_name
    return exp_name

def get_device(dev):
    if dev == "gpu":
        device = torch.device(device="cuda:0" if torch.cuda.is_available() else "cpu")
    elif dev == 'mps':
        device = torch.device(dev)
    else:
        device = torch.device(device="cpu")
    return device


def main(args):
    seed_everything(args.seed)
    exp_name = make_experiment_name(args)
    device = get_device(args.device) 

    beta = torch.randn((args.d,1))
    beta /= torch.linalg.norm(beta, keepdim=True)
    Beta = torch.tile(beta.unsqueeze(0), (args.instances,1,1))

    if args.pop_train:
        trainer = popTrainer(model, Beta, args.param_stream, args.device)
        dic = trainer.train(epochs=args.epochs, lr=args.lr, save_freq=args.save_freq, min_err=args.tol, verbose=args.verbose, weight_decay=reg)
        dic.update({"beta": beta.detach().cpu().numpy()})

    else:
        for n in args.n:
            save_dict = {}
            for reg in args.reg:
                model = deepLinear(L=args.layers, d=args.d, instances=args.instances, alpha=args.init_scale, init=args.init)
                X, y, dataset, dataloader = makeGaussianLinearDataset((args.instances, n, args.d), Beta.cpu(), noise=args.noise)
                trainer = empTrainer(model, Beta, args.param_stream, args.device)
                min_err = minimum_loss(X,y,device=device) + args.tol
                train_dict = trainer.train(epochs=args.epochs, lr=args.lr, dataloader=dataloader, save_freq=args.save_freq, min_err=min_err, verbose=args.verbose, weight_decay=reg)
                save_dict.update({reg: {"train_dict": train_dict, "X": X, "y": y, "beta": beta.detach().cpu().numpy()}})
                torch.save(model.state_dict(), args.save_path + exp_name + f'-n={n}-reg={reg}-model.pth') 
            pickle.dump(save_dict, open(args.save_path + exp_name + f"-n={n}-dict.p", "wb"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--layers", "-l", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--instances", type=int, nargs="?", default="1")
    parser.add_argument("--init_scale", type=float, nargs="?", default=1e-7)
    parser.add_argument("--init", nargs="?", choices=["rand", "zas"], default="rand")
    parser.add_argument("--save_path", type=str, nargs="?", default=os.getcwd() + '/')
    parser.add_argument("--pop_train", action="store_true")
    parser.add_argument("--param_stream", action="store_true")
    parser.add_argument("--epochs", type=int, nargs="?", default=1500)
    parser.add_argument("--lr", type=float, nargs="?", default=0.01)
    parser.add_argument("--save_freq", type=int, nargs="?", default=1)
    parser.add_argument("--tol", type=float, nargs="?", default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--noise", type=float, nargs="?", default=0)
    parser.add_argument("--n", type=int, nargs="*")
    parser.add_argument("--reg", type=float, nargs="*", default=[0])
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int, nargs="?", default=1)
    parser.add_argument("--exp_name", type=str, nargs="?", default="d")
    
    args = parser.parse_args()
    main(args)

