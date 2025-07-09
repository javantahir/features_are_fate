import os
import torch
import numpy as np 
from tqdm import tqdm 
import argparse
import random
import pickle
from copy import deepcopy

import train 
from model import deepLinear

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

    #define model
    model = deepLinear(L=args.layers, d=args.d, instances=1, alpha=args.init_scale, init=args.init)
    transfer_model = deepcopy(model)
    transfer_model.duplicate(args.instances)
    torch.save(transfer_model.state_dict(), args.save_path + exp_name + '-init_model.pth') #save stacked parallel model 

    #define source and target tasks
    beta_src = torch.randn((args.d,1))
    beta_src /= torch.linalg.norm(beta_src, keepdim=True)
    beta_orth = torch.randn((args.d,1))
    beta_orth -= beta_orth.T  @ beta_src * beta_src
    beta_orth /= torch.linalg.norm(beta_orth, keepdim=True)
    theta = 0.5*np.pi*args.theta #args.theta is in units of pi/2
    beta_tar = np.cos(theta)*beta_src + np.sin(theta)*beta_orth
    Beta_tar = torch.tile(beta_tar.unsqueeze(0), (args.instances,1,1))

    #train source task 
    print("pretraining\n")
    if args.n_src:
        model.duplicate(args.instances)
        Beta_src = torch.tile(beta_src.unsqueeze(0), (args.instances,1,1))
        X_src, y_src, dataset_src, dataloader_src = train.makeGaussianLinearDataset((args.instances, args.n_src, args.d), Beta_src.cpu(), noise=args.noise_src)
        src_trainer = train.empTrainer(model, Beta_src, param_stream=args.param_stream, device=args.device)
        src_dict = src_trainer.train(epochs=args.epochs, lr=args.lr, dataloader=dataloader_src, save_freq=args.save_freq, min_err=train.minimum_loss(X_src, y_src, device) + args.tol, verbose=args.verbose)
        src_dict.update({"beta_src": beta_src.detach().cpu().numpy(), "beta_tar": beta_tar.detach().cpu().numpy()})
        torch.save(model.state_dict(), args.save_path + exp_name + '-src_model.pth')
        pickle.dump(src_dict, open(args.save_path + exp_name + "-src_dict.p", "wb"))
    else:
        src_trainer = train.popTrainer(model, beta_src, param_stream=args.param_stream, device=args.device)
        src_dict = src_trainer.train(epochs=args.epochs, lr=args.lr, save_freq=args.save_freq, min_err=args.tol, verbose=args.verbose)
        src_dict.update({"beta_src": beta_src.detach().cpu().numpy(), "beta_tar": beta_tar.detach().cpu().numpy()})
        model.duplicate(args.instances)
        torch.save(model.state_dict(), args.save_path + exp_name + '-src_model.pth') #save stacked parallel model 
        pickle.dump(src_dict, open(args.save_path + exp_name + "-src_dict.p", "wb"))
    

    #train transfer 
    for n in args.n_tar:
        save_dict = {}

        X, y, dataset, dataloader = train.makeGaussianLinearDataset((args.instances, n, args.d), Beta_tar.cpu(), noise=args.noise)
        min_err = train.minimum_loss(X, y, device) + args.tol
        #scratch training
        if args.scratch:
            print(f"scratch training n={n}")
            transfer_model.load_state_dict(torch.load(args.save_path + exp_name + '-init_model.pth'))
            scratch_trainer = train.empTrainer(transfer_model, Beta_tar, param_stream=args.param_stream, device=args.device)
            scratch_dict = scratch_trainer.train(epochs=args.epochs, lr=args.lr, dataloader=dataloader, save_freq=args.save_freq, verbose=args.verbose, min_err=min_err, weight_decay=args.weight_decay)
            save_dict.update({"X": X, "y": y, "scratch_dict": scratch_dict})

        #transfer
        if args.fine_tune:
            print(f"fine tuning n={n}")
            transfer_model.load_state_dict(torch.load(args.save_path + exp_name + '-src_model.pth'))
            ft_trainer = train.empTrainer(transfer_model, Beta_tar, param_stream=args.param_stream, device=args.device)
            ft_dict = ft_trainer.train(epochs=args.epochs, lr=args.lr, dataloader=dataloader, save_freq=args.save_freq, verbose=args.verbose)
            save_dict.update({"ft_dict": ft_dict})

        if args.linear_transfer:
            print(f"linear transfer n={n}\n")
            transfer_model.load_state_dict(torch.load(args.save_path + exp_name + '-src_model.pth'))
            transfer_model.to(device)
            X = X.to(device)
            y = y.to(device)
            Beta_tar = Beta_tar.to(device)
            features = X @ transfer_model.getHiddenFeatures()
            lt_dict = {}
            for lam in args.lt_reg :
                if lam > 0:
                    Id = torch.tile(torch.eye(args.d).unsqueeze(0), (args.instances,1,1)).to(device)
                    A = features.transpose(-2,-1) @ features + n * lam * Id
                    c_lin = torch.linalg.inv(A) @ features.transpose(-2,-1) @ y.unsqueeze(-1)
                else:
                    #c_lin,_,_,_ = torch.linalg.lstsq(features, y.unsqueeze(-1))
                    Id = torch.tile(torch.eye(args.d).unsqueeze(0), (args.instances,1,1)).to(device)
                    A = features.transpose(-2,-1) @ features + n * 1e-5 * Id
                    c_lin = torch.linalg.inv(A) @ features.transpose(-2,-1) @ y.unsqueeze(-1)
                transfer_model.layers[-1] = torch.nn.Parameter(c_lin)
                lt_ge = train.population_loss(transfer_model.layers, Beta_tar).detach().cpu().numpy()
                lt_dict.update({lam: {"ge": lt_ge, "c_lt": c_lin.detach().cpu().numpy(), 'beta': transfer_model.getBeta().detach().cpu().numpy()}})
            save_dict.update({"lt_dict": lt_dict})

        #save
        pickle.dump(save_dict, open(args.save_path + exp_name + f'-n_tar={n}-dict.p', 'wb'))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--layers", "-l", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--instances", type=int, nargs="?", default="1")
    parser.add_argument("--init_scale", type=float, nargs="?", default=1e-7)
    parser.add_argument("--init", nargs="?", choices=["rand", "zas"], default="rand")
    parser.add_argument("--save_path", type=str, nargs="?", default=os.getcwd() + '/')
    parser.add_argument("--theta", type=float)
    parser.add_argument("--param_stream", action="store_true")
    parser.add_argument("--epochs", type=int, nargs="?", default=1500)
    parser.add_argument("--lr", type=float, nargs="?", default=0.01)
    parser.add_argument("--save_freq", type=int, nargs="?", default=1)
    parser.add_argument("--tol", type=float, nargs="?", default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--noise", type=float, nargs="?", default=0)
    parser.add_argument("--noise_src", type=float, nargs="?", default=0)
    parser.add_argument("--n_tar", type=int, nargs="*")
    parser.add_argument("--n_src", type=int)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--linear_transfer", action="store_true")
    parser.add_argument("--lt_reg", type=float, nargs="*")
    parser.add_argument("--scratch_reg", type=float, nargs="?", default=0)
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int, nargs="?", default=1)
    parser.add_argument("--exp_name", type=str, nargs="?", default="d-n_src")

    args = parser.parse_args()
    main(args)