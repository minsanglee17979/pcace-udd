#!/usr/bin/env python
# coding: utf-8

#**************************************************************************
# Import Statements
#**************************************************************************

#==== standard libraries ====

import sys
import torch
import logging

#==== pcace ====

import pcace
# basis
from pcace.basis import CutoffCos
from pcace.basis import RadialBesselJ
from pcace.basis import AngularBasis
from pcace.basis import AngularProduct
# ml
from pcace.ml import LossHuber, LossAsinh
from pcace.ml import NormT
from pcace.ml import Metrics
from pcace.ml import IERF
# optimization
from pcace.opt import TrainingTask
# mlp
from pcace.mlp import CACE
from pcace.mlp import ANN_SR, ANN_Pauli_Gauss, ANN_LDamp_Long
from pcace.mlp import NNH, NNS

member_seed = 202

# Preserve the RNG state used for ANN initialization.
def make_cace_preserving_rng(*args, **kwargs):
    with torch.random.fork_rng():
        return CACE(*args, **kwargs)

#**************************************************************************
# Global Variables
#**************************************************************************

# == double type ==
torch.set_default_dtype(torch.float64)
# == logging ==
pcace.tools.setup_logger(level='INFO')
# == device ==
device_str = 'cuda'
device = pcace.tools.init_device(device_str)

# == elements ==
z_list = [18] 
atomic_energies={18: -574.498250374157}

# == basis ==
rc = 6.0 # cutoff radius
nr = 6 # number of radial functions
dre = 6 # dimension - radial embedding
d_node_embed = 1 # node embedding dimension

# == nnh ==
activation = IERF() # activation function

# == loss ==
lw = 1.0e-2
loss_fn_energy = LossAsinh(a=lw)
loss_fn_force  = LossHuber(a=lw)

# == training ==
valid_fraction = 0.1
batch_size = 4
seed = 42
key_data = {
        'energy': 'energy_ref',
        'forces': 'forces_ref'
}
nepoch = 100

print("=========================================================")
print(f"Global Variables")
print(f"device        = {device_str}")
print(f"Elements:")
print(f"z_list        = {z_list}")
print(f"atom_energies = {atomic_energies}")
print(f"d_node_embed  = {d_node_embed}")
print(f"Basis:")
print(f"cutoff_radius = {rc}")
print(f"num_radial    = {nr}")
print(f"dim_radial_e  = {dre}")
print(f"NNH:")
print(f"activation    = {activation}")
print(f"Training:")
print(f"nepoch        = {nepoch}")
print(f"valid_frac    = {valid_fraction}")
print(f"batch_size    = {batch_size}")
print(f"random_seed   = {seed}")
print(f"key_data      = {key_data}")
print("=========================================================")

#**************************************************************************
# Data
#**************************************************************************

#==== load the data set ====
print("loading the data set")

# read data
path_data = sys.argv[1]
subset = pcace.data.read_dataset_xyz(
    # atom properties
    cutoff = rc,
    atomic_energies = atomic_energies,
    # paths
    path_train = path_data,
    path_valid = None,
    path_test  = None,
    # optimization
    valid_fraction = valid_fraction,
    seed = seed,
    key_data = key_data,
)
print(f"ntrain = {len(subset.train)}")
print(f"nval   = {len(subset.valid)}")

#==== create the data loaders ====

print("Creating the data loaders")

loader_train = pcace.data.load_data_loader(
    collection = subset,
    data_type = "train",
    batch_size = batch_size,
)
loader_valid = pcace.data.load_data_loader(
    collection = subset,
    data_type = "valid",
    batch_size = batch_size,
)

#**************************************************************************
# Neural Network Hamiltonian - Short Range
#**************************************************************************

torch.manual_seed(member_seed)

rep_sr = make_cace_preserving_rng(
    # atomic numbers
    z_list = z_list,
    # basis
    cutoff = CutoffCos(rc=rc),
    radial = RadialBesselJ(rc=rc, nr=nr, train=True),
    angular = AngularBasis(3),
    angprod = AngularProduct(2,3),
    # node/edge encoding/embedding
    dim_node_embed = d_node_embed,
    # radial embedding
    dim_radial_embed = dre,
    # message passing
    avg_num_neighbors=1,
)

ann_sr = ANN_SR(
    n_in = rep_sr.n_input,
    n_out = 1,
    n_hidden = [6,6],
    activation = activation,
    skip = False,
)

nnh_sr = NNH(
    rep = rep_sr,
    ann = ann_sr,
)

#**************************************************************************
# Neural Network Hamiltonian - Pauli
#**************************************************************************

rep_pauli = make_cace_preserving_rng(
    # atomic numbers
    z_list = z_list,
    # basis
    cutoff = CutoffCos(rc=rc),
    radial = RadialBesselJ(rc=rc, nr=nr, train=True),
    angular = AngularBasis(0),
    angprod = AngularProduct(1,0),
    # node/edge encoding/embedding
    dim_node_embed = d_node_embed,
    # radial embedding
    dim_radial_embed = dre,
    # message passing
    avg_num_neighbors=1,
)

ann_pauli = ANN_Pauli_Gauss(
    # neural network
    n_in = rep_pauli.n_input,
    n_out = 1,
    n_hidden = [3,3],
    activation = activation,
    skip = False,
    # radii - covalent
    radii = {18: 1.06}, 
    # potential parameters
    rc = rc,
)

nnh_pauli = NNH(
    rep = rep_pauli,
    ann = ann_pauli,
)

#**************************************************************************
# Atomic Neural Network - London (Damped)
#**************************************************************************

rep_ldamp = make_cace_preserving_rng(
    # atomic numbers
    z_list = z_list,
    # basis
    cutoff = CutoffCos(rc=rc),
    radial = RadialBesselJ(rc=rc, nr=nr, train=True),
    angular = AngularBasis(0),
    angprod = AngularProduct(1,0),
    # node/edge encoding/embedding
    dim_node_embed = d_node_embed,
    # radial embedding
    dim_radial_embed = dre,
    # message passing
    avg_num_neighbors=1,
)

ann_ldamp = ANN_LDamp_Long(
    # neural network
    n_in = rep_ldamp.n_input,
    n_out = 1,
    n_hidden = [3,3],
    activation = activation,
    skip = False,
    # radii - vdw
    radii = {18: 3.81},
    # kspace
    prec = 1.0e-6,
    rc = rc,
)

nnh_ldamp = NNH(
    rep = rep_ldamp,
    ann = ann_ldamp,
)

#**************************************************************************
# Neural Network Surface
#**************************************************************************

print("Creating the NNS")

nns = NNS(
    nnhl = torch.nn.ModuleList([
        nnh_pauli,
        nnh_ldamp,
        nnh_sr
    ])
)
logging.info(f"NNS: {nns}")

#**************************************************************************
# Loss/Metrics
#**************************************************************************

print("Creating loss functions")

# loss - energy
loss_energy = pcace.ml.LossMap(
    name_target  = 'energy',
    name_predict = nns.key_energy,
    loss_fn = loss_fn_energy,
    loss_wt = 1.0,
    normT = NormT.LINEAR,
)

# loss - force
loss_force = pcace.ml.LossMap(
    name_target = 'forces',
    name_predict = nns.key_forces,
    loss_fn = loss_fn_force,
    loss_wt = 1.0,
    normT = NormT.NONE,
)

print(loss_energy)
print(loss_force)

print("Creating metric functions")

# metric - energy
metric_energy = Metrics(
    name_target  = 'energy',
    name_predict = nns.key_energy,
    name_metric  = 'e/atom',
    per_atom     = True
)

# metric - force
metric_force = Metrics(
    name_target  = 'forces',
    name_predict = nns.key_forces,
    name_metric  = 'f'
)

print(metric_energy)
print(metric_force)

#**************************************************************************
# Training
#**************************************************************************

print("Creating optimizer")

# ==== optimizer ====
optimizer = torch.optim.NAdam
optimizer_args = {
    'lr': 1e-2,
}

# ==== scheduler ====
scheduler=torch.optim.lr_scheduler.OneCycleLR
scheduler_args = {
    'max_lr' : 1.0e-2,
    'total_steps' : None,
    'epochs' : nepoch,
    'steps_per_epoch' : len(loader_train),
    'pct_start' : 0.3,
    'anneal_strategy' : 'cos',
    'cycle_momentum' : True,
    'base_momentum' : 0.85,
    'max_momentum' : 0.95,
    'div_factor' : 25,
    'final_div_factor' : 40,
    'three_phase' : False,
    'last_epoch' : -1,
}

print(optimizer)
print(optimizer_args)
print(scheduler)
print(scheduler_args)

#**************************************************************************
# Training - Classical
#**************************************************************************

print("Creating training task - Classical")

for param in nns.nnhl[0].parameters(): # pauli
    param.requires_grad = True
for param in nns.nnhl[1].parameters(): # ldamp
    param.requires_grad = True
for param in nns.nnhl[2].parameters(): # sr
    param.requires_grad = False
nns.nnhl[0].ann.weight = torch.tensor(1.0,dtype=torch.get_default_dtype()) # pauli
nns.nnhl[1].ann.weight = torch.tensor(1.0,dtype=torch.get_default_dtype()) # ldamp
nns.nnhl[2].ann.weight = torch.tensor(0.0,dtype=torch.get_default_dtype()) # sr

parameters = []
parameters.extend(nns.nnhl[0].parameters())
parameters.extend(nns.nnhl[1].parameters())
optimizer_args = {
    'params': parameters,
    'lr': 1e-2,
}

task = TrainingTask(
    model = nns,
    losses = [loss_energy, loss_force],
    metrics = [metric_energy, metric_force],
    device = device,
    optimizer_cls = optimizer,
    optimizer_args = optimizer_args,
    scheduler_cls = scheduler,
    scheduler_args = scheduler_args,
    ema = False,
    ema_decay = 0.99,
    ema_start = 10,
    max_grad_norm = None,
    warmup_steps = 0,
)

print("Fitting the model - Classical")

task.fit(
    loader_train,
    loader_valid,
    epochs=nepoch,
    val_stride=1,
    print_stride=1,
)
task.save_model('model_cl.pth')

#**************************************************************************
# Training - SR
#**************************************************************************

print("Creating training task - SR")

for param in nns.nnhl[0].parameters(): # pauli
    param.requires_grad = True
for param in nns.nnhl[1].parameters(): # ldamp
    param.requires_grad = True
for param in nns.nnhl[2].parameters(): # sr
    param.requires_grad = True
nns.nnhl[0].ann.weight = torch.tensor(1.0,dtype=torch.get_default_dtype()) # pauli
nns.nnhl[1].ann.weight = torch.tensor(1.0,dtype=torch.get_default_dtype()) # ldamp
nns.nnhl[2].ann.weight = torch.tensor(1.0,dtype=torch.get_default_dtype()) # sr

parameters = []
parameters.extend(nns.nnhl[0].parameters())
parameters.extend(nns.nnhl[1].parameters())
parameters.extend(nns.nnhl[2].parameters())
optimizer_args = {
    'params': parameters,
    'lr': 1e-2,
}

task = TrainingTask(
    model = nns,
    losses = [loss_energy, loss_force],
    metrics = [metric_energy, metric_force],
    device = device,
    optimizer_cls = optimizer,
    optimizer_args = optimizer_args,
    scheduler_cls = scheduler,
    scheduler_args = scheduler_args,
    ema = False,
    ema_decay = 0.99,
    ema_start = 10,
    max_grad_norm = None,
    warmup_steps = 0,
)

print("Fitting the model - SR")

task.fit(
    loader_train,
    loader_valid,
    epochs=nepoch,
    val_stride=1,
    print_stride=1,
)
task.save_model('model_sr.pth')

#**************************************************************************
# Finish
#**************************************************************************

logging.info(f"Finished")
n_params_train = sum(p.numel() for p in nns.parameters() if p.requires_grad)
logging.info(f"Number of trainable parameters: {n_params_train}")
