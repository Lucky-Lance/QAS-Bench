
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from gate_utils import *
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    
def pks_apply_unitary_bmm(state, mat, wires,n_wires):
    # print(state.requires_grad)
    # exit(0)
    device_wires = wires
    # print(state)
    state=state.reshape([2]*n_wires)
    # print(state)
    devices_dims = [w  for w in device_wires]
    permute_to = list(range(state.dim()))
    # print(permute_to)
    for d in sorted(devices_dims, reverse=True):
        del permute_to[d]
    permute_to = permute_to+devices_dims
    permute_back = list(np.argsort(permute_to))
    
    
    permuted = state.permute(permute_to).reshape(
        [-1,mat.shape[-1],1])

    # if len(mat.shape) > 2:
    #     # both matrix and state are in batch mode
    #     new_state = mat.bmm(permuted)
    # else:
        # matrix no batch, state in batch mode
    bsz = permuted.shape[0]
    expand_shape = [bsz] + list(mat.shape)
    new_state = mat.expand(expand_shape).bmm(permuted)

    new_state = new_state.reshape([2]*n_wires).permute(permute_back).reshape([-1])
    return new_state

def pks_apply_unitary_bmm_batch(state, mat, wires,n_wires):

    device_wires = wires
    # print(state)
    state=state.reshape([-1]+[2]*n_wires)
    # print(state)
    devices_dims = [w+1  for w in device_wires]
    permute_to = list(range(state.dim()))
    # print(permute_to)
    for d in sorted(devices_dims, reverse=True):
        del permute_to[d]
    permute_to = permute_to+devices_dims
    permute_back = list(np.argsort(permute_to))
    
    
    permuted = state.permute(permute_to).reshape(
        [-1,mat.shape[-1],1])

    # if len(mat.shape) > 2:
    #     # both matrix and state are in batch mode
    #     new_state = mat.bmm(permuted)
    # else:
        # matrix no batch, state in batch mode
    bsz = permuted.shape[0]
    expand_shape = [bsz] + list(mat.shape)
    new_state = mat.expand(expand_shape).bmm(permuted)

    new_state = new_state.reshape([-1]+[2]*n_wires).permute(permute_back).reshape([-1]+[2**n_wires])
    return new_state


def pks_apply_unitary_on_matrix(matrix, mat, wires,n_wires):
    # print(matrix)
    return pks_apply_unitary_bmm_batch(matrix.permute([1,0]),mat,wires,n_wires).permute([1,0])


def pks_List_to_Matrix(Gatelist,n_wires):
    A = torch.eye(2**n_wires,dtype=torch.complex128).to(device=device)
    for gate in Gatelist:
        A=pks_apply_unitary_on_matrix(A,gate_matrix(gate),gate[1],n_wires)
    return A

if __name__=="__main__":
    X=torch.tensor(0.1,dtype=torch.float).to(device=device)
    print(pks_List_to_Matrix([['RX',[1],X],['RY',[0],X],['CNOT',[0,1]],['RX',[1],X],['RY',[0],X],['CNOT',[0,1]]],2))

