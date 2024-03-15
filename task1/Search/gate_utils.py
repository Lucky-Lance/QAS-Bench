import torch
import math
import numpy as np
import random
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
def get_CNOT():
    return torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=torch.complex128).to(device=device)


def get_H():
    H = 1 / math.sqrt(2) * torch.tensor([
        [1,  1],
        [1, -1]
    ], dtype=torch.cdouble).to(device=device)
    
    return H

def get_S():
    S = torch.tensor([
        [1, 0],
        [0, 1j]
    ], dtype=torch.cdouble).to(device=device)
    
    return S

def get_T():
    T = torch.tensor([
        [1, 0],
        [0, 1 / math.sqrt(2) * (1 + 1j)]
    ], dtype=torch.cdouble).to(device=device)
    
    return T

def get_I():
    I = torch.tensor([
        [1, 0],
        [0, 1]
    ], dtype=torch.cdouble).to(device=device)
    
    return I

def get_RX(theta):
    if type(theta)!=torch.tensor:
        theta=torch.tensor(theta,dtype=torch.double).to(device)
    a = torch.hstack((torch.cos(theta / 2), -1j * torch.sin(theta / 2)))
    b = torch.hstack((-1j * torch.sin(theta / 2), torch.cos(theta / 2)))
    RX = torch.vstack((a,b)).to(torch.cdouble).to(device)
    
    return RX

def get_RY(theta):
    if type(theta)!=torch.tensor:
        theta=torch.tensor(theta,dtype=torch.double).to(device)
    a = torch.hstack((torch.cos(theta / 2), -1 * torch.sin(theta / 2)))
    b = torch.hstack((torch.sin(theta / 2), torch.cos(theta / 2)))
    RY = torch.vstack((a,b)).to(torch.cdouble).to(device)
    
    return RY

def get_RZ(theta):
    if type(theta)!=torch.tensor:
        theta=torch.tensor(theta,dtype=torch.double).to(device)

    RZ_zero = torch.tensor(0).to(torch.cdouble).to(device)
    
    a = torch.hstack((torch.exp(-1j * theta / 2), RZ_zero))
    b = torch.hstack((RZ_zero, torch.exp(1j * theta / 2)))
    RZ = torch.vstack((a,b)).to(torch.cdouble).to(device)
    
    return RZ
Matrix_dict={'RX':get_RX,'RY':get_RY,'RZ':get_RZ,'I':get_I,'H':get_H,'S':get_S,'T':get_T,'CNOT':get_CNOT}
def gate_matrix(Gate):
    func=Matrix_dict[Gate[0]]
    if len(Gate)==3:
        return func(Gate[2])
    else:
        return func()

def Generate_randomgate(Candidate_set,qubit_num):
    # [ [gatename1,gatequbits1],[gatename2,gatequbits2]... ]
    Gate_name,Gate_qubits= random.choice(Candidate_set)
    # print(Gate_name,Gate_qubits)
    Qubit_List=random.sample(range(qubit_num),Gate_qubits)
    # print(Qubit_List)
    return [Gate_name,Qubit_List]