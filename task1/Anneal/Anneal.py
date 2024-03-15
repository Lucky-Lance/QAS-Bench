import argparse
import pickle
from simulation import *
import random
from copy import deepcopy
import sys
from tqdm import tqdm

def Anneal(U, qubit_num,layer_num,gate_type,seed=42):
    
    prepare_seed(seed)
    if gate_type!='single' and qubit_num!=1:
        choose_set_allC = ['H', 'S', 'T', 'I'] + ['CNOT'] 
    else:
        choose_set_allC = ['H','S','T','I']

    Gate_List = [['I',[0]] for _ in range(layer_num)]

    def loss(theU,U):
        return (theU - U).abs().sum().item()
        # return (theU@M.inverse()-layer2matrix(([0], ['I']))).abs().mean().item()

    def suppose_random_change_loss(Gate_List,qubit_num,U):
        '''
        Suppose a random change, return the change and loss.
        '''
        change_pos=random.randint(0,len(Gate_List)-1)
        change_gate=random.choice(choose_set_allC)
        if change_gate!='CNOT':
            change_qubit=[random.randint(0,qubit_num-1)]
        else:
            x=random.randint(0,qubit_num-1)
            y=(x+random.randint(1,qubit_num-1))%qubit_num
            change_qubit=[x,y]
        Final_gate=[change_gate,change_qubit]
        New_Gate_list=deepcopy(Gate_List)
        New_Gate_list[change_pos]=Final_gate
        new_loss=loss(pks_List_to_Matrix(New_Gate_list,qubit_num),U)
        return new_loss,change_pos,Final_gate
            
        
                

    T = 0.1
    alpha = 0.99999
    tt = 0
    prevloss = 10000
    cur_loss = loss(pks_List_to_Matrix(Gate_List,qubit_num),U)
    curbest = deepcopy(Gate_List)
    curbestloss = cur_loss
    for tt in tqdm( range(200000)):
        new_loss, change_pos , change_gate = suppose_random_change_loss(Gate_List,qubit_num,U)
        # print(tt, 'layers =', layers, 'cur_loss =' , cur_loss.item(), 'new_loss =', new_loss.item(), 'layers_0i_hat =', layers_0i_hat, 'layers_1i_hat =', layers_1i_hat, 'T =', T)
        if new_loss < cur_loss or random.random() < np.exp(-(new_loss - cur_loss) / T):
            # print('accept')
            cur_loss = new_loss
            Gate_List[change_pos]=change_gate
            if curbestloss > cur_loss:
                curbestloss = cur_loss
                curbest = deepcopy(Gate_List)
                if curbestloss<1e-10:
                    return curbest,curbestloss
        T *= alpha
        # if cur_loss < 0.00000001:
            # return layers
        if tt != 0 and tt % 20000 == 0:# previous 20w
            print(tt, cur_loss,  'T =', T)
            if abs(prevloss - cur_loss) <= 0.00000001:
                return curbest,curbestloss
            prevloss  = cur_loss
        # tt += 1
        # print(tt,curbest)
    
    return curbest,curbestloss

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproduction')
    parser.add_argument('--qubit_num', type=int, default=5, help='Qubits of the circuits')
    parser.add_argument('--layer_num', type=int, default=125)
    parser.add_argument('--candidate_set',type=str,default='hybrid')
    parser.add_argument('--epoch_num', type=int, default=2000)
    parser.add_argument('--matrix_path',type=str,required=True)
    parser.add_argument('--output_path',type=str,required=True)
    args = parser.parse_args(argv)
    return args


if __name__=='__main__':
    args = get_args(sys.argv[1:])
    U=pickle.load(open(args.matrix_path, "rb")).clone().detach().to(dtype=torch.complex128).to(device=device)
    Best_Circuit,Best_loss=Anneal(U,args.qubit_num,args.layer_num,args.candidate_set,args.seed)
    print(Best_Circuit,Best_loss)
    pickle.dump([Best_Circuit,Best_loss],open(args.output_path,"wb"))
