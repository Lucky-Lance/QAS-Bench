import argparse
import pickle
import sys
from simulation import *
import time
from copy import deepcopy
from tqdm import tqdm
def round_img(img, decimals = 10):
    return round(img.real + img.imag, decimals)
def Matrix_hash(U,Ubase,decimals=10):
    return round_img((U*Ubase).sum().item(),decimals)


def Brute_search(U, qubit_num,gate_type,seed=42,Maxstep=100000):
    prepare_seed(seed)
    print(gate_type)
    if gate_type!='single' and qubit_num>1:
        choose_set_allC = ['H', 'S', 'T'] + ['CNOT'] 
    else:
        choose_set_allC = ['H','S','T']
    print("Candidate set:",choose_set_allC)
    Gate_List_queue = [[]]

    def loss(theU,U):
        return (theU - U).abs().mean().item()
        # return (theU@M.inverse()-layer2matrix(([0], ['I']))).abs().mean().item()

    Hash_matrix1=torch.rand_like(U)
    Matrix_HashList={}

    curbest = deepcopy([])
    curbestloss = loss(pks_List_to_Matrix(curbest,qubit_num),U)
    for step in  tqdm(range(Maxstep)):
        if curbestloss<1e-10:
            break
        for gate in choose_set_allC:
            if gate!='CNOT':
                for Gate_pos in range(qubit_num):
                    Now_Gate_List=deepcopy(Gate_List_queue[step])
                    Now_Gate_List.append([gate,[Gate_pos]])
                    new_matrix=pks_List_to_Matrix(Now_Gate_List,qubit_num)
                    new_loss = loss(new_matrix,U)
                    if new_loss<curbestloss:
                        curbest = deepcopy(Now_Gate_List)
                        curbestloss = new_loss
                        
                    new_hash=Matrix_hash(new_matrix,Hash_matrix1)
                    if Matrix_HashList.get(new_hash) is None:
                        Matrix_HashList[new_hash]=True
                        Gate_List_queue.append(Now_Gate_List)
            else:
                for Gate_pos1 in range(qubit_num):
                    for Gate_pos2 in range(qubit_num):
                        if Gate_pos1==Gate_pos2:
                            continue
                        Now_Gate_List=deepcopy(Gate_List_queue[step])
                        Now_Gate_List.append([gate,[Gate_pos1,Gate_pos2]])
                        new_matrix=pks_List_to_Matrix(Now_Gate_List,qubit_num)
                        new_loss = loss(new_matrix,U)
                        if new_loss<curbestloss:
                            curbest = deepcopy(Now_Gate_List)
                            curbestloss = new_loss

                        new_hash=Matrix_hash(new_matrix,Hash_matrix1)
                        if Matrix_HashList.get(new_hash) is None:
                            Matrix_HashList[new_hash]=True
                            Gate_List_queue.append(Now_Gate_List)
    return curbest,curbestloss


def Bi_search(U, qubit_num,gate_type,seed=42,Maxstep=20000):
    prepare_seed(seed)
    # if Maxtime==-1 and Maxstep==-1:
    #     print("the maxtime have been set to 30s")
    #     Maxtime=30
    if gate_type!='single' and qubit_num>1:
        choose_set_allC = ['H', 'S', 'T'] + ['CNOT'] 
    else:
        choose_set_allC = ['H','S','T']
    print("Candidate set:",choose_set_allC)
    Gate_List_queue = [[]]

    def loss(theU,U):
        return (theU - U).abs().mean().item()
        # return (theU@M.inverse()-layer2matrix(([0], ['I']))).abs().mean().item()

    Hash_matrix1=torch.rand_like(U)
    Matrix_HashList={Matrix_hash(pks_List_to_Matrix(Gate_List_queue[0],qubit_num),Hash_matrix1):deepcopy([])}

    curbest = deepcopy([])
    curbestloss = loss(pks_List_to_Matrix(curbest,qubit_num),U)
    for step in  tqdm( range(Maxstep)):
        if curbestloss<1e-10:
            break
        for gate in choose_set_allC:
            if gate!='CNOT':
                for Gate_pos in range(qubit_num):
                    Now_Gate_List=deepcopy(Gate_List_queue[step])
                    Now_Gate_List.append([gate,[Gate_pos]])
                    new_matrix=pks_List_to_Matrix(Now_Gate_List,qubit_num)
                    new_loss = loss(new_matrix,U)
                    if new_loss<curbestloss:
                        curbest = deepcopy(Now_Gate_List)
                        curbestloss = new_loss
                    
                    Target_hash=Matrix_hash(torch.mm(U,new_matrix.adjoint()),Hash_matrix1)
                    if not (Matrix_HashList.get(Target_hash) is None):
                        print("Find the answer")
                        return Now_Gate_List + Matrix_HashList[Target_hash],loss(pks_List_to_Matrix(Now_Gate_List +[['I',[0]]]+ Matrix_HashList[Target_hash],qubit_num),U)
                        
                    Target_hash=Matrix_hash(torch.mm(new_matrix.adjoint(),U),Hash_matrix1)
                    if not (Matrix_HashList.get(Target_hash) is None):
                        print("Find the answer")
                        return Matrix_HashList[Target_hash]+ Now_Gate_List  ,loss(pks_List_to_Matrix(Matrix_HashList[Target_hash]+[['I',[0]]]+ Now_Gate_List,qubit_num),U)
                        
                    new_hash=Matrix_hash(new_matrix,Hash_matrix1)
                    if Matrix_HashList.get(new_hash) is None:
                        Matrix_HashList[new_hash]=deepcopy(Now_Gate_List)
                        
                        Gate_List_queue.append(Now_Gate_List)
                    
                    
            else:
                for Gate_pos1 in range(qubit_num):
                    for Gate_pos2 in range(qubit_num):
                        if Gate_pos1==Gate_pos2:
                            continue
                        Now_Gate_List=deepcopy(Gate_List_queue[step])
                        Now_Gate_List.append([gate,[Gate_pos1,Gate_pos2]])
                        new_matrix=pks_List_to_Matrix(Now_Gate_List,qubit_num)
                        new_loss = loss(new_matrix,U)
                        if new_loss<curbestloss:
                            curbest = deepcopy(Now_Gate_List)
                            curbestloss = new_loss

                        Target_hash=Matrix_hash(torch.mm(U,new_matrix.adjoint()),Hash_matrix1)
                        if not (Matrix_HashList.get(Target_hash) is None):
                            print("Find the answer")
                            return Now_Gate_List + Matrix_HashList[Target_hash],loss(pks_List_to_Matrix(Now_Gate_List +[['I',[0]]]+ Matrix_HashList[Target_hash],qubit_num),U)
                            
                        Target_hash=Matrix_hash(torch.mm(new_matrix.adjoint(),U),Hash_matrix1)
                        if not (Matrix_HashList.get(Target_hash) is None):
                            print("Find the answer")
                            return Matrix_HashList[Target_hash] + Now_Gate_List  ,loss(pks_List_to_Matrix(Matrix_HashList[Target_hash]+[['I',[0]]]+ Now_Gate_List,qubit_num),U)
                        
                        new_hash=Matrix_hash(new_matrix,Hash_matrix1)
                        if Matrix_HashList.get(new_hash) is None:
                            Matrix_HashList[new_hash]=deepcopy(Now_Gate_List)
                            Gate_List_queue.append(Now_Gate_List)
    return curbest,curbestloss

def valid_loss(vecx,vecy):
    return torch.sum(torch.abs(vecx)*torch.abs(vecy)) ** 2

def valid(Best_Circuit,qubit_num,valid_data):
    U=pks_List_to_Matrix(Best_Circuit,qubit_num)
    losssum=0
    for iter in valid_data[1:]:
        vecx=pks_apply_unitary_bmm_batch(iter['input'],U,list(range(qubit_num)),qubit_num)
        losssum+=valid_loss(vecx,iter['output'])
    return losssum/len(valid_data[1:])
def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproduction')
    parser.add_argument('--qubit_num', type=int, default=5, help='Qubits of the circuits')
    parser.add_argument('--layer_num', type=int, default=125)
    parser.add_argument('--candidate_set',type=str,default='hybrid')
    parser.add_argument('--epoch_num', type=int, default=20000)
    parser.add_argument('--matrix_path',type=str,required=True)
    parser.add_argument('--output_path',type=str,required=True)
    parser.add_argument('--valid_path',type=str,required=True)
    args = parser.parse_args(argv)
    return args

if __name__=='__main__':
    args = get_args(sys.argv[1:])
    U=pickle.load(open(args.matrix_path, "rb"))[0].clone().detach().to(dtype=torch.complex128).to(device=device)
    Best_Circuit,Best_loss=Brute_search(U,args.qubit_num,args.candidate_set,seed=args.seed)
    print(Best_Circuit,Best_loss)
    pickle.dump(Best_Circuit,open(args.output_path,"wb"))

    valid_data=pickle.load(open(args.valid_path,'rb'))
    valid_loss=valid(Best_Circuit,args.qubit_num,valid_data)
    print(Best_Circuit,valid_loss)
    pickle.dump([Best_Circuit,valid_loss],open(args.output_path,"wb"))