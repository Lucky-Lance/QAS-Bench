import argparse
import pickle
import sys
from simulation import *
import random
from copy import deepcopy
from tqdm import tqdm

def get_fitness(Gate_list,U,p,qubit_num):
    assert p == 1 or p == 2, print('p must be 1 or 2')    
    fitness = torch.norm(U - pks_List_to_Matrix(Gate_list,qubit_num), p)
    return fitness

def Genetic(U, qubit_num,layer_num,gate_type,epoch=2000,init_group_size=50,seed=42):
    prepare_seed(seed)
    if gate_type!='single' and qubit_num>1:
        choose_set_allC = [['H',1], ['S',1], ['T',1], ['I',1]] + [['CNOT',2]] 
    else:
        choose_set_allC = [['H',1], ['S',1], ['T',1], ['I',1]]

    init_group = []
    for i in range(init_group_size):
        single_chromosome = [Generate_randomgate(choose_set_allC,qubit_num) for _ in range(layer_num)]
        init_group.append(deepcopy(single_chromosome))

    fitness_best = deepcopy(init_group[0])
    
    for mode in range(2):
        # if mode == 0:
        #     print('pretrain for 2-norm')
        # elif mode == 1:
        #     print('finetune for 1-norm')
        p = 2 - mode
        for _ in tqdm(range(epoch)):
            fitness_list = torch.zeros(init_group_size)
            for i in range(init_group_size):
                fitness_list[i] = get_fitness(init_group[i],U, p,qubit_num)
            ## sort the fitness list
            _, index = fitness_list.sort()
            _, order = index.sort()
            fitness_best_idx = torch.argmin(fitness_list)
            fitness_best.append(fitness_list[fitness_best_idx])
            temp_best = deepcopy(init_group[fitness_best_idx])
            temp_best_score=fitness_list[fitness_best_idx]
            
            
            ## roulette wheel selection
            fitness_list = (torch.max(fitness_list)+0.0001) - fitness_list
            prob = fitness_list / torch.sum(fitness_list)
            assert np.array(torch.sum(prob)) > 0.999 and np.array(torch.sum(prob)) < 1.001
            for i in range(1, len(prob)):
                prob[i] = prob[i] + prob[i - 1]
            parent_list = []
            while True:
                sample_prob = random.random()
                for i in range(len(prob)):
                    if sample_prob < prob[i] and i not in parent_list:
                        parent_list.append(i)
                        break
                if len(parent_list) == init_group_size // 2:
                    break

            ## the best one is always in parent
            if fitness_best_idx not in parent_list:
                parent_list[random.randint(0, len(parent_list) - 1)] = fitness_best_idx
            replace_num = init_group_size // 2
            ## next generation
            
            operation = ['mutation', 'crossover', 'transpostion', 'insertion', 'deletion', 'reverse']
            for parent_idx in range(len(parent_list)):
                chrom_parent = init_group[parent_list[parent_idx]]
                while True:
                    another_parent_idx = random.randint(0, len(parent_list) - 1)
                    if another_parent_idx != parent_idx:
                        chrom_partner = init_group[parent_list[another_parent_idx]]
                        break
                opt_idx = random.randint(0, len(operation) - 1)
                if operation[opt_idx] == 'mutation':
                    new_chrome = chrom_parent.copy()
                    idx = random.randint(0, len(new_chrome) - 1)
                    change_to = Generate_randomgate(choose_set_allC,qubit_num)
                    new_chrome[idx] = change_to
                elif operation[opt_idx] == 'crossover':
                    idx_1 = random.randint(0, len(chrom_parent) - 1)
                    idx_2 = random.randint(0, len(chrom_partner) - 1)
                    new_chrome = chrom_parent[:idx_1] + chrom_partner[idx_2:]
                elif operation[opt_idx] == 'transpostion':
                    idx_1, idx_2 = random.randint(0, len(chrom_parent) - 1), random.randint(0, len(chrom_parent) - 1)
                    if idx_1 > idx_2:
                        temp = idx_1
                        idx_1 = idx_2
                        idx_2 = temp
                    idx_3, idx_4 = random.randint(0, len(chrom_partner) - 1), random.randint(0, len(chrom_partner) - 1)
                    if idx_3 > idx_4:
                        temp = idx_3
                        idx_3 = idx_4
                        idx_4 = temp
                    new_chrome = chrom_parent[:idx_1] + chrom_partner[idx_3:idx_4] + chrom_parent[idx_2:]
                elif operation[opt_idx] == 'insertion':
                    # sample_num = random.randint(1, 4) ## added by 4 at most
                    idx = random.randint(0, len(chrom_parent) - 1)
                    change_to = Generate_randomgate(choose_set_allC,qubit_num)
                    new_chrome = chrom_parent[:idx] + [change_to] + chrom_parent[idx:]
                elif operation[opt_idx] == 'reverse':
                    idx_1, idx_2 = random.randint(0, len(chrom_parent) - 1), random.randint(0, len(chrom_parent) - 1)
                    if idx_1 > idx_2:
                        temp = idx_1
                        idx_1 = idx_2
                        idx_2 = temp
                    new_chrome = chrom_parent[:idx_1] + chrom_parent[idx_1:idx_2][::-1] + chrom_parent[idx_2:]
                elif operation[opt_idx] == 'deletion':
                    idx_1, idx_2 = random.randint(0, len(chrom_parent) - 1), random.randint(0, len(chrom_parent) - 1)
                    if idx_1 > idx_2:
                        temp = idx_1
                        idx_1 = idx_2
                        idx_2 = temp
                    new_chrome = chrom_parent[:idx_1] + chrom_parent[idx_2:]
                else:
                    raise NotImplementedError
                init_group[order[replace_num]] = new_chrome
                replace_num += 1
            ## the best one is always in
            init_group[random.randint(0, len(init_group) - 1)] = temp_best
        
    return temp_best,temp_best_score
    
            

# if __name__=='__main__':
#     A=pks_List_to_Matrix([['CNOT',[0,1]],['H',[2]],['S',[1]],['CNOT',[2,0]],['CNOT',[0,1]],['H',[2]],['S',[1]],['CNOT',[2,0]],['CNOT',[0,1]],['H',[2]],['S',[1]],['CNOT',[2,0]],['CNOT',[0,1]],['H',[2]],['S',[1]],['CNOT',[2,0]],['CNOT',[0,1]],['H',[2]],['S',[1]],['CNOT',[2,0]],['CNOT',[0,1]],['H',[2]],['S',[1]],['CNOT',[2,0]],['CNOT',[0,1]],['H',[2]],['S',[1]],['CNOT',[2,0]]],5)
#     print(Genetic(A,5,125,'hybrid'))

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproduction')
    parser.add_argument('--qubit_num', type=int, default=5, help='Qubits of the circuits')
    parser.add_argument('--layer_num', type=int, default=125)
    parser.add_argument('--candidate_set',type=str,default='hybrid')
    parser.add_argument('--epoch_num', type=int, default=200)
    parser.add_argument('--init_group_size', type=int, default=50)
    parser.add_argument('--matrix_path',type=str,required=True)
    parser.add_argument('--valid_path',type=str,required=True)
    parser.add_argument('--output_path',type=str,required=True)
    args = parser.parse_args(argv)
    return args

def valid_loss(vecx,vecy):
    return torch.sum(torch.abs(vecx)*torch.abs(vecy)) ** 2

def valid(Best_Circuit,qubit_num,valid_data):
    U=pks_List_to_Matrix(Best_Circuit,qubit_num)
    losssum=0
    for iter in valid_data[1:]:
        vecx=pks_apply_unitary_bmm_batch(iter['input'],U,list(range(qubit_num)),qubit_num)
        losssum+=valid_loss(vecx,iter['output'])
    return losssum/len(valid_data[1:])
if __name__=='__main__':
    args = get_args(sys.argv[1:])
    U=pickle.load(open(args.matrix_path, "rb"))[0].clone().detach().to(dtype=torch.complex128).to(device=device)
    Best_Circuit,Best_loss=Genetic(U,args.qubit_num,args.layer_num,args.candidate_set,args.epoch_num,args.init_group_size,args.seed)

    valid_data=pickle.load(open(args.valid_path,'rb'))
    valid_loss=valid(Best_Circuit,args.qubit_num,valid_data)
    print(Best_Circuit,valid_loss)
    pickle.dump([Best_Circuit,valid_loss],open(args.output_path,"wb"))

    
    # U=torch.rand_like(pks_List_to_Matrix([['I',[0]]],5))/5
    # Best_Circuit=Genetic(U,5,125,'dense')
    # print(Best_Circuit)
    