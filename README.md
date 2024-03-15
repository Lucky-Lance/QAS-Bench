
# QAS-bench: rethinking quantum architecture search and a benchmark

## Datasets

You can download the dataset provided in thie paper at <a href='https://drive.google.com/drive/folders/1H4McsQkDzruVmYyhd2PaEep35C49LriB?usp=sharing'>this link</a>. The format of the dataset is as follows:

```
.
├── task1
│   ├── datamatrix
│   └── figure
└── task2
    ├── 2bit_arbit
    ├── 3bit_arbit
    ├── 4bit_arbit
    └── 5bit_arbit
```

## Baselines

We provide our source code for task1 and task2, containing our designed search algorithm, simulated annealing and genetic algorithm.

### For task1, the necessary inputs are qubit_num, layer_num, matrix_path, output_path.

The code can be called as follows:
```
python baseline/task1/Genetic/Genetic.py --qubit_num 2 --layer_num 4 --matrix_path data/task1/1bit/matrix_dense1_task0.pkl --output_path result.pkl
```

```
python baseline/task1/Search/Search.py --qubit_num 2 --layer_num 4 --matrix_path data/task1/1bit/matrix_dense1_task0.pkl --output_path result.pkl
```
```
python baseline/task1/Anneal/Anneal.py --qubit_num 2 --layer_num 4 --matrix_path data/task1/1bit/matrix_dense1_task0.pkl --output_path result.pkl
```

The circuits and calculated losses will be saved in ```result.pkl```.


### For Task2, the necessary inputs are qubit_num,layer_num,matrix_path,output_path.

The code can be called as follows:

```
python baseline/task2/Search/Search.py --qubit_num 3 --layer_num 10 --matrix_path data/task2/3bit_arbit/0/0_train.pkl --output_path result.pkl --valid_path data/task2/3bit_arbit/0/0_valid.pkl 
```


```
python baseline/task2/Genetic/Genetic.py --qubit_num 3 --layer_num 10 --matrix_path data/task2/3bit_arbit/0/0_train.pkl --output_path result.pkl --valid_path data/task2/3bit_arbit/0/0_valid.pkl 
```


```
python baseline/task2/Anneal/Anneal.py --qubit_num 3 --layer_num 10 --matrix_path data/task2/3bit_arbit/0/0_train.pkl --output_path result.pkl --valid_path data/task2/3bit_arbit/0/0_valid.pkl 
```



The circuits and calculated losses will be saved in ```result.pkl```.

### Other baselines

Reinforcement learning method is based on <a href="https://github.com/mostaszewski314/RL_for_optimization_of_VQE_circuit_architectures">this link</a>.

Hybrid Algorithm is based on <a href="https://github.com/yuxuan-du/Quantum_architecture_search">this link</a>.

Differentiable Algorithm is based on <a href="https://tensorcircuit.readthedocs.io/en/latest/tutorials/dqas.html">this link</a>.


## Citation
```
@inproceedings{lu2023qas,
  title={QAS-bench: rethinking quantum architecture search and a benchmark},
  author={Lu, Xudong and Pan, Kaisen and Yan, Ge and Shan, Jiaming and Wu, Wenjie and Yan, Junchi},
  booktitle={International Conference on Machine Learning},
  pages={22880--22898},
  year={2023},
  organization={PMLR}
}
```