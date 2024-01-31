
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')

import pandas as pd
import numpy  as np
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO


def obj(params : pd.DataFrame) -> np.ndarray:
    return ((params.values - 0.37)**2).sum(axis = 1).reshape(-1, 1)
        
space = DesignSpace().parse([{'name' : 'x', 'type' : 'num', 'lb' : -3, 'ub' : 3}])
opt   = HEBO(space)
for i in range(4):
    rec = opt.suggest(n_suggestions = 4)
    print("rec: ", rec) 
    opt.observe(rec, obj(rec))
    print('After %d iterations, best obj is %.2f' % (i, opt.y.min()))

# params = [
#     {'name' : 'hidden_size', 'type' : 'int', 'lb' : 16, 'ub' : 128},
#     {'name' : 'batch_size',  'type' : 'int', 'lb' : 16, 'ub' : 128},
#     {'name' : 'lr', 'type' : 'pow', 'lb' : 1e-4, 'ub' : 1e-2, 'base' : 10},
#     {'name' : 'use_bn', 'type' : 'bool'},
#     {'name' : 'activation', 'type' : 'cat', 'categories' : ['relu', 'tanh','sigmoid']},
#     {'name' : 'dropout_rate', 'type' : 'num', 'lb' : 0.1, 'ub' : 0.9},
#     {'name' : 'optimizer', 'type' : 'cat', 'categories' : ['sgd', 'adam', 'rmsprop']}
# ]

# space = DesignSpace().parse(params)
# space.sample(5)  # 随机抽样，返回一个pandas
 
 
# hebo_seq = HEBO(space, model_name = 'gpy', rand_sample = 4)
# for i in range(64):
#     rec_x = hebo_seq.suggest(n_suggestions=1) # n_suggestions可以决定每次采样多少个点来并行评估
#     hebo_seq.observe(rec_x, obj(rec_x)) # obj是目标函数
#     if i % 4 == 0:
#         print('Iter %d, best_y = %.2f' % (i, hebo_seq.y.min()))

# conv_hebo_seq = np.minimum.accumulate(hebo_seq.y) # 获取所有结果，用于绘图    