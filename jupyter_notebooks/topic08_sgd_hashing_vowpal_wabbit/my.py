import os
from tqdm import tqdm
from time import time
import numpy as np
from sklearn.metrics import mean_absolute_error
import itertools

best_mae = 5
best_str = ''
prev_str = ''

def train_vw_model(train_vw_file, model_filename,
                   ngram=1, bit_precision=27, passes=1,
                   learning_rate=0.5,
                   power_t = 0.5,
                   seed=17, quiet=True):
    init_time = time()
    global prev_str
    vw_call_string = ('vw {train_vw_file} ' + 
                       '-f {model_filename} -b {bit_precision} --random_seed {seed}' +
                      ' -l {learning_rate} --power_t {power_t} --loss_function quantile ' #--loss_function quantile
                     ).format(
                       train_vw_file=train_vw_file, learning_rate=learning_rate, 
                       model_filename=model_filename, bit_precision=bit_precision, seed=seed, power_t=power_t)
    if ngram > 1:
         vw_call_string += ' --ngram={}'.format(ngram)
            
    if passes > 1:
         vw_call_string += ' -k --passes={} --cache_file {}'.format(passes, 
                            model_filename.replace('.vw', '.cache'))
    if quiet:
        vw_call_string += ' --quiet'
    #`bits`, `learning_rate` и `power_t`
    
    print(vw_call_string) 
    prev_str = vw_call_string
    res = os.system(vw_call_string)
    print('Success. Elapsed: {} sec.'.format(round(time() - init_time, 2))
          if not res else 'Failed.')

def test_vw_model(model_filename, test_vw_file, prediction_filename,
                  true_labels, seed=17, quiet=True):
    global best_mae
    global best_str
    init_time = time()
    vw_call_string = ('vw -t -i {model_filename} {test_vw_file} ' + 
                       '-p {prediction_filename} --random_seed {seed}').format(
                       model_filename=model_filename, test_vw_file=test_vw_file, 
                       prediction_filename=prediction_filename, seed=seed)
    if quiet:
        vw_call_string += ' --quiet'
        
    print(vw_call_string) 
    res = os.system(vw_call_string)
    
    if not res: # the call resulted OK
        vw_pred = np.loadtxt(prediction_filename)
        print("MAE: {}. Elapsed: {} sec.".format(
            round(mean_absolute_error(true_labels, vw_pred), 6), 
            round(time() - init_time, 2)))
    else:
        print('Failed.')
    if best_mae > round(mean_absolute_error(true_labels, vw_pred), 6):
        best_mae = round(mean_absolute_error(true_labels, vw_pred), 6)
        best_str = prev_str
        print('MAE = ', best_mae)
        print('str:\n', best_str)
y_valid = np.loadtxt('train_zm_valid_labels_main.vw')

for i, (ngram, passes, bit_precision, learning_rate, power_t) in tqdm(enumerate(itertools.product([2],[1,3,5],[27],[0.1817],[0.855]))):
    train_vw_model('train_zm_main.vw', 
                   'vw_model{}_part.vw'.format(i), 
                   ngram=ngram, passes=passes,
                   bit_precision = bit_precision,
                   learning_rate = learning_rate, power_t = power_t, 
                   seed=17, quiet=True)
    test_vw_model(model_filename='vw_model{}_part.vw'.format(i), 
              test_vw_file='test_zm_main.vw', 
              prediction_filename='vw_valid_pred{}.csv'.format(i),
              true_labels=y_valid, seed=17, quiet=True)

print('MAE = ', best_mae)
print('str:\n', best_str)