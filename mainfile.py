import args_kdiverse

args_kdiverse.dat_ix = 3
args_kdiverse.FOLD = 5
args_kdiverse.test_index = 3
args_kdiverse.copy_no = 0

N_min = 5
N_max = 5

from kfold_dataset_generator import generate_ds

generate_ds(args_kdiverse.dat_ix, args_kdiverse.FOLD, args_kdiverse.test_index, args_kdiverse.copy_no)

from kdiverse_generator import generate_result

generate_result(False, K=3, N_min=N_min, N_max=N_max)
