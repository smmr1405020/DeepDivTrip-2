import args_kdiverse

args_kdiverse.dat_ix = 2
args_kdiverse.FOLD = 5
args_kdiverse.test_index = 1

from kfold_dataset_generator import generate_ds

generate_ds(args_kdiverse.dat_ix, args_kdiverse.FOLD, args_kdiverse.test_index)

from kdiverse_generator import generate_result

generate_result(False, 3)
