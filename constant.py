import os


dir_base = '/Users/sophie/workspace/data/contest'
active_year = '2023'
active_contest = '2'  # '1' 'dummy'
active_phase = 'B'  # B  'test'

dir_contest = os.path.join(dir_base, active_year, active_contest)
dir_train = os.path.join(dir_contest, 'train')
dir_test = os.path.join(dir_contest, active_phase)
dir_result = os.path.join(dir_contest, 'result')
dir_preprocess = os.path.join(dir_contest, 'preprocess')
dir_model = os.path.join(dir_contest, 'model')

file_name_train = 'train.p'
file_name_test = 'test.p'

random_states = [42, 2024, 29]
active_random_state = 29

LABEL = 'FLAG'
ID = 'CUST_NO'
