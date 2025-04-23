import torch
import pandas as pd
import csv
import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(current_dir))

df = pd.read_csv(project_dir + '/res/h100/spmm/tf32/filter_h100_spmm_tf32_result_new2_1015_128.csv')
df_libra = pd.read_csv(project_dir + '/eva100/plot/kernel_spmm/h100_fp16_128/rode_spmm_f32_n128.csv')
df_res = pd.merge(df, df_libra, on='dataSet', how='outer') 

#用于存储结果
file_name = project_dir + '/eva100/plot/kernel_spmm/h100_fp16_128/rode_spmm_f32_n256_0215.csv'
head = ['dataSet','rows_','columns_','nonzeros_','sputnik','Sputnik_gflops','cusparse','cuSPARSE_gflops','rode','ours_gflops']

with open(file_name, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(head)
count = 0

start_time = time.time()
for index, row in df.iterrows():
    count+=1
    
    data = [row['dataSet']]
    if data in ['adaptive', 'delaunay_n22', 'rgg_n_2_22_s0'] :
        continue
    with open(file_name, 'a', newline='') as csvfile:
        csvfile.write(','.join(map(str, data)))

    shell_command = project_dir + "/RoDe/build/eval/eval_spmm_f32_n256 " + "/public/home/shijinliang/Rode/data/" + row['dataSet'] + '/' + row['dataSet'] + ".mtx >> " + file_name
    
    print(row['dataSet'])
    subprocess.run(shell_command, shell=True)

    
    
end_time = time.time()
execution_time = end_time - start_time

dimN = 256
# Record execution time.
with open("execution_time_base.txt", "a") as file:
    file.write("spmm-" + str(dimN) + "-" + str(round(execution_time/60,2)) + " minutes\n")
