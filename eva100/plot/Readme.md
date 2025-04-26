# Evaluation
## Result
* Rode & Sputnik & cuSPARSE : 
    * /home/shijinliang/module/RoDe-main/res_sddmm_h100
    * /home/shijinliang/module/RoDe-main/res_sddmm_a800
* DTC-SpMM:
    * /home/shijinliang/module/DTC-SpMM_ASPLOS24-main/res
## How test Baselines?
### Rode:   
* Compile:
    * cmake .. -DCUDA_ARCHS="80;90" 
* SpMM & SDDMM:
    * python  /home/shijinliang/module/RoDe-main/script/eval_spmm_call_128.py
    * python /home/shijinliang/module/RoDe-main/script/eval_sddmm_call_128.py
* DTC-SpMM:
    * python /home/shijinliang/module/DTC-SpMM_ASPLOS24-main/scripts/DTCSpMM/run_DTC_SpMM_h100.py
### DTC-SpMM:
* Compile:
    * python setup.py install
* SpMM:
    * python /home/shijinliang/module/DTC-SpMM_ASPLOS24-main/scripts/DTCSpMM/run_DTC_SpMM_h100.py
## How test Libra?
* Complie:
    ** cd Libra-source5 & complie.sh
* SpMM & SDDMM:
    * python /home/shijinliang/module/Libra/eva100/kernel/spmm/spmm_h100_fp16_test_args.py
    * python /home/shijinliang/module/Libra/eva100/kernel/sddmm/sddmm_h100_fp16_test_args.py
* End-to-end:
    * python /home/shijinliang/module/Libra/eva100/kernel/gcn/spmm_h100_fp16_test_args.py
    * python /home/shijinliang/module/Libra/eva100/kernel/gat/sddmm_h100_fp16_test_args.py

## Reproduce the Figures and Tables :
### Figure 1: 
python /home/shijinliang/module/Libra/eva100/plot/motivaton/dstribution.py
python /home/shijinliang/module/Libra/eva100/plot/ablation/hybird/typical_profile/plot.py
### Figure 9: 
python /home/shijinliang/module/Libra/eva100/plot/kernel_spmm/h100_fp16_128/plot.py
### Figure 10: 
python /home/shijinliang/module/Libra/eva100/plot/kernel_sddmm/h100_fp16_32/plot.py
### Figure 11:
python /home/shijinliang/module/Libra/eva100/plot/gnn/plot_v2.py
python /home/shijinliang/module/Libra/eva100/plot/gnn/plot_v2_4090.py
### Figure 12:
python /home/shijinliang/module/Libra/eva100/plot/gnn/plot_huge.py
### Table 3:
python /home/shijinliang/module/Libra/eva100/plot/kernel_spmm/h100_fp16_128/profile.py
### Table 4: 
python /home/shijinliang/module/Libra/eva100/kernel/spmm/spmm_h100_fp16_test_nisght.py
### Table 5:
python /home/shijinliang/module/Libra/eva100/plot/kernel_sddmm/h100_fp16_32/profile.py
### Table 6 & 7:
* python /home/shijinliang/module/Libra/eva100/plot/ablation/hybird/count_matrix_strict_spmm.py
* python /home/shijinliang/module/Libra/eva100/plot/ablation/hybird/count_matrix_strict_sddmm.py
### Table 8:
python /home/shijinliang/module/Libra/eva100/plot/ablation/format_NG/plot_all_tf32.py