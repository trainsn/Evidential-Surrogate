python -u eval.py --dsp 28 --loss Evidential --resume models/model_Evidential_600.pth.tar --id 115
# python -u eval.py --dsp 28 --loss MSE --dropout --n-samples 75 --resume models/model_MSE_dp_1400.pth.tar --id 115

# python -u eval.py --dsp 28 --loss Evidential --resume models/model_Evidential_seed0_active_750.pth.tar --active --id 511
python -u eval.py --dsp 28 --loss Evidential --resume models/model_Evidential_seed0_activebase_150.pth.tar --active --id 511