python -u eval.py --dsp 28 --loss Evidential --resume models/model_Evidential_600.pth.tar --id 51
python -u eval.py --dsp 28 --loss MSE --dropout --n-samples 75 --resume models/model_MSE_dp_1400.pth.tar --id 51
python -u eval.py --dsp 28 --loss Gaussian --resume models/model_Gaussian_seed0_2000.pth.tar --id 51

python -u eval.py --dsp 28 --loss Evidential --resume models/model_Evidential_seed0_active_750.pth.tar --active --id 51
python -u eval.py --dsp 28 --loss Evidential --resume models/model_Evidential_seed0_active25_150.pth.tar --active --lam 25
python -u eval.py --dsp 28 --loss Evidential --resume models/model_Evidential_seed0_active50_750.pth.tar --active --lam 50 --id 51
python -u eval.py --dsp 28 --loss Evidential --resume models/model_Evidential_seed0_activebase_150.pth.tar --active --lam 0 --id 51
