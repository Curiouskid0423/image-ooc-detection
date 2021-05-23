### Anomaly Detection for Human Face (140k faces training, 10k faces validation)

### Dataset part disabled for Colab training
# gdown --id '1Uj248Ft4CcExkFx1DO-OXhO11H0otDW1' --output data-bin.tar.gz 
# tar zxvf data-bin.tar.gz
# rm data-bin.tar.gz

python main.py 
# python main.py --config brute_force_hp.yaml