"""
Run inference after training is complete.
"""
import pandas as pd

import torch
from torch import nn
from data.dataloader import CustomTensorDataset
from torch.utils.data import DataLoader, SequentialSampler


def inference(test_path=None, eval_batch_size=200, checkpoint_path=None, \
            out_file='PREDICTION_FILE.csv', model_type='cnn'):
    # build testing dataloader
    data = torch.tensor(test_path, dtype=torch.float32)
    test_dataset = CustomTensorDataset(data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
    eval_loss = nn.MSELoss(reduction='none')

    # default checkpoint_path: 'last_model_cnn.ckpt'
    if checkpoint_path is None:
        print("Checkpoint path cannot be None.")
        return 
    model = torch.load(checkpoint_path)
    model.eval()

        
    anomality = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader): 
            if model_type in ['cnn', 'vae']:
                img = data.float().cuda()
            else:
                img = data[0].cuda()

            output = model(img)

            if model_type in ['cnn']:
                output = output
            elif model_type in ['vae']: 
                output = output[0]
            
            loss = eval_loss(output, img).sum([1, 2, 3])
            anomality.append(loss)
    
    # Anomaly score (ROC_AUC score)
    anomality = torch.cat(anomality, axis=0)
    anomality = torch.sqrt(anomality).reshape(len(test_path), 1).cpu().numpy()

    df = pd.DataFrame(anomality, columns=['Predicted'])
    df.to_csv(out_file, index_label = 'Id')

