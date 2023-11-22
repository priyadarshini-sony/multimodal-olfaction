#!/bin/bash
echo "Welcome"
# CUDA_VISIBLE_DEVICES=1 python main.py --config-name config_MULTIMOLCLRMORDRED

#non-pretrained models
python main.py --config-name config_MLP #Mordred Features w/ MLP
python main.py --config-name config_MPNN #Graph Features w/ MPNN
python main.py --config-name config_TRANS #SMILES w/ Transformer

python main.py --config-name config_MULTI #Mordred + Graph + SMILES with MLP head

#pretrained models
python main.py --config-name config_EMBED #SMILES with pre-trained model
python main.py --config-name config_MOLCLR #Graph with pre-trained model

python main.py --config-name config_MOLCLREMBED_LR #SMILES + Graph with pre-trained Gransformer and GNN. Label balancer
python main.py --config-name config_MOLCLREMBED_MLP #SMILES + Graph with pre-trained Gransformer and GNN. MLP Head
