# trex_xmorl
Implementation of TREX: Trajectory Explanations for Multi-Objective Reinforcement Learning


## Training the MODT Expert

We use the Preference conditioned MODT from the PEDA Framework. The dataset (D4MORL) required to train the expert agent can be found in the original PEDA repository. (https://github.com/baitingzbt/PEDA/)

Update the config files in the "config" directory to set the training and evaluation parameters of the model.

- Run the following command to train the MODT(P).

    `python train_modt.py`


## Running the TREX Framework

Once the MODT is trained, choose the checkpoint of the saved model files and update the config file. 

The TREX framework is run in two parts.

- Run the 'trex_cluster.py' file to generate trajectories, encode and cluster them.

    `python trex_cluster.py`

- To train the complementary policies and attribution analysis, run the 'trex_complemetary_policies' file.

    `python trex_complemetary_policies.py`


## Saved models and artifacts

All the trained models, generated trajectories, embeddings and cluster information will be saved in the default "trained_models" directory. This can be changed in the config file if needed.