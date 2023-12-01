# AutoSG
This repository contains PyTorch Implementation of AutoSG:
  - **AutoSG**: Automatic Stacked Gate to Optimize Embedding for Click-Through Rate Prediction.

Notes: This repository is under debugging. We do not guarantee the reproducibility of our result on current version of code. We are actively debugging the reproducibility issue. Please check our code later.

### Data Preprocessing

You can prepare the Criteo data in the following format. Avazu and KDD12 datasets can be preprocessed by calling its own python file.

```
python dataset/criteo/criteo.py --store_stat --stats PATH_TO_STORE_STATS
		--dataset RAW_DATASET_FILE --record PATH_TO_PROCESSED_DATASET \
		--threshold 2 --ratio 0.8 0.1 0.1 \
```

Then you can find a `stats` folder under the `PATH_TO_STORE_STATS` folder and your processed files in the tfrecord format under the `PATH_TO_PROCESSED_DATASET` folder. You should update line 181-190 in `train.py` and line 200-209 in `evo.py` corresponding.


### Run Simple Stacked Gate(SSG)

Notes: For a fair comparison, you can select a model to run first in supernet.py and the model's corresponding initialization parameters will be 
        saved in model_init.pth. After setting Compare_with_MSG to True in retrain.py, the model will load the parameters in model_init.pth , which 
        can keep the initial parameters consistent.

Running SSG requires the following phase. First is to save the initial parameters of the selected model:
```
python supernet.py --use_gpu True  --mode_main run_ssg --dataset_path $YOUR_DATASET \
        --model_supernet_name $YOUR_MODEL \
        --batch_size 4096 --train_epoch 30 \
        --mlp_dims [1024, 1024, 1024] \
        --optim Adam --learning rate $LR --wd $WD \
        --stacked_num 5 --concat_mlp True \
        --init_name model_init.pth \
        --alpha $ALPHA \
```

Second is running the Simple Stacked Gate module:
```
python retrain.py --use_gpu True  --mode_ --dataset_path $YOUR_DATASET --compare_with_MSG True \  
        --model_supernet_name $YOUR_MODEL \
        --batch_size 4096 --train_epoch 30 \
        --mlp_dims [1024, 1024, 1024] \
        --optim Adam --learning rate $LR --wd $WD \
        --stacked_num 5 --concat_mlp True \
        --alpha $ALPHA \
```

### Run Automatic Stacked Gate(AutoSG)

Running AutoSG requires the following phase. First is to training the supernet and save the supernet's parameters and 
the initial parameters of the selected model:
```
python supernet.py --use_gpu True  --mode_supernet random --dataset_path $YOUR_DATASET \
        --model_supernet_name $YOUR_MODEL \
        --batch_size 4096 --train_epoch 30 \
        --mlp_dims [1024, 1024, 1024] \
        --optim Adam --learning rate $LR --wd $WD \
        --mask_num $Number of Masekd Bits \
        --stacked_num 5 --concat_mlp True \
        --init_name model_init.pth \
        --save_random_name model_random_train.pth \
        --alpha $ALPHA \
```

Second is evolutionary search:
```
python Evolution.py --dataset_path $YOUR_DATASET --compare_with_MSG True \  
        --model_supernet_name $YOUR_MODEL \
        --batch_size 4096 --search_epoch 30 \
        --mlp_dims [1024, 1024, 1024] --embed_dim $embedding_size\
        --optim Adam --learning rate $LR --wd $WD \
        --keep_num 0 --crossover_num 10 --mutation_num 10 --m_prob 0.1 \
        --mask_num $Number of Masekd Bits \
        --stacked_num 5 --concat_mlp True \
        --alpha $ALPHA \
```
Third is retraining the model:
```
python Retrain.py --use_gpu True  --mode_main retrain --dataset_path $YOUR_DATASET --compare_with_MSG True \  
        --model_supernet_name $YOUR_MODEL \
        --batch_size 4096 --search_epoch 30 \
        --mlp_dims [1024, 1024, 1024] \
        --optim Adam --learning rate $LR --wd $WD \
        --mask_num $Number of Masekd Bits \
        --stacked_num 5 --concat_mlp True \
        --alpha $ALPHA \
```

### Some important Hyperparameters
Notes: Due to the sensitivity of AutoSG, we do not guarantee that the following hyper-parameters will be 100% optimal in your own preprocessed dataset. Kindly tune the hyper-parameters a little bit. 
If you encounter any problems regarding hyperparameter tuning, you are welcomed to contact the first author directly.


Here we list and explain some important hyperparameters we used in SSG module and AutoSG framework:


| Hyperparameter | type | explain                                                                  |
|----------------|-----|--------------------------------------------------------------------------|
| Stacked_num    | int | The number of layers in Hierarchical Gating Network(HGN) of SSG module   |
| concat_mlp     | bool| Whether the model uses Concat_MLP or not                                 |
| mask_num       | int | The number of bits to mask in the embedding vector (Only AutoSG uses it) |
| mode_main      | str | This parameter is in retrain.py, which is used to select the type of model. <br/>run_ssg can run SSGmodule directly. <br/>run_original can run the original base model directly. <br/>retrain is used to retrain the model searched by the evolutionary algorithm, which belongs to AutoSG (you need to run supernet.py and Evolution.py first).                                                                   |




