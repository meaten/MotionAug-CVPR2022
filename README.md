# MotionAug

# Prerequisites
```
python3.6.9
```

# Dependencies
- please refer to [DeepMimic_repo](https://github.com/xbpeng/DeepMimic) to install followings
```
BulletPhysics
Eigen
OpenGL
freeglut
glew
swig
MPI
```
For ```BulletPhysics``` installation, do not forget the option ```USE_DOUBLE_PRECISION=OFF```.  
Please edit the ```DeepMimicCore/Makefile``` to specify path to libraries.


- Compile the simulation environment
```
cd DeepMimicCore
make python -j8
```

- Python environment
```
pip install -r requirements.txt
```

# Data preparation
```
bash prepare_data.sh
```
This command will unzip, align Left/Right, split HDM05 motion dataset, convert Npz format.

# Augmentation
Currently, we support following action classes  
kick, punch, walk, jog, sneak, grab, deposit, throw

- IK **without** motion correction
```
python generate_bvh_dataset.py --aug IK_kin --act_class {action class}
```

- VAE **without** motion correction
```
python vae_script.py --act_class {action_class} --gpu {gpu id}
python generate_bvh_dataset.py --aug VAE_kin --act_class {action class}
```

- IK **with** motion correction (take several days to finish)
```
python train_ik.py --act_class {act_class} --num_threads {total cpu threads to use}
python generate_bvh_dataset.py --aug IK_phys --act_class {act_class}
```

- VAE **with** motion correction (take several days to finish)
```
python train_vae.py --act_class {act_class} --num_threads {total cpu threads to use}
python generate_bvh_dataset.py --aug VAE_phys --act_class {act_class}
```

- IK&VAE **with** motion correction & motion debiasing (take several days to finish)
```
python train_ik.py --act_class {act_class} --num_threads {total cpu threads to use}
python train_vae.py --act_class {act_class} --num_threads {total cpu threads to use}
python generate_bvh_dataset.py --aug VAE_phys --act_class {act_class}
python generate_bvh_dataset.py --aug IK_phys --act_class {act_class}
python generate_bvh_dataset.py --aug Fixed_phys --act_class {act_class}
cd evaluate/DTW
python debias.py --debiaser_type NN --phys_data_npz ../dataset/dataset_Fixed_phys_{act_class}.npz --aug_data_npz ../dataset/dataset_VAE_phys_{act_class}.npz --act_class {act_class}
python debias.py --debiaser_type NN --phys_data_npz ../dataset/dataset_Fixed_phys_{act_class}.npz --aug_data_npz ../dataset/dataset_IK_phys_{act_class}.npz --act_class {act_class}
```
# Evaluation
So far, we prepared following augmentation options  
NOAUG, NOISE  
VAE, IK, VAE_IK  
VAE_PHYSICAL, IK_PHYSICAL, VAE_IK_PHYSICAL  
VAE_PHYSICAL_OFFSET_NN, IK_PHYSICAL_OFFSET_NN, VAE_IK_PHYSICAL_OFFSET_NN  

You can choose human motion prediction models from [RNN](https://github.com/enriccorona/human-motion-prediction-pytorch), [GCN](https://github.com/wei-mao-2019/LearnTrajDep), [Transformer](https://github.com/idiap/potr).  
RNN: seq2seq, GCN: GCN, Transformer: transformer  

please edit **aug_mode**, **actions**, **model_types** in ```evaluate/LearnTrajDep/run_train.py```.
```
cd evaluate/LearnTrajDep/
python run_train.py

