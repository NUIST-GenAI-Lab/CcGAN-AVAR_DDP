# [Imbalance-Robust and Sampling-Efficient Continuous Conditional GANs via Adaptive Vicinity and Auxiliary Regularization](https://arxiv.org/abs/XXXX)


--------------------------------------------------------
## Software Requirements
| Item | Version |
|---|---|
| OS | Ubuntu 22.04 |
| CUDA  | 12.8 |
| MATLAB | R2021 |
| Python | 3.12.7 |
| numpy | 1.26.4 |
| scipy | 1.13.1 |
| h5py | 3.11.0 |
| matplotlib | 3.9.2 |
| Pillow | 10.4.0 |
| torch | 2.7.0 |
| torchvision | 0.22.0 |
| accelearate | 1.6.0 |


--------------------------------------------------------
## Datasets
#### RC-49 (64x64)
[RC-49_64x64_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstI0OuDMqpEZA80tRQ?e=fJJbWw) <br />
[RC-49_64x64_BaiduYun_link](https://pan.baidu.com/s/1Odd02zraZI0XuqIj5UyOAw?pwd=bzjf) <br />

#### RC-49-I (64x64)
[RC-49-I_64x64_OneDrive_link](https://1drv.ms/u/c/907562db44a4f6b8/EbJrU1Vc_p9BjgSOeKS8QUgBOZLbGTBsnShRGLXlRC516g?e=scNBPW) <br />
[RC-49-I_64x64_BaiduYun_link](https://pan.baidu.com/s/1DgVy_AdQgFVVRbmTleggrQ?pwd=qfud) <br />

### The preprocessed UTKFace Dataset (h5 file)
#### UTKFace (64x64)
[UTKFace_64x64_Onedrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstIzurW-LCFpGz5D7Q?e=X23ybx) <br />
[UTKFace_64x64_BaiduYun_link](https://pan.baidu.com/s/1fYjxmD3tJG6QKw5jjXxqIg?pwd=ocmi) <br />
#### UTKFace (128x128)
[UTKFace_128x128_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstJGpTgNYrHE8DgDzA?e=d7AeZq) <br />
[UTKFace_128x128_BaiduYun_link](https://pan.baidu.com/s/17Br49DYS4lcRFzktfSCOyA?pwd=iary) <br />
#### UTKFace (192x192)
[UTKFace_192x192_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstY8hLN3lWEyX0lNLA?e=BcjUQh) <br />
[UTKFace_192x192_BaiduYun_link](https://pan.baidu.com/s/1KaT_k21GTdLqqJxUi24f-Q?pwd=4yf1) <br />

### The Steering Angle dataset (h5 file)
#### Steering Angle (64x64)
[SteeringAngle_64x64_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstIyDTDpGA0CNiONkA?e=Ui5kUK) <br />
[SteeringAngle_64x64_BaiduYun_link](https://pan.baidu.com/s/1ekpMJLC0mE08zVJp5GpFHQ?pwd=xucg) <br />
#### Steering Angle (128x128)
[SteeringAngle_128x128_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstJ0j7rXhDtm6y4IcA?e=bLQh2e) <br />
[SteeringAngle_128x128_BaiduYun_link](https://pan.baidu.com/s/1JVBccsr5vgsIdzC-uskx-A?pwd=4z5n) <br />





--------------------------------------------------------
## Preparation (Required!)
Download the evaluation checkpoints (zip file) from [OneDrive](https://1drv.ms/u/s!Arj2pETbYnWQvOQFAot2lzSWwOEgSQ?e=ZokUe5) or [BaiduYun](https://pan.baidu.com/s/1eIUieSpsFCay21ZrEjTCAA?pwd=d3e4), then extract the contents to `./CcGAN-AVAR/evaluation/eval_ckpts`.

--------------------------------------------------------
## Training


### (1) Auxiliary regression model training



### (2) CcGAN-AVAR training



--------------------------------------------------------
## Sampling and Evaluation






--------------------------------------------------------
## Acknowledge
- https://github.com/UBCDingXin/improved_CcGAN
- https://github.com/UBCDingXin/Dual-NDA
- https://github.com/UBCDingXin/CCDM