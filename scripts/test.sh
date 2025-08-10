# Table 8 of depthsplat paper
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
mode=test \
dataset.roots=[/hdd/u202320081001061/acid] \
dataset.view_sampler.index_path=assets/acid_8view.json \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=8 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vits \
checkpointing.pretrained_model=/home/u202320081001061/GaussinFuser/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints/epoch_38-step_50000.ckpt


# Table 8 of depthsplat paper
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
mode=test \
dataset.roots=[/hdd/u202320081001061/acid] \
dataset.view_sampler.index_path=assets/evaluation_index_acid_4v.json \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=4 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitl \
checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
mode=test \
dataset.roots=[/hdd/u202320081001061/acid] \
dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vits \
checkpointing.pretrained_model=/home/u202320081001061/GaussinFuser/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints/epoch_37-step_49000.ckpt


python -m src.main +experiment=re10k \
data_loader.train.batch_size=8 \
dataset.test_chunk_interval=10 \
trainer.max_steps=600000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_monodepth=/home/u202320081001061/GaussinFuser/depthsplat/pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=/home/u202320081001061/GaussinFuser/depthsplat/pretrained/gmflow-scale1-things-e9887eda.pth \
checkpointing.pretrained_model=/home/u202320081001061/GaussinFuser/depthsplat/pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
output_dir=checkpoints/re10k-256x256-depthsplat-small 


python -m src.main +experiment=re10k \
data_loader.train.batch_size=6 \
dataset.test_chunk_interval=10 \
trainer.max_steps=150000 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitl \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitl.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
output_dir=checkpoints/re10k-256x256-depthsplat-large \

python -m src.main +experiment=re10k \
data_loader.train.batch_size=4 \
dataset.test_chunk_interval=10 \
trainer.max_steps=150000 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitb.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
checkpointing.pretrained_model=/home/u202320081001061/GaussinFuser/depthsplat/pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth \
output_dir=checkpoints/re10k-256x256-depthsplat-base

