GPU_ID=0

TARGETS=("ballerina2")

length_targets=${#TARGETS[@]}

rotation_weight=0.01
scale_weight=0.05
shear_weight=0.1
translation_weight=1
predict_global_frame_deltas=1
lr_base_global=0.0001
lr_local=0.005
SEED=1000

num_iter=1000 # original is 1000
save_vid_iter=100 # original is 100

for (( i=0; i<length_targets; i++ ));
do
    TARGET="${TARGETS[$i]}"
    echo "===== $TARGET ====="
    wandb_run_name="${TARGET}_seed${SEED}"
    echo "${wandb_run_name}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python animate_svg.py \
        --target "svg_input/${TARGET}_scaled1" \
        --output_folder "tests/${wandb_run_name}" \
        --seed $SEED \
        -optim_points \
        -opt_points_with_mlp \
        --lr_base_points "${lr_local}" \
        --num_iter $num_iter \
        --save_vid_iter $save_vid_iter \
        --predict_global_frame_deltas $predict_global_frame_deltas \
        --inter_dim 128 \
        -augment_frames \
        --num_frames 24 \
        --guidance_scale 30 \
        --split_global_loss --guidance_scale_global 40 --lr_base_global "${lr_base_global}" \
        --rotation_weight "${rotation_weight}" \
        --scale_weight "${scale_weight}" \
        --shear_weight "${shear_weight}" \
        --translation_weight "${translation_weight}"
done
