#!/bin/bash

set -e

work_dir=v2v4real/OURS_4Agents_Heter_Simple
adapter_list=("convnext_crop_w_output_wholemap")
modality_list=("m2" "m4")

cp -r "/home/xiangbog/Folder/AdapterCAV/opencood/hypes_yaml/opv2v/MoreModality/$work_dir" \
    "/home/xiangbog/Folder/AdapterCAV/opencood/logs/"

CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --nproc_per_node=2 \
    --use_env opencood/tools/train_ddp.py \
    -y None \
    --model_dir "opencood/logs/$work_dir/protocol" \
    > "opencood/logs/$work_dir/protocol/train.log" 2>&1


for modality in "${modality_list[@]}"; do
    CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --nproc_per_node=2 \
        --use_env opencood/tools/train_ddp.py \
        -y None \
        --model_dir "opencood/logs/$work_dir/local/$modality" \
        > "opencood/logs/$work_dir/local/$modality/train.log" 2>&1
done

source_dir="opencood/logs/$work_dir/protocol"
file=$(find "$source_dir" -type f -name "net_epoch_bestval_at*.pth" | head -n 1)
if [ -z "$file" ]; then
  echo "Best checkpoint of protocol not found. Make sure training the protocol model first."
  exit 1
fi

target_file_adapter="opencood/logs/$work_dir/local_adapter/protocol.pth"
cp "$file" "$target_file_adapter"
echo "$file has been copied and renamed to $target_file_adapter"

target_file_single_eval="opencood/logs/$work_dir/single_eval/protocol"
cp "$file" "$target_file_single_eval"
echo "$file has been copied and renamed to $target_file_single_eval"

# Create symbolic links for each adapter and modality
for adapter in "${adapter_list[@]}"; do
    for modality in "${modality_list[@]}"; do
        mkdir -p "/home/xiangbog/Folder/AdapterCAV/opencood/logs/$work_dir/local_adapter/${adapter}/${modality}"
        ln -s "/home/xiangbog/Folder/AdapterCAV/opencood/logs/$work_dir/local_adapter/protocol.pth" \
            "/home/xiangbog/Folder/AdapterCAV/opencood/logs/$work_dir/local_adapter/${adapter}/${modality}/protocol.pth"
    done
done

for modality in "${modality_list[@]}"; do
    # Copy files for each adapter and modality
    source_dir="opencood/logs/$work_dir/local/$modality"
    
    file=$(find "$source_dir" -type f -name "net_epoch_bestval_at*.pth" | head -n 1)
    if [ -z "$file" ]; then
        echo "Best checkpoint of $modality not found. Make sure training the $modality model first."
        exit 1
    fi

    for adapter in "${adapter_list[@]}"; do
        target_file_adapter="opencood/logs/$work_dir/local_adapter/${adapter}/${modality}/ego.pth"
        cp "$file" "$target_file_adapter"
        echo "$file has been copied and renamed to $target_file_adapter"
        target_file_single_eval="opencood/logs/$work_dir/single_eval/${modality}/"
        cp "$file" "$target_file_single_eval"
        echo "$file has been copied and renamed to $target_file_single_eval"
    done
done



for adapter in "${adapter_list[@]}"; do
    for modality in "${modality_list[@]}"; do
        CUDA_VISIBLE_DEVICES=1 \
        python opencood/tools/train_adapter.py \
            -y None \
            --model_dir "opencood/logs/$work_dir/local_adapter/$adapter/$modality" \
            > "opencood/logs/$work_dir/local_adapter/$adapter/$modality/train.log" 2>&1
    done
done


for adapter in "${adapter_list[@]}"; do
    python opencood/tools/merge_model_w_adapter.py \
        --model_dir "opencood/logs/$work_dir" \
        --adapter_dir $adapter
    CUDA_VISIBLE_DEVICES=1 \
    python opencood/tools/inference_heter_task.py \
        --model_dir "opencood/logs/$work_dir/final_infer/$adapter" \
        --fusion_method intermediate \
        > "opencood/logs/$work_dir/final_infer/$adapter/inference.log" 2>&1 &
done

for modality in "${modality_list[@]}"; do
    CUDA_VISIBLE_DEVICES=1 \
    python opencood/tools/inference_heter_task.py \
        --model_dir "opencood/logs/$work_dir/single_eval/$modality" \
        --fusion_method intermediate \
        > "opencood/logs/$work_dir/single_eval/$modality/inference.log" 2>&1 &
done
