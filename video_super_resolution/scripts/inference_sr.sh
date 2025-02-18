#!/bin/bash

# Folder paths
# video_folder_path='./input/video'
video_folder_path='/data/chenxu/codes/star/input/video/dance'

# txt_file_path='./input/text/prompt.txt'
txt_file_path='/data/chenxu/codes/star/input/text/dance/view_05_seq00_crop.txt'

model_path='/data/chenxu/codes/star/pretrained_weight/I2VGen-XL-based/heavy_deg.pt'
model_type='I2VGen-XL-heavy'
# model_path='/data/chenxu/codes/star/pretrained_weight/I2VGen-XL-based/light_deg.pt'
# model_type='I2VGen-XL-light'
# model_path='/data/chenxu/codes/star/pretrained_weight/CogVideoX-5B-based/1/mp_rank_00_model_states.pt'
# model_type='CogVideoX-5B'
echo "Model type: $model_type"

# use model type as the postfix
save_dir="/data/chenxu/codes/star/results/${model_type}"
echo "Save directory: $save_dir"

# Get all .mp4 files in the folder using find to handle special characters
mapfile -t mp4_files < <(find "$video_folder_path" -type f -name "*.mp4")

# Print the list of MP4 files
echo "MP4 files to be processed:"
for mp4_file in "${mp4_files[@]}"; do
    echo "$mp4_file"
done

# Read lines from the text file, skipping empty lines
mapfile -t lines < <(grep -v '^\s*$' "$txt_file_path")

steps=15 # The number of steps for the optimization process. default: 15
upscale=4 # The upscale factor for the super-resolution model. default: 4
frame_length=12 # The number of video frames processed simultaneously during each denoising process. default: 32

# Debugging output
echo "Number of MP4 files: ${#mp4_files[@]}"
echo "Number of lines in the text file: ${#lines[@]}"

# Ensure the number of video files matches the number of lines
if [ ${#mp4_files[@]} -ne ${#lines[@]} ]; then
    echo "Number of MP4 files and lines in the text file do not match."
    exit 1
fi

# Loop through video files and corresponding lines
for i in "${!mp4_files[@]}"; do
    mp4_file="${mp4_files[$i]}"
    line="${lines[$i]}"

    # Extract the filename without the extension
    file_name=$(basename "$mp4_file" .mp4)

    echo "Processing video file: $mp4_file with prompt: $line"

    # Run Python script with parameters
    python \
        ./video_super_resolution/scripts/inference_sr.py \
        --solver_mode 'fast' \
        --steps ${steps} \
        --input_path "${mp4_file}" \
        --model_path ${model_path} \
        --prompt "${line}" \
        --upscale ${upscale} \
        --max_chunk_len ${frame_length} \
        --file_name "${file_name}.mp4" \
        --save_dir ${save_dir}
done

echo "All videos processed successfully."
