
gt_json=`realpath $1`

segway_dir=/n/groups/htem/Segmentation/shared-nondev/segway

# check if path is correct
if [[ $PYTHONPATH != */n/groups/htem/Segmentation/shared-nondev* ]]; then
    echo INFO: PYTHONPATH env does not have segway... adding it
    export PYTHONPATH=$PYTHONPATH:/n/groups/htem/Segmentation/shared-nondev
fi

alias 01_copy_raw_from_cb2_v2="python ${segway_dir}/gt_scripts/make_zarr_from_cb2_v2_zarr.py ${gt_json}"
alias 01_check_raw="python -i ${segway_dir}/gt_scripts/ng_check_raw.py ${gt_json}"

alias 02_run_segmentation="python ${segway_dir}/tasks/task_05_extract_segmentation_from_lut_blockwise.py ${gt_json}"
alias 02_check_segmentation="python -i ${segway_dir}/tools/ng_check_segmentation.py $gt_json"
alias print_catmaid_location="python ${segway_dir}/gt_scripts/print_catmaid_coordinate.py ${gt_json}"

alias 04_fetch_skeleton="python ${segway_dir}/gt_scripts/fetch_skeletons_cb2_from_json.py ${gt_json}"
alias 04_run_skeleton_correction="python ${segway_dir}/gt_scripts/fix_gt_with_skeleton.py ${gt_json}"
alias 04_check_skeleton_correction="python -i ${segway_dir}/gt_scripts/ng_check_skeleton_correction.py ${gt_json}"

alias 07_make_zarr_gt="python ${segway_dir}/gt_scripts/make_zarr_gt_background.py ${gt_json}"
alias 07_check_zarr_gt="python -i ${segway_dir}/gt_scripts/ng_check_gt.py ${gt_json}"
