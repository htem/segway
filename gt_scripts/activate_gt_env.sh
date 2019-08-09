
gt_json=`realpath $1`

segway_dir=/n/groups/htem/Segmentation/shared-nondev/segway

# check if path is correct
if [[ $PYTHONPATH != */n/groups/htem/Segmentation/shared-nondev* ]]; then
    echo INFO: PYTHONPATH env does not have segway... adding it
    export PYTHONPATH=$PYTHONPATH:/n/groups/htem/Segmentation/shared-nondev
fi

alias 01_copy_raw_from_cb2_v2="python ${segway_dir}/gt_scripts/make_zarr_from_cb2_v2_zarr.py ${gt_json}"
alias 01_check_raw="python -i ${segway_dir}/gt_scripts/ng_check_raw.py ${gt_json}"

02_add_segmentation_json() {
    if [ "$1" == "" ]; then echo ERROR: Please give the segmentation json file...; return 1; fi
    if [ ! -f "$1" ]; then echo ERROR: "$1" does not exist...; return 1; fi

    export SEGMENTATION_JSON=$1
}
alias 02_run_segmentation="python ${segway_dir}/tasks/task_05_extract_segmentation_from_lut_blockwise.py \${SEGMENTATION_JSON}"
alias 02_check_segmentation="python -i ${segway_dir}/tools/ng_check_segmentation.py \$SEGMENTATION_JSON"
alias print_catmaid_location="python ${segway_dir}/gt_scripts/print_catmaid_coordinate.py ${gt_json}"

alias 03_run_slicewise_segmentation="python ${segway_dir}/gt_scripts/extract_slicewise_segmentation_z.py ${gt_json}"
alias 03_check_slicewise_segmentation="python -i ${segway_dir}/gt_scripts/ng_check_slicewise_segmentation_z.py ${gt_json}"

alias 04_fetch_skeleton="python ${segway_dir}/gt_scripts/fetch_skeletons_cb2_from_json.py ${gt_json}"
alias 04_run_skeleton_correction="python ${segway_dir}/gt_scripts/fix_gt_with_skeleton.py ${gt_json}"
alias 04_check_skeleton_correction="python -i ${segway_dir}/gt_scripts/ng_check_skeleton_correction.py ${gt_json}"

alias 05a_add_labels_mask="python ${segway_dir}/gt_scripts/add_labels_mask_z.py ${gt_json}"
alias 05b_add_ignored_fragments_mask="python ${segway_dir}/gt_scripts/add_labels_mask_ignore_fragments.py ${gt_json}"
alias 05_check_labels_mask="python -i ${segway_dir}/gt_scripts/ng_check_labels_mask.py ${gt_json}"

alias 06a_add_unlabeled_mask="python ${segway_dir}/gt_scripts/add_unlabeled_using_skeleton.py ${gt_json}"
alias 06a_check_unlabeled_mask="python -i ${segway_dir}/gt_scripts/ng_check_unlabeled.py ${gt_json}"
alias 06b_add_myelin_mask="python ${segway_dir}/gt_scripts/add_myelin_gt.py ${gt_json}"

alias 07_make_zarr_gt="python ${segway_dir}/gt_scripts/make_zarr_gt3.py ${gt_json}"
alias 07_check_zarr_gt="python -i ${segway_dir}/gt_scripts/ng_check_gt.py ${gt_json}"

alias run_skeleton_correction_no_merge="python ${segway_dir}/gt_scripts/fix_gt_with_skeleton.py ${gt_json} --no_correct_merge"
alias run_skeleton_correction_no_split="python ${segway_dir}/gt_scripts/fix_gt_with_skeleton.py ${gt_json} --no_correct_split"

alias test_04_run_skeleton_correction="python ${segway_dir}/gt_scripts/fix_gt_with_skeleton2.py ${gt_json}"
