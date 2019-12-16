
f="/n/groups/htem/Segmentation/xray_segmentation/outputs/xray/191211_jlc_merge_full_sc2/setup01b/300000/output.zarr"
output_file=$f

ds="volumes/super_1x1x1_segmentation_0.500"
bash segway/tasks/pyramid/make_pyramid.sh $f $ds $output_file

ds="volumes/super_1x1x1_segmentation_0.400"
bash segway/tasks/pyramid/make_pyramid.sh $f $ds $output_file

ds="volumes/super_1x1x1_segmentation_0.600"
bash segway/tasks/pyramid/make_pyramid.sh $f $ds $output_file

ds="volumes/super_1x1x1_segmentation_0.700"
bash segway/tasks/pyramid/make_pyramid.sh $f $ds $output_file
