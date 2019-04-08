## Evaluating segmentation result with skeleton

run_evaluation could plot the graph with evaluation matrices number of split_merge_error, rand, voi OR just print out the coordinates of split error and merge error

### Simple illustration
you need to provide the configs file and mode("print","quickplot","plot"). You can check task_defaults.json to figure out the information to provide. For the mode, "print" means just print out the coordinates of split error or merge error. "quickplot" means plot all three matrices, "plot" means you can plot any combination of three matrices.

*-p*  means number of processes to use, default set to 1  

*-i*  means build the graph with interpolation or not, default is True

*-f*  just provide the name of graph file.

### A few things noteworthy
A few segmentation Markers are dismissed intentionally to provide clean figure. If you don't want that or the number of threshold is more than 9, you should provide your own markers. 

Also, make sure first seg_path file in list_seg_path have complete threshold_list. For example, in first seg_path, we have segmentation_0.1 to segmentation_0.9 and rest have segmentation_0.1 to segmentation_0.7. If it is not the case, please change the j value in function *compare_threshold* in *task_05_graph_evaluation_print_error.py*

Same thing for colors, if the number of model you would like to compare is more than 10, provide your own color. 


### Example code 
`python run_evaluation.py current_task.json print -p 5 -i False -f test01`
