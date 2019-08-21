In order to log GPU and/or memory utilization, the following steps must be taken:
-run job on 1, 2 or 4 GPU with qsub command > qsub subm_file.pbs 
-go inside into the actual GPU node > ssh r2i3n0 or ssh r2i3n1
-run logging_gpu_utilization_only script > /.logging_gpu_utilization_only
-exit the logging script execution manually by pressing Control+C after time for running job set in qsub has elapsed
-logged info is saved inside gpu_utillization.log
