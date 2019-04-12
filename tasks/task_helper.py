import datetime
import json
import logging
import multiprocessing
import hashlib
import subprocess
import os
import collections
import pymongo

import daisy
import ast

logger = logging.getLogger(__name__)

home = os.path.expanduser("~")
RUNNING_REMOTELY = os.path.isfile(home + "/CONFIG_LOCAL_DAISY")


class SlurmTask(daisy.Task):

    log_dir = daisy.Parameter()

    cpu_cores = daisy.Parameter(2)
    cpu_time = daisy.Parameter(0)
    cpu_mem = daisy.Parameter(4)

    def slurmSetup(
            self, config, actor_script,
            python_interpreter='python',
            **kwargs):
        '''Write config file and sbatch file for the actor, and generate
        `new_actor_cmd`. We also keep track of new jobs so to kill them
        when the task is finished.'''

        logname = (actor_script.split('.'))[-2].split('/')[-1]

        # assume that actor_script resides in the same folder
        actor_script = (os.path.dirname(os.path.realpath(__file__)) +
                        '/' + actor_script)
        # print("Actor script: %s" % actor_script)

        self.slurmtask_run_cmd, self.new_actor_cmd = generateActorSbatch(
            config,
            actor_script,
            python_interpreter=python_interpreter,
            log_dir=self.log_dir,
            logname=logname,
            cpu_cores=self.cpu_cores,
            cpu_time=self.cpu_time,
            cpu_mem=self.cpu_mem,
            **kwargs)
        self.started_jobs = multiprocessing.Manager().list()
        self.started_jobs_local = []

        # if self.dry_run:
        #     self.new_actor = lambda b: 0
        #     self.cleanup = lambda: 0

    def new_actor(self):
        '''Submit new actor job using sbatch'''
        context = os.environ['DAISY_CONTEXT']

        logger.info("Srun command: DAISY_CONTEXT={} CUDA_VISIBLE_DEVICES=0 {}".format(
                context,
                self.slurmtask_run_cmd))

        logger.info("Submit command: DAISY_CONTEXT={} {}".format(
                context,
                ' '.join(self.new_actor_cmd)))

        run_cmd = "cd %s" % os.getcwd() + "; "
        run_cmd += "source /home/tmn7/daisy/bin/activate;" + " "
        run_cmd += "DAISY_CONTEXT=%s" % context + " "
        run_cmd += ' '.join(self.new_actor_cmd)

        if RUNNING_REMOTELY:
            process_cmd = "ssh o2 " + "\"" + run_cmd + "\""
        else:
            process_cmd = run_cmd

        print(process_cmd)
        cp = subprocess.run(process_cmd,
                            stdout=subprocess.PIPE,
                            shell=True
                            )
        id = cp.stdout.strip().decode("utf-8")
        self.started_jobs.append(id)

    def cleanup(self):
        try:
            started_slurm_jobs = self.started_jobs._getvalue()
        except:
            try:
                started_slurm_jobs = self.started_jobs_local
            except:
                started_slurm_jobs = []

        # print(started_slurm_jobs._getvalue())
        # print(started_slurm_jobs._getvalue())
        if len(started_slurm_jobs) > 0:
            all_jobs = " ".join(started_slurm_jobs)
            if RUNNING_REMOTELY:
                cmd = "ssh o2 scancel {}".format(all_jobs)
            else:
                cmd = "scancel {}".format(all_jobs)
            print(cmd)
            subprocess.run(cmd, shell=True)
        else:
            print("No jobs to cleanup")

    def _periodic_callback(self):
        try:
            self.started_jobs_local = self.started_jobs._getvalue()
        except:
            pass


def generateActorSbatch(
        config, actor_script, log_dir, logname,
        python_interpreter,
        **kwargs):

    config_str = ''.join(['%s' % (v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))
    try:
        os.makedirs('.run_configs')
    except Exception:
        pass
    config_file = os.path.join(
        '.run_configs', '%s_%d.config' % (logname, config_hash))
    # log_out = os.path.join('.run_configs', '%d.out'%config_hash)
    # log_err = os.path.join('.run_configs', '%d.err'%config_hash)
    with open(config_file, 'w') as f:
        json.dump(config, f)

    run_cmd = ' '.join([
        python_interpreter,
        '%s' % actor_script,
        '%s' % config_file,
        ])

    sbatch_script = os.path.join('.run_configs', '%s_%d.sh'%(logname, config_hash))
    generateSbatchScript(
        sbatch_script, run_cmd, log_dir, logname,
        **kwargs)

    new_actor_cmd = [
        'sbatch',
        '--parsable',
        '%s' % sbatch_script
        ]

    return run_cmd, new_actor_cmd


def generateSbatchScript(
        sbatch_script,
        run_cmd,
        log_dir,
        logname,
        cpu_time=0,
        queue='short',
        cpu_cores=1,
        cpu_mem=6,
        gpu=None):
    text = []
    text.append("#!/bin/bash")
    text.append("#SBATCH -t %d:40:00" % cpu_time)

    if gpu is not None:
        text.append("#SBATCH -p gpu")
        if gpu == '' or gpu == 'any':
            text.append("#SBATCH --gres=gpu:1")
        else:
            text.append("#SBATCH --gres=gpu:{}:1".format(gpu))
    else:
        text.append("#SBATCH -p %s" % queue)
    text.append("#SBATCH -c %d" % cpu_cores)
    text.append("#SBATCH --mem=%dGB" % cpu_mem)
    text.append("#SBATCH -o {}/{}_%j.out".format(log_dir, logname))
    text.append("#SBATCH -e {}/{}_%j.err".format(log_dir, logname))
    # text.append("#SBATCH -o .logs_sbatch/{}_%j.out".format(logname))
    # text.append("#SBATCH -e .logs_sbatch/{}_%j.err".format(logname))

    text.append("")
    # text.append("$*")
    text.append(run_cmd)

    logger.info("Writing sbatch script %s" % sbatch_script)
    with open(sbatch_script, 'w') as f:
        f.write('\n'.join(text))


def parseConfigs(args, aggregate_configs=True):
    global_configs = {}
    user_configs = {}
    hierarchy_configs = collections.defaultdict(dict)

    # first load default configs if avail
    try:
        config_file = "segway/tasks/task_defaults.json"
        with open(config_file, 'r') as f:
            global_configs = {**json.load(f), **global_configs}
    except Exception:
        logger.info("Default task config not loaded")
        pass

    for config in args:
        print(config)
        if "=" in config:
            key, val = config.split('=')
            if '.' in key:
                task, param = key.split('.')
                hierarchy_configs[task][param] = val
            else:
                user_configs[key] = ast.literal_eval(val)
        else:
            with open(config, 'r') as f:
                print("\nhelper: loading %s" % config)
                new_configs = json.load(f)
                keys = set(list(global_configs.keys())).union(list(new_configs.keys()))
                for k in keys:
                    if k in global_configs:
                        if k in new_configs:
                            global_configs[k].update(new_configs[k])
                    else:
                        global_configs[k] = new_configs[k]
                print(list(global_configs.keys()))

    print("\nhelper: final config")
    print(global_configs)
    print(hierarchy_configs)
    global_configs = {**hierarchy_configs, **global_configs}
    if aggregate_configs:
        aggregateConfigs(global_configs)
    return (user_configs, global_configs)


def aggregateConfigs(configs):

    input_config = configs["Input"]
    network_config = configs["Network"]

    today = datetime.date.today()
    parameters = {}
    parameters['experiment'] = input_config['experiment']
    parameters['year'] = today.year
    parameters['month'] = '%02d' % today.month
    parameters['day'] = '%02d' % today.day
    parameters['network'] = network_config['name']
    parameters['iteration'] = network_config['iteration']

    for config in input_config:
        if isinstance(input_config[config], str):
            input_config[config] = input_config[config].format(**parameters)

    # add a hash based on directory path to the mongodb dataset
    # so that other users can run the same config without name conflicts
    # though if the db already exists, don't change it to avoid confusion
    db_host, db_name = (input_config['db_host'], input_config['db_name'])
    myclient = pymongo.MongoClient(db_host)
    if db_name not in myclient.database_names():
        output_path = os.path.abspath(input_config["output_file"])
        config_hash = hashlib.blake2b(
            output_path.encode(), digest_size=4).hexdigest()
        input_config['db_name'] = input_config['db_name'] + "_" + config_hash
        # print(config_hash)
    # print(input_config['db_name'])
    # exit(0)

    os.makedirs(input_config['log_dir'], exist_ok=True)

    if "PredictTask" in configs:
        config = configs["PredictTask"]
        config['raw_file'] = input_config['raw_file']
        config['raw_dataset'] = input_config['raw_dataset']
        if 'out_file' not in config:
            config['out_file'] = input_config['output_file']
        config['train_dir'] = network_config['train_dir']
        config['iteration'] = network_config['iteration']
        config['log_dir'] = input_config['log_dir']
        config['output_shape'] = network_config['output_shape']
        config['out_dtype'] = network_config['out_dtype']
        config['net_voxel_size'] = network_config['net_voxel_size']
        # config['effective_net_voxel_size'] = network_config['effective_net_voxel_size']
        config['input_shape'] = network_config['input_shape']
        config['out_dims'] = network_config['out_dims']
        config['predict_file'] = network_config['predict_file']
        if 'xy_downsample' in network_config:
            config['xy_downsample'] = network_config['xy_downsample']
        if 'roi_offset' in input_config:
            config['roi_offset'] = input_config['roi_offset']
        if 'roi_shape' in input_config:
            config['roi_shape'] = input_config['roi_shape']

    if "PredictMyelinTask" in configs:
        config = configs["PredictMyelinTask"]
        config['raw_file'] = input_config['raw_file']
        config['myelin_file'] = input_config['output_file']
        config['log_dir'] = input_config['log_dir']
        if 'roi_offset' in input_config:
            config['roi_offset'] = input_config['roi_offset']
        if 'roi_shape' in input_config:
            config['roi_shape'] = input_config['roi_shape']

    if "MergeMyelinTask" in configs:
        config = configs["MergeMyelinTask"]
        if 'affs_file' not in config:
            config['affs_file'] = input_config['output_file']
        config['myelin_file'] = input_config['output_file']
        config['merged_affs_file'] = input_config['output_file']
        config['log_dir'] = input_config['log_dir']

    if "ExtractFragmentTask" in configs:
        config = configs["ExtractFragmentTask"]
        if 'affs_file' not in config:
            config['affs_file'] = input_config['output_file']
        config['fragments_file'] = input_config['output_file']
        config['db_name'] = input_config['db_name']
        config['db_host'] = input_config['db_host']
        config['log_dir'] = input_config['log_dir']

    if "AgglomerateTask" in configs:
        config = configs["AgglomerateTask"]
        if 'affs_file' not in config:
            config['affs_file'] = input_config['output_file']
        config['fragments_file'] = input_config['output_file']
        config['db_name'] = input_config['db_name']
        config['db_host'] = input_config['db_host']
        config['log_dir'] = input_config['log_dir']

    if "SegmentationTask" in configs:
        config = configs["SegmentationTask"]
        config['fragments_file'] = input_config['output_file']
        config['out_file'] = input_config['output_file']
        config['db_name'] = input_config['db_name']
        config['db_host'] = input_config['db_host']
        config['log_dir'] = input_config['log_dir']

    if "GrowSegmentationTask" in configs:
        config = configs["GrowSegmentationTask"]
        config['fragments_file'] = input_config['output_file']
        config['out_file'] = input_config['output_file']
        # config['out_file'] = input_config['output_file']
        config['db_name'] = input_config['db_name']
        config['db_host'] = input_config['db_host']
        config['log_dir'] = input_config['log_dir']

    if "SparseSegmentationServer" in configs:
        config = configs["SparseSegmentationServer"]
        config['fragments_file'] = input_config['output_file']
        config['db_name'] = input_config['db_name']
        config['db_host'] = input_config['db_host']
        config['log_dir'] = input_config['log_dir']
        config['segment_file'] = input_config['output_file']

    if "BlockwiseSegmentationTask" in configs:
        config = configs["BlockwiseSegmentationTask"]
        config['fragments_file'] = input_config['output_file']
        config['db_name'] = input_config['db_name']
        config['db_host'] = input_config['db_host']
        config['log_dir'] = input_config['log_dir']
        config['out_file'] = input_config['output_file']

    if "SplitFixTask" in configs:
        config = configs["SplitFixTask"]
        config['fragments_file'] = input_config['output_file']
        config['segment_file'] = input_config['output_file']
        config['db_name'] = input_config['db_name']
        config['db_host'] = input_config['db_host']
        config['log_dir'] = input_config['log_dir']
        config['out_file'] = input_config['output_file']

    if "FixMergeTask" in configs:
        config = configs["FixMergeTask"]
        config['fragments_file'] = input_config['output_file']
        config['segment_file'] = input_config['output_file']
        config['db_name'] = input_config['db_name']
        config['db_host'] = input_config['db_host']
        config['log_dir'] = input_config['log_dir']
        # config['out_file'] = input_config['output_file']
