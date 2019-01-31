import datetime
import json
import logging
import multiprocessing
import hashlib
import subprocess
import os
import collections

import daisy

logger = logging.getLogger(__name__)


class SlurmTask(daisy.Task):

    log_dir = daisy.Parameter()
    started_jobs = []

    def slurmSetup(self, config, actor_script, **kwargs):
        '''Write config file and sbatch file for the actor, and generate
        `new_actor_cmd`. We also keep track of new jobs so to kill them
        when the task is finished.'''

        print(actor_script)
        logname = (actor_script.split('.'))[-2].split('/')[-1]
        self.slurmtask_run_cmd, self.new_actor_cmd = generateActorSbatch(
            config,
            actor_script,
            log_dir=self.log_dir,
            logname=logname,
            **kwargs)
        self.started_jobs = multiprocessing.Manager().list()

        # if self.dry_run:
        #     self.new_actor = lambda b: 0
        #     self.cleanup = lambda: 0

    def new_actor(self):
        '''Submit new actor job using sbatch'''
        context = os.environ['DAISY_CONTEXT']

        logger.info("Srun command: DAISY_CONTEXT={} {}".format(
                context,
                self.slurmtask_run_cmd))

        logger.info("Submit command: DAISY_CONTEXT={} {}".format(
                context,
                ' '.join(self.new_actor_cmd)))

        cp = subprocess.run(' '.join(self.new_actor_cmd),
                            stdout=subprocess.PIPE,
                            shell=True
                            )
        id = cp.stdout.strip().decode("utf-8")
        self.started_jobs.append(id)

    def cleanup(self):
        if len(self.started_jobs) > 0:
            all_jobs = " ".join(self.started_jobs)
            cmd = "scancel {}".format(all_jobs)
            subprocess.run(cmd, shell=True)


def generateActorSbatch(config, actor_script, log_dir, logname, **kwargs):

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
        'python -u',
        '%s' % actor_script,
        '%s' % config_file,
        ])

    sbatch_script = os.path.join('.run_configs', '%s_%d.sh'%(logname, config_hash))
    generateSbatchScript(sbatch_script, run_cmd, log_dir, logname, **kwargs)

    new_actor_cmd = [
        'sbatch',
        '--parsable',
        '%s' % sbatch_script
        ]

    return run_cmd,new_actor_cmd


def generateSbatchScript(
        sbatch_script,
        run_cmd,
        log_dir,
        logname,
        cpu_time=0,
        queue='short',
        num_core=1,
        cpu_mem=6,
        gpu=None):
    text = []
    text.append("#!/bin/bash")
    text.append("#SBATCH -t %d:30:00" % cpu_time)

    if gpu is not None:
        text.append("#SBATCH -p gpu")
        if gpu == '' or gpu == 'any':
            text.append("#SBATCH --gres=gpu:1")
        else:
            text.append("#SBATCH --gres=gpu:{}:1".format(gpu))
    else:
        text.append("#SBATCH -p %s" % queue)
    text.append("#SBATCH -c %d" % num_core)
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


def parseConfigs(args):
    global_configs = {}
    user_configs = {}
    hierarchy_configs = collections.defaultdict(dict)
    for config in args:
        if "=" in config:
            key, val = config.split('=')
            if '.' in key:
                task, param = key.split('.')
                hierarchy_configs[task][param] = val
            else:
                user_configs[key] = val
        else:
            with open(config, 'r') as f:
                global_configs = {**json.load(f), **global_configs}
    print(hierarchy_configs)
    global_configs = {**hierarchy_configs, **global_configs}
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
        print(input_config[config])
        input_config[config] = input_config[config].format(**parameters)

    print(input_config)
    os.makedirs(input_config['log_dir'], exist_ok=True)

    if "PredictTask" in configs:
        config = configs["PredictTask"]
        config['raw_file'] = input_config['raw_file']
        config['raw_dataset'] = input_config['raw_dataset']
        config['out_file'] = input_config['output_file']
        config['train_dir'] = network_config['train_dir']
        config['iteration'] = network_config['iteration']
        config['log_dir'] = input_config['log_dir']

    if "ExtractFragmentTask" in configs:
        config = configs["ExtractFragmentTask"]
        config['affs_file'] = input_config['output_file']
        config['fragments_file'] = input_config['output_file']
        config['db_name'] = input_config['db_name']
        config['db_host'] = input_config['db_host']
        config['log_dir'] = input_config['log_dir']

    if "AgglomerateTask" in configs:
        config = configs["AgglomerateTask"]
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

    if "SparseSegmentationTask" in configs:
        config = configs["SparseSegmentationTask"]
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
