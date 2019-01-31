import json
import logging
import lsd
import numpy as np
import daisy
import sys

from lsd.parallel_aff_agglomerate import agglomerate_in_block

from task_helper import *
from task_02_extract_fragments import ExtractFragmentTask

# logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


class AgglomerateTask(SlurmTask):
    '''
    Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.

    Args:

        in_file (``string``):

            The input file containing affs and fragments.

        affs_dataset, fragments_dataset (``string``):

            Where to find the affinities and fragments.

        block_size (``tuple`` of ``int``):

            The size of one block in world units.

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction and agglomeration,
            in world units.

        db_host (``string``):

            Where to find the MongoDB server.

        db_name (``string``):

            The name of the MongoDB database to use.

        num_workers (``int``):

            How many blocks to run in parallel.

        merge_function (``string``):

            Symbolic name of a merge function. See dictionary below.
    '''

    affs_file = daisy.Parameter()
    affs_dataset = daisy.Parameter()
    fragments_file = daisy.Parameter()
    fragments_dataset = daisy.Parameter()
    block_size = daisy.Parameter()
    context = daisy.Parameter()
    db_host = daisy.Parameter()
    db_name = daisy.Parameter()
    num_workers = daisy.Parameter()
    merge_function = daisy.Parameter()
    threshold = daisy.Parameter(default=1.0)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        waterz_merge_function = {
            'hist_quant_10': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>',
            'hist_quant_10_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>',
            'hist_quant_25': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
            'hist_quant_25_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>',
            'hist_quant_50': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
            'hist_quant_50_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>',
            'hist_quant_75': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
            'hist_quant_75_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>',
            'hist_quant_90': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>',
            'hist_quant_90_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>',
            'mean': 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
        }[self.merge_function]

        logging.info("Reading affs from %s", self.affs_file)
        affs = daisy.open_ds(self.affs_file, self.affs_dataset, mode='r')

        logging.info("Reading fragments from %s", self.fragments_file)
        fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset, mode='r')

        # open RAG DB
        logging.info("Opening RAG DB...")
        self.rag_provider = lsd.persistence.MongoDbRagProvider(
            self.db_name,
            host=self.db_host,
            mode='r+',
            edges_collection='edges_' + self.merge_function)
        logging.info("RAG DB opened")

        assert fragments.data.dtype == np.uint64

        # shape = affs.shape[1:]
        self.context = daisy.Coordinate(self.context)

        total_roi = affs.roi.grow(self.context, self.context)
        read_roi = daisy.Roi((0,)*affs.roi.dims(), self.block_size).grow(self.context, self.context)
        write_roi = daisy.Roi((0,)*affs.roi.dims(), self.block_size)

        config = {
            'affs_file': self.affs_file,
            'affs_dataset': self.affs_dataset,
            'fragments_file': self.fragments_file,
            'fragments_dataset': self.fragments_dataset,
            'block_size': self.block_size,
            'context': self.context,
            'db_host': self.db_host,
            'db_name': self.db_name,
            'num_workers': self.num_workers,
            'merge_function': self.merge_function,
            'threshold': self.threshold
        }
        self.slurmSetup(config, 'actor_agglomerate.py')

        self.schedule(
            total_roi,
            read_roi,
            write_roi,
            process_function=self.new_actor,
            check_function=(self.block_done, lambda b: True),
            num_workers=self.num_workers,
            read_write_conflict=False,
            fit='shrink')

    def block_done(self, block):
        return (
            self.rag_provider.has_edges(block.write_roi) or
            self.rag_provider.num_nodes(block.write_roi) == 0)

    def requires(self):
        return [ExtractFragmentTask()]


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('lsd.parallel_aff_agglomerate').setLevel(logging.DEBUG)

    configs = {}
    for config in sys.argv[1:]:
        with open(config, 'r') as f:
            configs = {**json.load(f), **configs}
    aggregateConfigs(configs)
    print(configs)

    daisy.distribute([{'task': AgglomerateTask(), 'request': None}],
        global_config=configs)
