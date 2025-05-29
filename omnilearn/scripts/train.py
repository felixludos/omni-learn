from .imports import *
try:
	import wandb
except ImportError:
	wandb = None

from .. import *
from ..op import *
from ..abstract import AbstractTrainer
from omnibelt import flatten


@fig.script('train', description='Iterative optimization on a dataset using a trainer')
def train(cfg: fig.Configuration):
	"""
	readme todo
	"""
	medium = cfg.pull('medium', where_am_i())
	pbar: bool = cfg.pull('pbar', medium != 'cluster')
	pstdout: bool = cfg.pull('pstdout', not pbar)

	record_step = cfg.pull('record-step', False)
	if record_step:
		Dataset._Batch = VizBatch
		# Trainer._Batch = VizBatch
		fig.component('mechanism')(VizMechanism)
		report_settings = cfg.pull('report-settings', {})

	use_wandb = cfg.pulls('use-wandb', 'wandb', default=wandb is not None)
	if use_wandb and wandb is None:
		raise ValueError('You need to install `wandb` to use `wandb`')

	out_root = cfg.pull('out-dir', None)
	if out_root:
		out_root = Path(out_root)
		out_root.mkdir(exist_ok=True)

	resume = cfg.pull('resume', None)
	if resume is None:
		cfg.push('trainer._type', 'default-trainer', overwrite=False, silent=True)
		ckptpath = None
	else:
		out_dir = out_root / resume
		assert out_dir.exists(), f'Cannot resume from {out_dir}, it does not exist'
		cfgpath = out_dir / 'config.yaml'
		assert cfgpath.exists(), f'Cannot resume from {out_dir}, config file does not exist'
		loadcfg = fig.create_config(cfgpath)
		loadcfg.update(cfg)
		# find/load checkpoint
		ckptpath = None

	trainer: AbstractTrainer = cfg.pull('trainer')

	if ckptpath is not None:
		trainer.load_checkpoint(path=ckptpath)

	pre_targets = cfg.pull('pre', None)
	if pre_targets is not None:
		pre_targets = flatten(pre_targets)
	log_targets = cfg.pull('log', None)
	if log_targets is not None:
		log_targets = flatten(log_targets)
	viz_targets = cfg.pull('viz', None)
	if viz_targets is not None:
		viz_targets = flatten(viz_targets)
	out_targets = cfg.pull('out', None)
	if out_targets is not None:
		out_targets = flatten(out_targets)

	ckpt_freq = cfg.pulls('ckpt-freq', 'ckpt', default=None)
	error_ckpt = cfg.pull('err-ckpt', True)

	trainer.prepare() # internally stages with scape
	out_dir = out_root.joinpath(trainer.name) if out_root else None
	if out_dir is not None:
		with out_dir.joinpath('config.yaml').open('w') as f:
			f.write(str(cfg))

	wandb_run = None
	if use_wandb:
		wandb_dir = out_dir.absolute()
		wandb_config = trainer.json()
		project_name = cfg.pull('project-name', '{dataset.name}')
		project_name = pformat(project_name, trainer=trainer, model=trainer.model, dataset=trainer.dataset,
							   config=wandb_config)
		wandb_id = None
		wandb_run = wandb.init(project=project_name, name=trainer.name, config=wandb_config, dir=wandb_dir, id=wandb_id)
		wandb_addr = f'{wandb_run.entity}/{wandb_run.project}/{wandb_run.id}'
		trainer.remember(wandb={'id': wandb_run.id, 'entity': wandb_run.entity, 'project': wandb_run.project})
	# sample_logger = None
	# if log_samples:
	# assert out_dir is not None, f'log-samples requires out-dir to be set'
	# logger = out_dir.joinpath('log.jsonl').open('a')

	limit = cfg.pull('limit', None)

	desc = trainer.describe()
	if desc is not None:
		print(desc)
		print()
	if out_dir is not None:
		print(f'Saving results to {out_dir}')
		print()

	artifacts = trainer.pre_loop(pre_targets)
	if artifacts is not None:
		if 'stats' in artifacts:
			print(f'Pre-loop stats:')
			print(tabulate(flatten(artifacts['stats']).items()))
		if use_wandb and 'study' in artifacts:
			tbl = {key: str(val) for key, val in flatten(artifacts['study']).items()}
			if isinstance(tbl, dict):
				# convert dict[str,str] to dataframe
				tbl = pd.DataFrame(tbl.items(), columns=['Key', 'Value'])
			if isinstance(tbl, pd.DataFrame):
				tbl = wandb.Table(dataframe=tbl)
			wandb_run.log({f'study': tbl})

	itr = trainer.enumerate(limit)
	past_iterations = trainer.past_steps
	remaining_iterations = trainer.expected_steps
	assert remaining_iterations is not None, 'Trainer must have expected_steps defined (set `max-steps` or `max-epochs` in the configuration)'
	num_digits = len(str(remaining_iterations)) + 1
	if pbar and not record_step: itr = tqdm(itr, initial=past_iterations, total=past_iterations + remaining_iterations)
	i = 0
	for i, batch in itr:
		try:
			terminate = trainer.learn(batch)
		except:
			if out_dir is not None and error_ckpt:
				trainer.checkpoint(out_dir / f'error{i}')
			raise

		if record_step:
			if pbar: itr.close()
			print(batch.report(**report_settings))
			break

		# # (internal) log progress
		# if sample_logger is not None:
		# 	if drop_keys:
		# 		drop_keys_in_sample(sample, drop_keys)
		# 	sample_logger.write(json.dumps(sample) + '\n')
		# 	sample_logger.flush()

		# log to stdout

		# log to wandb
		if wandb_run is not None:
			wandb_run.log(flatten(batch['log']), step=i)

		# validation step

		# checkpoint
		if out_dir is not None and ckpt_freq is not None and i > 0 and i % ckpt_freq == 0:
			trainer.checkpoint(out_dir / f'ckpt-{i+1:0{num_digits}}')

		# early stopping
		if terminate:
			if pbar: itr.close()
			break

	summary = trainer.summary()
	if summary is not None:
		print(summary)
		print()

	# if logger is not None:
	# 	logger.close()

	if out_dir is not None and (ckpt_freq is None or i % ckpt_freq != 0):
		trainer.checkpoint(out_dir / f'ckpt-{i+1:0{num_digits}}')

	output = trainer.post_loop(out_targets)

	if wandb_run is not None:
		# extract out from (full) output
		wandb.summary.update(flatten(output))
		wandb_run.finish()

	return output














