
import sys, os
import torch
import foundation as fd
import models
import torch.nn as nn

def load_model(path=None, args=None, optim=None, to_device=True):
	assert path is not None or args is not None, 'must specify the model'
	
	checkpoint = None
	if path is not None:
		if os.path.isdir(path):
			path = os.path.join(path, 'best.pth.tar')
		assert os.path.isfile(path), 'Could not find encoder:' + path
		
		checkpoint = torch.load(path)
		args = checkpoint['args']
		print('Loaded {}'.format(path))
		
	
	model = None
	config = args.data_type, args.model
	if args.dataset == 'mnist':

		model = fd.nets.make_MLP(input_dim=28*28, output_dim=10,
								hidden_dims=[64], nonlin=args.nonlin)

		model.normalized = False

	elif config == ('spec', 'naive'):
		model = models.MEL_Encoder(in_shape=args.in_shape, out_dim=args.out_dim, nonlin=args.nonlin,
								   use_batch_norm=args.use_batch_norm, fc_dims=args.fc_dims)
	
	elif config == ('wav', 'naive'):
		
		model = models.Wav_Encoder(in_shape=args.in_shape, out_dim=args.out_dim, nonlin=args.nonlin,
									rec_dim=args.rec_dim, rec_num_layers=args.rec_num_layers, rec_type=args.rec_type,
								   fc_dims=args.fc_dims, use_batch_norm=args.use_batch_norm, )
	
	elif config == ('wav', 'crnn'):
		
		model = models.Conv_RNN(in_shape=args.in_shape, out_dim=args.out_dim, nonlin=args.nonlin,
								hop=args.model_hop, ws=args.model_ws, channels=args.model_n,
								rec_dim=args.rec_dim, rec_num_layers=args.rec_num_layers, rec_type=args.rec_type,
								use_fc=args.use_fc, use_batch_norm=args.use_batch_norm, )
		
	elif config == ('spec', 'rnn'):
		
		model = models.MEL_RNN(in_shape=args.in_shape, out_dim=args.out_dim, rec_dim=args.rec_dim,
							   rec_num_layers=args.rec_num_layers, rec_type=args.rec_type,
							   use_fc=args.use_fc)
		
	elif config == ('wav', 'arc'):
		
		model = models.Conv_RNN_AutoEncoder(in_shape=args.in_shape, latent_dim=args.latent_dim, nonlin=args.nonlin,
								hop=args.model_hop, ws=args.model_ws, channels=args.model_n,
								rec_dim=args.rec_dim, rec_num_layers=args.rec_num_layers, rec_type=args.rec_type,
								use_batch_norm=args.use_batch_norm, noisy_rec=args.noisy_rec, cls_dim=args.out_dim)
	
	elif config == ('wav', 'gan'):
		
		judge = models.ConvRNN_Discriminator(in_shape=args.in_shape, cls_dim=args.out_dim if args.gen_use_cls else None, nonlin=args.nonlin,
								hop=args.model_hop, ws=args.model_ws, channels=args.model_n,
								rec_dim=args.rec_dim, rec_num_layers=args.rec_num_layers, rec_type=args.rec_type,
								use_batch_norm=args.use_batch_norm, )
		
		gen = models.Recurrent_Generator(seq_len=args.seq_len // args.gen_hop, latent_dim=args.latent_dim, cls_dim=args.out_dim if args.gen_use_cls else None,
		                                 nonlin=args.nonlin, use_batch_norm=args.use_batch_norm,
		                                 rec_dim=args.gen_rec_dim, device=args.device)
		
		model = nn.ModuleList([judge, gen])
	
	elif config == ('wav', 'gen-mel'):
		
		judge = models.ConvRNN_Discriminator(in_shape=args.in_shape, cls_dim=args.out_dim if args.gen_use_cls else None,
		                                     nonlin=args.nonlin,
		                                     hop=args.model_hop, ws=args.model_ws, channels=args.model_n,
		                                     rec_dim=args.rec_dim, rec_num_layers=args.rec_num_layers,
		                                     rec_type=args.rec_type,
		                                     use_batch_norm=args.use_batch_norm, )
		
		if not hasattr(args, 'gen_norm'):
			args.gen_norm=False
		
		gen = models.MEL_Generator(seq_len=args.seq_len, latent_dim=args.latent_dim,
		                                 cls_dim=args.out_dim if args.gen_use_cls else None, norm=args.gen_norm,
		                           ws=args.gen_ws, hop=args.gen_hop, n_mels=args.gen_n, use_fc=args.use_fc,
		                                 gain=args.gen_gain, use_batch_norm=args.use_batch_norm,
		                                 rec_dim=args.gen_rec_dim, device=args.device)
		
		model = nn.ModuleList([judge, gen])
		
	
	elif config == ('wav', 'judge-mel'):
		
		judge = models.MEL_Discriminator(mel_dim=args.gen_n, cls_dim=args.out_dim if args.gen_use_cls else None,
		                                     
		                                     rec_dim=args.rec_dim, rec_num_layers=args.rec_num_layers,
		                                     rec_type=args.rec_type, )
		
		gen = models.MEL_Generator(seq_len=args.seq_len, latent_dim=args.latent_dim, ret_mel=True,
		                           cls_dim=args.out_dim if args.gen_use_cls else None, norm=args.gen_norm,
		                           ws=args.gen_ws, hop=args.gen_hop, n_mels=args.gen_n, use_fc=args.use_fc,
		                           gain=args.gen_gain, use_batch_norm=args.use_batch_norm, rec_type=args.rec_type,
		                           rec_dim=args.gen_rec_dim, rec_num_layers=args.gen_rec_num_layers, device=args.device)
		
		model = nn.ModuleList([judge, gen])
	
	else:
		raise Exception('Unknown model config: {}'.format(config))
	
	if checkpoint is not None:
		
		#print(model)
		
		model.load_state_dict(checkpoint['model_state'])
		
		model.load_args = args
		
		
		try:
		
			if optim is not None:
				optim.load_state_dict(checkpoint['optim_state'])
		
		except:
			print('Failed to load optim')
		
		print('Saved params loaded')
	
	if to_device:
		model.to(args.device)
			
	return model




def save_model(info, save_dir, is_best=False, epoch=None):
	path = None
	if is_best:
		path = os.path.join(save_dir, 'best.pth.tar')
		torch.save(info, path)
	
	if epoch is not None:
		path = os.path.join(save_dir, 'checkpoint_{}.pth.tar'.format(epoch))
		torch.save(info, path)
		
	return path
		
		
