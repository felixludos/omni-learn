import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal as NormalDistribution
# from .. import framework as fm
from ..op import framework as fm

from .. import util

from omnibelt import InitWall, unspecified_argument
import omnifig as fig

class NSegReg_Base(fm.FunctionBase):
	'''1D Linear regression using N line segments in D dims'''
	def __init__(self, n_seg, dim=None, init_stds=1.5, init_noise=None, p=2, clamp_ends=True, eps=1e-8):
		super().__init__(din=1, dout=dim)
		
		self.init_stds = init_stds
		self.init_noise = init_noise
		
		self.n_seg = n_seg
		self.norm_p = p
		self.clamp_ends = clamp_ends
		self.eps = eps
		
		self.segs = None
		
	def init_params(self, pts, pt_wts=None):
		B, D = pts.size()
		
		U, S, V = torch.svd(pts.t() @ pts / (B - 1))
		
		S = S.sqrt()
		
		mu = pts.mean(0) if pt_wts is None else pts.mul(F.normalize(pt_wts,p=1).view(B,1)).sum(0)
		
		delta = U.mul(S.unsqueeze(1))[0]
		steps = torch.linspace(-self.init_stds, self.init_stds, self.n_seg + 1)
		anchors = mu.unsqueeze(0) + steps.unsqueeze(-1) * delta.unsqueeze(0)
		
		if self.init_noise is not None and self.init_noise > 0:
			mag = S[1:].mul(self.init_noise).unsqueeze(0).expand(self.n_seg+1, -1)
			noise = torch.normal(torch.zeros_like(mag), mag).unsqueeze(-1) * U[1:].view(1,-1,D)
			anchors += noise.sum(1)
			
		segs = anchors.clone()
		segs[1:] -= segs[:-1].clone()
		
		self.segs = nn.Parameter(segs, requires_grad=True)
		if self.dout is None:
			self.dout = D
		
		return self.segs
		
	def _point_wts(self, A, B):
		W = torch.cdist(A, B).clamp(min=self.eps).pow(-1)
		return W
	
	def resample_segments(self, pts, num=None):

		srcs = self.segs.detach().cumsum(0)
		W = self._point_wts(pts, srcs)
		
		# wt = W.sum(0)
		cnts = W.max(1)[1].bincount(minlength=self.n_seg+1)
		
		# scnts = resp.bincount()
		# wt = torch.zeros(self.n_seg).index_add(0, resp, dist).div(cnts.float().pow(cnts.gt(0)))
		
		if num is None:
			num = cnts.eq(0).sum().item()
		
		if num == 0:
			return
		
		bad = cnts.topk(num, largest=False)[1]
		good = torch.ones(W.size(1)).bool()
		good[bad] = 0
		
		removed = srcs[bad]
		cts = srcs[good]
		segs = cts.clone()
		segs[1:] -= segs[:-1].clone()
		
		self.segs.data.fill_(self.eps)[:len(segs)].copy_(segs)
		cts = list(cts)
		
		with torch.no_grad():
			# dist, resp = self._distance(pts)
			resp = self.compute_responsibilities(pts)
		
		rcnts = resp.clamp(max=len(cts)-2).bincount(minlength=len(cts))
		picks = rcnts.topk(num)[1].sort(descending=True)[0].tolist()
		
		new = []
		
		for idx in picks:
			p = cts[idx].add(cts[idx+1]).div(2)
			cts.insert(idx+1, p)
			new.append(p)
		
		segs = torch.stack(cts)
		segs[1:] -= segs[:-1].clone()
		self.segs.data.copy_(segs)
		
		return removed, torch.stack(new)
		
	def reorder_segments(self, pts, ):
		raise NotImplementedError
	
	def _vector_projection(self, U, V):
		
		locs = (U * V).sum(-1, keepdim=True).div(V.norm(dim=-1, keepdim=True))
		if self.clamp_ends:
			locs = locs.clamp(min=0, max=1)
		else:
			locs = torch.cat([locs[..., :1], locs[..., 1:-1].clamp(min=0, max=1), locs[..., -1:]], 1)

		return locs
	
	def _full_projection(self, pts, include_locs=False):
		D = pts.size(-1)
		
		dirs = self.segs[1:].unsqueeze(0)
		srcs = self.segs[:-1].cumsum(0).unsqueeze(0)
		pts = pts.reshape(-1, 1, D)
		
		locs = self._vector_projection(pts - srcs, dirs)
		
		prj = srcs + dirs * locs
		
		if include_locs:
			return prj, locs
		return prj
		
	def _space_dist_fn(self, p1, p2):
		return p1.sub(p2).norm(self.norm_p, dim=-1)
	
	def _distance(self, pts, prjs=None):
		if prjs is None:
			prjs = self._full_projection(pts)
		pts = pts.reshape(-1, 1, pts.size(-1))
		return self._space_dist_fn(pts, prjs).min(-1)
	
	def _batched_collect(self, opts, inds):
		'''
		
		:param opts: [B, N, D]
		:param inds: [B] batch of indices (0...N-1)
		:return: [B, D]
		'''
		D = opts.size(-1)
		return opts.gather(1,inds.view(-1,1,1).expand(-1,1,D)).squeeze(1)
	
	def compute_distances(self, pts, _prjs=None):
		dist = self._distance(pts, prjs=_prjs)[0]
		return dist.squeeze() if len(pts.size()) == 1 else dist
	
	def compute_responsibilities(self, pts, _prjs=None):
		resp = self._distance(pts, prjs=_prjs)[1]
		return resp.squeeze() if len(pts.size()) == 1 else resp

	def compute_projections(self, pts, _prjs=None, _resp=None):
		if _prjs is None:
			_prjs = self._full_projection(pts)
		if _resp is None:
			_resp = self.compute_responsibilities(pts, _prjs=_prjs)
		return self._batched_collect(_prjs, _resp)

	def inverse(self, pts):
		full, locs = self._full_projection(pts, include_locs=True)
		resp = self.compute_responsibilities(pts, _prjs=full)
		pts = self.compute_projections(pts, _prjs=full, _resp=resp)
		
		lens = self.segs[1:].norm(2,dim=-1).cumsum(0)
		lens = torch.cat([torch.zeros(1, dtype=lens.dtype), lens])
		
		src = self.segs[:-1].cumsum(0).unsqueeze(0)[resp]
		
		offset = pts.sub(src).norm(2,dim=-1,keepdim=True)
		if not self.clamp_ends:
			offset = (-1)**(self._batched_collect(locs, resp).lt(0)) * offset
		
		return lens.add(offset).div(lens[-1])
		

	def forward(self, t):
		assert self.segs is not None, f'segments have not been initialized'

		lens = self.segs[1:].norm(2,dim=-1).cumsum(0)
		lens = lens / lens[-1]
		
		inds = torch.searchsorted(lens, t)
		lens = torch.cat([torch.zeros(1, dtype=lens.dtype), lens])
		extra = t - lens[inds]
		
		return self.segs.cumsum(0)[inds] + extra.unsqueeze(-1) * \
		       F.normalize(self.segs[1:])[inds.clamp(max=self.n_seg-1)]
		



@fig.Component('n-seg-reg')
class NSegReg(fm.Function, fig.Configurable, NSegReg_Base):
	'''Linear regression using N line segments in D dims'''
	def __init__(self, A, n_seg=None, dim=unspecified_argument,
	             init_stds=unspecified_argument, init_noise=unspecified_argument, p=unspecified_argument,
	             clamp_ends=None, eps=None, **kwargs):
		
		if n_seg is None:
			n_seg = A.pull('n-seg', '<>N')
		
		if dim is None:
			dim = A.pull('dim', None)
		
		if init_stds is unspecified_argument:
			init_stds = A.pull('init-stds', 1.5)
		
		if init_noise is unspecified_argument:
			init_noise = A.pull('init-noise', None)
		
		if p is unspecified_argument:
			p = A.pull('norm-p', 2)
		
		if clamp_ends is None:
			clamp_ends = A.pull('clamp-ends', True)
		
		if eps is None:
			eps = A.pull('epsilon', '<>eps', 1e-8)
		
		super().__init__(A, n_seg=n_seg, dim=unspecified_argument,
	             init_stds=init_stds, init_noise=init_noise, p=p, clamp_ends=clamp_ends, eps=eps, **kwargs)


