"""Bounded dtype-consistent image proximal DC; no model, GT or score imports."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys

GSTMI=Path('/home/congcongliu/anaconda3/envs/GSTMI/bin/python')
if Path(sys.executable).resolve()!=GSTMI.resolve():
    raise ValueError('numerical physics requires actual GSTMI Python')
import torch
if torch.__version__!='2.7.1+cu118' or torch.version.cuda!='11.8':
    raise ValueError('require GSTMI Torch2.7.1+cu118/CUDA11.8')
ROOT=Path(__file__).resolve().parent


def _source(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


transforms=_source(ROOT.parent/'nonlocal_autoresearch/trainer/paired_gs/data/transforms.py','_proximal_source_transforms')
pcg=_source(ROOT.parent/'phase_firstorder/pcg.py','_proximal_reviewed_pcg')


@dataclass(frozen=True)
class ProxConfig:
    max_matvecs:int=32
    rtol:float=1e-5

    def __post_init__(self):
        if type(self.max_matvecs) is not int or not 1<=self.max_matvecs<=32 or self.rtol!=1e-5:
            raise ValueError('fixed rtol1e-5 and bounded1..32 CG matvecs required')


@dataclass
class ProxResult:
    image:torch.Tensor
    status:str
    record:dict
    work:dict


def _norm(value):
    return math.sqrt(float(value.real.double().square().sum()+value.imag.double().square().sum()))


def _dot(a,b):
    return float((a.real.double()*b.real.double()).sum()+(a.imag.double()*b.imag.double()).sum())


def verification_tolerances(dtype):
    if dtype==torch.complex64:
        return {'true_residual_tolerance':2e-5,'J_relative_tolerance':2e-6,'J_absolute_tolerance':1e-10}
    if dtype==torch.complex128:
        return {'true_residual_tolerance':2e-10,'J_relative_tolerance':5e-12,'J_absolute_tolerance':1e-12}
    raise ValueError('complex64/128 required')


class SenseOperator:
    """A=MFS and A^H, with original source FFT and operand-dtype accumulation.

    Mask is acquired binary M. No frequency/prior/GT or extra fitting is used.
    Source tensors are read-only views; counters record every attempted FFT.
    """
    def __init__(self,csm,measured,mask,*,chunk_size=4):
        if (not isinstance(csm,torch.Tensor) or csm.dtype not in (torch.complex64,torch.complex128)
                or csm.ndim!=4 or csm.shape[0]<1 or measured.shape!=csm.shape
                or measured.dtype!=csm.dtype or measured.device!=csm.device
                or tuple(mask.shape)!=tuple(csm.shape[1:]) or mask.device!=csm.device
                or mask.is_complex() or type(chunk_size) is not int or not 1<=chunk_size<=4):
            raise ValueError('SENSE tensors/dtypes/shapes/device/chunk mismatch')
        if not bool(((mask==0)|(mask==1)).all()):raise ValueError('acquired mask must be binary')
        self.csm,self.measured,self.mask=csm.detach(),measured.detach(),mask.bool().detach()
        self.shape=tuple(mask.shape);self.dtype=csm.dtype;self.device=csm.device;self.chunk_size=chunk_size
        power=torch.zeros(self.shape,dtype=torch.float64,device=self.device)
        for i in range(0,len(csm),chunk_size):
            s=csm[i:i+chunk_size];y=measured[i:i+chunk_size]
            if not bool(torch.isfinite(s).all()) or not bool(torch.isfinite(y).all()):
                raise ValueError('nonfinite CSM/data')
            power+=(s.real.double().square()+s.imag.double().square()).sum(0)
        self.support=power>0;self.L=float(power.max());self.lambda_requested=.1*self.L
        self.lambda_effective=float(torch.tensor(self.lambda_requested,dtype=csm.real.dtype))
        if (not math.isfinite(self.L) or self.L<=0 or not math.isfinite(self.lambda_effective)
                or self.lambda_effective<=0):raise ValueError('positive representable CSM-only lambda required; no floor')
        self._versions=(csm._version,measured._version,mask._version)
        self._original=(csm,measured,mask)
        self.work={key:0 for key in ('rhs_adjoint_calls','normal_calls','terminal_normal_calls','objective_calls',
                   'forward_calls','adjoint_calls','fft_batches','ifft_batches','coil_fft_transforms','coil_ifft_transforms',
                   'gaussian_renders','gaussian_vjps')}
        self.roles={}

    def _image(self,x):
        if x.shape!=self.shape or x.dtype!=self.dtype or x.device!=self.device or not bool(torch.isfinite(x).all()):
            raise ValueError('image shape/dtype/device/nonfinite mismatch')
        if tuple(t._version for t in self._original)!=self._versions:
            raise ValueError('physics input changed in place')

    def _fft(self,x,role):
        self.work['fft_batches']+=1;self.work['coil_fft_transforms']+=len(x)
        self.roles.setdefault(role,{'fft_batches':0,'ifft_batches':0})['fft_batches']+=1
        return transforms.fft3c(x)

    def _ifft(self,x,role):
        self.work['ifft_batches']+=1;self.work['coil_ifft_transforms']+=len(x)
        self.roles.setdefault(role,{'fft_batches':0,'ifft_batches':0})['ifft_batches']+=1
        return transforms.ifft3c(x)

    def forward(self,x,*,role='forward'):
        self._image(x);self.work['forward_calls']+=1
        return torch.cat([self._fft(s*x,role)*self.mask for s in self.csm.split(self.chunk_size)],0)

    def adjoint(self,k,*,role='adjoint'):
        if k.shape!=self.csm.shape or k.dtype!=self.dtype or k.device!=self.device:
            raise ValueError('adjoint tensor mismatch')
        self.work['adjoint_calls']+=1
        if role=='rhs':self.work['rhs_adjoint_calls']+=1
        acc=torch.zeros(self.shape,dtype=self.dtype,device=self.device)
        for i in range(0,len(self.csm),self.chunk_size):
            s=self.csm[i:i+self.chunk_size]
            acc+=(s.conj()*self._ifft(k[i:i+self.chunk_size]*self.mask,role)).sum(0)
        return acc

    def normal(self,x,*,role='normal'):
        self._image(x)
        self.work['terminal_normal_calls' if role=='terminal' else 'normal_calls']+=1
        acc=torch.zeros_like(x)
        for s in self.csm.split(self.chunk_size):
            k=self._fft(s*x,role)*self.mask
            acc+=(s.conj()*self._ifft(k,role)).sum(0)
        return acc

    def objective(self,x,anchor,*,role):
        self._image(x);self.work['objective_calls']+=1
        data=0.
        for i in range(0,len(self.csm),self.chunk_size):
            s=self.csm[i:i+self.chunk_size]
            residual=(self._fft(s*x,role)-self.measured[i:i+self.chunk_size])*self.mask
            data+=.5*_norm(residual)**2
        penalty=.5*self.lambda_effective*_norm(x-anchor)**2
        return {'J':data+penalty,'data_term':data,'anchor_penalty':penalty}


def proximal_dc(operator,raw_anchor,*,config=None):
    config=config or ProxConfig()
    if not isinstance(config,ProxConfig):raise ValueError('ProxConfig required')
    operator._image(raw_anchor)
    before=dict(operator.work);before_roles={k:dict(v) for k,v in operator.roles.items()}
    anchor=raw_anchor.detach().clone()*operator.support
    lam=operator.lambda_effective
    ah_y=operator.adjoint(operator.measured,role='rhs')
    rhs=ah_y+lam*anchor
    initial=operator.objective(anchor,anchor,role='objective_anchor')
    nonzero=bool((anchor!=0).any());initial_normal=None
    def apply(x):
        nonlocal initial_normal
        value=operator.normal(x,role='initial_normal' if nonzero and initial_normal is None else 'cg_normal')+lam*x
        if nonzero and initial_normal is None:initial_normal=value.detach().clone()
        return value
    try:
        cg=pcg.solve_pcg(apply,rhs,x0=anchor,maxiter=config.max_matvecs,rtol=config.rtol,
                         max_matvecs=config.max_matvecs)
        image=cg.x.detach()
        terminal=rhs-(operator.normal(image,role='terminal')+lam*image)
        rhs_norm=_norm(rhs);true_norm=_norm(terminal);relative=true_norm/max(rhs_norm,1e-30)
        final=operator.objective(image,anchor,role='objective_candidate')
        tolerance=verification_tolerances(image.dtype)
        gradient=(initial_normal-rhs) if initial_normal is not None else -rhs
        slope=_dot(gradient,image-anchor)
        J_tolerance=tolerance['J_absolute_tolerance']+tolerance['J_relative_tolerance']*max(abs(initial['J']),1e-30)
        J_ok=math.isfinite(final['J']) and final['J']<=initial['J']+J_tolerance
        support_ok=bool((image[~operator.support]==0).all())
        finite=bool(torch.isfinite(image).all()) and math.isfinite(relative) and math.isfinite(slope)
        verified=finite and support_ok and relative<=tolerance['true_residual_tolerance'] and J_ok
        if not cg.converged:status='budget_exhausted'
        elif not verified or (cg.iterations>0 and slope>=0):status='verification_failed'
        else:status='stationary' if cg.iterations==0 else 'converged'
        record={'status':status,'L':operator.L,'lambda_ratio':.1,'lambda_requested':operator.lambda_requested,
                'lambda_effective':lam,'dtype':str(image.dtype),'support_count':int(operator.support.sum()),
                'initial_anchor_nonzero':nonzero,'initial_residual_normal_evaluations':int(nonzero),
                'n_coils':len(operator.csm),'chunk_size':operator.chunk_size,
                'cg_max_matvecs':config.max_matvecs,'cg_rtol':config.rtol,'cg_converged':cg.converged,
                'cg_status':cg.status,'cg_iterations':cg.iterations,'cg_matvecs':cg.matvecs,
                'cg_residual_history':cg.residual_history,'rhs_norm':rhs_norm,'true_residual_norm':true_norm,
                'true_relative_residual':relative,**tolerance,'J_anchor':initial['J'],'J_after':final['J'],
                'data_anchor':initial['data_term'],'data_after':final['data_term'],'anchor_penalty':final['anchor_penalty'],
                'directional_derivative':slope,'J_check_tolerance':J_tolerance,'J_check_passed':J_ok,
                'outside_support_exact_zero':support_ok,'true_endpoint_passed':verified,
                'terminal_normal_evaluations':1,'training_updates':0}
    except Exception as error:
        error.proximal_work={k:operator.work[k]-before[k] for k in before}
        raise
    work={k:operator.work[k]-before[k] for k in before}
    work['roles']={k:{name:v[name]-before_roles.get(k,{}).get(name,0) for name in v} for k,v in operator.roles.items()}
    return ProxResult(image,status,record,work)
