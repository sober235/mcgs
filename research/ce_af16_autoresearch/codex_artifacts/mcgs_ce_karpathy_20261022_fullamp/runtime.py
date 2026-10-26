"""Numerical utilities; imported only by the GSTMI child or CPU tests."""
from __future__ import annotations
import math
import numpy as np
import torch


def normalized_acquired(kspace, mask, scale):
    if not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError('frozen normalization scale must be positive and finite')
    if not torch.is_complex(kspace) or tuple(kspace.shape[1:]) != tuple(mask.shape):
        raise ValueError('expected complex coils-first data and its unchanged spatial mask')
    if not bool(torch.isfinite(mask).all()) or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError('acquired mask must be finite and binary')
    if not bool(torch.isfinite(kspace).all()):
        raise ValueError('nonfinite acquisition')
    return (kspace / float(scale)) * mask.unsqueeze(0)


def zero_parity(candidate, reference):
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != reference.shape or candidate.ndim != 3:
        raise ValueError('canonical spatial shape/order differs')
    if candidate.dtype != np.complex64 or reference.dtype != np.complex64:
        raise ValueError('canonical arrays must be complex64')
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError('canonical arrays must be finite')
    a, b = candidate.astype(np.complex128), reference.astype(np.complex128)
    norm = float(np.linalg.norm(b.ravel()))
    if norm <= 0:
        raise ValueError('canonical reference has zero norm')
    relative = float(np.linalg.norm((a-b).ravel()) / norm)
    return {'passed': relative <= 1e-5, 'complex_relative_l2': relative,
            'tolerance': 1e-5, 'shape': list(a.shape), 'dtype': str(candidate.dtype),
            'array_equal': bool(np.array_equal(candidate, reference))}


def begin_frequency_step(state, resume_offset, iteration):
    if iteration != resume_offset+1+state['completed_steps'] or (
        state['current_iteration'] is not None and state['step_scans'] != ['pre','post']):
        raise ValueError('frequency observer clock or previous actual calls differ')
    state['current_iteration']=iteration
    state['step_scans']=[]
    state['curriculum_iteration']=None


def finish_frequency_step(state, resume_offset, iteration):
    if not (iteration == state['current_iteration'] == resume_offset+1+state['completed_steps']
            and state['curriculum_iteration'] == iteration and state['step_scans']==['pre','post']
            and state['loss_calls_total']==2*(state['completed_steps']+1)):
        raise ValueError('actual two-scan frequency calls are incomplete')
    result={'frequency_t':state['frequency_t'] if 'frequency_t' in state else 1.,
            'loss_call_scans':list(state['step_scans']),'loss_calls_total':state['loss_calls_total'],
            'absolute_iteration':iteration,'source_offset':resume_offset}
    state['completed_steps'] += 1
    state['final_frequency_use']=result
    return result


def acquired_dataset_class(source_class, scales):
    """Reuse source readers/mask/CSM without constructing an unacquired image."""
    from data.transforms import ifft3c

    class FrozenAcquiredDataset(source_class):
        @property
        def ground_truth_image(self):
            raise AttributeError('ground truth is unavailable in the numerical training process')

        def _process_data_simulated(self):
            if self.data_path not in scales or not self.csm_path or not self.mask_path:
                raise ValueError('only frozen acquisition paths, scales, CSM and external masks are allowed')
            if self.center_kspace_dc:
                raise ValueError('this frozen G2 acquisition must not be recentered')
            self._generate_mask()
            self.sensitivity_maps = self._load_external_csm(self.csm_path)
            self.normalization_scale = float(scales[self.data_path])
            measured = normalized_acquired(self.kspace_multicoil, self.mask, self.normalization_scale)
            self.kspace_multicoil = measured
            self.kspace_undersampled_complex = measured.contiguous()
            image = ifft3c(measured)
            zf = (self.sensitivity_maps.conj() * image).sum(0)
            self.zero_filled_complex = zf.contiguous()
            self.zero_filled_image = torch.stack((zf.real, zf.imag), dim=0)
            self.kspace_undersampled = torch.cat((measured.real, measured.imag), dim=0)
            self.kspace_full = None
            self.acquired_only_evidence = {'scale': self.normalization_scale,
                'ground_truth_constructed': False, 'mask_inferred': False,
                'acquired_count': int(self.mask.sum()), 'coils': int(measured.shape[0]),
                'shape': list(measured.shape[1:]), 'outside_mask_nonzero':
                int(torch.count_nonzero(measured * (1-self.mask)[None]))}

        def _process_data_presampled(self):
            raise ValueError('do not infer a new mask or normalization via the presampled pipeline')

        def _normalize_intensity_scale(self):
            raise RuntimeError('reference-derived normalization recomputation is forbidden')

    return FrozenAcquiredDataset
