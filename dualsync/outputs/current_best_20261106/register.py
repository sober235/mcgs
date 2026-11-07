"""Register Yuantian sharp (seed 1234) as the user's current optimal result. CPU only, no inference.

Copies the two existing .mat files byte-for-byte, exports magnitude PNGs with the same input p99.5
window and ROI as outputs/final_best_20261106, and writes manifest.json / README.md here.
Refuses to overwrite anything. The previous registration (final_best_20261106) is left untouched.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / 'outputs/yuantian_repro_20261106'
RUN = SRC / 'sharp_seed1234'
PREV = ROOT / 'outputs/final_best_20261106'
CASES = ('MSP-FSE-T/slice_06', 'DYF-FSE-T/slice_01')
INSTRUCTION = 'Record the "yuantian sharp" as the CURRENT optimal result'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    for name in ('manifest.json', 'README.md'):
        assert not (HERE / name).exists(), 'refusing to overwrite ' + name
    cfg = json.loads((RUN / 'yuantian_cfg.json').read_text())
    metrics = json.loads((SRC / 'metrics.json').read_text())
    prev = json.loads((PREV / 'manifest.json').read_text())
    prev_rec = {r['case']: r for r in prev['records']}
    sources = {str(RUN / 'yuantian_cfg.json'): sha(RUN / 'yuantian_cfg.json'),
               str(SRC / 'metrics.json'): sha(SRC / 'metrics.json'),
               str(PREV / 'manifest.json'): sha(PREV / 'manifest.json')}
    records = []
    for case in CASES:
        src = RUN / (case + '.mat')
        m = metrics[case]['Yuantian sharp']
        assert sha(src) == m['sha256'], 'source changed since evaluation: ' + case
        sources[str(src)] = m['sha256']
        mat = loadmat(str(src), simplify_cells=True)
        old = loadmat(prev_rec[case]['registered_mat'], simplify_cells=True)
        assert np.array_equal(mat['cg_sense'], old['cg_sense']), 'input differs from previous registration'
        dest = HERE / case
        dest.parent.mkdir(parents=True, exist_ok=True)
        matpath = dest.with_suffix('.mat')
        assert not matpath.exists()
        shutil.copy2(src, matpath)
        assert sha(matpath) == m['sha256']
        scale = prev_rec[case]['input_p99_5_scale']
        assert abs(scale - float(np.percentile(np.abs(mat['cg_sense']).astype(float), 99.5))) < 1e-6
        r, c = prev_rec[case]['roi_start']
        mag = np.abs(mat['diff']).astype(float) / scale
        images = {}
        for label, data in (('full', mag), ('roi', mag[r:r + 96, c:c + 96])):
            path = dest.parent / (dest.name + '_' + label + '.png')
            assert not path.exists()
            plt.imsave(path, data, cmap='gray', vmin=0, vmax=1, origin='upper')
            got = np.asarray(Image.open(path))
            np.testing.assert_array_equal(got, plt.get_cmap('gray')(np.clip(data, 0, 1), bytes=True))
            images[label] = dict(path=str(path), sha256=sha(path), shape=list(got.shape[:2]))
        records.append(dict(
            case=case, base_seed=1234, source_result=str(src), source_sha256=m['sha256'],
            registered_mat=str(matpath), registered_mat_sha256=sha(matpath),
            input_p99_5_scale=scale, roi_start=[r, c], images=images,
            metrics_vs_previous_final={k: m[k] for k in (
                'bg_snr_gain', 'inband_snr_gain', 'tuned_coarse_ssim', 'input_coarse_ssim',
                'tuned_input_coarse_ssim', 'strong_edge_amplitude_ratio', 'symmetric_edge_p95_pixels',
                'fg_mean_ratio', 'raw_gradient_retention', 'dc_rel')},
            frozen_gates_failed=m['failed']))
    for p, h in sources.items():
        assert sha(p) == h
    manifest = dict(
        status='user_selected_current_best',
        user_instruction=INSTRUCTION,
        decision_date='2026-11-06',
        method='Yuantian port, profile sharp (main_yuantian.py --profile sharp --seed 1234)',
        selection_basis='User choice after viewing outputs/yuantian_repro_20261106 comparison figures; '
                        'not a numerical optimum under the frozen gates.',
        frozen_gate_note='Fails the frozen structure gates on both slices: input coarse SSIM '
                         '(.934/.952 vs final .986/.988, limit drop .001) and coarse SSIM to the previous '
                         'final (.977/.987, limit .99). Recorded as the user decision; gates not changed.',
        yuantian_cfg=cfg['cfg'], timesteps=cfg['timesteps'], checkpoint=cfg['checkpoint'],
        source_code_sha256=cfg['source_sha256'], run_fingerprint=cfg['fingerprint'],
        scope='Two slices only (MSP06, DYF01), single reconstruction, seed 1234. '
              'No full-38 run of this profile exists yet.',
        supersedes=dict(path=str(PREV), manifest_sha256=sources[str(PREV / 'manifest.json')],
                        note='Previous registration kept unchanged as history.'),
        records=records, source_sha256=sources, registration_source_sha256=sha(__file__))
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    (HERE / 'README.md').write_text(
        '# 当前最优结果：Yuantian sharp\n\n'
        '2026-11-06 用户指示："' + INSTRUCTION + '"。\n\n'
        '方法：Yuantian 移植版 sharp 配置（切向 0.25、法向 0.10、每步更新上限 0.04、mu 10、t_jac 150、'
        'sigma_y 0.3、50 步），base seed 1234，单次重建。\n\n'
        '- MSP06：[完整图](MSP-FSE-T/slice_06_full.png) · [局部图](MSP-FSE-T/slice_06_roi.png) · '
        '[复数数据](MSP-FSE-T/slice_06.mat)\n'
        '- DYF01：[完整图](DYF-FSE-T/slice_01_full.png) · [局部图](DYF-FSE-T/slice_01_roi.png) · '
        '[复数数据](DYF-FSE-T/slice_01.mat)\n\n'
        '| 切片 | 背景 SNR | 带内 SNR | 与旧最终图粗尺度 SSIM | 与输入粗尺度 SSIM | 强边缘幅度 | 冻结门槛 |\n'
        '|---|---:|---:|---:|---:|---:|---|\n' + ''.join(
            '| %s | ×%.2f | ×%.2f | %.3f | %.3f（旧最终图 %.3f） | %.3f | 未通过：%s |\n' % (
                rec['case'], *(rec['metrics_vs_previous_final'][k] for k in (
                    'bg_snr_gain', 'inband_snr_gain', 'tuned_coarse_ssim', 'input_coarse_ssim',
                    'tuned_input_coarse_ssim', 'strong_edge_amplitude_ratio')),
                '、'.join(rec['frozen_gates_failed'])) for rec in records) +
        '\n说明：这是用户选择，不是冻结门槛下的数值最优；sharp 在两张图上都未通过粗尺度 SSIM 门槛。'
        '只有这两张切片、一个 seed，尚无 38 张全量结果。MAT 为源文件原字节副本；PNG 使用与旧登记相同的输入 p99.5 窗口和 ROI，'
        '未做额外滤波或推理。\n\n'
        '上一登记 [final_best_20261106](../final_best_20261106/README.md)（原始 Accepted Tuned）原样保留为历史记录。\n')
    print(json.dumps(dict(status=manifest['status'], images=len(records), mat_copies_byte_identical=True,
                          png_pixel_checks_passed=True)))


if __name__ == '__main__':
    main()
