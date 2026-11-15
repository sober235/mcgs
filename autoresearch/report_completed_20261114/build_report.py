#!/usr/bin/env python3
"""build_report.py -- build the PDF report of all completed MC-GS experiments on the server (CPU only).

Edition of 2026-11-14: the builder of ../report_completed_20261110 (branch `ablation`) with the figure list updated
and every clock / duration output removed (no build time in the manifest, no elapsed seconds in the log).

  /usr/bin/python3 ../gpu-tools/gpu.py run --cpu -- @python autoresearch/report_completed_20261114/build_report.py

Reads report.md next to this file and the figures of record listed in FIGS (read-only), writes everything into a
new directory OUT (default: codex_artifacts/mcgs_completed_report_20261114). Nothing outside OUT is written.

Resolution rule: every raster image must be at least 600 ppi at its printed size.
  - a PNG is placed no wider than pixels / 600 inch (no upsampling, no resampling; alpha is flattened onto white);
  - a PDF figure is placed no larger than natural size * (lowest embedded ppi / 600);
  - after the build, `pdfimages -list` of the finished PDF is the independent check. The build exits non-zero if any
    image is below 600 ppi. The measured per-figure values go into appendix A (second pass) and into manifest.json.

Tools used: pandoc (markdown -> LaTeX body), xelatex (ctexart + xeCJK, Noto CJK fonts), poppler utilities.
Nothing is installed.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
G = "/data0/congcong/code/GS"
R = G + "/Results_20260908"
A = G + "/3DGS/outputs/codex_artifacts"
F = "/home/congcongliu/figs"

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path(A) / "mcgs_completed_report_20261114")
ap.add_argument("--name", default="MCGS_completed_experiments_report_20261114")
ap.add_argument("--min_ppi", type=float, default=600.0)
ap.add_argument("--preview_dpi", type=int, default=80)
args = ap.parse_args()

OUT = args.out
BUILD = OUT / "build"
FIGDIR = BUILD / "figs"
MIN_PPI = args.min_ppi

# page geometry (mm): A4, margins 20 / 22
TEXT_W, TEXT_H = 170.0, 253.0
BOX = {
    "L": (TEXT_H - 2.0, TEXT_W - 20.0),   # landscape page: width x height left for the image
    "P": (TEXT_W, TEXT_H - 30.0),         # portrait page of its own
    "S": (150.0, 105.0),                  # small inline plot
    "M": (TEXT_W, 150.0),                 # medium inline figure
}

HIRES = R + "/T1FLAIR_AF12_16_20_24_figures_hires/pdf/t1flair_v3_af{af}_{c}_row3planes_compact_180_124_76_inset48_headerr_{p}_mcinr_spirit_zf_caps.pdf"
PLANES = "三个平面（横断 180、冠状 124、矢状 76），每列一个方法，下方为误差图，角上为 48 × 48 放大框"

FIGS = [
    # key, path, layout, caption
    ("jyz_trend_psnr", R + "/T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou/figures/jyz_t1flair_af12_16_20_24_head_psnr_lines.pdf", "S",
     "T1+FLAIR：头部 PSNR 随加速倍数的变化，12 个方法。"),
    ("jyz_trend_ssim", R + "/T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou/figures/jyz_t1flair_af12_16_20_24_head_ssim_lines.pdf", "S",
     "T1+FLAIR：头部 SSIM 随加速倍数的变化。"),
    ("widen_dumbbell", G + "/mcgs_widen_wt/figures/widen/widen_margin_dumbbell.pdf", "S",
     "widen 轮次：MC-GS 相对 GS-single 的裕量在各加速倍数上的变化。"),
    ("widen_ceilings", G + "/mcgs_widen_wt/figures/widen/widen_ceilings.pdf", "S",
     "widen 轮次：耦合上限探针。即使伙伴对比度的信息完全已知，联合相对公平单对比臂还能多得多少（上：T1，下：FLAIR）。"),
    ("widen_asym", G + "/mcgs_widen_wt/figures/widen/widen_asym_trend.pdf", "S",
     "widen 轮次：FLAIR 的加速倍数更高、T1 作为伙伴时，联合相对单对比的增益（头部 PSNR 与 HFEN）。"),
    ("ca_dose", F + "/coupling_attribution/ca_dose_lambda_edge.pdf", "M",
     "耦合归因：公共边缘先验权重 λ\\_edge 的剂量—响应。"),
    ("hp_sens", R + "/T1FLAIR_AF16_hparam_v3_jiangyizhou/figures/hp_sensitivity_psnr_ssim.pdf", "S",
     "超参数敏感性：高斯个数与私有点比例对头部 PSNR、SSIM 的影响（AF16）。"),
    ("cap_slice181", A + "/mcgs_hparam_capacity_fig_20261111/hp_capacity_slice181.pdf", "M",
     "高斯容量消融的记录图（用户 2026-11-11 指定）：横断 181 层，真值与 90 万 / 180 万 / 360 万 / 720 万个高斯的重建、48 × 48 放大框和误差图；下方为该层的 PSNR、SSIM、NMSE 和计算开销。"),
    ("noise_T1", R + "/T1FLAIR_AF16_noise0p5x_v3_jiangyizhou/figures/t1flair_v3_af16n05_T1_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit_zf_caps.pdf", "L",
     "外加噪声 0.5σ0 时的 T1 三平面对比（AF16）。"),
    ("noise_FLAIR", R + "/T1FLAIR_AF16_noise0p5x_v3_jiangyizhou/figures/t1flair_v3_af16n05_FLAIR_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit_zf_caps.pdf", "L",
     "外加噪声 0.5σ0 时的 FLAIR 三平面对比（AF16）。"),
    ("mask_record", A + "/mcgs_diffmask_masks_record_tra205_20261112/t1flair_v3_af16_masks_record_format_tra205_box64_600dpi.pdf", "P",
     "采样图案的记录图（用户 2026-11-13 指定）：横断 205，三种图案（变密度泊松、均匀密度、伪径向）× T1 / FLAIR 共六行，十二列为真值、零填充、九个对比方法和 MC-GS；每行下方是 64 × 64 框的放大和误差图。MC-GS 在六行的层面 PSNR、SSIM、NMSE 上全部第一（18 / 18）。"),
]
for af, p in (("12", "p14"), ("16", "p3"), ("20", "p3"), ("24", "p3")):
    for c in ("T1", "FLAIR"):
        FIGS.append((f"jyz_af{af}_{c}", HIRES.format(af=af, c=c, p=p), "L",
                     f"T1+FLAIR AF{af}，{c}：{PLANES}。{'P14' if p == 'p14' else 'AF16 phase-3'} 配方。"))
FIGS += [
    ("t1t2_T1", R + "/T1T2_AF12_retro/figures/t1t2_af12_T1_best3planes.png", "M",
     "T1+T2 AF12，T1：各方法的三平面对比。服务器上只有 PNG，按 600 dpi 放置，需放大查看。"),
    ("t1t2_T2", R + "/T1T2_AF12_retro/figures/t1t2_af12_T2_best3planes.png", "M",
     "T1+T2 AF12，T2：各方法的三平面对比。"),
    ("peds_T1w", R + "/Peds_TSC_MTP_AF12_retro/figures/peds_af12_T1w_row3planes_compact_46_149_171_inset48x48x16_box157_171_46_headerr_lesionB_allfirst_p5_zf_caps.pdf", "L",
     "儿童 TSC AF12，T1w：三平面对比，放大框位于病灶 B。"),
    ("peds_T2s", R + "/Peds_TSC_MTP_AF12_retro/figures/peds_af12_T2s_row3planes_compact_46_149_171_inset48x48x16_box157_171_46_headerr_lesionB_allfirst_p5_zf_caps.pdf", "L",
     "儿童 TSC AF12，T2*：三平面对比，放大框位于病灶 B。"),
    ("ce_prosp_enh", R + "/CE_pre_post_contrast/figures/ce_pair_recon_3planes_enhancement.pdf", "L",
     "CE 前瞻 AF5.33：11 个方法在三个病灶层面上的平扫、增强、增强减平扫。橙色轮廓为病灶核心；每列下方为各参考区域的强化量。没有真值。"),
    ("ce_sept_three", A + "/mcgs_completed_report_20261110/figs_rerender/sept_supfix/ce_af16_lesion_three_planes_201_131_48_prepost_Sept_recipe_DIP_INR_with_support_Oct_audit_settings_zoomerr.pdf", "P",
     "9 月 CE 回顾 AF16（BART 掩膜）：病灶层面横断 201、冠状 131、矢状 48 上的平扫和增强，全部方法；3D-DIP 与 3D-INR 为加头部支撑后的重跑。每个面板标注该层面的 PSNR / SSIM / NMSE，每行最好的值加粗。本次按 600 dpi 重新渲染。"),
    ("lesion_profile", R + "/CE_retro_cssense_af16/figures_seg/nature/ce_retro_af16_lesion_profile.pdf", "S",
     "9 月 CE 回顾 AF16：病灶边界两侧的强度剖面（平扫、增强），以及各方法的病灶对比度误差和 CNR 误差。"),
    ("lesion_metrics", R + "/CE_retro_cssense_af16/figures_seg/ce_retro_af16_lesion_metrics.png", "M",
     "各方法保留的病灶对比度与 CNR（平扫、增强）。服务器上只有 PNG。"),
    ("lesion_axial", R + "/CE_retro_cssense_af16/figures_seg/ce_retro_af16_lesion_axial_201_whole_b1.8_cnone_a7.png", "M",
     "横断 201 层，七个方法的平扫与增强。服务器上只有 PNG。"),
    ("ce_cmp_enh", R + "/CE_retro_cssense_af16_vdpois_cmp/figures/ce_retro_af16_enhancement_cmp.pdf", "L",
     "审计后的 CE vdPois AF16：各方法的平扫、增强、增强减平扫，三个病灶层面。"),
    ("ce_cmp_pre", R + "/CE_retro_cssense_af16_vdpois_cmp/figures/t1flair_v3_af16_pre_row3planes_compact_201_131_48_inset48_headerr_cmp_spirit_zf_caps.png", "L",
     "审计后的 CE vdPois AF16，平扫：三平面对比（横断 201、冠状 131、矢状 48）。服务器上只有 PNG，按 600 dpi 放置。"),
    ("ce_cmp_post", R + "/CE_retro_cssense_af16_vdpois_cmp/figures/t1flair_v3_af16_post_row3planes_compact_201_131_48_inset48_headerr_cmp_spirit_zf_caps.png", "L",
     "审计后的 CE vdPois AF16，增强：三平面对比。服务器上只有 PNG。"),
    ("ce_cmp_lesion", R + "/CE_retro_cssense_af16_vdpois_cmp/figures/ce_retro_af16_lesion_planes_zoom_t204_c118_s44_byplane_cmp.png", "M",
     "审计后的病灶层面放大对比（横断 204、冠状 118、矢状 44）。服务器上只有 PNG。"),
    ("ce_ts01_three", A + "/mcgs_completed_report_20261110/figs_rerender/ts01_s42/ce_af16_lesion_three_planes_196_130_41_prepost_TS01_seed_42_m625_zoomerr.pdf", "P",
     "11 月 keeper TS01（种子 42，第二阶段 625 步）：横断 196、冠状 130、矢状 41 三个病灶层面上的平扫和增强，对全部 10 个方法。面板数字取自冻结评估器，出图脚本重算后核对一致。本次按 600 dpi 渲染。"),
    ("knee_006_E1", R + "/KNEE_SKMTEA_AF16/figures_bm/testset_brainstyle/testset_MTR_006_E1.pdf", "L",
     "膝 BART 掩膜线，测试受试者 MTR\\_006，回波 1：各方法的重建、误差图和放大框。"),
    ("knee_240_E2", R + "/KNEE_SKMTEA_AF16/figures_bm/testset_brainstyle/testset_MTR_240_E2.pdf", "L",
     "膝 BART 掩膜线，测试受试者 MTR\\_240，回波 2。"),
    ("cardiac_ref", R + "/Cardiac_FreeRunning_MEGRE/figures/V07_ref.png", "M",
     "心脏 V07 的参考重建。服务器上只有 PNG。"),
    ("cardiac_compare", R + "/Cardiac_FreeRunning_MEGRE/figures/V07_compare_e0.png", "M",
     "V07 回波 0：不同加速倍数下的重建与参考的比较。服务器上只有 PNG。"),
    ("cardiac_selfcheck", R + "/Cardiac_FreeRunning_MEGRE/figures/V07_selfcheck_e0.png", "M",
     "V07 回波 0：评估口径自检。服务器上只有 PNG。"),
    ("af10_T1", F + "/af12_to_af10/af12_kz_pwc_vs_af10_T1.png", "M",
     "AF12 加已知零、按对比预白化后与 AF10 的比较，T1。服务器上只有 PNG。"),
    ("af10_FLAIR", F + "/af12_to_af10/af12_kz_pwc_vs_af10_FLAIR.png", "M",
     "同上，FLAIR。"),
    ("abl_ce_three", A + "/mcgs_framework_ablation_20261110/ce_slices/fig/ce_af16_lesion_three_planes_201_131_48_prepost_Sept_recipe_GS_single_independent_single_contrast_model_ablation_zoomerr.pdf", "P",
     "9 月 CE 回顾 AF16，层面 201 / 131 / 48：与图 \\ref{fig:ce_sept_three} 相同，只是 GS-single 一列换成框架级消融训练的独立单对比模型。每行最好的值加粗；18 项里 MC-GS 赢 5 项，其余 13 项的第一是独立单对比模型。本次按 600 dpi 渲染。"),
    ("tune_process", A + "/mcgs_framework_ablation_20261110/fig_process/t1t2tb_tuning_process.pdf", "M",
     "tangbing T1+T2 AF12 的调参过程（种子 42）：每轮保留配置的完整模型与当时最好的独立单对比模型的头部 PSNR，下方一行是两者之差。带 * 的改动只属于联合模型。"),
    ("abl_tb_three", A + "/mcgs_framework_ablation_20261110/fig_ablation/ablation_t1t2tb_three_planes_150_113_77.pdf", "P",
     "tangbing T1+T2 AF12，调参后的最终配置：横断 150、冠状 113、矢状 77 上的真值、零填充、完整模型和三种消融，每行下方是放大框和误差图。每个面板标注该层面的 PSNR / SSIM / NMSE，每行最好的值加粗。脚本核对了所画体数据的全头 PSNR 与各自的评分文件一致。"),
    ("ce_af12_three", A + "/mcgs_completed_report_20261114/figs_rerender/af12_supfix/ce_af12_lesion_three_planes_201_131_48_prepost_Sept_recipe_untuned_BART_Poisson_disc_AF12_zoomerr.pdf", "P",
     "CE 回顾 AF12（BART 掩膜）：病灶层面横断 201、冠状 131、矢状 48 上的平扫和增强，全部方法；3D-DIP 与 3D-INR 为加输出支撑的重跑。格式与图 \\ref{fig:ce_sept_three} 相同。本次按 600 dpi 渲染。"),
]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pdf_images(path):
    """[(page, kind, width_px, height_px, x_ppi, y_ppi)] from pdfimages -list."""
    out = run(["pdfimages", "-list", str(path)]).stdout.splitlines()[2:]
    rows = []
    for line in out:
        t = line.split()
        if len(t) < 14:
            continue
        try:
            rows.append((int(t[0]), t[2], int(t[3]), int(t[4]), float(t[12]), float(t[13])))
        except ValueError:
            continue
    return rows


def tex_path(s):
    """A server path as breakable typewriter text."""
    s = s.replace("\\", "/")
    for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    s = s.replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}").replace(".", r".\allowbreak{}")
    return r"{\ttfamily " + s + "}"


def short(path):
    for full, tag in ((A, "A"), (R, "R"), (G, "G"), (F, "F")):
        if path.startswith(full):
            return tag + path[len(full):]
    return path


# ---------------------------------------------------------------- figures
def prepare_figures():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, (key, src, layout, cap) in enumerate(FIGS, 1):
        src_p = Path(src)
        if not src_p.is_file():
            sys.exit(f"missing figure source: {src}")
        rec = {"key": key, "source": src, "layout": layout, "caption": cap, "sha256": sha256(src_p),
               "bytes": src_p.stat().st_size}
        if src_p.suffix.lower() == ".pdf":
            info = run(["pdfinfo", str(src_p)]).stdout
            m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info)
            w_mm, h_mm = float(m.group(1)) / 72 * 25.4, float(m.group(2)) / 72 * 25.4
            imgs = pdf_images(src_p)
            ppis = [min(r[4], r[5]) for r in imgs]
            src_ppi = min(ppis) if ppis else None
            limit = (src_ppi / MIN_PPI) if src_ppi else float("inf")
            local = FIGDIR / f"f{i:02d}_{key}.pdf"
            if local.exists() or local.is_symlink():
                local.unlink()
            os.symlink(src_p, local)
            rec.update(kind="pdf", n_images=len(imgs), src_min_ppi=src_ppi,
                       src_size=f"{w_mm:.0f} × {h_mm:.0f} mm")
        else:
            im = Image.open(src_p)
            px_w, px_h = im.size
            w_mm, h_mm = px_w / MIN_PPI * 25.4, px_h / MIN_PPI * 25.4   # size at exactly MIN_PPI
            limit = 1.0
            local = FIGDIR / f"f{i:02d}_{key}.png"
            if im.mode in ("RGBA", "LA", "P"):
                rgba = im.convert("RGBA")
                flat = Image.new("RGB", rgba.size, (255, 255, 255))
                flat.paste(rgba, mask=rgba.split()[3])
            else:
                flat = im.convert("RGB")
            flat.save(local, format="PNG", dpi=(MIN_PPI, MIN_PPI))
            rec.update(kind="png", n_images=1, src_min_ppi=None, src_size=f"{px_w} × {px_h} px",
                       px=(px_w, px_h))
        bw, bh = BOX[layout]
        scale = min(bw / w_mm, bh / h_mm, limit)
        place_w, place_h = w_mm * scale, h_mm * scale
        if rec["kind"] == "png":
            exp_ppi = rec["px"][0] / (place_w / 25.4)
        else:
            exp_ppi = (rec["src_min_ppi"] / scale) if rec["src_min_ppi"] else None
        rec.update(local=local.name, place_w_mm=round(place_w, 2), place_h_mm=round(place_h, 2),
                   expected_min_ppi=None if exp_ppi is None else round(exp_ppi, 1))
        recs.append(rec)
    return recs


def figure_tex(rec):
    inc = rf"\includegraphics[width={rec['place_w_mm']:.2f}mm]{{figs/{rec['local']}}}"
    cap = rf"{rec['caption']}"
    lab = rf"\label{{fig:{rec['key']}}}"
    if rec["layout"] == "L":
        return ("\\begin{landscape}\n\\thispagestyle{plain}\n\\vspace*{\\fill}\n\\begin{center}\n" + inc +
                "\n\\captionof{figure}{" + cap + "}" + lab + "\n\\end{center}\n\\vspace*{\\fill}\n\\end{landscape}\n")
    if rec["layout"] == "P":
        return ("\\clearpage\n\\vspace*{\\fill}\n\\begin{center}\n" + inc + "\n\\captionof{figure}{" + cap + "}" + lab +
                "\n\\end{center}\n\\vspace*{\\fill}\n\\clearpage\n")
    return ("\\begin{figure}[H]\n\\centering\n" + inc + "\n\\caption{" + cap + "}" + lab + "\n\\end{figure}\n")


def appendix_a(recs, measured):
    """LaTeX of appendix A. measured: key -> (page, min ppi, n images) from the finished PDF, or None on pass 1."""
    L = [r"\section*{附录 A：图清单与分辨率}", r"\addcontentsline{toc}{section}{附录 A：图清单与分辨率}", ""]
    L.append("下表逐图列出来源文件、类型、放置宽度，以及成品 PDF 里该图所在页上全部位图的最低分辨率。"
             "最低分辨率由 \\texttt{pdfimages -list} 对成品 PDF 实测（取水平和垂直方向的较小值）；"
             "同一页上有多幅图时，各图记的是该页的最小值。纯矢量图没有位图，记作「矢量」。"
             "来源一栏的路径前缀见附录 B（F = /home/congcongliu/figs）；完整路径和 SHA-256 在同目录的 \\texttt{manifest.json}。")
    L.append("")
    L.append(r"\begingroup\footnotesize")
    L.append(r"\begin{longtable}{@{}p{7mm}p{76mm}p{36mm}p{14mm}p{10mm}p{12mm}@{}}")
    head = r"图 & 来源 & 类型与源尺寸 & 放置宽度 & 位图数 & 最低 ppi\tabularnewline"
    L += [r"\toprule", head, r"\midrule", r"\endfirsthead", r"\toprule", head, r"\midrule", r"\endhead",
          r"\bottomrule", r"\endlastfoot"]
    for rec in recs:
        kind = ("PDF，" if rec["kind"] == "pdf" else "PNG，") + rec["src_size"]
        m = measured.get(rec["key"]) if measured else None
        if m is None:
            ppi, n = "待测", "—"
        elif m[2] == 0:
            ppi, n = "矢量", "0"
        else:
            ppi, n = f"{m[1]:.0f}", str(m[2])
        L.append(rf"\ref{{fig:{rec['key']}}} & \raggedright {tex_path(short(rec['source']))} & \raggedright {kind} & "
                 rf"{rec['place_w_mm']:.0f} mm & {n} & {ppi}\tabularnewline")
    L += [r"\end{longtable}", r"\endgroup", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- LaTeX
PREAMBLE = r"""\documentclass[UTF8,a4paper,zihao=5,fontset=none,scheme=chinese]{ctexart}
\usepackage[a4paper,left=20mm,right=20mm,top=22mm,bottom=22mm,headheight=14pt]{geometry}
\setmainfont{TeX Gyre Termes}
\setsansfont{TeX Gyre Heros}
\setmonofont{DejaVu Sans Mono}[Scale=0.80]
\setCJKmainfont{Noto Serif CJK SC}
\setCJKsansfont{Noto Sans CJK SC}
\setCJKmonofont{Noto Sans Mono CJK SC}[Scale=0.90]
\usepackage{graphicx,longtable,booktabs,array,etoolbox,float,pdflscape,xcolor,fancyhdr,enumitem}
\usepackage[font=small,labelfont=bf,labelsep=quad,skip=6pt]{caption}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=black,bookmarksnumbered=true]{hyperref}
\hypersetup{pdftitle={MC-GS 已完成实验结果报告},pdfsubject={MC-GS completed experiments, 2026-11-14}}
\tracinglostchars=2
\ctexset{
  section={format=\Large\bfseries\sffamily\raggedright,beforeskip=3.2ex plus .6ex minus .2ex,afterskip=1.6ex plus .2ex},
  subsection={format=\large\bfseries\sffamily\raggedright,beforeskip=2.4ex plus .4ex minus .2ex,afterskip=1.0ex plus .2ex},
  subsubsection={format=\normalsize\bfseries\sffamily\raggedright},
}
\providecommand{\tightlist}{\setlength{\itemsep}{2pt}\setlength{\parskip}{0pt}}
\setlist{leftmargin=2em,topsep=3pt}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\setlength{\LTcapwidth}{\textwidth}
\setlength{\LTpre}{8pt}\setlength{\LTpost}{10pt}
\AtBeginEnvironment{longtable}{\footnotesize}
\captionsetup[table]{position=top,skip=4pt}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\sffamily MC-GS 已完成实验结果报告}
\fancyhead[R]{\small\sffamily 2026-11-14}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\small\thepage}\renewcommand{\headrulewidth}{0pt}}
\setcounter{tocdepth}{2}
\emergencystretch=2em
\begin{document}
\begin{center}
{\LARGE\bfseries\sffamily MC-GS 已完成实验结果报告\par}
\vspace{6pt}
{\large\sffamily 多对比度 3D 高斯泼溅 MRI 重建：各研究线的结论、数字表与代表图\par}
\vspace{8pt}
{\small 截至 2026-11-14（2026-11-10 版的更新）\quad·\quad 仓库 \texttt{sober235/mcgs}\quad·\quad 图中位图在印刷尺寸下不低于 600 dpi\par}
\end{center}
\vspace{4pt}
\tableofcontents
\clearpage
"""


def fix_tables(body):
    """pandoc 2.9 gives minipage columns that sum to ~0.92 columnwidth and ignores \\tabcolsep; rescale to fit."""
    def one(m):
        block = m.group(0)
        spec = re.search(r"\\begin\{longtable\}\[\]\{@\{\}([lrc]+)@\{\}\}", block)
        if spec is None:
            return block
        n = len(spec.group(1))
        fr = [float(x) for x in re.findall(r"\\begin\{minipage\}\[[bt]\]\{([\d.]+)\\columnwidth\}", block)[:n]]
        if len(fr) < n:
            return block
        avail = 1.0 - (2 * n - 2) * 3.0 / (TEXT_W / 25.4 * 72.27)     # tabcolsep 3pt, @{} at both ends
        k = avail / sum(fr) * 0.995
        return re.sub(r"(\\begin\{minipage\}\[[bt]\]\{)([\d.]+)(\\columnwidth\})",
                      lambda mm: f"{mm.group(1)}{float(mm.group(2)) * k:.4f}{mm.group(3)}", block)
    return re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}", one, body, flags=re.S)


def fix_texttt(body):
    def one(m):
        s = m.group(1)
        s = s.replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")
        return r"\texttt{" + s + "}"
    return re.sub(r"\\texttt\{((?:[^{}]|\\[{}]|\{\})*)\}", one, body)


def build_tex(recs, measured):
    md = (HERE / "report.md").read_text(encoding="utf-8")
    by_key = {r["key"]: r for r in recs}
    used = []

    def sub(m):
        used.append(m.group(1))
        return "\n```{=latex}\n" + figure_tex(by_key[m.group(1)]) + "```\n"
    md = re.sub(r"^\{\{FIG:(\w+)\}\}\s*$", sub, md, flags=re.M)
    missing = [k for k in by_key if k not in used]
    if missing or len(used) != len(set(used)):
        sys.exit(f"figure placeholders do not match FIGS: unused {missing}, used {len(used)}")
    md = md.replace("{{APPENDIX_A}}", "\n```{=latex}\n" + appendix_a(recs, measured) + "\n```\n")
    if "{{" in md:
        sys.exit("unresolved placeholder in report.md")
    (BUILD / "report_expanded.md").write_text(md, encoding="utf-8")
    p = run(["pandoc", "-f", "markdown", "-t", "latex", "--columns=1", "--wrap=none",
             str(BUILD / "report_expanded.md")])
    if p.returncode != 0:
        sys.exit("pandoc failed:\n" + p.stderr)
    body = fix_texttt(fix_tables(p.stdout))
    (BUILD / "main.tex").write_text(PREAMBLE + body + "\n\\end{document}\n", encoding="utf-8")


def xelatex():
    # the PDF's creation date is pinned to the edition date (2026-11-14 12:00 UTC), not read from the machine's clock
    env = dict(os.environ, SOURCE_DATE_EPOCH="1794657600", FORCE_SOURCE_DATE="1")
    p = run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=BUILD, env=env)
    log = (BUILD / "main.log").read_text(encoding="utf-8", errors="replace") if (BUILD / "main.log").exists() else ""
    if p.returncode != 0 or not (BUILD / "main.pdf").exists():
        tail = "\n".join((p.stdout or "").splitlines()[-60:])
        sys.exit("xelatex failed:\n" + tail)
    print("[xelatex] pass done", flush=True)
    return log


def fig_pages():
    aux = (BUILD / "main.aux").read_text(encoding="utf-8", errors="replace")
    pages = {}
    for m in re.finditer(r"\\newlabel\{fig:(\w+)\}\{\{(\d+)\}\{(\d+)\}", aux):
        pages[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return pages


def measure(recs):
    """key -> (page, min ppi on that page, n images on that page), from the finished PDF."""
    imgs = pdf_images(BUILD / "main.pdf")
    per_page = {}
    for page, kind, w, h, xp, yp in imgs:
        per_page.setdefault(page, []).append(min(xp, yp))
    pages = fig_pages()
    out = {}
    for rec in recs:
        if rec["key"] not in pages:
            sys.exit(f"no page for figure {rec['key']}")
        pg = pages[rec["key"]][1]
        v = per_page.get(pg, [])
        out[rec["key"]] = (pg, min(v) if v else None, len(v))
    return out, imgs


def main():
    for tool in ("pandoc", "xelatex", "pdfinfo", "pdfimages", "pdffonts", "pdftoppm"):
        if shutil.which(tool) is None:
            sys.exit(f"missing tool: {tool}")
    OUT.mkdir(parents=True, exist_ok=True)
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    recs = prepare_figures()
    print(f"[figures] {len(recs)} prepared", flush=True)

    # pass 1: layout and page numbers, then measure the images of the finished PDF
    build_tex(recs, None)
    xelatex()
    xelatex()
    measured, _ = measure(recs)
    # pass 2: appendix A with the measured values (image placement does not change)
    build_tex(recs, measured)
    xelatex()
    log = xelatex()
    measured2, imgs = measure(recs)
    if {k: v[1:] for k, v in measured.items()} != {k: v[1:] for k, v in measured2.items()}:
        sys.exit("image resolution changed between passes")

    lost = sorted(set(re.findall(r"Missing character: There is no (\S+)", log)))
    undefined = re.findall(r"LaTeX Warning: Reference `([^']+)'", log)
    overfull = [float(x) for x in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)]
    low = [(pg, kind, w, h, xp, yp) for pg, kind, w, h, xp, yp in imgs if min(xp, yp) < MIN_PPI - 0.5]

    final = OUT / f"{args.name}.pdf"
    shutil.copyfile(BUILD / "main.pdf", final)
    info = run(["pdfinfo", str(final)]).stdout
    n_pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    fonts = run(["pdffonts", str(final)]).stdout.splitlines()[2:]
    not_embedded = [f for f in fonts if len(f.split()) >= 5 and f.split()[-5] == "no"]
    pages = fig_pages()
    for rec in recs:
        pg, ppi, n = measured2[rec["key"]]
        rec.update(figure_number=pages[rec["key"]][0], page=pg, measured_min_ppi_on_page=ppi, images_on_page=n)
    manifest = {
        "pdf": str(final), "sha256": sha256(final), "bytes": final.stat().st_size, "pages": n_pages,
        "min_ppi_required": MIN_PPI, "raster_images": len(imgs),
        "min_ppi_measured": min((min(r[4], r[5]) for r in imgs), default=None),
        "images_below_min_ppi": low, "fonts": len(fonts), "fonts_not_embedded": not_embedded,
        "missing_characters": lost, "undefined_references": undefined,
        "overfull_hbox_count": len(overfull), "overfull_hbox_max_pt": max(overfull, default=0.0),
        "report_md_sha256": sha256(HERE / "report.md"), "build_script_sha256": sha256(Path(__file__)),
        "figures": recs,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    for name in ("main.tex", "report_expanded.md"):
        shutil.copyfile(BUILD / name, OUT / name)
    shutil.copyfile(HERE / "report.md", OUT / "report.md")
    shutil.copyfile(Path(__file__), OUT / "build_report.py")

    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if art:
        art = Path(art)
        art.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUT / "manifest.json", art / "manifest.json")
        (art / "preview").mkdir(exist_ok=True)
        run(["pdftoppm", "-r", str(args.preview_dpi), "-jpeg", "-jpegopt", "quality=70", str(final),
             str(art / "preview" / "p")])
        (art / "log_tail.txt").write_text("\n".join(log.splitlines()[-200:]), encoding="utf-8")

    print(json.dumps({k: v for k, v in manifest.items() if k != "figures"}, ensure_ascii=False, indent=1))
    for rec in recs:
        print(f"fig {rec['figure_number']:>2} p{rec['page']:>3} {rec['kind']} {rec['layout']} "
              f"w={rec['place_w_mm']:.0f}mm imgs={rec['images_on_page']} min_ppi={rec['measured_min_ppi_on_page']} "
              f"{rec['key']}")
    print(f"[done] -> {final}")
    if low or lost or undefined or not_embedded:
        sys.exit("CHECK FAILED: see images_below_min_ppi / missing_characters / undefined_references / fonts_not_embedded")


if __name__ == "__main__":
    main()
