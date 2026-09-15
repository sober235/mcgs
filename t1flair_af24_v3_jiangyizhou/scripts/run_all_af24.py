#!/usr/bin/env python3
"""run_all_af24.py -- every arm of the jiangyizhou AF24 study, queued by priority onto whichever card is free (2026-09-14).

Why a queue and not fixed cards: when the study was prepared (21:40) five of the eight cards carried jobs of this user's
other sessions (cardiac radial ZS-SSDU / DIP, peds MC-GS, a detector); a fixed six-card plan would have left chains
waiting on cards that other sessions' schedulers re-take within seconds of freeing them.

Rules
- One GPU job of this user per card. A card is eligible when no compute process of this uid is on it (other sessions
  count: two MC-GS trainers on one card hung it twice on 2026-09-11), no job of this queue is running on it (covers the
  seconds before a new process shows in nvidia-smi), and it has the job's memory free.
- At most --max_gpu_jobs GPU jobs of this queue at once, on --cards.
- L1-ESPIRiT (BART pics on the CPU, 12 threads; its data and scoring pass on a GPU) runs in --max_cpu_jobs CPU slots and
  takes no GPU slot: it goes to a card where this queue runs a classical job, else the card with the most free memory
  (at AF20 it shared card 7 with the TV sweep).
- Priority: MC-GS joint; GS-single FLAIR; classical base T1, FLAIR; GS-single T1; converged-TV and joint-TV sweeps;
  ZS-SSDU; 3D-INR; 3D-DIP.
- A job whose result file exists is skipped. A job already running (its script and arguments found in a live process of
  this uid) is waited for, not relaunched. A non-MC-GS job that exits non-zero without its result is retried once; a
  failed MC-GS run is only logged (run_af24_mcgs.sh refuses to overwrite its outputs, so it needs a look first).
- When nothing is queued or running: finalize_af24_jyz.sh with the three MC-GS runs (their processes have exited, so the
  volume npz files are complete).
Arguments of every arm are those of the AF20 topic (run_classical_af20.sh, run_stage2_af20.sh, run_af20_mcgs.sh); the
lambda grids are the AF20 topic's final grids plus one point below each.
Launch detached from a bash script file:  setsid nohup PY -u scripts/run_all_af24.py > /dev/null 2>&1 < /dev/null &
Status: logs/run_all_af24.log, logs/run_all_af24_status.json, logs/RUN_ALL_AF24_DONE.
--dry_run prints the queue, the skips and the card eligibility once, and launches nothing.
--study draws (2026-09-15) queues the mask-draw variance study instead, as at AF20: per extra draw b, c, d
(scripts/prep_03b_pick_export_draws_af24.py) MC-GS joint with the frozen recipe (MASK_DRAW=<d>, no volumes) and
L1-ESPIRiT at the AF24 optimum and both neighbours of the main sweep (T1 0.0007 / 0.001 / 0.0015, FLAIR 0.0014 /
0.002 / 0.003; mask swapped through T1T2_MASK_<CONTRAST>, outputs suffixed _d<d>_lam<L>); at the end
scripts/score_mask_draws_af24.py instead of finalize. Logs logs/run_draws_af24{.log,_status.json}, logs/RUN_DRAWS_AF24_DONE.
--study extra (2026-09-15) queues the two arms the AF12 / 16 / 20 topics gained on 2026-09-14/15, so AF24 matches them:
MC-INR (Niessen 2025; mcinr_niessen2025 repo, run_volume.py --af 24 --resume --yield_to_others, GR_pt1.10 env; exit 3 =
it yielded to another process of this user and is queued again to resume) and the L1-SPIRiT sweep of the AF20 topic
(scripts/recon_spirit.py, ksize 3 / 5 x lam 0, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 150 it, chunk 24). No automatic post-step:
score.py --af 24, the registration and the figures are run by hand. Logs logs/run_extra_af24{.log,_status.json}.
"""
import argparse, json, os, subprocess, time
from pathlib import Path

J = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou")
F = Path("/data0/congcong/code/GS/mcgs_af16_wt")
PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
ENV = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
UID = os.getuid()
GS = {"joint": ("af24_p3_joint_full4k", ""), "singleFLAIR": ("af24_p3_singleFLAIR_full4k", "FLAIR"),
      "singleT1": ("af24_p3_singleT1_full4k", "T1")}
TV = {"T1": ["0.00035", "0.0005", "0.0007", "0.001", "0.0014"], "FLAIR": ["0.00025", "0.00035", "0.0005", "0.0007", "0.001"]}
JTV = ["0.00035", "0.0005", "0.0007", "0.001", "0.0014"]
L1 = {"T1": ["0.00035", "0.0005", "0.0007", "0.001", "0.0015", "0.0025", "0.004"],
      "FLAIR": ["0.0014", "0.002", "0.003", "0.005", "0.007"]}
CLASSICAL = ("base_", "tv_", "jtv_")
DRAWS = ("b", "c", "d")
DRAW_L1 = {"T1": ["0.0007", "0.001", "0.0015"], "FLAIR": ["0.0014", "0.002", "0.003"]}
MCINR = Path("/data0/congcong/code/GS/mcinr_niessen2025")
MCINR_TOPIC = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou")
PY_TCNN = "/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python"
SPIRIT_KS = ("3", "5")
SPIRIT_LAM = ("0", "0.0005", "0.001", "0.002", "0.004", "0.008")


def all_jobs():
    L = J / "logs"
    q = []

    def add(name, kind, need, cwd, cmd, done, log, script, args, absent=()):
        # (script, args, absent): the exact argv tokens that identify this job's live process
        q.append(dict(name=name, kind=kind, need=need, cwd=str(cwd), cmd=cmd, done=str(done), log=str(log),
                      script=script, args=args, absent=list(absent)))

    def gs(arm):
        exp, single = GS[arm]
        add(f"gs_{arm}", "gpu", 38, F, f"SINGLE={single} SAVE_VOLUMES=1 ./run_af24_mcgs.sh {exp} {{gpu}} 4000",
            F / f"results_arc/{exp}_score.json", L / f"sched_gs_{arm}.log", "arc/cycle.py", {"--exp": exp})

    def base(c):
        add(f"base_{c}", "gpu", 12, J, f"{PY} -u scripts/recon_classical.py --contrast {c} --gpu {{gpu}}",
            L / f"classical_{c}.json", L / f"classical_{c}.log", "recon_classical.py", {"--contrast": c}, absent=["--suffix"])

    gs("joint"); gs("singleFLAIR"); base("T1"); base("FLAIR"); gs("singleT1")
    for c in ("T1", "FLAIR"):
        for lam in TV[c]:
            add(f"tv_{c}_{lam}", "gpu", 12, J,
                f"{PY} -u scripts/recon_classical.py --contrast {c} --gpu {{gpu}} --methods cssense --admm_outer 200 --lam {lam} --suffix _o200_lam{lam}",
                L / f"classical_{c}_o200_lam{lam}.json", L / f"classical_{c}_o200_lam{lam}.log",
                "recon_classical.py", {"--contrast": c, "--suffix": f"_o200_lam{lam}"})
    for lam in JTV:
        add(f"jtv_{lam}", "gpu", 20, J,
            f"{PY} -u scripts/recon_joint_cssense.py --topic {J} --contrasts T1,FLAIR --gpu {{gpu}} --admm_outer 200 --lam {lam} --suffix _o200_lam{lam}",
            L / f"jcssense_o200_lam{lam}.json", L / f"jcssense_o200_lam{lam}.log",
            "recon_joint_cssense.py", {"--suffix": f"_o200_lam{lam}"})
    for c in ("T1", "FLAIR"):
        add(f"zsssdu_{c}", "gpu", 18, J, f"{PY} -u scripts/recon_zsssdu.py --epochs 100 --contrast {c} --gpu {{gpu}}",
            L / f"zsssdu_{c}.json", L / f"zsssdu_{c}.log", "recon_zsssdu.py", {"--contrast": c})
    for c in ("T1", "FLAIR"):
        add(f"inr_{c}", "gpu", 18, J,
            f"{PY} -u scripts/recon_dip_inr.py --method inr --iters 1500 --eval_every 50 --contrast {c} --gpu {{gpu}}",
            L / f"inr_{c}.json", L / f"inr_{c}.log", "recon_dip_inr.py", {"--method": "inr", "--contrast": c})
    for c in ("T1", "FLAIR"):
        add(f"dip_{c}", "gpu", 40, J,
            f"{PY} -u scripts/recon_dip_inr.py --method dip --iters 5000 --lr 3e-3 --eval_every 100 --contrast {c} --gpu {{gpu}}",
            L / f"dip_{c}.json", L / f"dip_{c}.log", "recon_dip_inr.py", {"--method": "dip", "--contrast": c})
    for c in ("T1", "FLAIR"):
        for lam in L1[c]:
            add(f"l1_{c}_{lam}", "cpu", 12, J,
                f"{PY} -u scripts/recon_l1espirit.py --topic {J} --contrast {c} --gpu {{gpu}} --lam {lam} --iters 100 --suffix _lam{lam} --threads 12",
                L / f"l1espirit_{c}_lam{lam}.json", L / f"l1espirit_{c}_lam{lam}.log",
                "recon_l1espirit.py", {"--contrast": c, "--suffix": f"_lam{lam}"})
    return q


def draw_jobs():
    L, q = J / "logs", []
    for d in DRAWS:
        exp = f"af24_p3_joint_d{d}_full4k"
        q.append(dict(name=f"gs_joint_d{d}", kind="gpu", need=38, cwd=str(F),
                      cmd=f"MASK_DRAW={d} ./run_af24_mcgs.sh {exp} {{gpu}} 4000",
                      done=str(F / f"results_arc/{exp}_score.json"), log=str(L / f"sched_gs_joint_d{d}.log"),
                      script="arc/cycle.py", args={"--exp": exp}, absent=[]))
    for d in DRAWS:
        for c in ("T1", "FLAIR"):
            for lam in DRAW_L1[c]:
                q.append(dict(name=f"l1_{c}_d{d}_{lam}", kind="cpu", need=12, cwd=str(J),
                              cmd=(f"T1T2_MASK_{c}={J}/data/mask_af24_d{d}_3d.h5 {PY} -u scripts/recon_l1espirit.py --topic {J} "
                                   f"--contrast {c} --gpu {{gpu}} --lam {lam} --iters 100 --suffix _d{d}_lam{lam} --threads 12"),
                              done=str(L / f"l1espirit_{c}_d{d}_lam{lam}.json"), log=str(L / f"l1espirit_{c}_d{d}_lam{lam}.log"),
                              script="recon_l1espirit.py", args={"--contrast": c, "--suffix": f"_d{d}_lam{lam}"}, absent=[]))
    return q


def extra_jobs():
    L, q = J / "logs", []
    q.append(dict(name="mcinr_af24", kind="gpu", need=10, cwd=str(MCINR), requeue_rc=3,
                  cmd=f"{PY_TCNN} -u scripts/run_volume.py --af 24 --gpu {{gpu}} --resume --yield_to_others",
                  done=str(MCINR_TOPIC / "recon/af24/meta.json"), log=str(MCINR_TOPIC / "logs/run_af24.log"),
                  script="run_volume.py", args={"--af": "24"}, absent=[]))
    for c in ("T1", "FLAIR"):
        for ks in SPIRIT_KS:
            for lam in SPIRIT_LAM:
                sfx = f"_ks{ks}_lam{lam}"
                q.append(dict(name=f"spirit_{c}{sfx}", kind="gpu", need=24, cwd=str(J),
                              cmd=(f"{PY} -u scripts/recon_spirit.py --contrast {c} --gpu {{gpu}} --ksize {ks} --lam {lam} "
                                   f"--iters 150 --chunk 24 --suffix {sfx}"),
                              done=str(L / f"spirit_{c}{sfx}.json"), log=str(L / f"spirit/{c}{sfx}.log"),
                              script="recon_spirit.py", args={"--contrast": c, "--suffix": sfx}, absent=[]))
    return q


def my_argvs():
    """argv token lists of this uid's live processes"""
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != UID:
                continue
            toks = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        out.append([t.decode(errors="replace") for t in toks if t])
    return out


def matches(argv, j):
    if not any(t.endswith(j["script"]) for t in argv):
        return False
    for k, v in j["args"].items():
        if not any(argv[i] == k and argv[i + 1] == v for i in range(len(argv) - 1)):
            return False
    return not any(k in argv for k in j["absent"])


def gpu_state():
    """per card index: free GiB and whether a compute process of this uid is on it"""
    st, bus2i = {}, {}
    for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,pci.bus_id,memory.free",
                                         "--format=csv,noheader,nounits"], text=True).strip().splitlines():
        i, b, fr = [x.strip() for x in line.split(",")]
        st[int(i)] = dict(free=float(fr) / 1024, mine=False); bus2i[b] = int(i)
    for line in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id",
                                         "--format=csv,noheader"], text=True).strip().splitlines():
        if not line.strip():
            continue
        pid, b = [x.strip() for x in line.split(",")]
        try:
            if b in bus2i and os.stat(f"/proc/{pid}").st_uid == UID:
                st[bus2i[b]]["mine"] = True
        except FileNotFoundError:
            pass
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max_gpu_jobs", type=int, default=4)
    ap.add_argument("--max_cpu_jobs", type=int, default=4)
    ap.add_argument("--poll_s", type=int, default=30)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--study", choices=("main", "draws", "extra"), default="main")
    a = ap.parse_args()
    cards = [int(x) for x in a.cards.split(",")]
    tag = {"main": "run_all_af24", "draws": "run_draws_af24", "extra": "run_extra_af24"}[a.study]
    logf = open(J / f"logs/{tag}.log", "a")

    def say(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        if not a.dry_run:
            logf.write(line + "\n"); logf.flush()

    queue, running, waiting, done, retried = {"main": all_jobs, "draws": draw_jobs, "extra": extra_jobs}[a.study](), [], [], {}, set()
    say(f"armed: {len(queue)} jobs, cards {cards}, max {a.max_gpu_jobs} GPU / {a.max_cpu_jobs} CPU jobs")
    while True:
        for r in list(running):                                   # reap
            rc = r["proc"].poll()
            if rc is None:
                continue
            running.remove(r); j = r["job"]; ok = Path(j["done"]).exists()
            done[j["name"]] = rc
            say(f"DONE {j['name']} gpu{r['gpu']} rc={rc} result={'yes' if ok else 'NO'} {(time.time() - r['t0']) / 60:.1f} min")
            if rc != 0 and not ok and rc == j.get("requeue_rc"):
                queue.insert(0, j); say(f"YIELDED {j['name']} (rc={rc}): queued again to resume")
            elif rc != 0 and not ok:
                if j["name"].startswith("gs_"):
                    say(f"FAILED {j['name']}: not retried (run_af24_mcgs.sh refuses to overwrite; look at its log)")
                elif j["name"] not in retried:
                    retried.add(j["name"]); queue.insert(0, j); say(f"RETRY {j['name']} once")
        argvs = my_argvs()
        for w in list(waiting):                                   # jobs found running outside this queue
            if not any(matches(v, w) for v in argvs):
                waiting.remove(w); done[w["name"]] = "external"
                say(f"EXTERNAL DONE {w['name']} result={'yes' if Path(w['done']).exists() else 'NO'}")
        for j in list(queue):
            if Path(j["done"]).exists():
                queue.remove(j); done[j["name"]] = "skip"; say(f"SKIP {j['name']} (result exists)")
            elif any(matches(v, j) for v in argvs):
                queue.remove(j); waiting.append(j); say(f"WAIT {j['name']} (already running outside this queue)")
        if not queue and not running and not waiting:
            break
        try:
            st = gpu_state()
        except Exception as e:                                    # nvidia-smi hiccup: try again next poll
            say(f"nvidia-smi failed: {e}"); time.sleep(a.poll_s); continue
        busy = {r["gpu"] for r in running if r["job"]["kind"] == "gpu"}
        n_gpu = sum(1 for r in running if r["job"]["kind"] == "gpu")
        eligible = sorted((g for g in cards if not st[g]["mine"] and g not in busy), key=lambda g: -st[g]["free"])
        if a.dry_run:
            for j in queue:
                print(f"  queued {j['name']:24s} {j['kind']} need {j['need']:>2} GB  -> {j['done']}")
            print("  cards:", {g: dict(free_gb=round(st[g]['free'], 1), mine=st[g]['mine']) for g in cards}, " eligible now:", eligible)
            return

        def launch(j, g):
            queue.remove(j)
            Path(j["log"]).parent.mkdir(parents=True, exist_ok=True)
            p = subprocess.Popen(j["cmd"].format(gpu=g), shell=True, cwd=j["cwd"], env=ENV, executable="/bin/bash",
                                 stdout=open(j["log"], "a"), stderr=subprocess.STDOUT)
            running.append(dict(job=j, gpu=g, proc=p, t0=time.time()))
            say(f"START {j['name']} gpu{g} (free {st[g]['free']:.0f} GB) -> {j['log']}")

        for g in eligible:                                        # GPU jobs: one per eligible card, highest priority that fits
            if n_gpu >= a.max_gpu_jobs:
                break
            j = next((x for x in queue if x["kind"] == "gpu" and x["need"] <= st[g]["free"]), None)
            if j:
                launch(j, g); n_gpu += 1
        n_cpu = sum(1 for r in running if r["job"]["kind"] == "cpu")
        for j in [x for x in queue if x["kind"] == "cpu"]:        # L1-ESPIRiT: CPU slots, data on a classical card
            if n_cpu >= a.max_cpu_jobs:
                break
            host = [r["gpu"] for r in running if r["job"]["name"].startswith(CLASSICAL) and st[r["gpu"]]["free"] >= j["need"]]
            spare = sorted((g for g in cards if st[g]["free"] >= j["need"]), key=lambda g: -st[g]["free"])
            g = host[0] if host else (spare[0] if spare else None)
            if g is None:
                break
            launch(j, g); n_cpu += 1
        json.dump(dict(time=time.strftime("%m-%d %H:%M:%S"),
                       running=[dict(name=r["job"]["name"], gpu=r["gpu"], min=round((time.time() - r["t0"]) / 60, 1)) for r in running],
                       waiting=[w["name"] for w in waiting], queued=[j["name"] for j in queue], done=done),
                  open(J / f"logs/{tag}_status.json", "w"), indent=1)
        time.sleep(a.poll_s)
    fails = {n: rc for n, rc in done.items() if isinstance(rc, int) and rc != 0}
    if a.study == "main":
        say(f"QUEUE EMPTY. failures: {fails}. finalize ...")
        exps = [GS[k][0] for k in ("joint", "singleT1", "singleFLAIR")]
        rc = subprocess.call(["bash", "scripts/finalize_af24_jyz.sh", *exps], cwd=J, env=ENV,
                             stdout=open(J / "logs/finalize_af24.log", "a"), stderr=subprocess.STDOUT)
    elif a.study == "extra":
        say(f"QUEUE EMPTY. failures: {fails}. no automatic post-step (score.py --af 24, registration, figures by hand)")
        rc = 0
    else:
        say(f"QUEUE EMPTY. failures: {fails}. score the draws ...")
        rc = subprocess.call([PY, "-u", "scripts/score_mask_draws_af24.py"], cwd=J, env=ENV,
                             stdout=open(J / "logs/mask_draws_af24_summary.txt", "w"), stderr=subprocess.STDOUT)
    say(f"FINALIZE rc={rc}")
    (J / f"logs/{tag.upper()}_DONE").write_text(json.dumps(dict(done=done, finalize_rc=rc), indent=1))


if __name__ == "__main__":
    main()
