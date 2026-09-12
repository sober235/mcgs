import numpy as np, os, glob

def cfl_map(base):
    with open(base + ".hdr") as f: dims=[int(x) for x in f.read().split("\n")[1].split()]
    n=int(np.prod(dims))
    return np.memmap(base+".cfl",dtype=np.complex64,mode='r',shape=(n,)).reshape(dims,order='F'), dims

def occ_plane(base):
    k,dims = cfl_map(base); nx,ny,nz,nc = dims[:4]
    p=np.zeros((ny,nz))
    for x in range(0,nx,16):
        p += (np.abs(np.asarray(k[x,:,:,:,0]))**2).sum(axis=-1)
    return p>0, dims

B="/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL"
for v in sorted(os.listdir(B)):
    d=os.path.join(B,v)
    if not os.path.isdir(d): continue
    print(f"=== {v}")
    for h in sorted(glob.glob(d+"/*.hdr")):
        base=h[:-4]
        try:
            occ,dims = occ_plane(base)
            ny,nz = occ.shape
            yy,zz = np.meshgrid(np.arange(ny)-ny//2, np.arange(nz)-nz//2, indexing='ij')
            r2 = (yy/(ny/2))**2 + (zz/(nz/2))**2
            ell = r2 <= 1.0
            inside_full = occ[ell].mean(); outside_any = occ[~ell].mean()
            print(f"  {os.path.basename(base):38s} dims={dims[:4]} occ={occ.mean():.4f} "
                  f"| inside-ellipse sampled={inside_full:.4f} outside sampled={outside_any:.4f}")
        except Exception as e:
            print(f"  {os.path.basename(base):38s} ERR {e}")
