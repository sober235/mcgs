
- Do not modify any historical source, CUDA binary, data, mask, checkpoint or score. New files only; all build products have a new dedicated directory and module identity. Retain negative evidence.
- Tensor axis order D,H,W. h=shape/2, C=(positions+1)*h-0.5; normalized displacement d=(v-C)/h; s=exp(log_scales); quaternion wxyz q/(norm(q)+1e-8); M=R diag((s+1e-8)^(-2)) R^T.
- Envelope exp(-d^T M d/2) inside mahal<=9, zero outside; support/tile decisions detached. Frequency cycles per voxel strictly (-0.5,0.5). Stable cos(theta)-1=-2*sin(theta/2)^2.
- CPU algebraic/autograd rtol=atol=1e-10; finite-difference rtol1e-5/atol1e-7 away from support crossings. CUDA bounded-fixture rtol2e-4/atol2e-5; zero correction and old-input correction gradients exactly zero at zero frequency; constructed frequency derivative nonzero.
- No MRI data or benchmark fitting in this phase, no GPU until source review and resource inspection. No fallback backend or implicit production compile. No claim of MRI gain from microgates.
- Use original seven-state rendering/training sources only as read-only reference. No integration scaffold, optimizer or selected MRI subset in this phase.
- Each worker owns only listed files/report, preserves others, does not commit or spawn reviewers. Controller stages/commits and dispatches fresh review. User review/fix cap3; preserve all evidence. Run focused checks, not unrelated legacy606 tests.

