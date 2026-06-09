# INT4 PTQ Activation Outlier Suppression

Course paper project for CIFAR-10 INT4 post-training quantization with layer-wise MSE-selected activation clipping.

See `Claude.md` for the project scope, module boundaries, experiment workflow, and AI collaboration rules.

## AutoDL workflow

The AutoDL PyTorch image already includes PyTorch, torchvision, CUDA, and Python. Use `requirements-autodl.txt` on the server so pip does not replace the CUDA-enabled torch wheels.

Local package only:

```powershell
.\scripts\package_for_autodl.ps1
```

Local package, upload, and unpack through SSH:

```powershell
.\scripts\sync_to_autodl.ps1 -HostName <host> -User root -Port <port> -RemoteDir ~/int4-ptq
```

On AutoDL:

```bash
cd ~/int4-ptq
bash scripts/autodl_setup.sh
bash scripts/autodl_smoke.sh
bash scripts/autodl_run_all.sh
```

`autodl_run_all.sh` writes the full paper artifacts and packages them into
`dist/autodl_results.zip`, including `outputs/results`, `outputs/logs`, and
`outputs/figures`.

When a remote run fails, copy the full command, traceback, and the last useful lines from `outputs/logs/*.log` back into Codex.
