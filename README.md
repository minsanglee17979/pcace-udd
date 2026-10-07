# Ar training and UDD

Train four pcace Ar models and compare ordinary committee MD with uncertainty-driven dynamics (UDD).

## Requirements

CUDA GPU, Python, pcace, PyTorch, NumPy, ASE, and MDAnalysis. Install pcace following [Mark DelloStritto's repository](https://github.com/markdellostritto/pcace).

Original runs used Python 3.11.4, PyTorch 2.14.0, NumPy 2.4.6, ASE 3.29.0, MDAnalysis 2.10.0, and pcace commit `962356945ddbd1a38eef0435ed0643c9bd8b6ee6`.

## Files

- `data/`: Ar training data.
- `training/bb-*/`: training inputs for seeds 101, 202, 303, and 404.
- `pretrained/bb-*/`: supplied trained models.
- `md/`, `udd/`: simulation inputs and initial PDB.

## 1 Train the committee

Activate your environment and run from the repository root on a GPU. These launchers run directly, not through Slurm; on HPC, use a GPU allocation.

Each member uses a different initialization seed and the same 90/10 training/validation split. Training has 100 classical-stage epochs followed by 100 full-model epochs.

```bash
for seed in 101 202 303 404; do
    bash "training/bb-$seed/run.sh" || break
done
```

The four trained models are saved as `training/bb-*/best_model.pth`.

## 2 Run MD and UDD

After all four training runs finish successfully, run MD and UDD using those `best_model.pth` files. The final argument, `40`, sets the target temperature to 40 K for both simulations:

```bash
bash md/run.sh "$(pwd)/training" 40
bash udd/run.sh "$(pwd)/training" 40
```

Both simulations start from the same 108-atom PDB, without replication, and generate new velocities with seed 42.

| Setting | Value |
| --- | --- |
| Ensemble | Nose-Hoover-chain NVT |
| Temperature | 40 K |
| Timestep | 1 fs |
| Duration | 100 ps |
| Thermostat damping | 100 fs |
| Bias amplitude A | MD: 0 eV; UDD: 15 eV |
| Bias width B | 0.0019311071298553674 eV |

Edit settings at the top of each simulation script. For a 1 ps trial, set `n_run = 1000`.

Outputs: `md.out`, `md.log`, `initial.pdb`, `md.dcd`, and `final.pdb`. Frames are saved every 0.1 ps. Existing outputs are protected; use a fresh simulation directory for another run.

### Optional use of supplied models

To skip training, use the supplied models instead:

```bash
bash md/run.sh
bash udd/run.sh
```

## UDD method

The Gaussian bias follows Kulichenko et al., Eqs. (2), (4), and (6)-(7). It is added to the committee-mean energy and force. A and B are Ar example settings, not universal parameters.

UDD is biased sampling; this example does not include an active-learning retraining loop. UDD energies and stresses in `md.log` include the bias.

## References

- Mark DelloStritto: [pcace](https://github.com/markdellostritto/pcace) and [pcace-fit](https://github.com/markdellostritto/pcace-fit) (revision `99b6036`). This example builds on the Ar training materials; the upstream CC0 notice is in `reference/pcace-fit-LICENSE`.
- Kulichenko, M. et al. **Uncertainty-driven dynamics for active learning of interatomic potentials.** Nature Computational Science **3**, 230-239 (2023). [doi:10.1038/s43588-023-00406-5](https://doi.org/10.1038/s43588-023-00406-5).
