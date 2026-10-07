import sys
from pathlib import Path

import numpy as np
import torch
import MDAnalysis as mda
from MDAnalysis.coordinates.DCD import DCDWriter
from ase import Atoms, units
from ase.cell import Cell
from ase.io import read, write
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pcace.calculators import CACECalculator

# Settings: run.sh supplies the model directory and temperature.
model_dir = Path(sys.argv[1])
temperature = float(sys.argv[2])
member_seeds = (101, 202, 303, 404)
n_run = 100000
save_every = 100
timestep_fs = 1.0
tdamp_fs = 100.0
velocity_seed = 42
A_eV = 15.0
B_eV = 0.0019311071298553674
atomic_energies = {18: -574.498250374157}

torch.set_default_dtype(torch.float64)

# Model prediction and UDD bias.
# Kulichenko et al., Nature Computational Science 3, 230-239 (2023).
# https://doi.org/10.1038/s43588-023-00406-5, Eqs. (2), (4), (6)-(7).
class ComponentInference(torch.nn.Module):

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.rc = base.rc
        self.key_energy = base.key_energy
        self.key_forces = base.key_forces
        self.key_stress = base.key_stress

    def forward(self, data, **kwargs):
        components = self.base.nnhl
        keys = (self.key_energy, self.key_forces, self.key_stress)
        totals = {}
        try:
            for component in components:
                if float(component.ann.weight.detach().cpu()) <= 0:
                    continue
                fresh = {
                    key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in data.items()
                }
                self.base.nnhl = torch.nn.ModuleList([component])
                result = self.base(
                    fresh, training=False, compute_forces=True,
                    compute_stress=True, compute_virials=True,
                )
                for key in keys:
                    value = result[key].detach()
                    totals[key] = totals[key] + value if key in totals else value.clone()
                del result, fresh
        finally:
            self.base.nnhl = components
        return totals


class CommitteeInference(torch.nn.Module):

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(
            [ComponentInference(model) for model in models]
        )
        first = self.models[0]
        self.rc = first.rc
        self.key_energy = first.key_energy
        self.key_forces = first.key_forces
        self.key_stress = first.key_stress

    def forward(self, data, **kwargs):
        predictions = [model(data, **kwargs) for model in self.models]
        keys = (self.key_energy, self.key_forces, self.key_stress)
        stacked = {
            key: torch.stack([prediction[key] for prediction in predictions])
            for key in keys
        }

        # CACECalculator evaluates one structure at a time.
        energies = stacked[self.key_energy].reshape(len(self.models))
        delta = energies - energies.mean()
        denominator = len(self.models) * len(conf_init) * B_eV**2
        q = 0.5 * torch.sum(delta**2)
        bias_energy = A_eV * torch.expm1(-q / denominator)
        coefficients = (
            -A_eV * torch.exp(-q / denominator) * delta / denominator
        )

        totals = {
            self.key_energy: stacked[self.key_energy].mean(dim=0) + bias_energy
        }
        for key in (self.key_forces, self.key_stress):
            values = stacked[key]
            weights = coefficients.reshape(
                (len(self.models),) + (1,) * (values.ndim - 1)
            )
            totals[key] = values.mean(dim=0) + (weights * values).sum(dim=0)
        return totals


# Initial structure and velocities: no supercell expansion.
for name in ('initial.pdb', 'final.pdb', 'md.dcd', 'md.log'):
    if Path(name).exists() or Path(name).is_symlink():
        raise RuntimeError(f'Existing output: {name}')

conf_init = read('start.pdb')
initial_cell = conf_init.cell.array.copy()
if not conf_init.pbc.all() or np.linalg.det(initial_cell) <= 0:
    raise ValueError('A valid periodic cell is required')

nnps = [
    torch.load(model_dir / f'bb-{seed}/best_model.pth',
               map_location='cuda', weights_only=False)
    for seed in member_seeds
]
for nnp in nnps:
    nnp.calc_stress = True

conf_init.calc = CACECalculator(
    model_path=CommitteeInference(nnps),
    device='cuda',
    compute_stress=True,
    atomic_energies=atomic_energies,
)
MaxwellBoltzmannDistribution(
    atoms=conf_init,
    temperature_K=temperature,
    rng=np.random.default_rng(velocity_seed),
)
dyn = NoseHooverChainNVT(
    conf_init,
    timestep=timestep_fs * units.fs,
    temperature_K=temperature,
    tdamp=tdamp_fs * units.fs,
    tchain=3,
    tloop=1,
)

# Rotate exported coordinates only, for PDB/DCD cell conventions.
display_cell = Cell.fromcellpar(conf_init.cell.cellpar()).array
rotation = np.linalg.solve(initial_cell, display_cell)
np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-10)


def snapshot():
    return Atoms(
        numbers=conf_init.numbers,
        positions=conf_init.positions @ rotation,
        cell=display_cell,
        pbc=conf_init.pbc,
    )


def check_state():
    temperature_now = conf_init.get_temperature()
    finite = all(np.isfinite(x).all() for x in (
        conf_init.positions, conf_init.cell.array, temperature_now,
    ))
    if not finite or temperature_now > 1000.0:
        raise RuntimeError(f'Invalid state at step {dyn.nsteps}: T={temperature_now}')
    if not np.array_equal(conf_init.cell.array, initial_cell):
        raise RuntimeError('The fixed NVT cell changed')


# Same 19-column thermo format as the legacy input.
COLUMNS = [
    ('Time[ps]', 12, '.4f'),
    ('Etot[eV]', 18, '.6f'),
    ('Epot[eV]', 18, '.6f'),
    ('Ekin[eV]', 14, '.6f'),
    ('T[K]', 10, '.3f'),
    ('Sxx[GPa]', 12, '.6f'),
    ('Syy[GPa]', 12, '.6f'),
    ('Szz[GPa]', 12, '.6f'),
    ('Syz[GPa]', 12, '.6f'),
    ('Sxz[GPa]', 12, '.6f'),
    ('Sxy[GPa]', 12, '.6f'),
    ('Volume[A^3]', 16, '.4f'),
    ('Density[g/cm^3]', 18, '.6f'),
    ('a[A]', 12, '.6f'),
    ('b[A]', 12, '.6f'),
    ('c[A]', 12, '.6f'),
    ('alpha[deg]', 14, '.6f'),
    ('beta[deg]', 14, '.6f'),
    ('gamma[deg]', 14, '.6f'),
]

check_state()
write('initial.pdb', snapshot(), format='proteindatabank')
universe = mda.Universe.empty(len(conf_init), trajectory=True)
print(f'NVT: T={temperature:g} K, A={A_eV:g} eV, B={B_eV:g} eV', flush=True)

with open('md.log', 'x') as thermo, DCDWriter(
    'md.dcd', n_atoms=len(conf_init), dt=timestep_fs * save_every / 1000.0,
) as trajectory:
    thermo.write(' '.join(f'{name:>{width}}' for name, width, fmt in COLUMNS) + '\n')

    def save_frame():
        epot = conf_init.get_potential_energy()
        ekin = conf_init.get_kinetic_energy()
        stress = conf_init.get_stress(include_ideal_gas=True) / units.GPa
        volume = conf_init.get_volume()
        density = conf_init.get_masses().sum() * 1.6605390666 / volume
        values = [
            dyn.get_time() / (1000.0 * units.fs),
            epot + ekin,
            epot,
            ekin,
            conf_init.get_temperature(),
        ]
        values += stress.tolist() + [volume, density] + conf_init.cell.cellpar().tolist()
        thermo.write(' '.join(
            format(value, f'{width}{fmt}')
            for value, (_, width, fmt) in zip(values, COLUMNS)
        ) + '\n')
        thermo.flush()

        frame = snapshot()
        universe.atoms.positions = frame.positions
        universe.dimensions = frame.cell.cellpar()
        trajectory.write(universe.atoms)
        print(f'step={dyn.nsteps} T={values[4]:.3f} K', flush=True)

    dyn.attach(check_state, interval=1)
    dyn.attach(save_frame, interval=save_every)
    dyn.run(n_run)

write('final.pdb', snapshot(), format='proteindatabank')
print(f'Completed {dyn.nsteps} steps.', flush=True)

