# Close-Proximity Human-Aware Locomotion — `plr_tasks`

Isaac Lab 2.1.1 extension that trains an **ANYmal-D** quadruped to navigate
while avoiding foothold placement on (simulated) humans, using a 2D binary
forbidden-zone map as an additional observation channel. The training stack is
the ETH Legged Robotics **SRU-enhanced rsl_rl fork**
([leggedrobotics/sru-navigation-learning](https://github.com/leggedrobotics/sru-navigation-learning)).

---

## Quick start

training the current main task (headless):

./isaaclab.sh -p source/plr/source/plr_tasks/scripts/rsl_rl/train.py --num_envs 2048 --task Isaac-PLR-Velocity-Flat-Anymal-D-v0 --headless

to play a model:

./isaaclab.sh -p source/plr/source/plr_tasks/scripts/rsl_rl/play.py --num_envs 16 --task Isaac-PLR-Velocity-Flat-Anymal-D-v0 --checkpoint logs/rsl_rl/anymal_navigation_ppo/DATE_XXX/model_X.pt

add --video to record a video

for the patch locomotion task:

python source/plr/source/plr_tasks/scripts/rsl_rl/train.py --task=Isaac-Anymal-Patch-v0 --debug_vis --video --headless

note: disable WANDB for this terminal session:

export WANDB_MODE=disabled

## Registered tasks

| Gym ID                                | Purpose               |
| ------------------------------------- | --------------------- |
| `Isaac-PLR-Velocity-Flat-Anymal-D-v0` | Current Latest Policy |

## Docs

| File                                 | Scope            |
| ------------------------------------ | ---------------- |
| [`docs/DOCKER.md`](docs/DOCKER.md)   | Docker Install   |
| [`docs/CLUSTER.md`](docs/CLUSTER.md) | Cluster Workflow |

## Credits

- Base env ported from Isaac Lab's `isaaclab_tasks/.../locomotion/velocity/anymal_d`.
- Custom `rsl_rl` fork: [sru-navigation-learning](https://github.com/leggedrobotics/sru-navigation-learning).
- Reference usage pattern: [sru-navigation-sim](https://github.com/leggedrobotics/sru-navigation-sim).
