# Human Aware Locomotion with the ANYmal Robot in the course Perception and Learning for Robotics at ETH


### useful commands

training the current main task (headless):

./isaaclab.sh -p source/plr/source/plr_tasks/scripts/rsl_rl/train.py --num_envs 2024 --task Isaac-PLR-Velocity-Flat-Anymal-D-v0 --headless

to play a model:

./isaaclab.sh -p source/plr/source/plr_tasks/scripts/rsl_rl/play.py --num_envs 16 --task Isaac-PLR-Velocity-Flat-Anymal-D-v0 --checkpoint logs/rsl_rl/anymal_navigation_ppo/DATE_XXX/model_X.pt

add --video to record a video