# Linux setup plan from openrt/25288

This directory reorganizes the Linux commands from the Naver Cafe post "001 ROS 2 개발 환경 구축" into smaller restartable steps.

Target environment from the post:

- OS: Linux Mint 20.x Cinnamon 64-bit
- Alternative OS: Ubuntu 20.04.x LTS (Focal Fossa)
- ROS: ROS 2 Foxy Fitzroy
- Simulator: Gazebo 11.x
- DDS: Fast DDS or Cyclone DDS
- IDE: Visual Studio Code

Recommended run order:

1. Read `00_environment.md`.
2. Install Linux Mint 20.x or Ubuntu 20.04.x manually.
3. Run `10_add_ros2_repository.sh`.
4. Run `20_install_ros2_foxy.sh`.
5. Open two terminals and run `30_verify_talker.sh` and `31_verify_listener.sh`.
6. Run `40_install_dev_tools.sh`.
7. Run `50_init_robot_ws.sh`.
8. Append `60_bashrc_snippet.txt` to `~/.bashrc`.
9. Optionally run `70_install_vscode_extensions.sh` after installing VS Code.

Notes:

- The post explicitly uses Ubuntu `focal` packages even on Linux Mint 20.x.
- The original environment is legacy. ROS 2 Foxy is end-of-life, so package availability may depend on mirrors or archived repositories.
- The post recommends installing the OS in English so terminal errors and build logs stay searchable.