# macOS setup options

This directory provides two macOS paths:

- `conda/`: recommended on current macOS and Apple Silicon. This matches the packages already observed on this machine: `robostack-jazzy`, `ros-jazzy-desktop`, `ros-jazzy-ros-gz`, `gz-sim8`, `ros-dev-tools`.
- `native/`: Homebrew + source build on the host macOS installation. This is best-effort and needs more troubleshooting.

Suggested choice:

1. Use `conda/` if you want the most reproducible path on macOS.
2. Use `native/` only if you specifically need a non-conda host install.

Common optional step:

- Run `70_install_vscode_extensions.sh` after VS Code is installed.

Do not mix the `native` shell snippet and the `conda` workflow in the same shell session.