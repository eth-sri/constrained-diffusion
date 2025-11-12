set -ex

sudo apt update -y
sudo apt install python3-pip python3-venv git wget -y #  nvitop
python3 -m venv ~/venv
source ~/venv/bin/activate
# set up huggingface
pip install -U "huggingface_hub[cli]"
huggingface-cli login --add-to-git-credential

# set up typesafe_llm
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" || echo "SSH key already exists"
cat ~/.ssh/id_ed25519.pub
echo "Add the above SSH key to your GitLab account at"
echo "https://gitlab.inf.ethz.ch/-/user_settings/ssh_keys"
read -n 1 -p "Press Enter after adding the SSH key to continue..."
git clone git@gitlab.inf.ethz.ch:nmuendler/diffusion_constraint.git
cd diffusion_constraint
bash install_rust.sh
. "$HOME/.cargo/env"
pip install -r requirements.txt
echo "Restart the shell again to complete the installation"

echo "export HF_HUB=/ephemeral/hub" >> "$HOME/.bashrc"