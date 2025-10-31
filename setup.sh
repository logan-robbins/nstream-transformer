# 1) Remove older system nodejs if present (to avoid conflicts)
sudo apt-get remove -y nodejs npm

# 2) Install nvm (as ubuntu user)
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# 3) Load nvm in current shell
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# 4) Install & use LTS version (e.g., v24)
nvm install 24
nvm alias default 24
nvm use 24

# 5) Verify
node -v
npm -v

# 6) Now you can install your global package
npm install -g @openai/codex

