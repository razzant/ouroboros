# Written by scripts/promote_stable.py into main's promotion commit for this
# Stable release, from its published proof. Do not edit by hand.
cask "ouroboros" do
  version "7.12.1"
  sha256 "8aeee7b4489ca7d853f255bb2b9d629d1ffb39915a300f222169ffd759c394a6"

  url "https://github.com/razzant/ouroboros/releases/download/v#{version}/Ouroboros-#{version}.dmg"
  name "Ouroboros"
  desc "Self-creating AI agent"
  homepage "https://ouroboros-agent.ai/"

  # A plain `brew upgrade` first compares the installed app's own version.
  auto_updates true
  depends_on arch: :arm64
  depends_on macos: :monterey

  app "Ouroboros.app"

  # Asks a running app to quit before an upgrade or uninstall. No launchctl or
  # zap: launchctl also runs on upgrade and would drop the sign-in registration,
  # and ~/Ouroboros holds the owner's repository, memory and projects.
  uninstall quit: "com.ouroboros.agent"

end
