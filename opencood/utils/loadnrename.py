import torch
from tqdm import tqdm

def rename_state_dict(state_dict):
  """
  Renames all modules in a state_dict with names containing "_m1" to "_m3".

  Args:
    state_dict: A state_dict to be renamed.

  Returns:
    A new state_dict with renamed modules.
  """

  new_state_dict = {}
  for key, value in tqdm(state_dict.items()):
    if "_m0" in key:
      new_key = key.replace("_m0", "_m4")
      new_state_dict[new_key] = value
    else:
      new_state_dict[key] = value
  return new_state_dict

# Load the state_dict
state_dict = torch.load("opencood/logs/OURS_4Agents_Heter_Simple/local/m4/net_epoch_bestval_at29_m0.pth")

# Rename the state_dict
new_state_dict = rename_state_dict(state_dict)

# Save the renamed state_dict
torch.save(new_state_dict, "opencood/logs/OURS_4Agents_Heter_Simple/local/m4/net_epoch_bestval_at29.pth")