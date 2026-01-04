# RL Meta-Controller Deployment Notes

- Candidate checkpoint: `production/ddqn_meta_controller_prod.pt`
- Origin: `checkpoints/rl_meta_controller/ddqn_meta_controller_epoch100.pt` (trained on combined_141)
- Summary: `checkpoints/rl_meta_controller/training_summary_epoch100.json`

Usage example (inference):
```bash
PYTHONPATH=/home/dwijenayake/pv_forecast_30d python -c "from src.rl.rl_meta_controller import MetaController, RLConfig; from src.rl.training import load_checkpoint; c=MetaController(state_dim=35, config=RLConfig(mode='rl')); load_checkpoint(c,'production/ddqn_meta_controller_prod.pt', device='cpu')"
```

Notes:
- This model was trained to 100 epochs (resumed to 100 from 50 → total 100), final loss ~28.93.
- If deploying to GPU, ensure `device='cuda'` and move optimizer state appropriately when loading.
- Keep the `training_summary_epoch100.json` with the checkpoint for provenance.

