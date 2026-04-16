import pathlib
root = pathlib.Path('C:/Users/wwwSC/OneDrive/Desktop/smart_microgrid')
files = {
    'data': root / 'data' / 'Renewable_energy_dataset.csv',
    'lstm': root / 'models' / 'saved' / 'lstm_solar_model.keras',
    'xgb': root / 'models' / 'saved' / 'xgb_load_model.pkl',
    'dqn': root / 'models' / 'saved' / 'dqn_agent.keras',
}
for k,p in files.items():
    print(k, p.exists(), p)
