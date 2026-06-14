import zipfile
import json
import os

model_path = r"model/best_model.keras"
patched_path = r"model/temp_patched.keras"

with zipfile.ZipFile(model_path, 'r') as z_in:
    with zipfile.ZipFile(patched_path, 'w') as z_out:
        for item in z_in.infolist():
            if item.filename == 'config.json':
                config_data = z_in.read(item.filename).decode('utf-8')
                
                def remove_quant(d):
                    if isinstance(d, dict):
                        d.pop('quantization_config', None)
                        for v in d.values():
                            remove_quant(v)
                    elif isinstance(d, list):
                        for v in d:
                            remove_quant(v)
                            
                config = json.loads(config_data)
                remove_quant(config)
                z_out.writestr(item, json.dumps(config))
            else:
                z_out.writestr(item, z_in.read(item.filename))

os.replace(patched_path, model_path)
print("Successfully patched best_model.keras!")
