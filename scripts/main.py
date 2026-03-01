from sd_image_encryption import log, password, app
from modules.script_callbacks import on_app_started

EncryptVersion = 3.1

if password == '':
    log.error('Disabled - empty password provided')
elif not password:
    log.error('Disabled - no password argument provided')
else:
    log.success(f"Enabled V{EncryptVersion}")
    on_app_started(app)