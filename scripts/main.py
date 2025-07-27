from modules.script_callbacks import on_app_started
from sd_image_encryption import ImageEncryptionLogger, password, app

EncryptVersion = 2.6

if password == '':
    ImageEncryptionLogger.log('Disabled - empty password provided', 'error')
elif not password:
    ImageEncryptionLogger.log('Disabled - no password argument provided', 'error')
else:
    ImageEncryptionLogger.log(f"Enabled V{EncryptVersion}", 'success')
    on_app_started(app)