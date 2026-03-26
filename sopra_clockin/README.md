# SopraGP4U Automatic Clock In/Out Automation

Programa automático para registrar entrada/salida en SopraGP4U mediante Python y Selenium.

## Descripción

Este proyecto automatiza el proceso de registro de entrada/salida en el portal SopraGP4U:
- **CLOCK-IN**: Se ejecuta automáticamente **antes de las 10:00 AM**
- **CLOCK-OUT**: Se ejecuta automáticamente **después de las 5:00 PM (17:00)**

Todas las acciones se registran en archivos de log detallados para auditoría y depuración.

## Características

✅ Automatización de entrada/salida basada en horarios
✅ Logging detallado en archivo y consola
✅ Logica de reintentos en caso de fallos
✅ Modo DRY-RUN para pruebas sin efectuar acciones
✅ Compatible con Windows Task Scheduler
✅ Interfaz en segundo plano (headless)
✅ Gestión de archivos de log con rotación

## Requisitos

### Software requerido
- Python 3.8 o superior
- Google Chrome instalado (requiere el driver ChromeDriver)
- Windows 7 o superior (para usar Task Scheduler)

### Dependencias Python
```
selenium>=4.0.0
webdriver-manager>=3.8.0
```

## Instalación

### 1. Instalar dependencias Python

Abre una terminal PowerShell **como administrador** y navega a la carpeta del proyecto:

```powershell
cd "c:\Users\eguillemsimon\Documents\IA\IA\sopra_clockin"
pip install -r requirements.txt
```

**Nota**: Si tienes múltiples versiones de Python, usa:
```powershell
python3 -m pip install -r requirements.txt
```

### 2. Configurar credenciales (IMPORTANTE)

Las credenciales se deben establecer como variables de entorno de Windows. **Nunca las comitas al repositorio**.

#### Opción A: Establecer variables de entorno permanentemente

1. Abre **Variables de entorno** (Windows + R → `sysdm.cpl` → pestaña "Avanzadas" → "Variables de entorno")
2. Haz clic en "Nueva..." bajo "Variables de usuario"
3. Crea las siguientes variables:
   - **SOPRA_USERNAME**: Tu usuario de SopraGP4U
   - **SOPRA_PASSWORD**: Tu contraseña de SopraGP4U

#### Opción B: Establecer variables antes de ejecutar

```powershell
$env:SOPRA_USERNAME = "tu_usuario"
$env:SOPRA_PASSWORD = "tu_contraseña"
python src/sopra_clockin.py
```

#### Opción C: Crear un archivo `.env` (no recomendado en producción)

Crea `config/.env.ps1`:
```powershell
$env:SOPRA_USERNAME = "tu_usuario"
$env:SOPRA_PASSWORD = "tu_contraseña"
```

Luego ejecuta:
```powershell
& config/.env.ps1
python src/sopra_clockin.py
```

## Uso

### Ejecución manual

```bash
cd c:\Users\eguillemsimon\Documents\IA\IA\sopra_clockin
python src/sopra_clockin.py
```

### Ejecución mediante Task Scheduler

#### Opción 1: Programar mediante GUI (Recomendado)

1. **Abre Task Scheduler** (Windows + R → `taskschd.msc`)

2. **Crea una nueva tarea**: 
   - Click derecho en "Tareas programadas" → "Crear tarea..."

3. **Pestaña "General"**:
   - Nombre: `SopraGP4U Clock In`
   - Descripción: `Automatic clock in/out for SopraGP4U`
   - ☑ Ejecutar independientemente de que el usuario haya iniciado sesión o no
   - ☑ Ejecutar con privilegios máximos

4. **Pestaña "Desencadenadores"** - Crea dos tareas:

   **Tarea 1 - CLOCK IN**:
   - Tipo: Diariamente
   - Hora: 08:00 (8:00 AM - antes de las 10:00 AM)
   - Repetir cada día

   **Tarea 2 - CLOCK OUT**:
   - Tipo: Diariamente
   - Hora: 17:15 (5:15 PM - después de las 5:00 PM)
   - Repetir cada día

5. **Pestaña "Acciones"**:
   - Acción: Iniciar un programa
   - Programa: `C:\Users\eguillemsimon\Documents\IA\IA\sopra_clockin\scheduled_clockin.bat`
   - Directorio de inicio: `C:\Users\eguillemsimon\Documents\IA\IA\sopra_clockin`

6. **Pestaña "Condiciones"**:
   - ☐ Iniciar la tarea solo si se está ejecutando en CA
   - ☐ Detener si la tarea se ejecuta más de: (sin límite)

7. **Pestaña "Configuración"**:
   - Permitir que la tarea se ejecute a petición
   - Si la tarea no se ejecuta en la hora programada, reintentar cada: 5 minutos (máx. 3 intentos)

#### Opción 2: Programar mediante PowerShell (Avanzado)

```powershell
# Crear tarea de CLOCK IN
$action = New-ScheduledTaskAction -Execute "C:\Users\eguillemsimon\Documents\IA\IA\sopra_clockin\scheduled_clockin.bat"
$trigger = New-ScheduledTaskTrigger -Daily -At 08:00
Register-ScheduledTask -TaskName "SopraGP4U Clock In" -Action $action -Trigger $trigger -RunLevel Highest

# Crear tarea de CLOCK OUT
$trigger = New-ScheduledTaskTrigger -Daily -At 17:15
Register-ScheduledTask -TaskName "SopraGP4U Clock Out" -Action $action -Trigger $trigger -RunLevel Highest
```

## Configuración

Edita `config/config.py` para personalizar:

```python
SOPRA_URL = "https://sprportal-mcp.soprahronline.com/SopraGP4U/"
MENU_LINK_TEXT = "Registro de entrada/salida"
CLOCK_IN_THRESHOLD = 10      # Hora límite para CLOCK-IN (10:00 AM)
CLOCK_OUT_THRESHOLD = 17     # Hora mínima para CLOCK-OUT (5:00 PM)
HEADLESS_MODE = True         # Ejecutar sin interfaz visual
DRY_RUN = False              # True para modo simulación
MAX_RETRIES = 3              # Número de reintentos
RETRY_DELAY = 5              # Segundos entre reintentos
```

## Archivos de Log

Los logs se guardan en la carpeta `logs/`:

- **`sopra_clockin.log`**: Log principal con todas las acciones
- **`schedule_run.log`**: Log de ejecuciones programadas

### Ejemplo de log

```
2026-03-26 09:45:32 - __main__ - INFO - ================================================================================
2026-03-26 09:45:32 - __main__ - INFO - Starting SopraGP4U Clock In/Out Automation
2026-03-26 09:45:32 - __main__ - INFO - Current time: 2026-03-26 09:45:32
2026-03-26 09:45:32 - __main__ - INFO - Action type: CLOCK_IN
2026-03-26 09:45:32 - __main__ - INFO - ================================================================================
2026-03-26 09:45:32 - __main__ - INFO - Attempt 1/3
2026-03-26 09:45:32 - __main__ - INFO - Setting up Chrome WebDriver...
2026-03-26 09:45:34 - __main__ - INFO - Chrome WebDriver initialized successfully
2026-03-26 09:45:34 - __main__ - INFO - Navigating to https://sprportal-mcp.soprahronline.com/SopraGP4U/
2026-03-26 09:45:36 - __main__ - INFO - Successfully navigated to portal
2026-03-26 09:45:36 - __main__ - INFO - Attempting to login...
2026-03-26 09:45:38 - __main__ - INFO - Login successful
2026-03-26 09:45:38 - __main__ - INFO - Looking for menu link: 'Registro de entrada/salida'
2026-03-26 09:45:39 - __main__ - INFO - Successfully clicked on menu link: Registro de entrada/salida
2026-03-26 09:45:39 - __main__ - INFO - Looking for CLOCK_IN button...
2026-03-26 09:45:40 - __main__ - INFO - Successfully clicked CLOCK_IN button
2026-03-26 09:45:40 - __main__ - INFO - ================================================================================
2026-03-26 09:45:40 - __main__ - INFO - Automation completed successfully!
2026-03-26 09:45:40 - __main__ - INFO - ================================================================================
```

## Modo DRY-RUN (Pruebas)

Para probar sin efectuar cambios reales, edita `config/config.py`:

```python
DRY_RUN = True
```

En este modo, los logs mostrarán `[DRY RUN]` pero no se ejecutarán clicks reales.

## Solución de problemas

### Error: "Python is not installed or not in PATH"

1. Verifica que Python esté instalado:
```powershell
python --version
```

2. Si no aparece, reinstala Python desde https://www.python.org/ e incluye "Add Python to PATH"

### Error: "Username field not found"

La estructura HTML del portal cambió. Necesitas inspeccionar la página:

1. Abre el portal en Google Chrome
2. Presiona F12 para abrir Developer Tools
3. Inspecciona los elementos `<input>` y `<button>`
4. Actualiza los selectores en `config/config.py` (líneas con `By.ID`, `By.XPATH`, etc.)

### Chrome WebDriver se cuelga

1. Verifica que Google Chrome esté actualizado
2. Asegúrate de que `webdriver-manager` está instalado:
```powershell
pip install --upgrade webdriver-manager
```

### Task Scheduler no ejecuta la tarea

1. Verifica que el archivo `scheduled_clockin.bat` existe
2. Prueba a ejecutar manualmente desde PowerShell como administrador
3. Revisa `logs/schedule_run.log` para errores
4. Asegúrate de que la contraseña de Windows no ha expirado

## Estructura del proyecto

```
sopra_clockin/
├── config/
│   ├── __init__.py
│   └── config.py                 # Configuración principal
├── src/
│   ├── __init__.py
│   ├── logger_config.py          # Configuración de logging
│   └── sopra_clockin.py          # Script principal
├── logs/                         # Archivos de log (se crean automáticamente)
│   ├── sopra_clockin.log
│   └── schedule_run.log
├── requirements.txt              # Dependencias Python
├── scheduled_clockin.bat         # Script de ejecución para Task Scheduler
└── README.md                     # Este archivo
```

## Seguridad

⚠️ **IMPORTANTE**: 

- ❌ **NO** guardes credenciales en código
- ❌ **NO** comitas archivos `.env` con credenciales
- ✅ **USA** variables de entorno de Windows
- ✅ **PROTEGE** el acceso a tu máquina
- ✅ Usa contraseñas fuertes

## Limitaciones conocidas

1. Requiere que Chrome esté instalado
2. No funciona en máquinas sin interfaz gráfica (headless solo en Linux/macOS con Xvfb)
3. Cambios en la estructura HTML del portal pueden requerir actualización de selectores
4. Requiere credenciales válidas de SopraGP4U

## Próximas mejoras

- [ ] Soporte para API REST (si SopraGP4U la proporciona)
- [ ] Notificaciones por email en caso de fallo
- [ ] Interfaz gráfica para configurar parámetros
- [ ] Soporte para múltiples usuarios
- [ ] Integración con otros portales de asistencia

## Soporte

Para reportar problemas:
1. Revisa los logs en `logs/sopra_clockin.log`
2. Ejecuta en modo DRY-RUN para diagnosticar
3. Inspecciona el portal con Chrome Developer Tools
4. Documenta los errores con logs adjuntos

## Licencia

Este proyecto es de uso personal. No se debe compartir sin permiso explícito.

---

**Autor**: Emilio Guillem Simón  
**Creado**: 2026-03-26  
**Última actualización**: 2026-03-26
