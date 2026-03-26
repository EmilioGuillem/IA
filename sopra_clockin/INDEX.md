# ÍNDICE Y ESTRUCTURA DEL PROYECTO
# SopraGP4U Clock In/Out Automation

## 📁 Estructura de Archivos

```
sopra_clockin/
│
├── 📄 README.md                      ← Documentación completa
├── 📄 QUICK_START.md                 ← Guía de inicio rápido
├── 📄 IMPORTANT_NOTES.md             ← Notas importantes
├── 📄 INDEX.md                       ← Este archivo
├── 📄 requirements.txt               ← Dependencias Python
├── 📄 .gitignore                     ← Archivos a ignorar en Git
│
├── 🐍 SCRIPTS DE EJECUCIÓN
│   ├── 📄 setup.bat                  ← Instalación inicial (Windows)
│   ├── 📄 setup.ps1                  ← Setup helper (PowerShell)
│   ├── 📄 quick_setup.py             ← Wizard interactivo
│   ├── 📄 deploy.py                  ← Deployment automático
│   └── 📄 scheduled_clockin.bat      ← Ejecución desde Task Scheduler
│
├── 📁 config/
│   ├── 📄 __init__.py               ← Módulo Python
│   └── 📄 config.py                 ← ⚙️ CONFIGURACIÓN PRINCIPAL
│
├── 📁 src/
│   ├── 📄 __init__.py               ← Módulo Python
│   ├── 📄 sopra_clockin.py          ← 🎯 SCRIPT PRINCIPAL
│   ├── 📄 logger_config.py          ← Configuración de logging
│   ├── 📄 test_setup.py             ← Tests de conectividad
│   ├── 📄 inspect_portal_elements.py ← Herramienta de inspección
│   └── 📄 deploy.py                 ← (copia del deploy principal)
│
└── 📁 logs/                         ← 📊 Logs automáticos
    ├── sopra_clockin.log            ← Log principal (rotativo)
    └── schedule_run.log             ← Log de ejecuciones programadas
```

## 🚀 INICIO RÁPIDO

```powershell
# 1. En PowerShell como Administrador:
cd "c:\Users\eguillemsimon\Documents\IA\IA\sopra_clockin"

# 2. Opción A - Setup Automático (Recomendado):
python quick_setup.py

# Opción B - Setup Manual:
pip install -r requirements.txt
# ... configurar credenciales ...
python src/test_setup.py
python src/sopra_clockin.py
```

## 📋 ARCHIVOS CLAVE

### 🎯 config/config.py
**USO**: Configuración central del proyecto
**EDITABLE**: ✓ SÍ
**IMPORTANTE**: Actualiza los selectores HTML después de inspeccionar el portal

```python
# Parámetros principales:
SOBRA_URL = "..."                    # URL del portal
CLOCK_IN_THRESHOLD = 10              # Antes de qué hora hacer CLOCK-IN
CLOCK_OUT_THRESHOLD = 17             # Después de qué hora hacer CLOCK-OUT
SOPRA_USERNAME = os.getenv("...")    # De variables de entorno
SOPRA_PASSWORD = os.getenv("...")    # De variables de entorno
DRY_RUN = False                      # True para simular
HEADLESS_MODE = True                # False para ver el navegador
```

### 🎯 src/sopra_clockin.py
**USO**: Script principal de automatización
**EDITABLE**: ❌ NO (a menos que entiendas Selenium)
**DESCRIPCIÓN**: Ejecuta la lógica de clock-in/out

### 🎯 src/test_setup.py
**USO**: Verificar que todo está configurado correctamente
**CÓMO USAR**: `python src/test_setup.py`
**O QÚTIL**: Detecta problemas antes de ejecutar automation

### 🎯 src/inspect_portal_elements.py
**USO**: Encontrar selectores HTML del portal
**CÓMO USAR**: `python src/inspect_portal_elements.py`
**NECESARIO**: Cuando cambia el HTML del portal

### 🎯 scheduled_clockin.bat
**USO**: Script que ejecuta Task Scheduler
**EDITABLE**: ❌ NO (a menos que sepas batch)
**IMPORTANTE**: Define cómo se ejecuta desde Task Scheduler

## 🔧 WORKFLOW DE CONFIGURACIÓN

```
1. Instala dependencias
   ↓
2. Configura credenciales en variables de entorno
   ↓
3. Inspecciona portal (F12 en Chrome)
   ↓
4. Actualiza config/config.py con selectores
   ↓
5. Corre tests: python src/test_setup.py
   ↓
6. Prueba con DRY_RUN=True
   ↓
7. Prueba con DRY_RUN=False
   ↓
8. Configura Task Scheduler
   ↓
9. ¡LISTO!
```

## 📚 DOCUMENTACIÓN

| Archivo | Contenido | Lectura |
|---------|-----------|---------|
| README.md | Documentación completa | ⭐⭐⭐ Recomendada |
| QUICK_START.md | Guía rápida | ⭐⭐⭐ Inicio |
| IMPORTANT_NOTES.md | Seguridad y limitaciones | ⭐⭐ Importante |
| INDEX.md | Este archivo | ⭐ Referencia |

## 🔍 ARCHIVOS DE LOG

Se guardan en `logs/`:

- **sopra_clockin.log** (principal)
  - Rotación: cada 10MB
  - Backup: 10 archivos anteriores
  - Contenido: Todas las acciones

- **schedule_run.log** (Task Scheduler)
  - Contenido: Ejecuciones programadas
  - Útil para: Debugging de Task Scheduler

## 🛠️ COMANDOS ÚTILES

```powershell
# Ver logs en tiempo real
Get-Content logs/sopra_clockin.log -Tail 20 -Wait

# Ejecutar manualmente
python src/sopra_clockin.py

# Ejecutar tests
python src/test_setup.py

# Inspeccionar portal
python src/inspect_portal_elements.py

# Ver tareas programadas
Get-ScheduledTask -TaskName *SopraGP4U*

# Ejecutar tarea manualmente
Start-ScheduledTask -TaskName "SopraGP4U Clock In"

# Deshabilitar tarea
Disable-ScheduledTask -TaskName "SopraGP4U Clock In"

# En Linux/Mac (si tienes WSL):
bash setup.sh
```

## ⚙️ CONFIGURACIÓN PERSONALIZADA

### Cambiar horarios

Edita `config/config.py`:

```python
CLOCK_IN_THRESHOLD = 9      # Antes de 9:00 AM (en lugar de 10)
CLOCK_OUT_THRESHOLD = 18    # Después de 6:00 PM (en lugar de 5)
```

### Cambiar horarios de Task Scheduler

Abre Task Scheduler:
1. Busca "SopraGP4U Clock In"
2. Propiedades → Desencadenadores
3. Edita la hora (por defecto: 08:00 para CLOCK-IN, 17:15 para CLOCK-OUT)

### Ver navegador durante ejecución

Edita `config/config.py`:

```python
HEADLESS_MODE = False  # Ver ventana de Chrome
```

### Modo simulación (DRY-RUN)

Edita `config/config.py`:

```python
DRY_RUN = True  # No hace clicks reales, solo simula
```

## 🆘 PROBLEMAS FRECUENTES

| Problema | Solución |
|----------|----------|
| "Python not found" | Reinstala con "Add to PATH" |
| "Selenium timeout" | Portal lento o selectores incorrectos |
| "Login fails" | Credenciales incorrectas o VPN |
| "Element not found" | Portal cambió, corre inspect_portal_elements.py |
| "Task no ejecuta" | Usuario desconectado o privilegios |

Ver [IMPORTANT_NOTES.md](IMPORTANT_NOTES.md) para más detalles.

## 📊 MONITOREO

Revisa `logs/sopra_clockin.log` para:

```
✓ [OK] Acciones exitosas
⚠ [WARNING] Advertencias
✗ [ERROR] Errores
```

Busca los últimos eventos:

```powershell
Get-Content logs/sopra_clockin.log | Select-String "ERROR" -Context 3
```

## 🔐 SEGURIDAD

- ✅ Credenciales en variables de entorno
- ✗ No en código
- ✓ .gitignore protege archivos sensibles
- ✓ Logs sin contraseñas
- ✓ Task Scheduler con privilegios limitados

## 📝 NOTAS

- La primera ejecución puede ser lenta (descarga ChromeDriver)
- Los logs se rotan automáticamente
- Compatible con Windows 7+
- Requiere Python 3.8+

## 🎯 PRÓXIMAS VERSIONES

- Soporte para API (si está disponible)
- Notificaciones por email en caso de fallo
- Dashboard web de monitoreo
- Soporte multi-usuario

---

**Creado**: 2026-03-26
**Versión**: 1.0
**Estado**: Production Ready ✓
