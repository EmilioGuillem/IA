# GUÍA DE INICIO RÁPIDO - SopraGP4U Clock In/Out

## 1. Instalación (5 minutos)

### Opción A: Setup Automático (Recomendado)

Abre PowerShell **como administrador** en la carpeta del proyecto:

```powershell
# Navega a la carpeta del proyecto
cd "c:\Users\eguillemsimon\Documents\IA\IA\sopra_clockin"

# Ejecuta el setup interactivo
python quick_setup.py
```

This script will guide you through all setup steps automatically.

### Opción B: Setup Manual

1. **Instalar dependencias**:
```powershell
pip install -r requirements.txt
```

2. **Configurar credenciales** como variables de entorno:
   - Windows + R → `sysdm.cpl` → Avanzadas → Variables de entorno
   - Nueva variable: `SOPRA_USERNAME` = Tu usuario
   - Nueva variable: `SOPRA_PASSWORD` = Tu contraseña

3. **Inspeccionar portal** para encontrar selectores:
```powershell
python src/inspect_portal_elements.py
```

4. **Actualizar config** con los selectores encontrados:
   - Edita `config/config.py`
   - Actualiza los `By.ID`, `By.XPATH`, etc. con los valores encontrados

5. **Probar**:
```powershell
python src/sopra_clockin.py
```

## 2. Verificación

Antes de programar en Task Scheduler, verifica que funciona:

```powershell
# Test 1: Verificar conectividad
python src/test_setup.py

# Test 2: Run automation (en DRY-RUN para no hacer clicks reales)
# Edita config/config.py y pon: DRY_RUN = True
python src/sopra_clockin.py

# Test 3: Run con acciones reales
# Edita config/config.py y pon: DRY_RUN = False
python src/sopra_clockin.py
```

## 3. Configurar Windows Task Scheduler

### Opción A: Via GUI (Fácil)

1. Abre **Programador de tareas** (Windows + R → `taskschd.msc`)

2. Click derecho → "Crear tarea"

3. **General**:
   - Nombre: `SopraGP4U Clock In`
   - ✓ Ejecutar con privilegios máximos

4. **Desencadenadores** → Nueva entrada:
   - Tipo: Diariamente
   - Hora: 08:00 (para CLOCK-IN)
   - Aceptar

5. **Acciones** → Nueva entrada:
   - Programa: `C:\Users\eguillemsimon\Documents\IA\IA\sopra_clockin\scheduled_clockin.bat`
   - Aceptar

6. **Crear segunda tarea** igual pero a las 17:15 (para CLOCK-OUT)

### Opción B: Via PowerShell (Automático)

Abre PowerShell como administrador:

```powershell
cd "c:\Users\eguillemsimon\Documents\IA\IA\sopra_clockin"
.\setup.ps1 -CreateScheduledTasks
```

## 4. Verificar Logs

Los logs se guardan en `logs/sopra_clockin.log`:

```powershell
# Ver últimas líneas del log
Get-Content logs/sopra_clockin.log -Tail 50

# Monitor en tiempo real (como tail -f)
Get-Content logs/sopra_clockin.log -Tail 10 -Wait
```

## 5. Solución de Problemas Rápida

| Problema | Solución |
|----------|----------|
| "Python not found" | Reinstala Python desde https://www.python.org/ (marca "Add to PATH") |
| "Selenium not installed" | `pip install -r requirements.txt` |
| "Chrome not found" | Instala Google Chrome desde https://www.google.com/chrome/ |
| "Element not found" | Corre `python src/inspect_portal_elements.py` para encontrar selectores correctos |
| "Login fails" | Verifica credenciales en variables de entorno (`echo $env:SOPRA_USERNAME`) |
| "Task Scheduler no ejecuta" | Verifica que el archivo .bat existe y que las variables de entorno están en el nivel correcto |

## 6. Próximos Pasos

✓ Instalación completada
✓ Credenciales configuradas  
✓ Selectores encontrados
✓ Tests realizados
✓ Task Scheduler programado

**Listo para uso en producción** ✓

---

## Comandos Útiles

```powershell
# Ejecutar script manualmente
python src/sopra_clockin.py

# Ejecutar con DRY-RUN (simulación)
# Primero edita config/config.py: DRY_RUN = True
python src/sopra_clockin.py

# Testear setup
python src/test_setup.py

# Inspeccionar portal
python src/inspect_portal_elements.py

# Ver logs
Get-Content logs/sopra_clockin.log

# Ver tareas programadas
Get-ScheduledTask -TaskName *SopraGP4U*

# Ejecutar tarea manualmente
Start-ScheduledTask -TaskName "SopraGP4U Clock In"
```

---

**¿Problemas?** Lee: [README.md](README.md)
