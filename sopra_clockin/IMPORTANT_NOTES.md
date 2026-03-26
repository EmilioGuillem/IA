# NOTAS IMPORTANTES - SopraGP4U Clock In/Out Automation

## ⚠️ SEGURIDAD

### Credenciales
- **NUNCA** guardes credenciales directamente en código
- **NUNCA** comitas `.env` a Git
- **USA** variables de entorno de Windows
- **PROTEGE** acceso a tu máquina

### Acceso a Task Scheduler
- Programa debe ejecutarse con **privilegios de administrador**
- Contraseña de usuario debe estar activa (no expirada)
- Máquina debe estar **encendida** a las horas programadas

## ⚠️ LIMITACIONES

1. **Requiere Chrome instalado** → Selenium usa WebDriver de Chrome
2. **No funciona sin interfaz visual** → Headless mode solo en Linux
3. **Cambios en HTML → actualizar selectores**
4. **VPN corporativa** → Asegúrate de estar conectado

## ⚠️ VERIFICACIÓN PREVIA

Antes de programar en Task Scheduler:

```powershell
# 1. Verifica Python
python --version

# 2. Verifica dependencias
pip list | Select-String "selenium|webdriver"

# 3. Verifica credenciales
echo $env:SOPRA_USERNAME
echo $env:SOPRA_PASSWORD

# 4. Corre test
python src/test_setup.py

# 5. Corre con DRY-RUN
python src/sopra_clockin.py  # con DRY_RUN = True en config.py
```

## 📝 CAMBIOS EN EL PORTAL

Si SopraGP4U se actualiza y la automatización falla:

1. Abre el portal en Chrome
2. Presiona F12 (Developer Tools)
3. Inspecciona los elementos HTML
4. Busca los nuevos IDs/selectores
5. Actualiza `config/config.py` con los nuevos selectores

**Ejemplo:**
```python
# Antiguo:
CLOCK_IN_BUTTON_ID = "CLOCK-IN"

# Nuevo (después de inspeccionar):
CLOCK_IN_BUTTON_ID = "btn-clock-in"  # o cambiar el método de búsqueda
```

##📋 CHECKLIST DE IMPLEMENTACIÓN

- [ ] Python 3.8+ instalado
- [ ] `pip install -r requirements.txt` ejecutado
- [ ] Variables de entorno configuradas (SOPRA_USERNAME, SOPRA_PASSWORD)
- [ ] Portal inspeccionado y selectores actualizados en config.py
- [ ] `python src/test_setup.py` pasó todos los tests
- [ ] `python src/sopra_clockin.py` ejecutado exitosamente (mode DRY-RUN)
- [ ] `python src/sopra_clockin.py` ejecutado exitosamente (con acciones reales)
- [ ] Task Scheduler programado con scheduled_clockin.bat
- [ ] Tareas de test ejecutadas manualmente en Task Scheduler
- [ ] Logs revisados y verificados

## 🔧 MODO MANTENIMIENTO

Si necesitas pausar la automatización:

```powershell
# Deshabilitar tarea
Disable-ScheduledTask -TaskName "SopraGP4U Clock In"
Disable-ScheduledTask -SopraGP4U Clock Out"

# Re-habilitar
Enable-ScheduledTask -TaskName "SopraGP4U Clock In"
Enable-ScheduledTask -TaskName "SopraGP4U Clock Out"

# Ver estado
Get-ScheduledTask -TaskName "*SopraGP4U*"
```

## 📊 MONITOREO

Para monitorear si las tareas están ejecutándose:

```powershell
# Ver historial de ejecuciones
Get-ScheduledTaskInfo -TaskName "SopraGP4U Clock In"

# Ver últimas líneas de log
Get-Content logs/sopra_clockin.log -Tail 20

# Monitor en vivo
Get-Content logs/sopra_clockin.log -Tail 5 -Wait
```

## 🆘 CASOS DE ERROR COMUNES

### Error 1: "Element not found"
- Probablemente el HTML del portal cambió
- Solución: Run `python src/inspect_portal_elements.py`
- Actualiza config.py con los nuevos selectores

### Error 2: "Login failed"
- Credenciales incorrectas o expiradas
- VPN no conectada
- Solución: Verifica credenciales y conexión de red

### Error 3: "Chrome WebDriver not found"
- ChromeDriver version incompatible
- Solución: `pip install --upgrade webdriver-manager`

### Error 4: "Task no ejecuta"
- Task Scheduler deshabilitada
- Usuario ocioso/sesión cerrada
- Contraseña del usuario expirada
- Solución: Verifica el historial de Task Scheduler

## 🎯 MEJORAS FUTURAS POSIBLES

- [ ] Integración con MongoDB para histórico
- [ ] API REST propia para monitoreo
- [ ] Notificaciones por email/Slack en caso de fallo
- [ ] Dashboard web para visualizar attendence
- [ ] Soporte para múltiples usuarios
- [ ] Integración con Outlook/Teams para recordatorios

## 📞 SOPORTE TÉCNICO

Para reportar problemas:

1. **Revisa el log**: `logs/sopra_clockin.log`
2. **Ejecuta el test**: `python src/test_setup.py`
3. **Verifica la documentación**: [README.md](README.md)
4. **Inspecciona el portal**: 
   - F12 en Chrome
   - Verifica selectores
   - Busca cambios en HTML

---

**Última actualización**: 2026-03-26
**Versión**: 1.0
**Estado**: Production Ready ✓
