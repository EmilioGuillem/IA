# 🎉 ¡BIENVENIDO! - SopraGP4U Clock In/Out Automation

Hemos creado un **sistema completo de automatización** para tu registro de entrada/salida en SopraGP4U.

## ✅ QUÉ INCLUYE TU PROYECTO

### 📦 Código Python Produccional
- **sopra_clockin.py**: Script principal con Selenium y logging avanzado
- **logger_config.py**: Logging a archivo y consola
- **config.py**: Configuración centralizada
- **test_setup.py**: Tests de conectividad
- **inspect_portal_elements.py**: Herramienta para encontrar selectores HTML

### 🔧 Herramientas de Automatización
- **setup.bat**: Instalación en Windows (1 clic)
- **setup.ps1**: Setup avanzado via PowerShell
- **quick_setup.py**: Wizard interactivo
- **deploy.py**: Deployment automático
- **scheduled_clockin.bat**: Integración con Task Scheduler

### 📚 Documentación Completa
- **README.md**: Guía completa (50+ páginas)
- **QUICK_START.md**: Inicio en 5 minutos
- **IMPORTANT_NOTES.md**: Seguridad y limitaciones
- **INDEX.md**: Índice del proyecto

### 📊 Logging Automático
- Carpeta `logs/` con rotación automática
- Logs detallados de cada acción
- Compatible con auditoría y debugging

## 🚀 COMENZAR EN 3 PASOS

### Paso 1: Instalar (2 minutos)

Abre **PowerShell como Administrador** en tu carpeta:

```powershell
cd "c:\Users\eguillemsimon\Documents\IA\IA\sopra_clockin"
python quick_setup.py
```

### Paso 2: Configurar (2 minutos)

El wizard te pide:
- Usuario SopraGP4U
- Contraseña SopraGP4U
- Inspeccionar portal (para encontrar selectores)

### Paso 3: Probar (1 minuto)

```powershell
python src/sopra_clockin.py
```

## 📋 CARACTERÍSTICAS

✅ **Automatización Inteligente**
- CLOCK-IN antes de 10:00 AM
- CLOCK-OUT después de 5:00 PM
- Reintentos automáticos

✅ **Logging Profesional**
- Archivo y consola
- Rotación automática
- Auditoría completa

✅ **Seguridad**
- Credenciales en variables de entorno
- No en código
- .gitignore protege secretos

✅ **Windows Task Scheduler Compatible**
- Scripts .bat listos
- Ejecución automática diaria
- Manejo de errores

✅ **Fácil de Mantener**
- Código limpio y comentado
- Configuración centralizada
- Tests de diagnóstico

## 📍 UBICACIÓN

```
c:\Users\eguillemsimon\Documents\IA\IA\sopra_clockin\
```

## 🎯 PRÓXIMOS PASOS

1. Abre PowerShell como Administrador
2. Navega a la carpeta del proyecto
3. Ejecuta: `python quick_setup.py`
4. Sigue las instrucciones del wizard
5. ¡Listo!

## 📖 DOCUMENTACIÓN

| Documento | Para qué |
|-----------|----------|
| [QUICK_START.md](QUICK_START.md) | Empezar rápido |
| [README.md](README.md) | Documentación completa |
| [IMPORTANT_NOTES.md](IMPORTANT_NOTES.md) | Seguridad e info importante |
| [INDEX.md](INDEX.md) | Mapa del proyecto |

## 🔧 REQUISITOS

- ✅ Python 3.8+ (descargable desde python.org)
- ✅ Google Chrome (descargable desde google.com/chrome)
- ✅ Windows 7 o superior
- ✅ Válido de SopraGP4U

## ✨ CARACTERÍSTICA DESTACADA: DRY-RUN

Antes de activar la automatización real, puedes hacer PRUEBAS sin efectuar cambios:

```python
# En config/config.py cambiar:
DRY_RUN = True
```

Los logs mostrarán `[DRY RUN]` pero no habrá clicks reales.

## 🎓 APRENDIZAJE

El código es un buen ejemplo de:
- ✓ Automatización web con Selenium
- ✓ Logging profesional en Python
- ✓ Integración con Windows Task Scheduler
- ✓ Gestión de credenciales segura
- ✓ Arquitectura modular y mantenible

## 💪 ANTES DE EMPEZAR

Asegúrate de tener:
1. Python instalado y en PATH
2. Google Chrome
3. Conexión a internet
4. Credenciales SopraGP4U válidas
5. Acceso a Windows Task Scheduler (admin)

## 🆘 ¿PROBLEMAS?

```powershell
# Ver documentación
notepad README.md

# Ejecutar tests
python src/test_setup.py

# Ver logs
Get-Content logs/sopra_clockin.log -Tail 50
```

## 📞 SOPORTE

Lee primero: [IMPORTANT_NOTES.md](IMPORTANT_NOTES.md)
Luego: [README.md](README.md)
Por último: Ejecuta `python src/test_setup.py`

---

## 🎉 ¡ESTÁS LISTO!

Todo está preparado para ser uso INMEDIATO.

**Próximo paso**: Abre PowerShell y ejecuta `python quick_setup.py`

---

**Versión**: 1.0 Production Ready ✓  
**Creado**: 2026-03-26  
**Por**: Tu asistente IA
