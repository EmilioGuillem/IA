<!-- AI_DISCLAIMER v1.0 -->
# MyBoda Online

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

Paquete de despliegue para publicar la aplicación de boda de `MyBoda`.

## Preparar el paquete

Desde PowerShell, situado en esta carpeta, ejecuta:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\prepare.ps1
```

El script copia el código actual de `../MyBoda` y excluye `node_modules`, `data`, `.env`, builds y logs. Después de prepararlo, `MyBoda-Online` es una copia independiente y desplegable.

## Opción recomendada: Render

1. Crea un repositorio Git con el contenido de `MyBoda-Online` después de ejecutar `prepare.ps1`.
2. En Render, crea un **Web Service** conectado al repositorio.
3. Usa:
   - Build command: `npm install && npm run build`
   - Start command: `npm run start`
4. Añade `ADMIN_PASSWORD` como secret.
5. Añade `SUPABASE_URL` y `SUPABASE_SERVICE_ROLE_KEY` como secrets.

Render sirve el frontend compilado desde Express y proporciona HTTPS. El plan gratuito puede dormir tras un periodo de inactividad.

## Persistencia real

La aplicación usa Supabase/PostgreSQL. El servidor ya no necesita SQLite ni almacenamiento persistente local; las respuestas sobreviven a reinicios y despliegues.

Ejecuta `supabase-schema.sql` en el SQL Editor de Supabase antes de publicar.

## Comprobación local de producción

```powershell
npm.cmd install
npm.cmd run build
Copy-Item .env.example .env
npm.cmd run start
```

Abre `http://localhost:3001`. El servidor servirá `dist` directamente cuando exista.

## Variables

Consulta `.env.example`. Nunca subas `.env`, `SUPABASE_SERVICE_ROLE_KEY` ni la contraseña del administrador.