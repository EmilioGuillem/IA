<!-- AI_DISCLAIMER v1.0 -->
# Guía de despliegue futuro

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

## Antes de publicar

- Ejecutar `prepare.ps1` para sincronizar la última versión de `MyBoda`.
- Ejecutar `npm.cmd install` y `npm.cmd run build`.
- Cambiar `ADMIN_PASSWORD` por una contraseña aleatoria y privada.
- Crear la tabla ejecutando `supabase-schema.sql` en Supabase.
- Configurar `SUPABASE_URL` y `SUPABASE_SERVICE_ROLE_KEY` solo como secrets del servidor.
- No incluir `.env`, `data/` ni ficheros `.sqlite` en Git.

## Render

`render.yaml` define un Web Service Node con build y start commands. El plan gratuito es adecuado para una boda de pequeño volumen, aunque puede suspender el servicio por inactividad. La persistencia está fuera de Render, en Supabase.

## Seguridad de Supabase

La API usa `@supabase/supabase-js` en el servidor con la service role key. Esa clave nunca debe enviarse al navegador ni añadirse a Git. RLS permanece activado y no se crea una política pública de lectura; solo el servidor autorizado accede a los invitados.

## Verificación posterior

- Abrir la URL pública y completar un RSVP de prueba sintético.
- Entrar en administración y confirmar que aparecen estado y alergias.
- Editar y borrar la respuesta de prueba.
- Eliminar los datos de prueba antes de compartir el enlace.