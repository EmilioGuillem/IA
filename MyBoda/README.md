<!-- AI_DISCLAIMER v1.0 -->
# MyBoda

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

Aplicación RSVP para la boda del 29 de mayo de 2027.

## Desarrollo local

1. Instala Node.js 20 o superior.
2. Ejecuta `npm install` dentro de esta carpeta.
3. Copia `.env.example` como `.env` y cambia `ADMIN_PASSWORD`.
4. Ejecuta `npm run dev`.
5. Abre `http://localhost:5173`.

La base de datos SQLite se crea automáticamente en `data/myboda.sqlite`. Esa carpeta no debe publicarse en el repositorio si contiene respuestas reales.

## Funcionalidades

- Formulario público para confirmar, rechazar o dejar pendiente la asistencia.
- Añadir varios invitados en una misma respuesta.
- Detección previa de nombres duplicados.
- Registro de alergias e intolerancias por invitado, con resumen en administración.
- Login de administrador mediante `ADMIN_PASSWORD`.
- Panel con búsqueda, filtros, estadísticas, edición y borrado.

## Publicación

El servidor necesita un volumen persistente para la base SQLite y HTTPS para proteger la sesión de administración. Antes de publicar, cambia siempre la contraseña por defecto y configura `DATABASE_PATH` en una ruta persistente.