# Netlify frontend deployment

The root `netlify.toml` sets the base directory to `app/web`, the publish directory to `.`, and no build command. This folder contains `index.html`, `style.css`, `app.js`, and `research.js`. No Python dependencies need to be installed to publish these static files.

Push this configuration to the branch connected to Netlify and trigger a new production deploy. For a manual drag-and-drop deployment, upload the `app/web` folder itself. Repository configuration only affects Git-based builds; it does not change an already uploaded manual deployment.

This fixes the missing homepage when the wrong folder was published. It does not host the Python API. The current frontend requests `/api/*`; these will remain unavailable until an API backend is deployed and a same-origin proxy is configured. Do not add a catch-all HTML redirect for API requests: it would conceal missing endpoints as JSON parsing errors.

Before enabling the full application publicly, the FastAPI runtime must be installed and tested; its loopback-only host/origin checks must be configured for the actual deployment; session cookies must be configured for HTTPS; and both SQLite files need persistent private storage (or a database migration). Keep credentials and databases outside the static publish folder. The existing local server must not be treated as a production deployment simply by changing its bind address.

No live backend URL, hosting account or persistent disk has been configured in this repository. There is no claim that publishing the frontend alone enables live data, sign-in or review writes.
