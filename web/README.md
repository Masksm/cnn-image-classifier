# Terrain AI Web Frontend

Next.js frontend for the CNN image classifier. Set `NEXT_PUBLIC_API_URL` to the public URL of the separate FastAPI prediction service, without a trailing slash.

Run locally:

```bash
npm install
npm run dev
```

The interface uploads images to `POST {NEXT_PUBLIC_API_URL}/predict` using multipart field `file`. The API must return `{"predictions":[{"class":"forest","confidence":0.98}]}`.
