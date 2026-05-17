import express, { NextFunction, Request, Response } from "express";
import path from "node:path";
import { PUBLIC_DIR } from "./config/serverConfig";
import { enrollmentRoutes } from "./routes/enrollmentRoutes";

export function createApp(): express.Express {
  const app = express();

  app.use(express.json({ limit: "20mb" }));
  app.use(express.urlencoded({ extended: true }));

  app.use(express.static(PUBLIC_DIR, {
    setHeaders: (res, filePath) => {
      if (/\.(html|js|css|json)$/i.test(filePath)) {
        res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate");
      }
    },
  }));

  app.get("/", (_req, res) => {
    res.sendFile(path.join(PUBLIC_DIR, "index.html"));
  });

  app.use("/api", enrollmentRoutes);

  app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
    console.error(err);
    res.status(500).json({ ok: false, error: err.message || "Server error" });
  });

  return app;
}
